import tensorflow as tf
import math
from train import batch_size, na, nl, img_size, stride, balance
from train import loss_box, loss_obj, loss_cls
from train import batch_no_constant, anchor_no_constant, anchors_reshape, anchor_t, anchors_constant, layer_no_constant
from train import val_batch_no_constant, val_layer_no_constant
from util import *
from params import *

#In param: 
#    p - predictions of the model, list of three detection level.
#    labels - the label of the object, dimension [batch, boxnum, 5(class, xywh)]
#Out param:
#    results - list of the suggest positive samples for three detection level. 
#        dimension for each element: [sample_number, 5(batch_no, anch_no, x, y, class)]
#    anch - list of the anchor wh ratio for the positive samples
#        dimension for each element: [sample_number, anchor_w, anchor_h]
@tf.function(
    input_signature=(
        [tf.TensorSpec(shape=[batch_size, None, 5], dtype=tf.float32)]
    )
)
def tf_find_3_positive(labels):
    batch_no = tf.zeros_like(labels)[...,0:1] + batch_no_constant
    targets = tf.concat((batch_no, labels), axis=-1)    #targets dim [batch,box_num,6]
    targets = tf.reshape(targets, [batch_size, 1, -1, 6])   #targets dim [batch,1,box_num,6]
    targets = tf.tile(targets, [1,na,1,1])
    anchor_no = anchor_no_constant + tf.reshape(tf.zeros_like(batch_no), [batch_size, 1, -1, 1])
    targets = tf.concat([targets,anchor_no], axis=-1)   #targets dim [batch,na,box_num,7(batch_no, cls, xywh, anchor_no)]

    g = 0.5  # bias
    offsets = tf.expand_dims(tf.constant([[0.,0.], [-1.,0.], [0.,-1.], [1.,0.], [0.,1.]]), axis=0)  #offset dim [1,5,2]

    gain = tf.constant([[1.,1.,80.,80.,80.,80.,1.], [1.,1.,40.,40.,40.,40.,1.], [1.,1.,20.,20.,20.,20.,1.]])

    results = tf.TensorArray(tf.int32, size=nl, dynamic_size=False)
    anch = tf.TensorArray(tf.float32, size=nl, dynamic_size=False)

    for i in tf.range(nl):
        t = targets * tf.gather(gain, i)
        r = t[..., 4:6] / tf.gather(anchors_reshape, i)
        r_reciprocal = tf.math.reciprocal_no_nan(r)      #1/r
        r_max = tf.reduce_max(tf.math.maximum(r, r_reciprocal), axis=-1)
        mask_t = tf.logical_and(r_max<anchor_t, r_max>0)
        t = t[mask_t]
        # Offsets
        gxy = t[:, 2:4]  # grid xy
        #gxi = gain[[2, 3]] - gxy  # inverse    
        gxi = tf.gather(gain, i)[2:4] - gxy
        mask_xy = tf.concat([
            tf.ones([tf.shape(t)[0], 1], dtype=tf.bool),
            ((gxy % 1. < g) & (gxy > 1.)),
            ((gxi % 1. < g) & (gxi > 1.))
        ], axis=1)
        t = tf.repeat(tf.expand_dims(t, axis=1), 5, axis=1)[mask_xy]
        offsets_xy = (tf.expand_dims(tf.zeros_like(gxy, dtype=tf.float32), axis=1) + offsets)[mask_xy]
        xy = t[...,2:4] + offsets_xy
        from_which_layer = tf.ones_like(t[...,0:1]) * tf.dtypes.cast(i, tf.float32)
        results = results.write(i, tf.dtypes.cast(tf.concat([t[...,0:1], t[...,-1:], xy[...,1:2], xy[...,0:1], t[...,1:2], from_which_layer], axis=-1), tf.int32))
        anch = anch.write(i, tf.gather(tf.gather(anchors_constant, i), tf.dtypes.cast(t[...,-1], tf.int32)))
    return results.concat(), anch.concat()

@tf.function(
    input_signature=([
        tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 4], dtype=tf.float32)
    ])
)
def box_iou(box1, box2):
    area1 = (box1[:,2]-box1[:,0])*(box1[:,3]-box1[:,1])
    area2 = (box2[:,2]-box2[:,0])*(box2[:,3]-box2[:,1])

    intersect_wh = tf.math.minimum(box1[:,None,2:], box2[:,2:]) - tf.math.maximum(box1[:,None,:2], box2[:,:2])
    intersect_wh = tf.clip_by_value(intersect_wh, clip_value_min=0, clip_value_max=img_size)
    intersect_area = intersect_wh[...,0]*intersect_wh[...,1]
    
    iou = intersect_area/(area1[:,None]+area2-intersect_area)
    return iou

@tf.function(
    input_signature=([
        tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 4], dtype=tf.float32)
    ])
)
def bbox_ciou(box1, box2):
    eps=1e-7
    b1_x1, b1_x2 = box1[:,0]-box1[:,2]/2, box1[:,0]+box1[:,2]/2
    b1_y1, b1_y2 = box1[:,1]-box1[:,3]/2, box1[:,1]+box1[:,3]/2
    b2_x1, b2_x2 = box2[:,0]-box2[:,2]/2, box2[:,0]+box2[:,2]/2
    b2_y1, b2_y2 = box2[:,1]-box2[:,3]/2, box2[:,1]+box2[:,3]/2
    
    # Intersection area
    inter = tf.clip_by_value(
        tf.math.minimum(b1_x2, b2_x2) - tf.math.maximum(b1_x1, b2_x1), 
        clip_value_min=0, 
        clip_value_max=tf.float32.max) * tf.clip_by_value(
        tf.math.minimum(b1_y2, b2_y2) - tf.math.maximum(b1_y1, b2_y1), 
        clip_value_min=0, 
        clip_value_max=tf.float32.max)
    
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    
    cw = tf.math.maximum(b1_x2, b2_x2) - tf.math.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = tf.math.maximum(b1_y2, b2_y2) - tf.math.minimum(b1_y1, b2_y1)  # convex height
    
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
    
    v = (4 / math.pi ** 2) * tf.math.pow(tf.math.atan(w2 / (h2 + eps)) - tf.math.atan(w1 / (h1 + eps)), 2)
    alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)

@tf.function(
    input_signature=([
        tf.TensorSpec(shape=[batch_size, na, None, 85], dtype=tf.float32),
        tf.TensorSpec(shape=[batch_size, None, 5], dtype=tf.float32)
    ])
)
def tf_build_targets(p, labels):
    results, anch = tf_find_3_positive(labels)

    #stride = tf.constant([8., 16., 32.])
    grids = tf.dtypes.cast(img_size/stride, tf.int32)

    pxyxys = tf.TensorArray(tf.float32, size=nl, dynamic_size=False)
    p_obj = tf.TensorArray(tf.float32, size=nl, dynamic_size=True, element_shape=[None, 1])
    p_cls = tf.TensorArray(tf.float32, size=nl, dynamic_size=False)
    all_idx = tf.TensorArray(tf.int32, size=nl, dynamic_size=False)
    from_which_layer = tf.TensorArray(tf.int32, size=nl, dynamic_size=False)
    all_anch = tf.TensorArray(tf.float32, size=nl, dynamic_size=False)
    
    matching_idxs = tf.TensorArray(tf.int32, size=batch_size, dynamic_size=False)
    matching_targets = tf.TensorArray(tf.float32, size=batch_size, dynamic_size=False)
    matching_anchs = tf.TensorArray(tf.float32, size=batch_size, dynamic_size=False)
    matching_layers = tf.TensorArray(tf.int32, size=batch_size, dynamic_size=False)

    for i in tf.range(nl):
        idx_mask = results[...,-1]==i
        idx = tf.boolean_mask(results, idx_mask)
        layer_mask = layer_no_constant[...,0]==i
        grid_no = tf.gather(grids, i)
        pl = tf.boolean_mask(p, layer_mask)
        pl = tf.reshape(pl, [batch_size, na, grid_no, grid_no, -1])
        pi = tf.gather_nd(pl, idx[...,0:4])
        anchors_p = tf.boolean_mask(anch, idx_mask)
        p_obj = p_obj.write(i, pi[...,4:5])
        p_cls = p_cls.write(i, pi[...,5:])
        gij = tf.dtypes.cast(tf.concat([idx[...,3:4], idx[...,2:3]], axis=-1), tf.float32)
        pxy = (tf.math.sigmoid(pi[...,:2])*2-0.5+gij)*tf.dtypes.cast(tf.gather(stride, i), tf.float32)
        pwh = (tf.math.sigmoid(pi[...,2:4])*2)**2*anchors_p*tf.dtypes.cast(tf.gather(stride, i), tf.float32)
        pxywh = tf.concat([pxy, pwh], axis=-1)
        pxyxy = xywh2xyxy(pxywh)
        pxyxys = pxyxys.write(i, pxyxy)
        all_idx = all_idx.write(i, idx[...,0:4])
        from_which_layer = from_which_layer.write(i, idx[..., -1:])
        all_anch = all_anch.write(i, tf.boolean_mask(anch, idx_mask))

    pxyxys = pxyxys.concat()
    p_obj = p_obj.concat()
    p_cls = p_cls.concat()
    all_idx = all_idx.concat()
    from_which_layer = from_which_layer.concat()
    all_anch = all_anch.concat()

    for i in tf.range(batch_size):
        batch_mask = all_idx[...,0]==i
        if tf.math.reduce_sum(tf.dtypes.cast(batch_mask, tf.int32)) > 0:
            pxyxy_i = tf.boolean_mask(pxyxys, batch_mask)
            target_mask = labels[i][...,3]>0
            target = tf.boolean_mask(labels[i], target_mask)
            txywh = target[...,1:] * img_size
            txyxy = xywh2xyxy(txywh)
            pair_wise_iou = box_iou(txyxy, pxyxy_i)
            pair_wise_iou_loss = -tf.math.log(pair_wise_iou + 1e-8)

            top_k, _ = tf.math.top_k(pair_wise_iou, tf.math.minimum(10, tf.shape(pair_wise_iou)[1]))
            dynamic_ks = tf.clip_by_value(
                tf.dtypes.cast(tf.math.reduce_sum(top_k, axis=-1), tf.int32),
                clip_value_min=1, 
                clip_value_max=10)

            gt_cls_per_image = tf.tile(
                tf.expand_dims(
                    tf.one_hot(
                        tf.dtypes.cast(target[...,0], tf.int32), nc),
                    axis = 1),
                [1,tf.shape(pxyxy_i)[0],1])

            num_gt = tf.shape(target)[0]
            cls_preds_ = (
                tf.math.sigmoid(tf.tile(tf.expand_dims(tf.boolean_mask(p_cls, batch_mask), 0), [num_gt, 1, 1])) *
                tf.math.sigmoid(tf.tile(tf.expand_dims(tf.boolean_mask(p_obj, batch_mask), 0), [num_gt, 1, 1])))    #dimension [labels_number, positive_targets_number, 80]
            y = tf.math.sqrt(cls_preds_)
            pair_wise_cls_loss = tf.math.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels = gt_cls_per_image,
                    logits = tf.math.log(y/(1-y))),
                axis = -1)

            cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = tf.zeros_like(cost)      #dimension [labels_number, positive_targets_number]

            matching_idx = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
            for gt_idx in tf.range(num_gt):
                _, pos_idx = tf.math.top_k(
                    -cost[gt_idx], k=dynamic_ks[gt_idx], sorted=True)
                X,Y = tf.meshgrid(gt_idx, pos_idx)
                matching_idx = matching_idx.write(gt_idx, tf.dtypes.cast(tf.concat([X,Y], axis=-1), tf.int64))

            matching_idx = matching_idx.concat()
            '''
            matching_matrix = tf.scatter_nd(
                matching_idx, 
                tf.ones(tf.shape(matching_idx)[0]), 
                tf.dtypes.cast(tf.shape(cost), tf.int64))
            '''
            matching_matrix = tf.sparse.to_dense(
                tf.sparse.reorder(
                    tf.sparse.SparseTensor(
                        indices=tf.dtypes.cast(matching_idx, tf.int64), 
                        values=tf.ones(tf.shape(matching_idx)[0]), 
                        dense_shape=tf.dtypes.cast(tf.shape(cost), tf.int64))
                )
            )

            anchor_matching_gt = tf.reduce_sum(matching_matrix, axis=0)    #dimension [positive_targets_number]
            mask_1 = anchor_matching_gt>1     #it means one target match to several ground truths

            if tf.reduce_sum(tf.dtypes.cast(mask_1, tf.int32)) > 0:   #There is at least one positive target that predict several ground truth  
                #Get the lowest cost of the serveral ground truth of the target
                #For example, there are 100 targets and 10 ground truths.
                #The #5 target match to the #2 and #3 ground truth, the related cost are 10 for #2 and 20 for #3
                #Then it will select #2 gound truth for the #5 target.
                #mask_1 dimension [positive_targets_number]
                #tf.boolean_mask(cost, mask_1, axis=1), dimension [ground_truth_numer, targets_predict_sevearl_GT_number]
                cost_argmin = tf.math.argmin(
                    tf.boolean_mask(cost, mask_1, axis=1), axis=0)  #in above example, the cost_argmin is [2]
                m = tf.dtypes.cast(mask_1, tf.float32)
                _, target_indices = tf.math.top_k(
                    m, 
                    k=tf.dtypes.cast(tf.math.reduce_sum(m), tf.int32))  #in above example, the target_indices is [5]
                #So will set the index [2,5] of matching_matrix to 1, and set the other elements of [:,5] to 0
                target_matching_gt_indices = tf.concat(
                    [tf.reshape(tf.dtypes.cast(cost_argmin, tf.int32), [-1,1]), tf.reshape(target_indices, [-1,1])], 
                    axis=1)          
                matching_matrix = tf.multiply(
                    matching_matrix,
                    tf.repeat(tf.reshape(tf.dtypes.cast(anchor_matching_gt<=1, tf.float32), [1,-1]), tf.shape(cost)[0], axis=0))
                target_value = tf.sparse.to_dense(
                    tf.sparse.reorder(
                        tf.sparse.SparseTensor(
                            indices=tf.dtypes.cast(target_matching_gt_indices, tf.int64),
                            values=tf.ones(tf.shape(target_matching_gt_indices)[0]),
                            dense_shape=tf.dtypes.cast(tf.shape(matching_matrix), tf.int64)
                        )
                    )
                )
                matching_matrix = tf.add(matching_matrix, target_value)

            fg_mask_inboxes = tf.math.reduce_sum(matching_matrix, axis=0)>0.  #The mask for the targets that will use to predict
            if tf.shape(tf.boolean_mask(matching_matrix, fg_mask_inboxes, axis=1))[0]>0:
                matched_gt_inds = tf.math.argmax(tf.boolean_mask(matching_matrix, fg_mask_inboxes, axis=1), axis=0)  #Get the related gt number for the target

                all_idx_i = tf.boolean_mask(tf.boolean_mask(all_idx, batch_mask), fg_mask_inboxes)
                from_which_layer_i = tf.boolean_mask(tf.boolean_mask(from_which_layer, batch_mask), fg_mask_inboxes)
                all_anch_i = tf.boolean_mask(tf.boolean_mask(all_anch, batch_mask), fg_mask_inboxes)

                matching_idxs = matching_idxs.write(i, all_idx_i)
                matching_layers = matching_layers.write(i, from_which_layer_i)
                matching_anchs = matching_anchs.write(i, all_anch_i )
                matching_targets = matching_targets.write(i, tf.gather(target, matched_gt_inds))
            else:
                matching_idxs = matching_idxs.write(i, tf.constant([[-1,-1,-1,-1]], dtype=tf.int32))
                matching_layers = matching_layers.write(i, tf.constant([[-1]], dtype=tf.int32))
                matching_anchs = matching_anchs.write(i, tf.constant([[-1, -1]], dtype=tf.float32))
                matching_targets = matching_targets.write(i, tf.constant([[-1, -1, -1, -1, -1]], dtype=tf.float32))                                    
        
        else:
            matching_idxs = matching_idxs.write(i, tf.constant([[-1,-1,-1,-1]], dtype=tf.int32))
            matching_layers = matching_layers.write(i, tf.constant([[-1]], dtype=tf.int32))
            matching_anchs = matching_anchs.write(i, tf.constant([[-1, -1]], dtype=tf.float32))
            matching_targets = matching_targets.write(i, tf.constant([[-1, -1, -1, -1, -1]], dtype=tf.float32))
        
    matching_idxs = matching_idxs.concat()
    matching_layers = matching_layers.concat()
    matching_anchs = matching_anchs.concat()
    matching_targets = matching_targets.concat()
    filter_mask = matching_idxs[:,0]!=-1
    matching_idxs = tf.boolean_mask(matching_idxs, filter_mask)
    matching_layers = tf.boolean_mask(matching_layers, filter_mask)
    matching_anchs = tf.boolean_mask(matching_anchs, filter_mask)
    matching_targets = tf.boolean_mask(matching_targets, filter_mask)
    
    #return pxyxys, all_idx, matching_idx, matching_matrix, all_idx_i, cost, pair_wise_iou, from_which_layer_i
    return matching_idxs, matching_layers, matching_anchs, matching_targets

@tf.function(
    input_signature=([
        tf.TensorSpec(shape=[batch_size, na, None, 85], dtype=tf.float32),
        tf.TensorSpec(shape=[batch_size, None, 5], dtype=tf.float32)
    ])
)
def tf_loss_func(p, labels):
    matching_idxs, matching_layers, matching_anchs, matching_targets = tf_build_targets(p, labels)
    lcls, lbox, lobj = tf.zeros(1), tf.zeros(1), tf.zeros(1)
    
    grids = img_size//stride
    for i in tf.range(nl):
        layer_mask = layer_no_constant[...,0]==i
        grid = tf.gather(grids, i)
        pi = tf.reshape(tf.boolean_mask(p, layer_mask), [batch_size, na, grid, grid, -1])
        matching_layer_mask = matching_layers[:,0]==i
        if tf.reduce_sum(tf.dtypes.cast(matching_layer_mask, tf.int32))==0:
            continue
        m_idxs = tf.boolean_mask(matching_idxs, matching_layer_mask)
        if tf.shape(m_idxs)[0]==0:
            continue
        m_targets = tf.boolean_mask(matching_targets, matching_layer_mask)
        m_anchs = tf.boolean_mask(matching_anchs, matching_layer_mask)
        ps = tf.gather_nd(pi, m_idxs)
        pxy = tf.math.sigmoid(ps[:,:2])*2-0.5
        pwh = (tf.math.sigmoid(ps[:,2:4])*2)**2*m_anchs
        pbox = tf.concat([pxy,pwh], axis=-1)
        #selected_tbox = tf.gather_nd(labels, matching_targets[i])[:, 1:]
        selected_tbox = m_targets[:, 1:]
        selected_tbox = tf.multiply(selected_tbox, tf.dtypes.cast(grid, tf.float32))
        tbox_grid = tf.concat([
            tf.dtypes.cast(m_idxs[:,3:4], tf.float32),
            tf.dtypes.cast(m_idxs[:,2:3], tf.float32),
            tf.zeros((tf.shape(m_idxs)[0],2))], 
            axis=-1)
        selected_tbox = tf.subtract(selected_tbox, tbox_grid)
        iou = bbox_ciou(pbox, selected_tbox)
        lbox += tf.math.reduce_mean(1.0 - iou)  # iou loss

        # Objectness
        tobj = tf.sparse.to_dense(
            tf.sparse.reorder(
                tf.sparse.SparseTensor(
                    indices = tf.dtypes.cast(m_idxs, tf.int64),
                    values = (1.0 - gr) + gr * tf.clip_by_value(tf.stop_gradient(iou), clip_value_min=0, clip_value_max=tf.float32.max),
                    dense_shape = tf.dtypes.cast(tf.shape(pi[..., 0]), tf.int64)
                )
            ), validate_indices=False
        )

        # Classification

        tcls = tf.one_hot(
            indices = tf.dtypes.cast(m_targets[:,0], tf.int32),
            depth = 80,
            dtype = tf.float32
        )
        
        lcls += tf.math.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = tcls,
                logits = ps[:, 5:]
            )
        )
        '''
        lcls += tf.math.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = tf.dtypes.cast(m_targets[:,0], tf.int32),
                logits = ps[:, 5:]
            )    
        )
        '''
        obji = tf.math.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = tobj,
                logits = pi[..., 4]
            )
        )

        lobj += obji * tf.gather(balance, i) 
        
    lbox *= loss_box
    lobj *= loss_obj
    lcls *= loss_cls

    loss = (lbox + lobj + lcls) * batch_size

    return loss

@tf.function(
    input_signature=([
        tf.TensorSpec(shape=[None, na, 8400, 85], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, 5], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 2], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.int32),
    ])
)
def tf_predict_func(predictions, labels, imgs_hw, imgs_id):
    grids = img_size // stride
    batch_size = tf.shape(predictions)[0]
    confidence_threshold = 0.2
    probabilty_threshold = 0.8
    all_predict_result = tf.TensorArray(tf.float32, size=nl, dynamic_size=False) 
    boxes_result = tf.TensorArray(tf.float32, size=0, dynamic_size=True) 
    imgs_info = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    for i in tf.range(nl):
        grid = tf.gather(grids, i)
        grid_x, grid_y = tf.meshgrid(tf.range(grid, dtype=tf.float32), tf.range(grid, dtype=tf.float32))
        grid_x = tf.reshape(grid_x, [-1, 1])
        grid_y = tf.reshape(grid_y, [-1, 1])
        #grid_xy = tf.concat([grid_y, grid_x], axis=-1)
        grid_xy = tf.concat([grid_x, grid_y], axis=-1)
        grid_xy = tf.reshape(grid_xy, [1,1,-1,2])
        layer_mask = val_layer_no_constant[...,0]==i
        #grid = tf.gather(grids, i)
        predict_layer = tf.boolean_mask(predictions, layer_mask)
        predict_layer = tf.reshape(predict_layer, [batch_size, na, -1, 85])
        predict_conf = tf.math.sigmoid(predict_layer[...,4:5])
        predict_xy = (tf.math.sigmoid(predict_layer[...,:2])*2-0.5 + \
            tf.dtypes.cast(grid_xy,tf.float32))*tf.dtypes.cast(tf.gather(stride, i), tf.float32)
        predict_wh = (tf.math.sigmoid(predict_layer[...,2:4])*2)**2*\
            tf.reshape(tf.gather(anchors_constant,i), [1,na,1,2])*\
            tf.dtypes.cast(tf.gather(stride, i), tf.float32)
        predict_xywh = tf.concat([predict_xy, predict_wh], axis=-1)
        predict_xyxy = xywh2xyxy(predict_xywh)
        predict_cls = tf.reshape(tf.argmax(predict_layer[...,5:], axis=-1), [batch_size, na, -1, 1])
        predict_cls = tf.dtypes.cast(predict_cls, tf.float32)
        predict_proba = tf.nn.sigmoid(
            tf.reduce_max(
                predict_layer[...,5:], axis=-1, keepdims=True
            )
        )
        batch_no = tf.expand_dims(tf.tile(tf.gather(val_batch_no_constant, tf.range(batch_size)), [1,na,grid*grid]), -1)
        predict_result = tf.concat([batch_no, predict_conf, predict_xyxy, predict_cls, predict_proba], axis=-1)
        mask = tf.math.logical_and(
            predict_result[...,1]>=confidence_threshold,
            predict_result[...,-1]>=probabilty_threshold
        )
        predict_result = tf.boolean_mask(predict_result, mask)
        #tf.print(tf.shape(predict_result))
        if tf.shape(predict_result)[0] > 0:
            all_predict_result = all_predict_result.write(i, predict_result)
            #tf.print(tf.shape(predict_result))
        else:
            all_predict_result = all_predict_result.write(i, tf.zeros(shape=[1,8]))
    all_predict_result = all_predict_result.concat()
    #return all_predict_result
          
    for i in tf.range(batch_size):
        batch_mask = tf.math.logical_and(
            all_predict_result[...,0]==tf.dtypes.cast(i, tf.float32),
            all_predict_result[...,1]>0
        )
        predict_true_box = tf.boolean_mask(all_predict_result, batch_mask)
        if tf.shape(predict_true_box)[0]==0:
            continue
        original_hw = tf.dtypes.cast(tf.gather(imgs_hw, i), tf.float32)
        ratio = tf.dtypes.cast(tf.reduce_max(original_hw/img_size), tf.float32)
        predict_classes, _ = tf.unique(predict_true_box[:,6])
        #predict_classes_list = tf.unstack(predict_classes)
        #for class_id in predict_classes_list:
        for j in tf.range(tf.shape(predict_classes)[0]):
            #class_mask = tf.math.equal(predict_true_box[:, 6], class_id)
            class_mask = tf.math.equal(predict_true_box[:, 6], tf.gather(predict_classes, j))
            predict_true_box_class = tf.boolean_mask(predict_true_box, class_mask)
            predict_true_box_xy = predict_true_box_class[:, 2:6]
            predict_true_box_score = predict_true_box_class[:, 7]*predict_true_box_class[:, 1]
            #predict_true_box_score = predict_true_box_class[:, 1]
            selected_indices = tf.image.non_max_suppression(
                predict_true_box_xy,
                predict_true_box_score,
                100,
                iou_threshold=0.2
                #score_threshold=confidence_threshold
            )
            #Shape [box_num, 7]
            selected_boxes = tf.gather(predict_true_box_class, selected_indices) 
            #boxes_result = boxes_result.write(boxes_result.size(), selected_boxes)
            boxes_xyxy = selected_boxes[:,2:6]*ratio
            boxes_x1 = tf.clip_by_value(boxes_xyxy[:,0:1], 0., original_hw[1])
            boxes_x2 = tf.clip_by_value(boxes_xyxy[:,2:3], 0., original_hw[1])
            boxes_y1 = tf.clip_by_value(boxes_xyxy[:,1:2], 0., original_hw[0])
            boxes_y2 = tf.clip_by_value(boxes_xyxy[:,3:4], 0., original_hw[0])
            boxes_w = boxes_x2 - boxes_x1
            boxes_h = boxes_y2 - boxes_y1
            boxes = tf.concat([selected_boxes[:,0:2], boxes_x1, boxes_y1, boxes_w, boxes_h, selected_boxes[:,6:8]], axis=-1)
            boxes_result = boxes_result.write(boxes_result.size(), boxes)
        img_id = tf.gather(imgs_id, i)
        imgs_info = imgs_info.write(imgs_info.size(), tf.reshape(tf.stack([i, img_id]), [-1,2]))
    if boxes_result.size()==0:
        boxes_result = boxes_result.write(0, tf.zeros(shape=[1,8]))
    if imgs_info.size()==0:
        imgs_info = imgs_info.write(0, tf.dtypes.cast(tf.zeros(shape=[1,2]), tf.int32))

    return boxes_result.concat(), imgs_info.concat()