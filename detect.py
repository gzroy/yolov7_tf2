import tensorflow as tf
from tensorflow.keras import mixed_precision
from util import xywh2xyxy, load_image
import argparse
from yolo import create_model
import cv2
import numpy as np
import random
import os
from params import classnames
from matplotlib import pyplot as plt
mixed_precision.set_global_policy('mixed_float16')

class Detect:
    def __init__(self, batch_size=1, img_size=640, stride=[8,16,32], max_nms=30000, max_det=300, ckpt=None, conf_thres=0.25, iou_thres=0.65):
        self.img_size = 640
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.nl = len(stride)
        self.stride = tf.constant(stride)
        self.max_nms = max_nms
        self.max_det = max_det
        self.grid_xy_layer = []
        self.grids = img_size // self.stride
        self.na = 3
        self.anchors_constant = tf.constant(
            [[[ 1.50000,  2.00000],
            [ 2.37500,  4.50000],
            [ 5.00000,  3.50000]],
            [[ 2.25000,  4.68750],
            [ 4.75000,  3.43750],
            [ 4.50000,  9.12500]],
            [[ 4.43750,  3.43750],
            [ 6.00000,  7.59375],
            [14.34375, 12.53125]]])
        self.batch_size = batch_size
        self.set_batch(batch_size)
        self.model = create_model()
        ema = tf.train.ExponentialMovingAverage(decay=0.9999, num_updates=0)
        ema.apply(self.model.trainable_variables)

        averages = [ema.average(var) for var in self.model.trainable_variables]
        model_trainable_vars = [var for var in self.model.trainable_variables]
        model_non_trainable_vars = [var for var in self.model.non_trainable_variables]
        checkpoint = tf.train.Checkpoint(
            model_trainable_weights=model_trainable_vars, 
            model_non_trainable_weights=model_non_trainable_vars, 
            averaged_weights=averages)

        loadStatus = checkpoint.restore(ckpt)
        for i, avg_var in enumerate(averages):
            model_trainable_vars[i].assign(avg_var)
        print("Successfully loaded checkpoint {}".format(ckpt))

    def set_batch(self, batch_size):
        self.batch_size = batch_size
        self.batch_no_constant = tf.reshape(tf.range(batch_size, dtype=tf.float32), [batch_size, 1, 1])
        self.anchor_no_constant = tf.reshape(tf.tile(tf.range(self.na, dtype=tf.float32), [batch_size]), [batch_size, self.na, 1, 1])
        layer_no_constant = tf.repeat([x for x in range(self.nl)], [a*a for a in [int(self.img_size/b) for b in self.stride]])
        layer_no_constant = tf.reshape(layer_no_constant, [1,-1,1])
        layer_no_constant = tf.repeat(layer_no_constant, [batch_size*self.na], axis=0)
        self.layer_no_constant = tf.reshape(layer_no_constant, [batch_size, self.na, -1, 1])


    @tf.function
    def predict(self, predictions, imgs_hw):
        all_predict_result = tf.TensorArray(tf.float32, size=self.nl, dynamic_size=False) 
        boxes_result = tf.TensorArray(tf.float32, size=0, dynamic_size=True) 
        for i in tf.range(self.nl):
            #grid_xy = self.grid_xy_layer[i]
            grid = tf.gather(self.grids, i)
            grid_x, grid_y = tf.meshgrid(tf.range(grid, dtype=tf.float32), tf.range(grid, dtype=tf.float32))
            grid_x = tf.reshape(grid_x, [-1, 1])
            grid_y = tf.reshape(grid_y, [-1, 1])
            grid_xy = tf.concat([grid_x, grid_y], axis=-1)
            grid_xy = tf.reshape(grid_xy, [1,1,-1,2])
            layer_mask = self.layer_no_constant[...,0]==i
            predict_layer = tf.boolean_mask(predictions, layer_mask)
            predict_layer = tf.reshape(predict_layer, [self.batch_size, self.na, -1, 85])
            predict_conf = tf.math.sigmoid(predict_layer[...,4:5])
            predict_xy = (tf.math.sigmoid(predict_layer[...,:2])*2-0.5 + \
                tf.dtypes.cast(grid_xy,tf.float32))*tf.dtypes.cast(tf.gather(self.stride, i), tf.float32)
            predict_wh = (tf.math.sigmoid(predict_layer[...,2:4])*2)**2*\
                tf.reshape(tf.gather(self.anchors_constant,i), [1,self.na,1,2])*\
                tf.dtypes.cast(tf.gather(self.stride, i), tf.float32)
            predict_xywh = tf.concat([predict_xy, predict_wh], axis=-1)
            predict_xyxy = xywh2xyxy(predict_xywh)
            predict_cls_conf = tf.nn.sigmoid(predict_layer[...,5:]) * predict_conf
            batch_no = tf.expand_dims(tf.tile(tf.gather(self.batch_no_constant, tf.range(self.batch_size)), [1,self.na,grid*grid]), -1)
            predict_result = tf.concat([batch_no, predict_xyxy], axis=-1)
            mask = predict_conf[..., 0] > self.conf_thres
            predict_result = tf.boolean_mask(predict_result, mask)
            predict_cls_conf = tf.boolean_mask(predict_cls_conf, mask)
            mask = predict_cls_conf[..., 0:] > self.conf_thres
            mask_indices = tf.where(mask)
            predict_result = tf.gather_nd(predict_result, mask_indices[..., :-1])
            predict_cls = tf.dtypes.cast(mask_indices[..., -1:], tf.float32)
            predict_score = tf.expand_dims(tf.gather_nd(predict_cls_conf, mask_indices), axis=-1)
            predict_result = tf.concat([predict_result, predict_cls, predict_score], axis=-1)
            if tf.shape(predict_result)[0] > 0:
                all_predict_result = all_predict_result.write(i, predict_result)
            else:
                all_predict_result = all_predict_result.write(i, tf.zeros(shape=[1,7]))
        all_predict_result = all_predict_result.concat()
            
        for i in tf.range(self.batch_size):
            batch_mask = tf.math.logical_and(
                all_predict_result[...,0]==tf.dtypes.cast(i, tf.float32),
                all_predict_result[...,-1]>0
            )
            predict_true_box = tf.boolean_mask(all_predict_result, batch_mask)
            if tf.shape(predict_true_box)[0]==0:
                continue
            if tf.shape(predict_true_box)[0]>self.max_nms:
                sort_result = tf.math.top_k(predict_true_box[..., -1], k=self.max_nms, sorted=False)
                predict_true_box = tf.gather(predict_true_box, sort_result.indices)
            boxes_result_img = tf.TensorArray(tf.float32, size=0, dynamic_size=True) 
            original_hw = tf.dtypes.cast(tf.gather(imgs_hw, i), tf.float32)
            ratio = tf.dtypes.cast(tf.reduce_max(original_hw/self.img_size), tf.float32)
            predict_classes, _ = tf.unique(predict_true_box[:,5])

            for j in tf.range(tf.shape(predict_classes)[0]):
                class_mask = tf.math.equal(predict_true_box[:, 5], tf.gather(predict_classes, j))
                predict_true_box_class = tf.boolean_mask(predict_true_box, class_mask)
                predict_true_box_xy = predict_true_box_class[:, 1:5]
                predict_true_box_score = predict_true_box_class[:, -1]
                selected_indices = tf.image.non_max_suppression(
                    predict_true_box_xy,
                    predict_true_box_score,
                    self.max_det,
                    iou_threshold=self.iou_thres,
                    score_threshold=self.conf_thres
                )

                selected_boxes = tf.gather(predict_true_box_class, selected_indices) 
                boxes_xyxy = selected_boxes[:,1:5]*ratio
                boxes_x1 = tf.clip_by_value(boxes_xyxy[:,0:1], 0., original_hw[1])
                boxes_x2 = tf.clip_by_value(boxes_xyxy[:,2:3], 0., original_hw[1])
                boxes_y1 = tf.clip_by_value(boxes_xyxy[:,1:2], 0., original_hw[0])
                boxes_y2 = tf.clip_by_value(boxes_xyxy[:,3:4], 0., original_hw[0])
                boxes_w = boxes_x2 - boxes_x1
                boxes_h = boxes_y2 - boxes_y1
                boxes = tf.concat([selected_boxes[:,0:1], selected_boxes[:,5:6], boxes_x1, boxes_y1, boxes_w, boxes_h, selected_boxes[:,-1:]], axis=-1)
                boxes_result_img = boxes_result_img.write(boxes_result_img.size(), boxes)
            boxes_result_img = boxes_result_img.concat()
            if tf.shape(boxes_result_img)[0]>self.max_det:
                sort_boxes_result = tf.math.top_k(boxes_result_img[..., -1], k=self.max_det, sorted=False)
                boxes_result_img = tf.gather(boxes_result_img, sort_boxes_result.indices)
            boxes_result = boxes_result.write(boxes_result.size(), boxes_result_img)
        if boxes_result.size()==0:
            boxes_result = boxes_result.write(0, tf.zeros(shape=[1,7]))
        return boxes_result.concat()

    def load_image(self, filename, augment, img_size=640):
        img = cv2.imread(filename)
        full_img = np.ones((img_size, img_size, 3), np.uint8) * 114
        h0, w0 = img.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h1, w1 = img.shape[:2]
        full_img[:h1, :w1, :] = img
        full_img = full_img[:, :, ::-1].transpose(2,0,1)
        full_img = full_img/255.
        full_img = np.expand_dims(full_img, axis=0)
        return full_img, (h0, w0), (h1, w1)  # img, hw_original, hw_resized

    def detect(self, filelist):
        imgs = []
        imgs_orighw = []
        for f in filelist:
            img, orighw, _ = self.load_image(f, False)
            imgs.append(img)
            imgs_orighw.append(orighw)
        if len(imgs) != self.batch_size:
            self.set_batch(len(imgs))
        imgs = np.concatenate(imgs, axis=0)
        imgs_orighw = np.array(imgs_orighw)
        predictions = self.model(imgs, training=False)
        detection_results = self.predict(predictions, imgs_orighw)
        return detection_results.numpy()

def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yolo')
    parser.add_argument('--ckpt', type=str, help='Specify the CKPT name to load the model')
    parser.add_argument('--input', type=str, help='Specify the input folder of imgs for detection')
    parser.add_argument('--output', type=str, help='Specify the output folder to save the detection imgs')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='Specify the confidence threshold for detection')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='Specify the iou threshold for nms')
    args = parser.parse_args()

    detect = Detect(ckpt=args.ckpt, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
    input_imgs = os.listdir(args.input)
    input_imgs = [(args.input+'/'+a) for a in input_imgs]
    results = detect.detect(input_imgs)
    results = results.astype(int)

    colors = class_colors(classnames)

    for i, imgfile in enumerate(input_imgs):
        bboxes = results[results[..., 0]==i]
        if bboxes.shape[0] == 0:
            continue
        img = cv2.imread(imgfile)
        for j in range(bboxes.shape[0]):
            cv2.rectangle(
                img, 
                (bboxes[j, 2], bboxes[j, 3]), ((bboxes[j, 2]+bboxes[j, 4]), (bboxes[j, 3]+bboxes[j,5])), 
                colors[classnames[bboxes[j, 1]]], 1)
            size = cv2.getTextSize(classnames[bboxes[j, 1]], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            txt_w, txt_h = size[0]
            top = bboxes[j, 3] - txt_h + 4
            if top < 0:
                top = bboxes[j, 3]+txt_h+1
            txt_top = bboxes[j, 3] - txt_h -5
            txt_bottom = bboxes[j, 3]
            if txt_top < 0:
                txt_top = bboxes[j, 3]
                txt_bottom =  bboxes[j, 3] + txt_h + 5
            cv2.rectangle(img, (bboxes[j, 2], txt_top), (bboxes[j, 2]+txt_w, txt_bottom), colors[classnames[bboxes[j, 1]]], -1)
            cv2.putText(img, classnames[bboxes[j, 1]], (bboxes[j, 2], top), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        savefilename = imgfile.split('/')[-1]
        img = img[:, :, ::-1]
        plt.imsave(args.output + '/'+savefilename, img)