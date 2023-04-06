import tensorflow as tf

hyp = {
    'degrees': 0.,
    'translate': 0.2,
    'scale': 0.9,
    'shear': 0.,
    'perspective': 0.,
    'mixup': 0.15,
    'paste_in': 0.15,
    'fliplr': 0.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4
}

img_size = 640
mosaic_border = [-img_size // 2, -img_size // 2]

nl = 3    # number of detection layer
na = 3    # number of anchors per detection layer
nc = 80   # number of classes
anchor_t = 4.
weight_decay = 0.0005

gr = 1.0
cn = 0.0
cp = 1.0

balance = tf.constant([4.0, 1.0, 0.4])    #weights for the three detection layers
loss_box = 0.05
loss_obj = 0.7
loss_cls = 0.3

batch_size = 32
val_batch_size = 8
stride = [img_size//80, img_size//40, img_size//20]
stride = tf.constant(stride)

anchors_constant = tf.constant(
    [[[ 1.50000,  2.00000],
     [ 2.37500,  4.50000],
     [ 5.00000,  3.50000]],
    [[ 2.25000,  4.68750],
     [ 4.75000,  3.43750],
     [ 4.50000,  9.12500]],
    [[ 4.43750,  3.43750],
     [ 6.00000,  7.59375],
     [14.34375, 12.53125]]])
anchors_reshape = tf.reshape(anchors_constant, [nl, 1, na, 1, 2]) 
batch_no_constant = tf.reshape(tf.range(batch_size, dtype=tf.float32), [batch_size, 1, 1])
anchor_no_constant = tf.reshape(tf.tile(tf.range(na, dtype=tf.float32), [batch_size]), [batch_size, na, 1, 1])
layer_no_constant = tf.repeat([x for x in range(nl)], [a*a for a in [int(img_size/b) for b in stride]])
layer_no_constant = tf.reshape(layer_no_constant, [1,-1,1])
val_layer_no_constant = tf.repeat(layer_no_constant, [val_batch_size*na], axis=0)
val_layer_no_constant = tf.reshape(val_layer_no_constant, [val_batch_size, na, -1, 1])
layer_no_constant = tf.repeat(layer_no_constant, [batch_size*na], axis=0)
layer_no_constant = tf.reshape(layer_no_constant, [batch_size, na, -1, 1])
val_batch_no_constant = tf.reshape(tf.range(val_batch_size, dtype=tf.float32), [val_batch_size, 1, 1])

classnames = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}