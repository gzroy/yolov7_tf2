import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

from multiprocessing import shared_memory, Queue
import multiprocessing as mp
from dataloader import Dataloader
import random
from random import sample
import time
import cv2
import pickle
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from yolo import create_model
from tensorflow.keras import mixed_precision
from params import *
from loss import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import argparse
from tqdm import tqdm

mixed_precision.set_global_policy('mixed_float16')

initial_warmup_steps = 1000
initial_lr = 0.01
total_train_images = 120000    #coco train images, around 11K images
maximum_batches = 120000//batch_size*300
#maximum_batches = 1200000
power = 4
 
train_image_dir = 'coco/images/train2017'
train_label_dir = 'coco/labels/train2017'
train_files = os.listdir(train_image_dir)
imgid_train = [int(filename[:-4]) for filename in train_files]
data_size = batch_size * 50
sample_len = data_size * 5
subprocess_num = 4
q = Queue()
shared_memory_size = int(1.5*data_size*1024*1024)   #500M
imgid_num_process = sample_len//subprocess_num

val_image_dir = 'coco/images/val2017'
val_label_path = 'coco/labels/val2017/'
val_imgids = [139,285,632,724,776,785,802,872,885,1000,1268,1296]

def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}

def augment_data(imgids, datasize, memory_name, offset, q):
    dataset = Dataloader(img_size, train_image_dir, train_label_dir, imgids, hyp)
    traindata = dataset.generateTrainData(datasize)
    traindata_obj = pickle.dumps(traindata, protocol=pickle.HIGHEST_PROTOCOL)
    existing_shm = shared_memory.SharedMemory(name=memory_name)
    existing_shm.buf[offset:offset+len(traindata_obj)] = traindata_obj 
    q.put((offset, offset+len(traindata_obj)))
    existing_shm.close()

def merge_subprocess(q, subprocess_num, memory_name):
    results = []
    while(True):
        msg = q.get()
        if msg is not None:
            results.append(msg)
        if len(results)>=subprocess_num:
            break
        else:
            time.sleep(1)
    existing_shm = shared_memory.SharedMemory(name=memory_name)
    merge_data =  []
    for result in results:
        merge_data.extend(pickle.loads(existing_shm.buf[result[0]:result[1]]))
    merge_data_obj = pickle.dumps(merge_data, protocol=pickle.HIGHEST_PROTOCOL)
    existing_shm.buf[:len(merge_data_obj)] = merge_data_obj
    existing_shm.close()
    q.put(len(merge_data_obj))

def prepare_traindata(memory_name):
    sample_imgid = sample(imgid_train, sample_len)
    subprocess_list = []
    for i in range(subprocess_num):
        subprocess_list.append(
            mp.Process(
                target=augment_data, 
                args=(sample_imgid[i*imgid_num_process:(i+1)*imgid_num_process], data_size//subprocess_num, memory_name, i*shared_memory_size//subprocess_num, q, )
            )
        )
    for p in subprocess_list:
        p.start()
    p0 = mp.Process(target=merge_subprocess, args=(q, subprocess_num, memory_name,))
    p0.start()
    return p0

def traindata_gen():
    global traindata
    i = 0
    while i<len(traindata):
        yield traindata[i][0]/255., traindata[i][1]
        i += 1

def map_val_fn(t: tf.Tensor):
    filename = str(t.numpy(), encoding='utf-8')
    imgid = int(filename[20:32])
    # Load image
    img, (h0, w0), (h, w) = load_image(filename)
    #augment_hsv(img, hgain=hsv_h, sgain=hsv_s, vgain=hsv_v)

    # Labels
    label_filename = val_label_path + filename.split('/')[-1].split('.')[0] + '.txt'
    labels, _ = load_labels(label_filename)
    labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, 0, 0)  # normalized xywh to pixel xyxy format
    labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
    labels[:, 1:5] /= img_size  # normalized height 0-1
    
    #img = img/255.
    img = img[:, :, ::-1].transpose(2,0,1)
    img = img/255.
    
    #img_hw = tf.reshape(tf.concat([h0, w0], axis=0), [-1,2])
    img_hw = tf.concat([h0, w0], axis=0)
    return img, labels, img_hw, imgid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yolo')
    parser.add_argument('--resume', type=str, help='Specify the CKPT name for resume training')
    parser.add_argument('--verify', type=int, default=16, help='Whether to generate predictions on the first 16 pics of val dataset')
    parser.add_argument('--startepoch', type=int, default=0, help='If resume training then specify the epoch to continue')
    parser.add_argument('--numepoch', type=int, default=1, help='Specify the number of epochs to train')
    parser.add_argument('--stepsepoch', type=int, default=10000, help='Specify the steps of epoch')
    args = parser.parse_args()

    START_EPOCH = args.startepoch
    NUM_EPOCH = args.numepoch
    STEPS_EPOCH = args.stepsepoch
    STEPS_OFFSET = START_EPOCH*STEPS_EPOCH+1

    colors = class_colors(classnames)

    model = create_model()

    ema = tf.train.ExponentialMovingAverage(decay=0.9999, num_updates=0)
    ema.apply(model.trainable_variables)

    averages = [ema.average(var) for var in model.trainable_variables]
    model_trainable_vars = [var for var in model.trainable_variables]
    model_non_trainable_vars = [var for var in model.non_trainable_variables]
    checkpoint = tf.train.Checkpoint(
        model_trainable_weights=model_trainable_vars, 
        model_non_trainable_weights=model_non_trainable_vars, 
        averaged_weights=averages)

    if args.resume:
        loadStatus = checkpoint.restore(args.resume)
        print("Successfully loaded checkpoint {}, continue training".format(args.resume))

    dataset_val = tf.data.Dataset.list_files("coco/images/val2017/*.jpg", shuffle=False)
    dataset_val = dataset_val.map(
        lambda x: tf.py_function(func=map_val_fn, inp=[x], Tout=[tf.float32, tf.float32, tf.int32, tf.int32]), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_val = dataset_val\
        .padded_batch(val_batch_size, padded_shapes=([3, img_size, img_size], [None, 5], [2], []))\
        .prefetch(tf.data.experimental.AUTOTUNE)

    dataset_val_len = dataset_val.cardinality().numpy()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.937, nesterov=True)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    @tf.function(
        input_signature=([
            tf.TensorSpec(shape=[batch_size, 3, img_size, img_size], dtype=tf.float32),
            tf.TensorSpec(shape=[batch_size, None, 5], dtype=tf.float32)
        ])
    )
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            pred_loss = tf_loss_func(predictions, labels)
            regularization_loss = tf.math.add_n(model.losses)
            loss = pred_loss + regularization_loss
            scaled_loss = optimizer.get_scaled_loss(loss)
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        #gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, model.trainable_variables) if grad is not None)
        ema.apply(model.trainable_variables)
        return loss, predictions

    dataset_val = tf.data.Dataset.list_files("coco/images/val2017/*.jpg", shuffle=False)
    dataset_val = dataset_val.map(
        lambda x: tf.py_function(func=map_val_fn, inp=[x], Tout=[tf.float32, tf.float32, tf.int32, tf.int32]), 
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_val = dataset_val\
        .padded_batch(val_batch_size, padded_shapes=([3, img_size, img_size], [None, 5], [2], []), padding_values=(144/255., 0., 0, 0))\
        .prefetch(tf.data.experimental.AUTOTUNE)

    #Initialize cocoapi for evalutation
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    #initialize COCO ground truth api
    dataDir='coco'
    dataType='val2017'
    annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
    cocoGt=COCO(annFile)
    #imgIds=sorted(cocoGt.getImgIds())
    #The val folder contain 4952 images.
    imgIds = []
    val_imageFiles = os.listdir(val_image_dir)
    for filename in val_imageFiles:
        imgIds.append(int(filename[:-4]))
    imgIds = sorted(imgIds)
    cocoid_mapping_labels = coco80_to_coco91_class()

    dataset = None
    for epoch in range(START_EPOCH, (START_EPOCH+NUM_EPOCH)):
        start_step = tf.keras.backend.get_value(optimizer.iterations)+STEPS_OFFSET
        print("Epoch_{} training starts:".format(epoch))
        steps = start_step
        loss_sum = 0
        start_time = time.time()

        while True:
            image_cache = shared_memory.SharedMemory(name="dataset", create=True, size=shared_memory_size)
            merge_proc = prepare_traindata("dataset")

            if dataset:
                for inputs, labels in dataset:
                    step_loss, predictions = train_step(inputs, labels)
                    step_loss = step_loss.numpy()[0]
                    loss_sum += step_loss
                    if steps <= initial_warmup_steps:
                        lr = initial_lr * math.pow(steps/initial_warmup_steps, power)
                        tf.keras.backend.set_value(optimizer.lr, lr)
                    else:
                        lr = initial_lr * math.pow((1.-steps/maximum_batches), power)
                        tf.keras.backend.set_value(optimizer.lr, lr)
                    if steps%100 == 0:
                        elasp_time = time.time()-start_time
                        print("Step:{}, Loss:{:4.2f}, LR:{:5f}, Time:{:3.1f}s".format(steps, loss_sum/1, lr, elasp_time))
                        loss_sum = 0
                        start_time = time.time()
                    steps += 1
            
            merge_proc.join()
            msg = q.get()
            if msg>0:
                traindata = pickle.loads(image_cache.buf[:msg])
            else:
                print("Could not load training data.")
                image_cache.close()
                image_cache.unlink()
                break
            image_cache.close()
            image_cache.unlink()

            dataset = tf.data.Dataset.from_generator(
                traindata_gen,
                output_types=(tf.float32, tf.float32), 
                output_shapes=((3, img_size, img_size), (None, 5)))
            dataset = dataset.padded_batch(batch_size, padded_shapes=([3, img_size, img_size], [None, 5]))
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

            if (steps-start_step)>=STEPS_EPOCH:
                break

        checkpoint.save('training_ckpt')

        #Validate result
        print("Start evaluating")
        result_json = []
        for imgs, labels, imgs_hw, imgs_id in tqdm(dataset_val):
            predictions = model(imgs, training=False)
            predict_results, imgs_info = tf_predict_func_v2(predictions, labels, imgs_hw, imgs_id, 0.001, 0.65)
            results = predict_results.numpy()
            for i in range(results.shape[0]):
                item = results[i]
                result = {}
                result['image_id'] = int(imgs_id[int(item[0])].numpy())
                result['category_id'] = cocoid_mapping_labels[int(item[6])]
                result['bbox'] = item[2:6].tolist()
                result['bbox'] = [int(a*10)/10 for a in result['bbox']]
                result['score'] = int(item[6]*1000)/1000
                if result['score'] == 0:
                    continue
                result_json.append(result)
        
        all_result_str = ','.join([json.dumps(item) for item in result_json])
        all_result_str = '['+all_result_str+']'
        savepath = 'valresult/epoch_'+str(epoch)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        result_filename = savepath + '/validation_result.json'
        result_file = open(result_filename, 'w')
        result_file.write(all_result_str)
        result_file.close()

        cocoDt = cocoGt.loadRes(result_filename)
        # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        if args.verify>0:
            #val_imgids = imgIds[:args.verify]
            plt.figure(figsize=(15,10))    
            with open(result_filename, 'r') as f:
                results = json.load(f)
            for imgid in val_imgids[:args.verify]:
                filename = '0'*(12-len(str(imgid)))+str(imgid)
                img_filename = val_image_dir + '/' + filename+'.jpg'
                img = cv2.imread(img_filename)
                for item in results:
                    if item['image_id']==imgid and item['score']>(0.25*0.65):
                        bbox = []
                        bbox_tmp = item['bbox']
                        bbox.append(int(bbox_tmp[0]))
                        bbox.append(int(bbox_tmp[1]))
                        bbox.append(int(bbox_tmp[0]+bbox_tmp[2]))
                        bbox.append(int(bbox_tmp[1]+bbox_tmp[3]))
                        objcls = item['category_id']
                        cv2.rectangle(
                            img, 
                            (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 
                            colors[coco_id_name_map[objcls]], 1)
                        size = cv2.getTextSize(coco_id_name_map[objcls], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        txt_w, txt_h = size[0]
                        top = int(bbox[1]) - txt_h + 4
                        if top < 0:
                            top = int(bbox[1])+txt_h+1
                        txt_top = int(bbox[1]) - txt_h -5
                        txt_bottom = int(bbox[1])
                        if txt_top < 0:
                            txt_top = int(bbox[1])
                            txt_bottom =  int(bbox[1]) + txt_h + 5
                        cv2.rectangle(img, (int(bbox[0]), txt_top), (int(bbox[0])+txt_w, txt_bottom), colors[coco_id_name_map[objcls]], -1)
                        cv2.putText(img, coco_id_name_map[objcls], (int(bbox[0]), top), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                plt.imsave(savepath + '/valresult_epoch_'+str(epoch)+'_'+str(imgid)+'.png', img)

