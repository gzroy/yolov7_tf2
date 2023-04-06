# yolov7_tf2
Yolov7 implementation on tensorflow 2.x

## Prepare data
Run the script get_coco.sh under the scripts folder to download the images and labels. Place the data under the coco images/ and labels/ folder.

## Train the model from scratch
Run python train.py --startepoch 0 --numepoch 1 --stepepoch 3700 to train the model for one epoch. After each epoch training, it will output the validation result use the pycocotools for the validiation dataset, and store the prediction result of the validation images in the valresult/ folder.

## Resume training
Run python train.py --resume 'training_ckpt-1' --startepoch 1 --numepoch 1 --stepepoch 3700 to continue training.

## Test on the input images
To be implement later...
