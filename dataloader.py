import cv2
from util import *
import random
from random import sample
import math

class Dataloader:
    def __init__(self, img_size, img_dir, label_dir, img_ids, hyp):
        self.img_size = img_size
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_ids = img_ids
        self.hyp = hyp
        self.images = {}
        self.labels = {}
        self.segments = {}
        self._loadImages()

    def _loadImages(self):
        for imgid in self.img_ids:
            filename = '0'*(12-len(str(imgid)))+str(imgid)
            img_filename = self.img_dir + '/' + filename+'.jpg'
            img = cv2.imread(img_filename)
            self.images[imgid] = img
            label_filename = self.label_dir + '/' + filename+'.txt'
            with open(label_filename, 'r') as f:
                lines = f.readlines()
            boxes = []
            classes = []
            segments = []
            for l in lines:
                data = [np.float32(x) for x in l.strip().split(' ')]
                classes.append(data[0])
                segment = np.array(data[1:]).reshape((-1, 2))
                boxes.append([segment[:,0].min(), segment[:,1].min(), segment[:,0].max(), segment[:,1].max()])
                segments.append(np.array(l.strip().split(' ')[1:], dtype=np.float32).reshape(-1, 2))
            self.segments[imgid] = segments
            boxes = xyxy2xywh(np.array(boxes))
            classes = np.array(classes).reshape((-1,1))
            self.labels[imgid] = np.concatenate([classes, boxes], axis=-1)

    def random_perspective(self, img, targets=(), segments=(), degrees=10, translate=0., scale=0., shear=10, perspective=0.0,
                        border=(0, 0)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # targets = [cls, xyxy]

        height = img.shape[0] + border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1.1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # Visualize
        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
        # ax[0].imshow(img[:, :, ::-1])  # base
        # ax[1].imshow(img2[:, :, ::-1])  # warped

        # Transform label coordinates
        n = len(targets)
        if n:
            use_segments = any(x.any() for x in segments)
            #use_segments = False
            new = np.zeros((n, 4))
            if use_segments:  # warp segments
                segments = resample_segments(segments)  # upsample
                for i, segment in enumerate(segments):
                    xy = np.ones((len(segment), 3))
                    xy[:, :2] = segment
                    xy = xy @ M.T  # transform
                    xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                    # clip
                    new[i] = segment2box(xy, width, height)

            else:  # warp boxes
                xy = np.ones((n * 4, 3))
                xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = xy @ M.T  # transform
                xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # clip
                new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
                new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            # filter candidates
            i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
            targets = targets[i]
            targets[:, 1:5] = new[i]

        return img, targets   

    def load_mosaic(self, selectedids):
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        #for i, id in enumerate(selected_ids):
        for i, id in enumerate(selectedids):
            img = self.images[id]
            h, w, _ = img.shape
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels = self.labels[id].copy()
            segments = self.segments[id].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments_tmp = [xyn2xy(x, w, h, padw, padh) for x in segments]
                segments_tmp = resample_segments(segments_tmp)  # upsample
                segments = []
                selected_index = []
                for i,item in enumerate(segments_tmp):
                    x, y = item.T
                    inside = (x >= 0) & (y >= 0) & (x <= 2*s) & (y <= 2*s)
                    x, y, = x[inside], y[inside]
                    if x.shape[0]>0:
                        segments.append(np.stack([x,y]).T)
                        selected_index.append(i)
            if len(selected_index)>0:
                labels4.append(np.stack([labels[i] for i in selected_index]))
            if len(segments)>0:
                segments4.extend(segments)
            
        labels4 = np.concatenate(labels4, axis=0)
        np.clip(labels4[:,1:], 0, 2*s, labels4[:,1:])
            
        img4, labels4 = self.random_perspective(img4, labels4, segments4, self.hyp['degrees'], self.hyp['translate'], self.hyp['scale'], self.hyp['shear'], self.hyp['perspective'], self.mosaic_border)
        
        return img4, labels4

    def load_mosaic9(self, selectedids):
        # loads images in a 9-mosaic
        labels9, segments9 = [], []
        s = self.img_size
        for i, id in enumerate(selectedids):
            # Load image
            img = self.images[id]
            h, w, _ = img.shape

            # place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

            # Labels
            labels = self.labels[id].copy()
            segments = self.segments[id].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc

        c = np.array([xc, yc])  # centers
        #segments9 = [x - c for x in segments9]

        segments_tmp = [x - c for x in segments9]
        segments_tmp = resample_segments(segments_tmp)  # upsample
        segments9 = []
        selected_index = []
        for i,item in enumerate(segments_tmp):
            x, y = item.T
            inside = (x >= 0) & (y >= 0) & (x <= 2*s) & (y <= 2*s)
            x, y, = x[inside], y[inside]
            if x.shape[0]>0:
                segments9.append(np.stack([x,y]).T)
                selected_index.append(i)
        if len(selected_index)>0:
            labels9 = labels9[selected_index]

        #np.clip(labels9[:,1:], 0, 2*s, out=labels9[:,1:])
        labels9_clipped = labels9.copy()
        np.clip(labels9[:,1:], 0, 2*s, labels9_clipped[:,1:])
        i = box_candidates(box1=labels9[:, 1:5].T, box2=labels9_clipped[:, 1:5].T, area_thr=0.1)
        labels9 = labels9_clipped[i]
        segments_tmp = []
        for j,flag in enumerate(i):
            if flag:
                segments_tmp.append(segments9[j])
        segments9 = segments_tmp

        #for x in (labels9[:, 1:], *segments9):
        #    np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        # Augment
        img9, labels9 = self.random_perspective(img9, labels9, segments9,
                                        degrees=self.hyp['degrees'],
                                        translate=self.hyp['translate'],
                                        scale=self.hyp['scale'],
                                        shear=self.hyp['shear'],
                                        perspective=self.hyp['perspective'],
                                        border=self.mosaic_border)  # border to remove
        return img9, labels9
    
    def load_samples(self, selectedids):
        # loads images in a 4-mosaic

        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        for i, id in enumerate(selectedids):
            img = self.images[id]
            h, w, _ = img.shape

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = self.labels[id].copy()
            segments = self.segments[id].copy()
            #labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments) 

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        #img4, labels4, segments4 = remove_background(img4, labels4, segments4)
        sample_labels, sample_images, sample_masks = self.sample_segments(img4, labels4, segments4, probability=0.5)

        return sample_labels, sample_images, sample_masks

    def sample_segments(self, img, labels, segments, probability=0.5):
        # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
        n = len(segments)
        sample_labels = []
        sample_images = []
        sample_masks = []
        if probability and n:
            h, w, c = img.shape  # height, width, channels
            for j in random.sample(range(n), k=round(probability * n)):
                l, s = labels[j], segments[j]
                box = l[1].astype(np.int32).clip(0,w-1), l[2].astype(np.int32).clip(0,h-1), l[3].astype(np.int32).clip(0,w-1), l[4].astype(np.int32).clip(0,h-1) 
                
                #print(box)
                if (box[2] <= box[0]) or (box[3] <= box[1]):
                    continue
                
                sample_labels.append(l[0])
                
                mask = np.zeros(img.shape, np.uint8)
                
                cv2.drawContours(mask, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
                sample_masks.append(mask[box[1]:box[3],box[0]:box[2],:])
                
                result = cv2.bitwise_and(src1=img, src2=mask)
                i = result > 0  # pixels to replace
                mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
                #print(box)
                sample_images.append(mask[box[1]:box[3],box[0]:box[2],:])

        return sample_labels, sample_images, sample_masks

    def pastein(self, image, labels, sample_labels, sample_images, sample_masks):
        # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
        h, w = image.shape[:2]

        # create random masks
        scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
        for s in scales:
            if random.random() < 0.2:
                continue
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)   
            
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            if len(labels):
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area     
            else:
                ioa = np.zeros(1)
            
            if (ioa < 0.30).all() and len(sample_labels) and (xmax > xmin+20) and (ymax > ymin+20):  # allow 30% obscuration of existing labels
                sel_ind = random.randint(0, len(sample_labels)-1)
                hs, ws, cs = sample_images[sel_ind].shape
                r_scale = min((ymax-ymin)/hs, (xmax-xmin)/ws)
                r_w = int(ws*r_scale)
                r_h = int(hs*r_scale)
                
                if (r_w > 10) and (r_h > 10):
                    r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                    r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                    temp_crop = image[ymin:ymin+r_h, xmin:xmin+r_w]
                    m_ind = r_mask > 0
                    if m_ind.astype(np.int32).sum() > 60:
                        temp_crop[m_ind] = r_image[m_ind]
                        box = np.array([xmin, ymin, xmin+r_w, ymin+r_h], dtype=np.float32)
                        if len(labels):
                            labels = np.concatenate((labels, [[sample_labels[sel_ind], *box]]), 0)
                        else:
                            labels = np.array([[sample_labels[sel_ind], *box]])
                                
                        image[ymin:ymin+r_h, xmin:xmin+r_w] = temp_crop

        return labels

    def augment_hsv(self, img, hgain=0.5, sgain=0.5, vgain=0.5):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    def generateTrainData(self, datasize):
        traindata = []
        for i in range(datasize):
            if random.random() < 0.8:
                selected_ids = sample(self.img_ids, 4)
                img, labels = self.load_mosaic(selected_ids)
            else:
                selected_ids = sample(self.img_ids, 9)
                img, labels = self.load_mosaic9(selected_ids)
    
            if random.random() < self.hyp['mixup']:
                if random.random() < 0.8:
                    selected_ids = sample(self.img_ids, 4)
                    img2, labels2 = self.load_mosaic(selected_ids)
                else:
                    selected_ids = sample(self.img_ids, 9)
                    img2, labels2 = self.load_mosaic9(selected_ids)

                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

            self.augment_hsv(img, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'], vgain=self.hyp['hsv_v'])

            if random.random() < self.hyp['paste_in']:
                sample_labels, sample_images, sample_masks = [], [], [] 
                while len(sample_labels) < 30:
                    selected_ids = sample(self.img_ids, 4)
                    sample_labels_, sample_images_, sample_masks_ = self.load_samples(selected_ids)
                    sample_labels += sample_labels_
                    sample_images += sample_images_
                    sample_masks += sample_masks_
                    #print(len(sample_labels))
                    if len(sample_labels) == 0:
                        break
                labels = self.pastein(img, labels, sample_labels, sample_images, sample_masks)

            nL = len(labels)  # number of labels
            if nL:
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
                labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
                labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1
                
            if random.random() < self.hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
            
            img = img[:, :, ::-1].transpose(2,0,1)
            traindata.append((img, labels))  
        return traindata

        
