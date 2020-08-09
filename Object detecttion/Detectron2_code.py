
# %cd '/content/drive/My Drive/gridPersonal/IMat'

!ls
import torch
torch.cuda.is_available()

!pip install pyyaml==5.1 pycocotools>=2.0.1
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version
# opencv is pre-installed on colab

# install detectron2: (colab has CUDA 10.1 + torch 1.6)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
assert torch.__version__.startswith("1.6")
!pip install detectron2==0.2.1 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html

import collections
import json
import torch, torchvision
import os
import cv2
import random
import gc
import pycocotools
import torch.nn.functional as F

import numpy as np 
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

# import some common detectron2 utilities
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

img_list = []
for dirname, _, filenames in os.walk('train/'):
    for filename in filenames:
        img_list.append(os.path.join(dirname, filename))

data_dir = Path('IMat/')
img_dir = Path('train/')
df = pd.read_csv(data_dir/'train.csv')

# Get label descriptions
with open(data_dir/'label_descriptions.json', 'r') as file:
    label_desc = json.load(file)
df_categories = pd.DataFrame(label_desc['categories'])
df_attributes = pd.DataFrame(label_desc['attributes'])

dirname = 'train/' #name of training image directory
def data_cleaner(df, dirname = 'train/', class_int = 1):
    """
    Takes a dataframe and clean outliers in classId
    Our product vertical is T-shirt, so maximize the t-shirt instances for training
    class_int = 0-46, 1 for t-shirt
    """
    df2 = df[df['ClassId'].astype('float') == class_int] # checks for t-shirt instances
    boo = df.ImageId.isin(df2.ImageId)
    df_fin = df2 # subset df to maximimze class_int instances

    # taking 1000 images due to memory retsrictions. Can be trained on full dataset too!
    df_copy = df_fin[:10000].copy()
    df_copy['ImageId'] = dirname + df_copy['ImageId']
    return df_copy[0:2000]
df_copy = data_cleaner(df, dirname, class_int = 1)

# helper functions to get boundary boxes from run-length encoded masks

def rle_decoder(string, h, w):
    mask = np.full(h*w, 0, dtype=np.uint8)
    annotation = [int(x) for x in string.split(' ')]
    for i, start_pixel in enumerate(annotation[::2]):
        mask[start_pixel: start_pixel+annotation[2*i+1]] = 1
    mask = mask.reshape((h, w), order='F')

    
    return mask

def get_bbox(rle, shape):
    '''
    Get a bbox from mask required for Detectron2
    rle: run-length encoded image mask, as string
    shape: (height, width) of image on which RLE was produced
    Returns (x0, y0, x1, y1) tuple describing the bounding box of the rle mask
    
    Note on image vs np.array dimensions:
    
        np.array implies the `[y, x]` indexing order in terms of image dimensions,
        so the variable on `shape[0]` is `y`, and the variable on the `shape[1]` is `x`,
        hence the result would be correct (x0,y0,x1,y1) in terms of image dimensions
        for RLE-encoded indices of np.array
    '''
    
    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2))  # an array of (start, length) pairs
    a[:,0] -= 1  # `start` is 1-indexed
    
    y0 = a[:,0] % shape[0]
    y1 = y0 + a[:,1]
    if np.any(y1 > shape[0]):
        # got `y` overrun, meaning that there are a pixels in mask on 0 and shape[0] position
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)
    
    x0 = a[:,0] // shape[0]
    x1 = (a[:,0] + a[:,1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)
    
    if x1 > shape[1]:
        # just went out of the image dimensions
        raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
            x1, shape[1]
        ))

    return x0, y0, x1, y1

def bbox(df):
    """
    calls get_bbox and adds column of x0,y0,x1,y1 in df
    """
    bboxes = [get_bbox(c.EncodedPixels, (c.Height, c.Width)) for n, c in df.iterrows()]
    assert len(bboxes) == df.shape[0]
    bboxes_arr = np.array(bboxes)
    df['x0'], df['y0'], df['x1'], df['y1'] = bboxes_arr[:,0], bboxes_arr[:,1], bboxes_arr[:,2], bboxes_arr[:,3]
    # saving up some memory
    del bboxes
    gc.collect()
    return df
df_copy = bbox(df_copy)

from detectron2.structures import BoxMode
import pycocotools

def get_fashion_dict(df):
    
    dataset_dicts = []
    
    for idx, filename in enumerate(df['ImageId'].unique().tolist()):
        
        record = {}
        
        # Convert to int otherwise evaluation will throw an error
        record['height'] = int(df[df['ImageId']==filename]['Height'].values[0])
        record['width'] = int(df[df['ImageId']==filename]['Width'].values[0])
        
        record['file_name'] = filename
        record['image_id'] = idx
        
        objs = []
        for index, row in df[(df['ImageId']==filename)].iterrows():
            
            # Get binary mask
            mask = rle_decoder(row['EncodedPixels'], row['Height'], row['Width'])
            
            # opencv 4.2+
            # Transform the mask from binary to polygon format
            contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
            
            # opencv 3.2
            # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
            #                                            cv2.CHAIN_APPROX_SIMPLE)
            
            segmentation = []

            for contour in contours:
                contour = contour.flatten().tolist()
                # segmentation.append(contour)
                if len(contour) > 4:
                    segmentation.append(contour)    
            
            # Data for each mask
            obj = {
                'bbox': [row['x0'], row['y0'], row['x1'], row['y1']],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': int(row['ClassId']),
                'segmentation': segmentation,
                'iscrowd': 0
            }
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

from time import time
start = time()
fashion_dict = get_fashion_dict(df_copy)
print((time() - start)/60)

# split into train and validation set
df_copy_train = df_copy[:1600].copy()
df_copy_test = df_copy[-400:].copy()

from detectron2.data import DatasetCatalog, MetadataCatalog

# Register the train set metadata
for d in ['train','test']:
    if d == 'train':
        DatasetCatalog.register('1fashion_' + d, lambda d=df_copy_train: get_fashion_dict(d))
    else:
        DatasetCatalog.register('1fashion_' + d, lambda d=df_copy_test: get_fashion_dict(d))
    MetadataCatalog.get("1fashion_" + d).set(thing_classes=['top, t-shirt, sweatshirt'])
    fashion_metadata = MetadataCatalog.get("1fashion_" + d)

import random
for d in random.sample(fashion_dict, 10):
    plt.figure(figsize=(10,10))
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], fashion_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    plt.imshow(vis.get_image()[:, :, ::-1])

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("1fashion_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 50
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)

# Lets Train!

trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
cfg.DATASETS.TEST = ('1fashion_test',)
predictor = DefaultPredictor(cfg)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
with torch.no_grad():
    evaluator = COCOEvaluator("1fashion_test", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "1fashion_test")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way is to use trainer.test

from detectron2.utils.visualizer import ColorMode
plt.figure(figsize=(12,12))
for d in random.sample(fashion_dict, 2):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=fashion_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(vis.get_image()[:, :, ::-1])

im = cv2.imread('25ea31fc-3af7-4230-a8d2-dddd6a261f01.jpeg')
plt.imshow(im)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
               metadata=fashion_metadata, 
               scale=0.5,)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.figure(figsize = (18, 10))
plt.imshow(out.get_image()[:, :, ::-1])

import torch, torchvision
import math
from PIL import Image

class ResizeMe(object):
    #resize and center image in desired size 
    def __init__(self,desired_size):
        
        self.desired_size = desired_size
        
    def __call__(self,img):
    
        img = np.array(img).astype(np.uint8)
        
        desired_ratio = self.desired_size[1] / self.desired_size[0]
        actual_ratio = img.shape[0] / img.shape[1]

        desired_ratio1 = self.desired_size[0] / self.desired_size[1]
        actual_ratio1 = img.shape[1] / img.shape[0]

        if desired_ratio < actual_ratio:
            img = cv2.resize(img,(int(self.desired_size[1]*actual_ratio1),self.desired_size[1]),None,interpolation=cv2.INTER_AREA)
        elif desired_ratio > actual_ratio:
            img = cv2.resize(img,(self.desired_size[0],int(self.desired_size[0]*actual_ratio)),None,interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img,(self.desired_size[0], self.desired_size[1]),None, interpolation=cv2.INTER_AREA)
            
        h, w, _ = img.shape

        new_img = np.zeros((self.desired_size[1],self.desired_size[0],3))
        
        hh, ww, _ = new_img.shape

        yoff = int((hh-h)/2)
        xoff = int((ww-w)/2)
        
        new_img[yoff:yoff+h, xoff:xoff+w,:] = img

        
        return Image.fromarray(new_img.astype(np.uint8))

class MakeLandscape():
    #flip if needed
    def __init__(self):
        pass
    def __call__(self,img):
        
        if img.height> img.width:
            img = np.rot90(np.array(img))
            img = Image.fromarray(img)
        return img


def get_cropped(rotrect,box,image):
    
    width = int(rotrect[1][0])
    height = int(rotrect[1][1])

    src_pts = box.astype("float32")
    # corrdinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


class MakeLandscape():
    #flip if needed
    def __init__(self):
        pass
    def __call__(self,img):
        
        if img.height> img.width:
            img = np.rot90(np.array(img))
            img = Image.fromarray(img)
        return img


def get_cropped(rotrect,box,image):
    
    width = int(rotrect[1][0])
    height = int(rotrect[1][1])

    src_pts = box.astype("float32")
    # corrdinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped




def get_cropped_leaf(img,predictor,return_mapping=False,resize=None):
    #convert to numpy    
    img = np.array(img)[:,:,::-1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    #get prediction
    outputs = predictor(img)
    
    #get boxes and masks
    ins = outputs["instances"]
    pred_masks = ins.get_fields()["pred_masks"]
    boxes = ins.get_fields()["pred_boxes"]    
    classes = ins.get_fields()["pred_classes"]
    #get main leaf mask if the area is >= the mean area of boxes and is closes to the centre 
    masker = pred_masks[np.argmin([calculateDistance(x[0], x[1], int(img.shape[1]), int(img.shape[0])) for i,x in enumerate(boxes.get_centers()) if (boxes[i].area()==torch.max(boxes.area()).to("cpu")).item()])].to("cpu").numpy().astype(np.uint8)
    #mask image
    mask_out = cv2.bitwise_and(img, img, mask=masker)
#     plt.imshow(mask_out)
    #find contours and boxes
    contours, hierarchy = cv2.findContours(masker.copy() ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[np.argmax([cv2.contourArea(x) for x in contours])]
    rotrect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)
    

    #crop image
    cropped = get_cropped(rotrect,box,mask_out)
    #resize
    rotated = MakeLandscape()(Image.fromarray(cropped))
    
    if not resize == None:
        resized = ResizeMe((resize[0],resize[1]))(rotated)
    else:
        resized = rotated
        
    if return_mapping:
        img = cv2.drawContours(img, [box], 0, (0,0,255), 10)
        img = cv2.drawContours(img, contours, -1, (255,150,), 10)
        return resized, ResizeMe((int(resize[0]),int(resize[1])))(Image.fromarray(img))
    
    return resized

def calculateDistance(x1,y1,x2,y2):  
    dist = math.hypot(x2 - x1, y2 - y1)
    return dist

