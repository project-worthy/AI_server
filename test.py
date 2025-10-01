from pathlib import Path
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import numpy as np
import cv2

import socketio
import cv2
import pickle
import os
import numpy as np
import time

from headDetect.myFROZEN_GRAPH_HEAD import FROZEN_GRAPH_HEAD

import matplotlib.pyplot as plt

import asyncio
from IOT.IOT_Calibrater import IOT_Calibrater
from utils.measure import get_square_distance_map,trilateration_3d,degrees_to_radians,rotate_point_2d,calculate_distance
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from tf_bodypix.api import download_model, load_model, BodyPixModelPaths

import tensorflow as tf

# setup input and output paths
# output_path = Path('./data/example-output')
# output_path.mkdir(parents=True, exist_ok=True)

bodypix_model = load_model(download_model(
    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
))
img = cv2.imread('serious-black-businesswoman-sitting-at-desk-in-office-5669603-small.jpg',cv2.IMREAD_COLOR)
head_dectect = FROZEN_GRAPH_HEAD('headDetect/models/HEAD_DETECTION_300x300_ssd_mobilenetv2.pb')


result = bodypix_model.predict_single(img)


im_height, im_width, _ = img.shape


mask = result.get_mask(threshold=0.5)
colored_mask = result.get_colored_part_mask(mask)

img_copy, heads = head_dectect.run(img.copy(), im_width, im_height)
left = heads[0]["left"]
right = heads[0]["right"]
bottom = heads[0]["bottom"]
top = heads[0]["top"]
detection_width = heads[0]["width"]
a = colored_mask[top:bottom,left:right]

bounding_top = top
bounding_bottom = bottom

# get top pixel point 
for row_idx, row in enumerate(a):
  unique_pixels, counts = np.unique(row, axis=0, return_counts=True)
  linePixelDict = dict(zip(map(tuple, unique_pixels), counts))
  coloredCount = linePixelDict.get((110,64,170)) + linePixelDict.get((143,61,178))
  colored_ratio = coloredCount / detection_width
  
  print(f"Row {row_idx}: {coloredCount} {coloredCount / detection_width}")
  if(colored_ratio > 0.5): #colored threshold
    print(f"start with index {top + row_idx}")
    bounding_top = top + row_idx
    break

# get bottom pixel point 
for row_idx, row in enumerate(reversed(a)):
  unique_pixels, counts = np.unique(row, axis=0, return_counts=True)
  linePixelDict = dict(zip(map(tuple, unique_pixels), counts))
  coloredCount = linePixelDict.get((110,64,170)) + linePixelDict.get((143,61,178))
  colored_ratio = coloredCount / detection_width
  
  print(f"Row {row_idx}: {coloredCount} {coloredCount / detection_width}")
  if(colored_ratio > 0.5): #colored threshold
    print(f"start with index {bottom - row_idx}")
    bounding_bottom = bottom - row_idx
    break

# added_image = cv2.addWeighted(img_copy,0,colored_mask.astype(np.uint8),1,0)

cv2.rectangle(img,(left,bounding_top),(right,bounding_bottom),(0,255,0),2)
cv2.rectangle(img,(left,top),(right,bottom),(255,255,0),2)
cv2.imshow("img",img)
cv2.waitKey(0)
