import cv2
import numpy as np
from functools import partial,wraps

import sys
import os

# sys.path.append(os.path.join(os.getcwd(),"../"))
# print(sys)

from headDetect.myFROZEN_GRAPH_HEAD import FROZEN_GRAPH_HEAD

PATH_TO_CKPT_HEAD='headDetect/models/HEAD_DETECTION_300x300_ssd_mobilenetv2.pb'
head_detector=FROZEN_GRAPH_HEAD(PATH_TO_CKPT_HEAD)

START_SEC = 7
TARGET_IMG_SIZE= (1280,720)

frame_index = 0
temp_arr = []
dp_img = None

def click_event(event,x,y,flags,params):
   if(event == cv2.EVENT_LBUTTONDOWN):
      print(f"frame_num:{frame_index} x:{x} y:{y}")
      if(len(temp_arr) <= 0):
         return
      temp_arr[-1]['heads'].append(dict({"x":x,"y":y}))
      print(temp_arr[-1])
      if dp_img is not None:
        cv2.circle(dp_img,(x,y),2,(0,255,0),2)
        cv2.imshow('frame',dp_img)

# def handle_callback(func,**kwargs):
#    @wraps(func)
#    def wrapper(*a,**k):
#       return partial(func,**kwargs)(*a,**k)
#    return wrapper



if __name__ == "__main__":
  blank_image = np.zeros(TARGET_IMG_SIZE)
  dp_img = None
  cv2.imshow('frame',blank_image)
  cv2.waitKey(1)
  cv2.setMouseCallback('frame',click_event)

  cap = cv2.VideoCapture('./video/video2.mp4')
  fps = cap.get(cv2.CAP_PROP_FPS)
  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  print(f"{length} photo {fps}fps {width}x{height}")
  while(cap.isOpened()):
      ret, frame = cap.read()
      image = cv2.resize(frame,TARGET_IMG_SIZE)
      dp_img = image.copy()
      is_before_time = START_SEC * fps > frame_index
      cv2.imshow('frame',image)

      im_height,im_width,_ = image.shape

      if(not is_before_time):
         print("next")
         print(im_width,im_height)
         _,head_predict_result = head_detector.run(image,im_width,im_height)
        #  print(head_predict_result["heads"])
         temp_arr.append(dict({"frame_index":frame_index,"heads":[]}))

      keypress = cv2.waitKey(1 if is_before_time else 0)
      if keypress & 0xFF == ord('q'):
        break
      frame_index+=1
        
  cap.release()
  cv2.destroyAllWindows()