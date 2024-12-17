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

from utils.plt_math import get_rotation_matrix

load_dotenv(verbose=True)

distances = [0,0,0]

PATH_TO_CKPT_HEAD=os.getenv("PATH_TO_CKPT_HEAD")
CAMERA_DEGREE=int(os.getenv("CAMERA_DEGREE"))
WIDTH=int(os.getenv("WIDTH")) # room size
HEIGHT=int(os.getenv("HEIGHT")) # room size
THRESHOLD_DISTANCE=float(os.getenv("THRESHOLD_DISTANCE"))



point1 = (0,WIDTH,0)
point2 = (HEIGHT,WIDTH,0)
point3 = (HEIGHT,0,0)

tic = time.time()

class SocketManager:
  def __init__(self,plt_datas):
    self._socketInit()
    self.plt_datas = plt_datas
    self.clientMap = dict()
    self.devices = dict()
    self.segmentModal = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_75_STRIDE_16))

    self.head_detector = FROZEN_GRAPH_HEAD(PATH_TO_CKPT_HEAD)
    self.IOT_Calibrater = IOT_Calibrater()
    self.ranges = get_square_distance_map(WIDTH,HEIGHT)

    
  def _socketInit(self):
    self.sio_server = socketio.AsyncServer(
      async_mode="asgi",
      cors_allowed_origins="*"
    )
    self.sio_app = socketio.ASGIApp(socketio_server=self.sio_server,socketio_path="/")


    # init_video_connections(self)

    @self.sio_server.event(namespace="/video")
    async def connect(sid, environ, auth):
      self.clientMap.update({sid:len(self.clientMap.values())})
      print("connected",self.clientMap)
      intrinsic = pickle.load(open("intrinsic.pkl","rb"))
      print(intrinsic)
      extrinsic = pickle.load(open("extrinsic.pkl","rb"))

      cameraMatrix = intrinsic['cameraMatrix']

      self.f_x = cameraMatrix[0][0] # f_x
      self.f_y = cameraMatrix[1][1] # f_y
      self.c_x = cameraMatrix[0][2] # c_x
      self.c_y = cameraMatrix[1][2] # c_y
      self.rvecs = np.array(extrinsic['rvecs'])
      self.tvecs = np.array(extrinsic['tvecs'])
      print(self.rvecs)
      self.camera_position = np.matmul(-1 * cv2.Rodrigues(self.rvecs)[0].T,self.tvecs)
      
      print("cameraMatrix\n",cameraMatrix)
      print("f_x,f_y",self.f_x,self.f_y)
      print("c_x,c_y",self.c_x,self.c_y)
      print("rvecs\n",self.rvecs)
      print("tvecs\n",self.tvecs)
      print("camera position",self.camera_position)


    @self.sio_server.event(namespace="/video")
    async def disconnect(sid):
      cv2.destroyWindow("img"+str(self.clientMap.get(sid)))
      del(self.clientMap[sid])
      print("disconnect",sid)


    @self.sio_server.on("ws:photo",namespace="/video")
    async def on_message(sid,data):
      data = pickle.loads(data)
      img =cv2.imdecode(data,cv2.IMREAD_COLOR)
      a_t = time.time()

      t_start = time.time()

      im_height, im_width, _ = img.shape

      result = self.segmentModal.predict_single(img)
      img_copy, heads = self.head_detector.run(img, im_width, im_height)

      fps = 1 / (time.time() - t_start)

      cv2.putText(img_copy, "FPS: {:.2f}".format(fps), (10, 30), 0, 5e-3 * 130, (0,0,255), 2)
      mask = result.get_mask(threshold=0.5)
      colored_mask = result.get_colored_part_mask(mask)
      target_colors = [(110,64,170),(143,61,178)]

      # cv2.line(img,(0,int(im_height/2)),(im_width-1,int(im_height/2)),(255,255,255),1,1)
      h = 1
      added_image = img
      humans_locate = []

      if len(heads) > 0:
        added_image = img_copy
        for head in heads:
          left = head["left"]
          right = head["right"]
          bottom = head["bottom"]
          top = head["top"]
          detection_width = head["width"]
          height = head["height"]
          a = colored_mask[top:bottom,left:right]

          bounding_top = top
          bounding_bottom = bottom

          init_pos = np.array([left,top])
          # if(a.size > 0):
          #   # # 기존 방식
          #   # a = np.array([np.argwhere(np.all(a == target_color, axis=-1))[[0,-1]] for target_color in target_colors])
          #   # flattened = a.ravel()
          #   # values = flattened[[0,2,4,6]]
          #   # min_value = np.min(values)
          #   # max_value = np.max(values)


          #   results = []
          #   for target_color in target_colors:
          #       # Find all locations of the target color
          #       indices = np.argwhere(np.all(a == target_color, axis=-1))
                
          #       # Check if there are any matches
          #       if indices.size > 0:
          #           # Store the first and last occurrence
          #           first_last = indices[[0, -1]]
          #           results.append(first_last)

          #   flattened = np.array(results).ravel()

          #   if(len(flattened) > 0):
          #     values = flattened[np.arange(0,len(flattened),2)]
          #     min_value = np.min(values)
          #     max_value = np.max(values)
          #     # print(values,np.arange(0,len(flattened),2))
          #     bounding_bottom = max_value + top
          #     bounding_top = min_value + top


          # print("1")
          # added_image = cv2.addWeighted(added_image,0.5,colored_mask.astype(np.uint8),0.5,0)


 
          # cv2.rectangle(added_image,(left,bounding_top),(right,bounding_bottom),(0,255,255),2)

          # h = head['height']
          h = np.abs(bounding_top - bounding_bottom)

          H = 0.23 # head size(m)

          u = head['left']
          v = head['top']
          w = head['width']
          zc = self.f_y/h * H

          xc = zc / self.f_x * (u + w /2 - self.c_x)
          yc = zc / self.f_y * (v + h/2 - self.c_y)


          cameraCoord = np.array([[xc],[yc],[zc]])
          R,_ = cv2.Rodrigues(self.rvecs)

          # worldCoord = np.dot(R.T,cameraCoord - self.tvecs) # world_coord = []
          # worldCoord = np.dot(R.T,cameraCoord) # world_coord = []

          cv2.circle(added_image,(int(u + w /2),int(v + h/2)),1,(255,255,0),1)
          cv2.circle(added_image,(int(self.c_x),int(self.c_y)),1,(0,0,255),3)
          
          # abs_camera_x = abs(self.camera_position[0])
          # abs_camera_z = abs(self.camera_position[2])
          # cv2.putText(added_image,'{},{}'.format(worldCoord[0]+abs_camera_x,worldCoord[2]+abs_camera_z),(u,v + h + 60),0,5e-3 * 130,(0,0,255),2)
          # cv2.putText(added_image,'{},{}'.format(worldCoord[0],worldCoord[2]),(u,v + h + 30),0,5e-3 * 130,(0,0,255),2)
          # cv2.putText(added_image,'{},{}'.format(cameraCoord[0],cameraCoord[2]),(u,v + h + 90),0,5e-3 * 130,(0,0,255),2)

          rad = degrees_to_radians(CAMERA_DEGREE)
          # print(rotate_point_2d(human_coord_list,rad))


          rotation_matrix = get_rotation_matrix(0,self.rvecs.flatten()[1],0)
          rotated_cameraCoord = rotation_matrix @ cameraCoord
          # print(rotation_matrix @ cameraCoord)
          cv2.putText(added_image,'{},{}'.format(rotated_cameraCoord[0],rotated_cameraCoord[2]),(u,v + h + 90),0,5e-3 * 130,(0,0,255),2)

          human_coord = rotate_point_2d(rotated_cameraCoord[[0,2]],rad)
          humans_locate.append(human_coord)

          for device in self.devices.keys():
            device_coord = self.devices[device]["coordinate"]["point_on_line"][0:2]
            device_coord = [device_coord[0] * 0.01, device_coord[1] * 0.01]
            print(device_coord,human_coord)
            

            a = calculate_distance(human_coord,device_coord)
            if(a < THRESHOLD_DISTANCE):
              print("on")
              await self.sio_server.emit("turn_on","turn_on",namespace="/temp")
            else:
              print("off")
              await self.sio_server.emit("turn_off","turn_off",namespace="/temp")
        self.plt_datas["humans_locations"] = humans_locate
      # else 
        
      b_t = time.time()

      print(f"loop delayed: {b_t - a_t}sec")
      cv2.imshow("img"+str(self.clientMap.get(sid)),added_image)
      cv2.waitKey(10)





    @self.sio_server.on("distances",namespace="/coordinate")
    async def on_message(sid,data):
      # print(data)
      distances = list(dict(data).values())
      self.plt_datas["distances"] = distances
      device_array = []

      if(sid not in self.devices):
        self.devices[sid] = dict()
      if(not all(value == 0 for value in distances)):
        self.devices[sid]["coordinate"] = trilateration_3d(point1,point2,point3,*distances)

      for device in self.devices.keys():
        if("coordinate" not in self.devices[device]): 
          break
        device_coord = np.array(self.devices[device]["coordinate"]["point_on_line"][0:2])
        device_array.append(device_coord)
      print("device_array:",device_array)
      self.plt_datas["device_coordinates"] = device_array


    @self.sio_server.event(namespace="/coordinate")
    async def connect(sid, environ, auth):
      print("coordinate connected")

    @self.sio_server.event(namespace="/coordinate")
    async def disconnect(sid):
      print("coordinate disconnected")
      self.IOT_Calibrater.remove(sid)


    @self.sio_server.event(namespace="/iot_calibrate")
    async def connect(sid, environ, auth):
      print("iot_calibrate connected")

    @self.sio_server.event(namespace="/iot_calibrate")
    async def disconnect(sid):
      print("iot_calibrate disconnected")
      self.IOT_Calibrater.remove(sid)
    
    # sending for calibrating distances
    @self.sio_server.on("initial",namespace="/iot_calibrate")
    async def on_message(sid,data):
      print(data)
      conditionSatisfied = self.IOT_Calibrater(sid,dict(data))
      if(conditionSatisfied):
        while(True):
          pair = self.IOT_Calibrater.get_combinations()
          print("calibrate pair:",pair)
          await asyncio.sleep(3)
          if(pair is None): 
            break
          

          print("run calibrate")
          startId = self.IOT_Calibrater.sidDict[pair[0]]["id"]
          endId = self.IOT_Calibrater.sidDict[pair[1]]["id"]

          print("calibrating ids:",startId,endId)
          data2 = dict()
          data2["run_type"] = 1 # as anchor
          data2["poll_msg"] = self.IOT_Calibrater.sidDict[pair[0]]["poll_msg"]
          data2["resp_msg"] = self.IOT_Calibrater.sidDict[pair[0]]["resp_msg"]
          # data2["distance"] = self.ranges[startId][endId]
          data2["distance"] = 3.8


          await self.sio_server.emit("calibrate",data2,to=pair[1],namespace="/iot_calibrate")
          await self.IOT_Calibrater.wait_anchor()

          data1 = dict()
          data1["run_type"] = 2 # as tag
          data1["poll_msg"] = self.IOT_Calibrater.sidDict[pair[0]]["poll_msg"]
          data1["resp_msg"] = self.IOT_Calibrater.sidDict[pair[0]]["resp_msg"]
          # data2["distance"] = self.ranges[startId][endId]
          data2["distance"] = 3.8

          self.tag_sid = pair[0]
          self.anchor_sid = pair[1]

          await self.sio_server.emit("calibrate",data1,to=pair[0],namespace="/iot_calibrate")
          await self.IOT_Calibrater.wait_received()
          print("received")

          await self.sio_server.emit("end_calibrate",to=pair[0],namespace="/iot_calibrate")
          await self.sio_server.emit("end_calibrate",to=pair[1],namespace="/iot_calibrate")

        await self.boardcast_all_devices()


    @self.sio_server.on("ready_tag",namespace="/iot_calibrate")
    async def on_message(sid,data):
      print(data)
      self.IOT_Calibrater.resolve()

    @self.sio_server.on("ready_anchor",namespace="/iot_calibrate")
    async def on_message(sid,data):
      print(data)
      self.IOT_Calibrater.resolve_anchor()


    @self.sio_server.on("calibrate",namespace="/iot_calibrate")
    async def on_message(sid,data):
      print(data)
      self.IOT_Calibrater.resolve()
      self.IOT_Calibrater.add(data["delay"])


    @self.sio_server.event(namespace="/temp")
    async def connect(sid, environ, auth):
      print("temp connected")

    @self.sio_server.event(namespace="/iot_calibrate")
    async def disconnect(sid):
      print("temp disconnected")

    @self.sio_server.on("turn_on",namespace="/temp")
    async def on_message(sid,data):
      await self.sio_server.emit("turn_on","turn_on",namespace="/temp")

    @self.sio_server.on("turn_off",namespace="/temp")
    async def on_message(sid,data):
      print("turn_off")
      await self.sio_server.emit("turn_off","turn_off",namespace="/temp")

  async def boardcast_all_devices(self):
    b = [self.IOT_Calibrater.sidDict[sid]["delay"] for sid in self.IOT_Calibrater.sidDict.keys()]

    for (index,sid) in list(enumerate(self.IOT_Calibrater.sidDict.keys())):
      c = dict()
      c["index"] = index
      c["delays"] = b
      await self.sio_server.emit("terminate_calibrate",c,to=sid,namespace="/iot_calibrate")

  def getSocketApp(self):
    return self.sio_app

