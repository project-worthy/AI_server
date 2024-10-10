import socketio
import cv2
import pickle

import numpy as np
import tensorflow as tf
import time

import math

from headDetect.myFROZEN_GRAPH_HEAD import FROZEN_GRAPH_HEAD


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# should go to env file
PATH_TO_CKPT_HEAD = 'headDetect/models/HEAD_DETECTION_300x300_ssd_mobilenetv2.pb'

THRESHOLD_DISTANCE = 50
get_pos_first = True
# m 단위
anchor_coord = np.array([
  [0,0.17,0],
  [0,0,0],
  [0.19,0,0]
])
k = np.zeros(3)
AInv = np.zeros((2,2))

avg_dis = np.zeros(3)
def get_position(distances):
  global get_pos_first
  global anchor_matrix
  global k
  global AInv
  d = list(distances.values())
  b = np.zeros(3)
  print(d)
  current_tag_position = np.zeros(3)

  [d1,d2,d3] = d
  avg_dis[0] = (d[0] + avg_dis * 2) / 2
  avg_dis[1] = (d[0] + avg_dis * 2) / 2
  avg_dis[2] = (d[0] + avg_dis * 2) / 2

  D =np.zeros(3)

  [x1,y1,z1] = anchor_coord[0]
  [x2,y2,z2] = anchor_coord[1]
  [x3,y3,z3] = anchor_coord[2]
  # 방정식 1 (x2, y2, z2)와 비교
  A1 = 2 * (x2 - x1)
  B1 = 2 * (y2 - y1)
  C1 = 2 * (z2 - z1)
  D1 = d1**2 - d2**2 - x1**2 + x2**2 - y1**2 + y2**2 - z1**2 + z2**2

  # 방정식 2 (x3, y3, z3)와 비교
  A2 = 2 * (x3 - x1)
  B2 = 2 * (y3 - y1)
  C2 = 2 * (z3 - z1)
  D2 = d1**2 - d3**2 - x1**2 + x3**2 - y1**2 + y3**2 - z1**2 + z3**2
  # z를 0으로 놓고 평면 방정식 풀이
  A = np.array([[A1, B1], [A2, B2]])
  D = np.array([D1, D2])
  # print(A)
  # print(D)
  # z=0일 때 평면의 (x, y) 좌표 계산
  x, y = np.linalg.solve(A, D)

  # z 좌표는 첫 번째 거리 방정식으로부터 계산
  z = np.sqrt(d1**2 - (x - x1)**2 - (y - y1)**2)
  # print(x,y,z)

  return (x,y,z)

def get_distance_3d(a, b):
    x1, y1, z1 = a
    x2, y2, z2 = b
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

class SocketManager:
  def __init__(self):
    self._socketInit()
    self.clientMap = dict()
    self.devices = dict()
    self.head_detector = FROZEN_GRAPH_HEAD(PATH_TO_CKPT_HEAD)

    

  def _socketInit(self):
    self.sio_server = socketio.AsyncServer(
      async_mode="asgi",
      cors_allowed_origins="*"
    )
    self.sio_app = socketio.ASGIApp(socketio_server=self.sio_server,socketio_path="/")

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
      self.rvecs = extrinsic['rvecs']
      self.tvecs = extrinsic['tvecs']
      self.camera_position = np.matmul(-1 * cv2.Rodrigues(self.rvecs[0])[0].T,self.tvecs[0])
      
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

    @self.sio_server.event(namespace="/calibration")
    async def on_message(sid,data):
      data = pickle


    @self.sio_server.on("ws:photo",namespace="/video")
    def on_message(sid,data):
      data = pickle.loads(data)
      img =cv2.imdecode(data,cv2.IMREAD_COLOR)



      t_start = time.time()


      im_height, im_width, _ = img.shape
      img = cv2.flip(img, 1)

      # Head-detection run model
      img, heads = self.head_detector.run(img, im_width, im_height)


      fps = 1 / (time.time() - t_start)
      cv2.putText(img, "FPS: {:.2f}".format(fps), (10, 30), 0, 5e-3 * 130, (0,0,255), 2)
      cv2.line(img,(0,int(im_height/2)),(im_width-1,int(im_height/2)),(255,255,255),1,1)
      h = 1
      if len(heads) > 0:

        for head in heads:
          h = head['height']

          H = 0.25 # head size(m)

          u = head['left']
          v = head['top']
          w = head['width']
          zc = self.f_y/h * H

          xc = zc / self.f_x * (u + w /2 - self.c_x)
          yc = zc / self.f_y * (v + h/2 - self.c_y)


          cameraCoord = np.array([[xc],[yc],[zc]])
          R,_ = cv2.Rodrigues(self.rvecs)

          worldCoord = np.dot(R.T,cameraCoord - self.tvecs) # world_coord = []

          cv2.circle(img,(int(u + w /2),int(v + h/2)),1,(255,255,0),1)
          cv2.circle(img,(int(self.c_x),int(self.c_y)),1,(0,0,255),3)
          
          cv2.putText(img,'{},{}'.format(worldCoord[0],worldCoord[2]),(u,v + h + 30),0,5e-3 * 130,(0,0,255),2)
          cv2.putText(img,'{},{}'.format(cameraCoord[0],cameraCoord[2]),(u,v + h + 60),0,5e-3 * 130,(0,0,255),2)


          for device in self.devices.values():
            dis = get_distance_3d(tuple(worldCoord),device)
            if(dis < THRESHOLD_DISTANCE):
              print("on")
      
      cv2.imshow("img"+str(self.clientMap.get(sid)),img)
      cv2.waitKey(10)


    @self.sio_server.on("distances",namespace="/coordinate")
    def on_message(sid,data):
      device_info = dict(data)
      (x,y,z) = get_position(dict(data))
      deviceMacAddress = device_info.MAC
      self.devices[deviceMacAddress] = (x,y,z)


  def getSocketApp(self):
    return self.sio_app
