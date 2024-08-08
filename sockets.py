import socketio
import cv2
import pickle

import numpy as np
import tensorflow as tf
import time

from headDetect.myFROZEN_GRAPH_HEAD import FROZEN_GRAPH_HEAD


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# should go to env file
PATH_TO_CKPT_HEAD = 'headDetect/models/HEAD_DETECTION_300x300_ssd_mobilenetv2.pb'

class SocketManager:
  def __init__(self):
    self._socketInit()
    self.clientMap = dict()
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
      cameraMatrix = pickle.load(open("calibrationData.pkl","rb"))
      rmat = pickle.load(open("rmat.pkl","rb"))

      cameraMatrix = cameraMatrix

# * 2.5
# * 2.5
      self.f_x = cameraMatrix[0][0] # f_x
      self.f_y = cameraMatrix[1][1] # f_y
      self.c_x = cameraMatrix[0][2] # c_x
      self.c_y = cameraMatrix[1][2] # c_y
      self.rmat = rmat # R^-1
      print("cameraMatrix\n",cameraMatrix)
      print("f_x,f_y",self.f_x,self.f_y)
      print("c_x,c_y",self.c_x,self.c_y)
      print("rmat\n",rmat)



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
        # if self.f_x is None:
        #   return
        # print(heads[0])

        for head in heads:
          h = head['height']

          H = 0.25 # head size

          u = head['left']
          v = head['top']
          w = head['width']
          zc = self.f_y/h * H

          xc = zc / self.f_x * (u + w /2 - self.c_x)
          yc = zc / self.f_y * (v + h/2 - self.c_y)


          cameraCoord = np.array([[xc],[yc],[zc]])
          mat_invert = np.linalg.inv(self.rmat)
          cv2.circle(img,(int(u + w /2),int(v + h/2)),1,(255,255,0),1)
          cv2.circle(img,(int(self.c_x),int(self.c_y)),1,(0,0,255),3)
          multipled = np.matmul(mat_invert,cameraCoord)
          cv2.putText(img,'{},{}'.format(multipled[0],multipled[2]),(u,v + h + 30),0,5e-3 * 130,(0,0,255),2)
          cv2.putText(img,'{},{}'.format(cameraCoord[0],cameraCoord[2]),(u,v + h + 60),0,5e-3 * 130,(0,0,255),2)


        # self.ax.clear()
        # self.ax.scatter(xc,zc)
        # self.ax.draw()
        # plt.pause(0.0001)
        # plt.clf()
        # plt.scatter(xc,zc)
        # plt.draw()
        # plt.pause(0.0001)
        # self.ax.pause(0.05)
        # plt.show()
      
      cv2.imshow("img"+str(self.clientMap.get(sid)),img)
      cv2.waitKey(10)
      # cv2.imshow("img"+str(self.clientMap.get(sid)),img)
      # cv2.waitKey(10)
      # print("message received")


  def getSocketApp(self):
    return self.sio_app
