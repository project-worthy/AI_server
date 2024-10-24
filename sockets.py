import socketio
import cv2
import pickle

import numpy as np
import tensorflow as tf
import time


from headDetect.myFROZEN_GRAPH_HEAD import FROZEN_GRAPH_HEAD


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import math

from itertools import combinations, permutations;
import asyncio

from utils.measure import get_square_distance_map

# should go to env file
PATH_TO_CKPT_HEAD = 'headDetect/models/HEAD_DETECTION_300x300_ssd_mobilenetv2.pb'


WIDTH = 3.05 #m
HEIGHT = 2.05 #m

THRESHOLD_DISTANCE = 50
get_pos_first = True
# cm 단위
anchor_coord = np.array([
  [0,0,0],
  [0.135,0.215,0],
  [0.135,0,0]
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

def trilaterate(p1, p2, p3, r1, r2, r3):
    """
    Trilaterate a point in 3D space given three reference points and their distances to the target point.
    
    Parameters:
    - p1, p2, p3: Arrays or lists of the coordinates of the three reference points (e.g., [x, y, z]).
    - r1, r2, r3: Distances from the target point to each of the three reference points.
    
    Returns:
    - target_point: The calculated coordinates of the target point.
    """
    # Convert inputs to numpy arrays
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    # Vector distances between the points
    ex = (p2 - p1) / np.linalg.norm(p2 - p1)
    i = np.dot(ex, p3 - p1)
    ey = (p3 - p1 - i * ex) / np.linalg.norm(p3 - p1 - i * ex)
    ez = np.cross(ex, ey)
    
    # Distances
    d = np.linalg.norm(p2 - p1)
    j = np.dot(ey, p3 - p1)
    
    # Calculate coordinates
    x = (r1**2 - r2**2 + d**2) / (2 * d)
    y = (r1**2 - r3**2 + i**2 + j**2) / (2 * j) - (i / j) * x
    y_new = (p3[0]**2 + p3[1]**2 + r1**2 - r3**2 - 2*x*p3[0])/2*p3[1]
    print(x,y)
    print(x,y_new)
    # z = np.sqrt(r1**2 - x**2 - y**2)
    
    # Calculate the target point in 3D
    # target_point = p1 + x * ex + y * ey + z * ez
    # return target_point


def get_a(a, b, c, d_a, d_b, d_c):
  a = a*100
  b = b*100
  c = c*100
  var_lambda = a[0]**2 + a[1]**2 - d_a**2 - b[0]**2 - b[1]**2 + d_b**2 + (b[1] - a[1]) * (d_a**2 - d_c**2 - a[1]**2 + c[1]**2 + c[0]**2 - a[0]**2)/(c[1] - a[1])
  var_delta = 2*((b[1] - a[1]) * (c[0] - a[0])*(c[0] - a[0]) - (b[0] - a[0]) * (c[1]-a[0]))
  print(var_lambda,var_delta)
  coord_x = var_lambda * (c[1] - a[1]) / var_delta
  coord_y = (d_a**2 - d_c **2 - a[1]**2 + c[1]**2 + c[0]**2 - a[0]**2 - 2*(c[0] - a[0])*coord_x)/2 * (c[1] - a[1])
  print(coord_x,coord_y)

def get_distance_3d(a, b):
    x1, y1, z1 = a
    x2, y2, z2 = b
    # return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2 - p1[0]**2 - 2*(p3[0] - p1[0])*)


class IOT_Calibrater:
  def __init__(self):
    self.sidDict = dict()
    self.tagCount = 0
    self.anchorCount = 0
    self.combinations = None
    self.combinationCount = 0


  def __call__(self,sid,json):

    self.sidDict[sid] = json
    self.sidDict[sid]["delay"] = 0


    if(json["type"] == 0):
      self.anchorCount += 1 
    if(json["type"] == 1):
      self.tagCount += 1
    

    print(self.tagCount,self.anchorCount)

    if self.anchorCount == 3 and self.tagCount == 1:
      pass
      print("starting calibrating")
      return True
    # else:
    #   self.reset()
    
    return False

  def remove(self,sid):
    jsonData = self.sidDict.get(sid)
    if(jsonData):
      if(jsonData["type"] == 0):
        self.anchorCount -= 1
      if(jsonData["type"] == 1):
        self.tagCount -= 1
  
  def get_combinations(self):

    if(not self.combinations):
      self.combinations = list(combinations(self.sidDict.keys(),2))
    if(self.combinationCount >= len(self.combinations)):
      return None
    result = self.combinations[self.combinationCount]
    self.combinationCount += 1

    self.from_sid = result[0]
    self.to_sid = result[1]

    return result
  
  def wait_received(self):
    self.received_promise = asyncio.Future()
    return self.received_promise 
  
  def wait_anchor(self):
    self.await_anchor = asyncio.Future()
    return self.await_anchor
  
  def resolve(self):
    self.received_promise.set_result(True)

  def resolve_anchor(self):
    self.await_anchor.set_result(True)

  def eject(self):
    self.received_promise.set_result(False)

  def reset(self):
    self.sidDict = dict()
    self.tagCount = 0
    self.anchorCount = 0
    self.combinations = None
    self.combinationCount = 0

  def add(self,delay):
    avg_delay = self.sidDict[self.from_sid]["delay"]
    delay_total = avg_delay * 2 + delay * 2
    if(avg_delay == 0):
      delay_total /= 2
    else:
      delay_total /= 4
    self.sidDict[self.from_sid]["delay"] = delay_total

    avg_delay = self.sidDict[self.to_sid]["delay"]
    delay_total = avg_delay * 2 + delay * 2
    if(avg_delay == 0):
      delay_total /= 2
    else:
      delay_total /= 4
    self.sidDict[self.to_sid]["delay"] = delay_total

    print(self.sidDict)

class SocketManager:
  def __init__(self):
    self._socketInit()
    self.clientMap = dict()
    self.devices = dict()
    self.head_detector = FROZEN_GRAPH_HEAD(PATH_TO_CKPT_HEAD)
    self.IOT_Calibrater = IOT_Calibrater()
    self.ranges = get_square_distance_map(WIDTH,HEIGHT)

    
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
    def on_message(sid,data):
      data = pickle.loads(data)
      img =cv2.imdecode(data,cv2.IMREAD_COLOR)



      t_start = time.time()


      im_height, im_width, _ = img.shape
      img = cv2.flip(img, 1)

      # Head-detection run model
      _, heads = self.head_detector.run(img, im_width, im_height)


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
          
          abs_camera_x = abs(self.camera_position[0])
          abs_camera_z = abs(self.camera_position[2])
          # cv2.putText(img,'{},{}'.format(worldCoord[0]+abs_camera_x,worldCoord[2]+abs_camera_z),(u,v + h + 30),0,5e-3 * 130,(0,0,255),2)
          # cv2.putText(img,'{},{}'.format(cameraCoord[0],cameraCoord[2]),(u,v + h + 60),0,5e-3 * 130,(0,0,255),2)


          for device in self.devices.values():
            dis = get_distance_3d(tuple(worldCoord),device)
            if(dis < THRESHOLD_DISTANCE):
              print("on")
      
      cv2.imshow("img"+str(self.clientMap.get(sid)),img)
      cv2.waitKey(10)

    @self.sio_server.event(namespace="/iot_calibrate")
    async def connect(sid, environ, auth):
      print("iot_calibrate connected")

    @self.sio_server.event(namespace="/iot_calibrate")
    async def disconnect(sid):
      print("iot_calibrate disconnected")
      self.IOT_Calibrater.remove(sid)



    @self.sio_server.on("distances",namespace="/coordinate")
    async def on_message(sid,data):
      print(data)

    @self.sio_server.event(namespace="/coordinate")
    async def connect(sid, environ, auth):
      print("coordinate connected")

    @self.sio_server.event(namespace="/coordinate")
    async def disconnect(sid):
      print("coordinate disconnected")
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
          data2["distance"] = self.ranges[startId][endId]

          await self.sio_server.emit("calibrate",data2,to=pair[1],namespace="/iot_calibrate")
          await self.IOT_Calibrater.wait_anchor()

          data1 = dict()
          data1["run_type"] = 2 # as tag
          data1["poll_msg"] = self.IOT_Calibrater.sidDict[pair[0]]["poll_msg"]
          data1["resp_msg"] = self.IOT_Calibrater.sidDict[pair[0]]["resp_msg"]
          data2["distance"] = self.ranges[startId][endId]

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


  async def boardcast_all_devices(self):
    # a = map(lambda x: {x:x["delay"]},[i for i in self.IOT_Calibrater.sidDict.keys()])
    # print(dict(a))
    b = [self.IOT_Calibrater.sidDict[sid]["delay"] for sid in self.IOT_Calibrater.sidDict.keys()]

    for (index,sid) in list(enumerate(self.IOT_Calibrater.sidDict.keys())):
      c = dict()
      c["index"] = index
      c["delays"] = b
      await self.sio_server.emit("terminate_calibrate",c,to=sid,namespace="/iot_calibrate")

  def getSocketApp(self):
    return self.sio_app

