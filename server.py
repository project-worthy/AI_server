from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np
import pickle
from contextlib import asynccontextmanager
from layout.layout import layout_loop

from multiprocessing import Process, Manager

from dotenv import load_dotenv
import os

class Item(BaseModel):
  camMatrix:list
  distCoeff: list

class RotationMatrixDto(BaseModel):
  rvecs:list
  tvecs:list

from sockets import SocketManager

load_dotenv(verbose=True)
load_dotenv(dotenv_path=".secrets.env",verbose=True)

CAMERA_DEGREE = int(os.getenv("CAMERA_DEGREE"))
WIDTH = int(os.getenv("WIDTH"))
HEIGHT = int(os.getenv("HEIGHT"))
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))
THRESHOLD_DISTANCE=float(os.getenv("THRESHOLD_DISTANCE"))



@asynccontextmanager
async def lifespan(app: FastAPI):
  work_dict = Manager().dict({
    "distances":[0,0,0],
    "device_coordinates":[],
    "humans_locations":[],
    "camera_degree":CAMERA_DEGREE,
    "width":WIDTH,
    "height":HEIGHT,
    "threshold_distance":THRESHOLD_DISTANCE
    })
  layoutProcess = Process(target=layout_loop,args=(work_dict,))
  layoutProcess.start()
  socketManager = SocketManager(work_dict)
  app.mount("/ws",app=socketManager.getSocketApp())
  yield
  layoutProcess.terminate()
  layoutProcess.join()

app = FastAPI(lifespan=lifespan)


@app.get("/")
def get_home():
  return {"message":"hello world"}

@app.get("/test")
def get_home():
  return {"message":"hello world"}

@app.post("/intrinsic")
def read_item(item:Item):
  print(item.model_dump())
  itemJson = item.model_dump()
  cameraMatrix = np.array(itemJson['camMatrix'])
  distCoeff = np.array(itemJson["distCoeff"])
  pickle.dump({"cameraMatrix":cameraMatrix,"distCoeff":distCoeff},open("intrinsic.pkl","wb"))


@app.post("/extrinsic")
def read_martrix(rmat:RotationMatrixDto):
  itemJson = rmat.model_dump()
  rvecs = np.array(itemJson['rvecs'])
  tvecs = np.array(itemJson['tvecs'])
  pickle.dump({"rvecs":rvecs,"tvecs":tvecs},open("extrinsic.pkl","wb"))
  print(itemJson)


if __name__ == "__main__":
  uvicorn.run("server:app",reload=True,host=HOST,port=PORT)
