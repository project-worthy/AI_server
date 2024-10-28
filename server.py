from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np
import pickle
from contextlib import asynccontextmanager
from layout.layout import layout_loop

from multiprocessing import Process, Manager

class Item(BaseModel):
  camMatrix:list
  distCoeff: list

class RotationMatrixDto(BaseModel):
  rvecs:list
  tvecs:list

from sockets import SocketManager





@asynccontextmanager
async def lifespan(app: FastAPI):
  work_dict = Manager().dict({"distances":[0,0,0],"device_coordinate":[250,50]})
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
  uvicorn.run("server:main",reload=True,host="172.22.173.67",port=8001)
