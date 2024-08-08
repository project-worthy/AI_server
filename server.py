from typing import Union
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np
import pickle

class Item(BaseModel):
  cameraMat:list
  dist: list

class RotationMatrixDto(BaseModel):
  rmat:list

from sockets import SocketManager

app = FastAPI()
socketManager = SocketManager()
app.mount("/ws",app=socketManager.getSocketApp())

@app.get("/")
def get_home():
  return {"message":"hello world"}

@app.get("/test")
def get_home():
  return {"message":"hello world"}

@app.post("/calibration")
def read_item(item:Item):
  print(item.model_dump())
  itemJson = item.model_dump()
  cameraMatrix = np.array(itemJson['cameraMat'])
  pickle.dump(cameraMatrix,open("calibrationData.pkl","wb"))
  # return {item['name']}
  # return { "item_id": item_id, "q":q}


@app.post("/rotationMatrix")
def read_martrix(rmat:RotationMatrixDto):
  itemJson = rmat.model_dump()
  rmat = np.array(itemJson['rmat'])
  pickle.dump(rmat,open("rmat.pkl","wb"))
  print(itemJson)

if __name__ == "__main__":
  uvicorn.run("server:app",reload=True,host="172.22.173.67",port=8001)
