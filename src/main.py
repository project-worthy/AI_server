import numpy as np
import cv2
import inquirer  # noqa
from dotenv import load_dotenv

from socketio.async_client import asyncio

from client import socketConnect
from network import NetworkManager
from socketioClient import SocketManager
from calibration import  takePhoto,calibrate

import os
import glob

import shutil


import requests

load_dotenv(verbose=True)
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")

async def initializeMatrix(result):
    task1 = asyncio.create_task(postCalibrationData(result))
    await task1


async def postCalibrationData(result):
    rvecs_list = [list(map(lambda a: a.tolist(),list(result['rvecs'])))]
    tvecs_list = [list(map(lambda a: a.tolist(),list(result['tvecs'])))]
    print(result['rmat'])
    mbobj = {
        "cameraMat":result['cameraMatrix'].tolist(),
        "dist":result['dist'].tolist(),
        "rvecs":rvecs_list,
        "tvecs":tvecs_list,
        "rmat": result['rmat'].tolist()
    }

    res = requests.post("http://{}:{}/calibration".format(HOST,PORT),json=mbobj)
    return res.json()
     
questions =[
    inquirer.Confirm("instrinicParamCam",message="Do you want to set Internal Parameter?",default=False),
    inquirer.Confirm("exstrinicParamCam",message="Does position of your camera changed?",default=False)
]
questions_camera=[
    inquirer.List(
            "photoMethod",
            message="Method to take photo:",
            choices=["manual", "auto"],
        )
    ]

# socketManager = SocketManager(HOST,PORT)
networkManager = NetworkManager(HOST,PORT);

answer = inquirer.prompt(questions)
print(answer)
if answer and answer["instrinicParamCam"]:
    takePhoto(10,photoMethod="manual",save=True,path="images/calibrations")
camMatrix,distCoeff = calibrate(showPics=True,path="images/calibrations",paramType="intrinsic")
intrinsicData = {
    "camMatrix":camMatrix.tolist(),
    "distCoeff":distCoeff.tolist(),
}
networkManager.post("intrinsic",intrinsicData)

if answer and answer["exstrinicParamCam"]:
    takePhoto(1,photoMethod="manual",save=True,path="images/extrinsic")

# 내부 파라미터를 찍은 사진과 외부 파리미터를 위해 찍은 사진을 하나로 합친다.

for jpgfile in glob.iglob(os.path.join("images/extrinsic","*.png")):
    if(os.path.isfile("images/total/get.png")):
        os.remove("images/total/get.png");
    shutil.copy(jpgfile,"images/total/get.png")

rvecs,tvecs = calibrate(showPics=True,path="images/total",paramType="extrinsic",camMat=camMatrix,dist=distCoeff)
print("camera position\n",np.matmul(-1 * cv2.Rodrigues(rvecs[0])[0].T,tvecs[0]))
print("rvecs\n",rvecs[0])
print("tvecs\n",[[x[0] * 0.01] for x in tvecs[0].tolist()])
extrinsicData={
    "rvecs":rvecs[0].tolist(),
    "tvecs":[[x[0] * 0.01] for x in tvecs[0].tolist()]
}
networkManager.post("extrinsic",extrinsicData)
# socketManager.connect()
socketConnect(HOST,PORT)
