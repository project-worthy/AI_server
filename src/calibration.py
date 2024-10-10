from time import time
from cv2.typing import MatLike
import numpy as np
import cv2
import glob
from utils import countDown,stackTick
import os 

def takePhoto(totalPhotoCount = 10,photoMethod = "auto",save=True,path="images"):
    cap = cv2.VideoCapture(0)
    a = os.getenv("IMG_WIDTH") or "480"
    cap.set(3,int(a))
    cap.set(4,int(os.getenv("IMG_HEIGHT") or 480))

    num = 0
    sec = 5
    pictureArr = []

    a = time()
    while cap.isOpened():

        _, img = cap.read()
        k = cv2.waitKey(40)

        if photoMethod == "auto" :
            if(stackTick(1)):
                print("{} sec unitl taking photo...".format(sec))
                sec = sec - 1
                if sec == 0:
                    if save:
                        cv2.imwrite('{}/img{}.png'.format(path,num), img)
                    pictureArr.append(img)
                    print("image saved! [{}/{}]".format(num + 1,totalPhotoCount))
                    num = num + 1
                    sec = 5
            if k == 27:
                break
        elif photoMethod == "manual":
            if k == 27:
                break
            elif k == ord('s'): # wait for 's' key to save and exit
                if save:
                    cv2.imwrite('{}/img{}.png'.format(path,num), img)
                pictureArr.append(img)
                print("image saved!")
                num = num + 1
            message = "Press s to save image, esc to cancel " + '{}/{} left'.format(num,totalPhotoCount)
            cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Img',img)

        if(num == totalPhotoCount):
            break

    cap.release()
    cv2.destroyAllWindows()
    return pictureArr

def calibrate(showPics=True,path="demoImages/calibration",paramType="intrinsic",camMat=None,dist=None):
    if paramType != "intrinsic" and paramType != "extrinsic":
        raise Exception("parmType must be 'intrinsic' or 'extrinsic'")
    root = os.getcwd()
    calibrateDir = os.path.join(root,path)
    imgPathList = glob.glob(os.path.join(calibrateDir,'*.png'))
    print(os.path.join(calibrateDir,'*.png'))
    print(len(imgPathList), "images found")

    nRows = 9
    nCols = 6
    termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows*nCols,3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)
    worldPtsList = []
    imgPtsList = []
    shape = ()
    print(imgPathList)
    for curImgPath in imgPathList:
        imgBGR = cv2.imread(curImgPath)
        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        shape = imgGray.shape[::-1]
        cornersFound, cornersOrg = cv2.findChessboardCorners(imgGray, (nCols,nRows), None)

        if cornersFound == True:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv2.cornerSubPix(imgGray, cornersOrg, (11,11), (-1,-1), termCriteria)
            imgPtsList.append(cornersRefined)
            if showPics:
                cv2.drawChessboardCorners(imgBGR, (nCols,nRows), cornersRefined, cornersFound)
                cv2.imshow('img', imgBGR)
                cv2.waitKey(500)
    cv2.destroyAllWindows()

    repError, camMatrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(worldPtsList, imgPtsList, shape, camMat, dist) #type: ignore
    print("Camera Matrix:\n",camMatrix)
    print("Reroj Error (pixels): {:.4f}".format(repError))

    paramPath = os.path.join(root, './src/calibrationParams.npz')
    np.savez(paramPath, camMatrix=camMatrix, distCoeffs=distCoeff)

    if(paramType == "extrinsic"):
        return rvecs, tvecs
    return camMatrix, distCoeff


def check_calibration(result,objpoints,imgpoints):
    cameraMatrix = result['cameraMatrix']
    dist = result['dist']
    rvecs = result['rvecs']
    tvecs = result['tvecs']
    rmat = result['rmat']

    # cameraMatrix,dist,tvecs,rvecs = pickle.load(open("calibration.pkl","rb"))
    img = cv2.imread('images/img0.png')
    h,  w = img.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# Undistort
    dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('caliResult1.png', dst)



# Undistort with Remapping
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5) #type: ignore
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('caliResult2.png', dst)




# Reprojection Error
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )
