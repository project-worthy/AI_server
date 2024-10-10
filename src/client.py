
import cv2
import pickle
import time
from socketioClient import SocketManager

def socketConnect(host,port):
    socketManager = SocketManager(host,port)
    socketManager.connect()
    

    cap = cv2.VideoCapture(0)
    # cap.set(3, 640)  # width
    # cap.set(4, 480)  # height
    while cap.isOpened():
        ret, img = cap.read()
        cv2.imshow('img', img)

        ret, buffer = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        x_as_bytes = pickle.dumps(buffer)
        time.sleep(1/60)
        socketManager.sendData(x_as_bytes);
        
        if cv2.waitKey(int(1000 / 10)) & 0xFF == 27:
            break

    cv2.destroyWindow("img")
    cap.release()
    socketManager.disconnect()

    
if __name__ == "__main__":
    socketConnect("","")
