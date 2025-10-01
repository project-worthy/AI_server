
import cv2
import pickle
import time
from socketioClient import SocketManager

def socketConnect(host,port):
    print(host,port)
    socketManager = SocketManager(host,port)
    socketManager.connect()
    prev_time = 0
    

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        t_start = time.time()

        ret, img = cap.read()

        ret, buffer = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        x_as_bytes = pickle.dumps(buffer)
        # time.sleep(1/30)
        current_time = t_start - prev_time
        if(current_time > 1./5):
            prev_time = time.time()
            socketManager.sendData(x_as_bytes)
            print("send")
        fps = 1 / (time.time() - t_start) 
        cv2.putText(img,"FPS {:.2f}".format(fps),(10,30),0,5e-3 * 130, (0,0,255),2)
        cv2.imshow('img', img)

        # cv2.waitKey(int(1000 / 10))
        if cv2.waitKey(int(1000 / 1)) & 0xFF == 27:
            break

    cv2.destroyWindow("img")
    cap.release()
    socketManager.disconnect()

    
if __name__ == "__main__":
    socketConnect("","")
