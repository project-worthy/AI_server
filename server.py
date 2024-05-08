import socket
import cv2
import pickle
from _thread import *
import mmcv
import struct
import numpy as np
from testing import VideoManager, process_one_image
from mmpose.evaluation.functional import nms
from mmpose.apis import inference_topdown, MMPoseInferencer
from mmpose.structures import merge_data_samples, split_instances


ip = "0.0.0.0"


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1000000)
port =  6124
s.bind((ip, port))
s.listen()


def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # print(msglen)
    # Read the message data
    return recvall(sock, msglen)


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def threaded(client_socket, addr):
    print('>> Connected by :', addr[0], ':', addr[1])
    videoManager = VideoManager()

    # process until client disconnect #
    while True:
        try:
            # send client if data recieved(echo) #

            data = recv_msg(client_socket)
            if type(data) is type(None):
                print('>> Disconnected by ' + addr[0], ':', addr[1])
                # cv2.destroyWindow("image")
                cv2.destroyAllWindows() 
                break
            data = pickle.loads(data)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)



            ## 수정 전 부분
            pred_instance = process_one_image(videoManager.options, img, videoManager.detector,
                                                videoManager.pose_estimator, videoManager.visualizer,
                                                0.001)

            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == 0, # 0 det_cat_id
                                          pred_instance.scores > 0.3)] # 0.3 bbox_thr
            bboxes = bboxes[nms(bboxes, 0.3 ), :4] #0.3 nms_thr

            # predict keypoints
            pose_results = inference_topdown(videoManager.pose_estimator, img, bboxes)
            # print(pose_results)
            data_samples = merge_data_samples(pose_results)

            # show the results
            if isinstance(img, str):
                img = mmcv.imread(img, channel_order='rgb')
            elif isinstance(img, np.ndarray):
                img = mmcv.bgr2rgb(img)

            videoManager.visualizer.add_datasample(
                'result',
                img,
                data_sample=data_samples,
                draw_gt=False,
                draw_heatmap= videoManager.options.draw_heatmap, # default: false - draw_heatmap
                draw_bbox=videoManager.options.draw_bbox, # draw_bbox
                show_kpt_idx=videoManager.options.show_kpt_idx, # default: false
                skeleton_style=videoManager.options.skeleton_style, # choice: mmpose , openpose - Skeleton style selection
                show=videoManager.options.show, # default: False
                wait_time=videoManager.options.show_interval,
                kpt_thr=videoManager.options.kpt_thr) # default:0.3 kpt_thr

            # 수정 전 부분 끝

            # chat to client connecting client #
            # chat to client connecting client except person sending message #
            for client in client_sockets:
                if client != client_socket:
                    client.send(data)

        except ConnectionResetError as e:
            print(e)
            cv2.destroyWindow("result")

            print('>> Disconnected by ' + addr[0], ':', addr[1])
            break

    if client_socket in client_sockets:
        client_sockets.remove(client_socket)
        print('remove client list : ', len(client_sockets))

    client_socket.close()


print('>> server started with: ', ip, ':', port)
client_sockets = []
while True:
    try:
        while True:
            print('>> Wait')

            client_socket, addr = s.accept()
            client_sockets.append(client_socket)
            start_new_thread(threaded, (client_socket, addr))
            print("join count : ", len(client_sockets))
    except Exception as e:
        print('error : ', e)

    finally:
        s.close()
