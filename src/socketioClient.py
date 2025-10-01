import socketio

class SocketManager:
    def __init__(self,host,port):
        print(host,port)
        self.sio = socketio.Client(handle_sigint=True)
        self.host = host
        self.port = port

        # @self.sio.on("ack_frame",)
        

    def connect(self):
        self.sio.connect("http://{}:{}/ws".format(self.host,self.port),socketio_path="/ws/socket.io",namespaces="/video")
    
    def disconnect(self):
        self.sio.disconnect()

    def sendData(self,msg):
        self.sio.emit("ws:photo",msg,namespace="/video")



def socket():
    pass

if __name__ == "__main__":
    socket()
