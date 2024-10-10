import requests
class NetworkManager:
    def __init__(self,host,port):
        self.host = host;
        self.port = port;
        pass

    def post(self,path,data):
        if path[0] == '/':
            path = path[1:]
        try:
            return requests.post("http://{}:{}/{}".format(self.host,self.port,path),json=data)
        except:
            print('can not send data to server')

        pass

    def get(self,path):
        pass
