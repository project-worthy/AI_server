import { io } from "socket.io-client"

const socket = io("http://118.47.179.23:8001/video",{path:"/ws",autoConnect:false});

//,{path: "/sockets"}

socket.connect();
socket.on("connectoreac",() => {
  console.log("print")
  socket.emit("ws:photo","testing");
})
