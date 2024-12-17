import { io } from "socket.io-client"
import readline from "readline/promises"
import { stdin, stdout } from "process"

const socket = io("http://HOSTNAME/temp",{path:"/ws",autoConnect:false});

//,{path: "/sockets"}

socket.connect();

const rl = readline.createInterface({input:stdin,output:stdout})

while(true){
  control_led()
  console.log("print");
}

async function control_led(){
  const answer = await rl.question("turn on? : y/n");
  if(answer === "y"){
    socket.emit("turn_on","turn_on");
  }
  else {
    socket.emit("turn_off","turn_off");
  }
}
