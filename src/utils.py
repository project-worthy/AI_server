import time
import re
tick = None 
prevTime = None
def getStartIndex(message:str):
    messageReg = re.findall(r'\{(.*?)\}', message)
    startNum = []
    for i in range(len(messageReg)):
        if str(i) not in messageReg:
            startNum.append(i)
    return startNum,len(messageReg)

def putIndexInFormatMessage(startNum=None,message="",*args):
    if(startNum == None or message == None):
        raise Exception("need to pass in all 3 arguments")

    for i in range(len(startNum)):
        res = re.search(r'\{\s?\}',message);
        if not res:
            break;
        searchIndex = res.span()
        message = message[:searchIndex[0]] + "{" + str(startNum[i]) + "}" + message[searchIndex[1]:]
    return message

def countDown(sec:int,message:str,*args):
    startNum,findLength = getStartIndex(message)

    if findLength != len(args) + 1:
        raise Exception("need to match the number of {} in the message. {0} is for sec to display")

    message = putIndexInFormatMessage(startNum,message,*args)

    for i in range(sec, 0, -1):
        print(message.format(i,*args))
        time.sleep(1)
    print('')

def stackTick(sec):
    global tick
    global prevTime
    if tick is None:
        tick = 0
    if prevTime is None:
        prevTime= time.time()
    current = time.time()
    tick = tick + (current - prevTime)
    prevTime = current
    if(tick > 1):
        tick = 0
        return True
    else:
        return False
