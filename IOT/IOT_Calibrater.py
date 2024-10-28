import asyncio
from itertools import combinations;

class IOT_Calibrater:
  def __init__(self):
    self.sidDict = dict()
    self.tagCount = 0
    self.anchorCount = 0
    self.combinations = None
    self.combinationCount = 0


  def __call__(self,sid,json):

    self.sidDict[sid] = json
    self.sidDict[sid]["delay"] = 0


    if(json["type"] == 0):
      self.anchorCount += 1 
    if(json["type"] == 1):
      self.tagCount += 1
    

    print(self.tagCount,self.anchorCount)

    if self.anchorCount == 1 and self.tagCount == 1:
      pass
      print("starting calibrating")
      return True
    
    return False

  def remove(self,sid):
    jsonData = self.sidDict.get(sid)
    if(jsonData):
      if(jsonData["type"] == 0):
        self.anchorCount -= 1
      if(jsonData["type"] == 1):
        self.tagCount -= 1
  
  def get_combinations(self):

    if(not self.combinations):
      self.combinations = list(combinations(self.sidDict.keys(),2))
    if(self.combinationCount >= len(self.combinations)):
      return None
    result = self.combinations[self.combinationCount]
    self.combinationCount += 1

    self.from_sid = result[0]
    self.to_sid = result[1]

    return result
  
  def wait_received(self):
    self.received_promise = asyncio.Future()
    return self.received_promise 
  
  def wait_anchor(self):
    self.await_anchor = asyncio.Future()
    return self.await_anchor
  
  def resolve(self):
    self.received_promise.set_result(True)

  def resolve_anchor(self):
    self.await_anchor.set_result(True)

  def eject(self):
    self.received_promise.set_result(False)

  def reset(self):
    self.sidDict = dict()
    self.tagCount = 0
    self.anchorCount = 0
    self.combinations = None
    self.combinationCount = 0

  def add(self,delay):
    avg_delay = self.sidDict[self.from_sid]["delay"]
    delay_total = avg_delay * 2 + delay * 2
    if(avg_delay == 0):
      delay_total /= 2
    else:
      delay_total /= 4
    self.sidDict[self.from_sid]["delay"] = delay_total

    avg_delay = self.sidDict[self.to_sid]["delay"]
    delay_total = avg_delay * 2 + delay * 2
    if(avg_delay == 0):
      delay_total /= 2
    else:
      delay_total /= 4
    self.sidDict[self.to_sid]["delay"] = delay_total

    print(self.sidDict)