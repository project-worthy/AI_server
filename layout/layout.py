import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from utils.plt_math import slope_from_angle


plt.ion() ## Note this correction
fig=plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.axis([-10,500,-10,500])

# ax.draw()

def layout_loop(work_dict):
  camera_degree = work_dict.get("camera_degree") or 0
  slope_value = slope_from_angle(camera_degree)
  between_devices_height = work_dict.get("height")
  between_devices_width = work_dict.get("width")

  point3 = (between_devices_height,0)
  point2 = (between_devices_height,between_devices_width)
  point1 = (0,between_devices_width)

  plt_coord = math.ceil(max(between_devices_height,between_devices_width) / 100) * 100

  while(True):
    print(work_dict)
    # initialize
    ax.cla()
    ax.axis([-plt_coord,plt_coord,-10,plt_coord * 2 - 10])

    # draw watching camera degrees

    if(camera_degree in (0,180,360)):
      ax.axvline(x=0,c="teal",lw=3)
      ax.axhline(y=0,linestyle="dashed",c="teal",lw=3)
    elif(camera_degree in (90,270)):
      ax.axvline(x=0,linestyle="dashed",c="teal",lw=3)
      ax.axhline(y=0,c="teal",lw=3)     
    else:
      ax.axline((0,0),slope=slope_value,c="teal",lw=3)
      ax.axline((0,0),slope=-1/slope_value,linestyle="dashed",c="teal",lw=3)

    cir = patches.Circle(point1,radius=work_dict["distances"][0] * 100,edgecolor='g',facecolor="none")
    ax.add_patch(cir)
    ax.scatter(*point1,c="g")
    

    cir = patches.Circle(point2,radius=work_dict["distances"][1] * 100,edgecolor='r',facecolor="none")
    ax.add_patch(cir)
    ax.scatter(*point2,c="r")

    cir = patches.Circle(point3,radius=work_dict["distances"][2] * 100,edgecolor='b',facecolor="none")
    ax.add_patch(cir)
    ax.scatter(*point3,c="b")


    for device in work_dict["device_coordinates"]:
      # if(device is not None):
      ax.scatter(*device,c="purple")
      cir = patches.Circle(device,radius=work_dict["threshold_distance"] * 100,edgecolor='purple',facecolor="none")
      ax.add_patch(cir)
      # ax.scatter(work_dict["device_coordinate"][0],work_dict["device_coordinate"][1],c="purple")
    ax.scatter(0,0,c="black")

    for human in work_dict["humans_locations"]:
      ax.scatter(human[0] * 100,human[1] * 100,c="lightsalmon")

    plt.pause(2)
    time.sleep(2)