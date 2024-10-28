import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches


plt.ion() ## Note this correction
fig=plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)
ax.axis([-10,500,-10,500])

# ax.draw()

def layout_loop(work_dict):
  while(True):
    print(work_dict)

    ax.cla()
    ax.axis([-10,500,-10,500])


    cir = patches.Circle((290,0),radius=work_dict["distances"][0] * 100,edgecolor='g',facecolor="none")
    ax.add_patch(cir)
    ax.scatter(290,0,c="g")
    

    cir = patches.Circle((290,316),radius=work_dict["distances"][1] * 100,edgecolor='r',facecolor="none")
    # plt.gca().add_patch(cir)
    ax.scatter(290,316,c="r")

    cir = patches.Circle((0,316),radius=work_dict["distances"][2] * 100,edgecolor='b',facecolor="none")
    ax.add_patch(cir)
    ax.scatter(0,316,c="b")


    ax.scatter(work_dict["device_coordinate"][0],work_dict["device_coordinate"][1],c="purple")
    ax.scatter(0,0,c="black")


    # plt.scatter(human_coord[0],human_coord[1],c="#777777")
    # plt.show()
    plt.pause(2)
    time.sleep(2)