import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
prev_time = 0

long = 4 * np.pi

leng = [10, 10]

plt.ion()
fig, ax = plt.subplots()

track_x = []
track_y = []

while 1:
    millis = time.time() - start_time
    tim = millis - prev_time

    px = 16 * (np.sin(tim) ** 3)
    py = (13 * np.cos(tim)) - (5 * np.cos(2 * tim)) - (2 * np.cos(3 * tim)) - np.cos(4 * tim)

    track_x.append(px)
    track_y.append(py)

    if tim >= long:
        break

    rad = np.sqrt((px ** 2) + (py ** 2))

    theta2 = np.acos(((rad ** 2) - (leng[0] ** 2) - (leng[1] ** 2)) / (2 * leng[0] * leng[1])) * 180 / np.pi
    theta1 = (np.atan2(py, px) - np.atan2((leng[1] * np.sin(theta2 * np.pi / 180)), ((leng[1] * np.cos(theta2 * np.pi / 180)) + leng[0]))) * 180 / np.pi

    x1 = leng[0] * np.cos(theta1 * np.pi / 180)
    x2 = leng[0] * np.cos(theta1 * np.pi / 180) + leng[1] * np.cos((theta1 + theta2) * np.pi / 180)
    y1 = leng[0] * np.sin(theta1 * np.pi / 180)
    y2 = leng[0] * np.sin(theta1 * np.pi / 180) + leng[1] * np.sin((theta1 + theta2) * np.pi / 180)

    ax.cla()
    ax.plot([0, x1, x2], [0, y1, y2], '-o', label="Robot Arm")
    ax.plot(track_x, track_y, '-r', label="Writing Track")
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.grid()
    ax.legend()

    plt.pause(0.01)

plt.ioff()
plt.show()