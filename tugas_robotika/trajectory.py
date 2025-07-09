import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
prev_time = 0

long = 2

tracks = [
    [0, 20, 0],
    [-14, 3, 0],
    [-14, 13, 0],
    [-10, 10, 0],
    [-6, 13, 0],
    [-6, 3, 0],
    [-2, 3, 1],
    [2, 3, 0],
    [0, 3, 0],
    [0, 13, 0],
    [-2, 13, 0],
    [2, 13, 0],
    [6, 3, 1],
    [10, 13, 0],
    [14, 3, 0],
    [12, 8, 1],
    [8, 8, 0],
    [20, 0, 0],
    [20, 0, 0]
]

leng = [10, 10]

plt.ion()
fig, ax = plt.subplots()
data = 0

track_x_1 = []
track_x_2 = []
track_x_3 = []

track_y_1 = []
track_y_2 = []
track_y_3 = []

while 1:
    millis = time.time() - start_time
    tim = millis - prev_time

    px = tracks[data][0] + (((tracks[data+1][0] - tracks[data][0]) * tim) / long)
    py = tracks[data][1] + (((tracks[data+1][1] - tracks[data][1]) * tim) / long)

    if data >= 1 and data <= 4:
        track_x_1.append(px)
        track_y_1.append(py)
    if data >= 6 and data <= 10:
        track_x_2.append(px)
        track_y_2.append(py)
    if data >= 12 and data < 16:
        track_x_3.append(px)
        track_y_3.append(py)

    if tim >= long:
        prev_time = millis
        data = data + 1

    if data == len(tracks) - 1:
        break

    rad = np.sqrt((px ** 2) + (py ** 2))

    acos_value = ((rad ** 2) - (leng[0] ** 2) - (leng[1] ** 2)) / (2 * leng[0] * leng[1])
    if -1 <= acos_value <= 1:
        theta2 = np.acos(acos_value) * 180 / np.pi
    else:
        theta2 = 0
    theta1 = (np.atan2(py, px) - np.atan2((leng[1] * np.sin(theta2 * np.pi / 180)), ((leng[1] * np.cos(theta2 * np.pi / 180)) + leng[0]))) * 180 / np.pi

    x1 = leng[0] * np.cos(theta1 * np.pi / 180)
    x2 = leng[0] * np.cos(theta1 * np.pi / 180) + leng[1] * np.cos((theta1 + theta2) * np.pi / 180)
    y1 = leng[0] * np.sin(theta1 * np.pi / 180)
    y2 = leng[0] * np.sin(theta1 * np.pi / 180) + leng[1] * np.sin((theta1 + theta2) * np.pi / 180)

    ax.cla()
    ax.plot([0, x1], [0, y1], '-o', color="Blue", label="Robot Arm 1")
    ax.plot([x1, x2], [y1, y2], '-o', color="Green", label="Robot Arm 2")
    ax.plot(track_x_1, track_y_1, '-r', label="Writing Track")
    ax.plot(track_x_2, track_y_2, '-r')
    ax.plot(track_x_3, track_y_3, '-r')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.grid()
    ax.legend()
    ax.set_aspect('equal')

    plt.pause(0.001)

plt.ioff()
plt.show()