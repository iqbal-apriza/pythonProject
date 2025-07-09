import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()
prev_time = 0
long = 3

tracks = [
    [64.75036, 64.75036, -104.85473+10],
    [64.75036, 64.75036+30, -104.85473],
    [64.75036, 64.75036, -104.85473],
    [64.75036, 64.75036-30, -104.85473],
    [64.75036, 64.75036, -104.85473+10],
]

len1 = 35
len2 = 67
len3 = 70

data = 0

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

track_x1 = []

while 1:
    millis = time.time() - start_time
    tim = millis - prev_time

    px = tracks[data][0] + (((tracks[data+1][0] - tracks[data][0]) * tim) / long)
    py = tracks[data][1] + (((tracks[data+1][1] - tracks[data][1]) * tim) / long)
    pz = tracks[data][2] + (((tracks[data+1][2] - tracks[data][2]) * tim) / long)

    if tim >= long:
        prev_time = millis
        data = data + 1

    if data == len(tracks) - 1:
        data = 0

    rad = np.sqrt((px ** 2) + (py ** 2))
    leng = np.sqrt(((rad - len1) ** 2) + (pz ** 2))

    coxa = np.atan2(py, px) * 180 / np.pi
    femur = (np.acos(((len2 ** 2) + (leng ** 2) - (len3 ** 2)) / (2 * leng * len2)) + np.atan2(pz, (rad - len1))) * 180 / np.pi
    tibia = -np.acos(((leng ** 2) - (len2 ** 2) - (len3 ** 2)) / (2 * len2 * len3)) * 180 / np.pi
    femur2 = (np.atan2(pz, (rad - len1)) - np.atan2(len3 * np.sin(tibia * np.pi / 180), (len3 * np.cos(tibia * np.pi / 180)) + len2)) * 180 / np.pi

    x0_1 = len1 * np.cos(coxa * np.pi / 180)
    x1_1 = (len1 + (len2 * np.cos(femur * np.pi/180))) * np.cos(coxa * np.pi / 180)
    x2_1 = (len1 + (len2 * np.cos(femur * np.pi/180)) + (len3 * np.cos((femur + tibia) * np.pi/180))) * np.cos(coxa * np.pi / 180)
    z1_1 = (len2 * np.sin(femur * np.pi/180))
    z2_1 = (len2 * np.sin(femur * np.pi/180)) + (len3 * np.sin((femur + tibia) * np.pi/180))

    ax1.cla()
    ax1.plot([0, x0_1], [0, 0], '-o', color="red", label="Cocsa")
    ax1.plot([x0_1, x1_1], [0, z1_1], '-o', color="blue", label="Femur")
    ax1.plot([x1_1, x2_1], [z1_1, z2_1], '-o', color="green", label="Tibia")
    ax1.set_xlim(-150, 150)
    ax1.set_ylim(-150, 150)
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.axvline(0, color='gray', linestyle='--')
    ax1.grid()
    ax1.legend()

    x0_2 = len1 * np.cos(coxa * np.pi / 180)
    y0 = len1 * np.sin(coxa * np.pi / 180)
    y1 = (len1 + (len2 * np.cos(femur * np.pi/180))) * np.sin(coxa * np.pi / 180)
    y2 = (len1 + (len2 * np.cos(femur * np.pi/180)) + (len3 * np.cos((femur + tibia) * np.pi/180))) * np.sin(coxa * np.pi / 180)

    ax2.cla()
    ax2.plot([x1_1, x2_1], [y1, y2], '-o', color="green", label="Tibia")
    ax2.plot([x0_1, x1_1], [y0, y1], '-o', color="blue", label="Femur")
    ax2.plot([0, x0_1], [0, y0], '-o', color="red", label="Cocsa")
    ax2.set_xlim(-150, 150)
    ax2.set_ylim(-150, 150)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.axvline(0, color='gray', linestyle='--')
    ax2.grid()
    ax2.legend()

    plt.pause(0.001)

    coxa_bit = int(512 + ((coxa / 300) * 1023))
    femur_bit = int(636 + ((femur / 300) * 1023))
    tibia_bit = int(309 - ((tibia / 300) * 1023))

    # print(np.round(tim, 2))
    print(px, py, pz)
    print(coxa, femur, tibia, femur2)
    print(coxa_bit, femur_bit, tibia_bit)
    # print(data)
    print("")
    time.sleep(0.01)

plt.ioff()
plt.show()