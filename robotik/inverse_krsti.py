import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()
prev_time = 0
long = 1

tracks_ka = [
    [0, 0, -188],
    [100, 0, -188],
    [0, 0, -188],
    [-100, 0, -188],
    [0, 0, -188]
]

tracks_ki = [

]

paha = 95
betis = 93

loop = 0

# px = 0
# py = 0
# pz = -188

data = 0

def ease_in_out(tm):
    return (3 * (tm**2)) - (2 * (tm**3))

def ease_in_out_cubic(tm):
    return 4 * (tm**3) if tm < 0.5 else 1 - np.pow(-2 * tm + 2, 3) / 2

def ease_out_cubic(tm):
    return 1 - np.pow(1 - tm, 3)

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

while 1:
    millis = time.time() - start_time
    time_now = millis - prev_time

    s = ease_in_out_cubic(time_now / long)
    s2 = ease_out_cubic(time_now / long)

    if data == 0 or data == 2:
        px = tracks_ka[data][0] + ((tracks_ka[data + 1][0] - tracks_ka[data][0]) * s2)
    else:
        px = tracks_ka[data][0] + ((tracks_ka[data + 1][0] - tracks_ka[data][0]) * s)

    py = tracks_ka[data][1] + (((tracks_ka[data + 1][1] - tracks_ka[data][1]) * time_now) / long)
    pz = tracks_ka[data][2] + ((tracks_ka[data + 1][2] - tracks_ka[data][2]) * s)

    if time_now >= long:
        prev_time = millis
        data = data + 1

    if data == len(tracks_ka) - 1:
        data = 0
        loop = loop + 1

    if loop == 5:
        break

    leng = np.sqrt((py ** 2) + (pz ** 2))

    acos_val = ((leng ** 2) - (paha ** 2) - (betis ** 2)) / (2 * paha * betis)
    if acos_val < 1 and acos_val > -1:
        ang2 = -np.acos(acos_val) * 180 / np.pi
    else:
        ang2 = 0
    ang1 = (90 + ((np.atan2(pz, py) + np.atan2((betis * np.sin(np.abs(ang2) * np.pi / 180)), (betis * np.cos(np.abs(ang2) * np.pi / 180)) + paha)) * 180 / np.pi))
    ang3 = 90 - ((90 - ang1) - ang2)
    ang4 = np.asin(px / pz) * 180 / np.pi

    y1 = paha * np.sin(ang1 * np.pi / 180)
    y2 = paha * np.sin(ang1 * np.pi / 180) + betis * np.sin((ang2 + ang1) * np.pi / 180)
    y3 = paha * np.sin(ang1 * np.pi / 180) + betis * np.sin((ang2 + ang1) * np.pi / 180) + 30
    z1 = paha * np.cos(ang1 * np.pi / 180)
    z2 = paha * np.cos(ang1 * np.pi / 180) + betis * np.cos((ang2 + ang1) * np.pi / 180)
    z3 = paha * np.cos(ang1 * np.pi / 180) + betis * np.cos((ang2 + ang1) * np.pi / 180)

    ax1.cla()
    ax1.plot([0, y1], [0, -z1], '-o', color="Blue", label="Robot Arm 1")
    ax1.plot([y1, y2], [-z1, -z2], '-o', color="Green", label="Robot Arm 2")
    ax1.plot([y2, y3], [-z2, -z3], '-o', color="Red", label="Robot Arm 3")
    ax1.set_xlim(-200, 200)
    ax1.set_ylim(-200, 200)
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.axvline(0, color='gray', linestyle='--')
    ax1.grid()
    ax1.legend()
    ax1.set_aspect('equal')

    x2_1 = betis * np.sin(ang4 * np.pi / 180) - 50
    x1_1 = paha * np.sin(ang4 * np.pi/180) + x2_1

    x2_2 = betis * np.sin(ang4 * np.pi / 180) + 50
    x1_2 = paha * np.sin(ang4 * np.pi/180) + x2_2

    z1_2 = z2 - (z2 * np.cos(ang4 * np.pi/180))
    z2_2 = z1 + (z1 - (z1 * np.cos(ang4 * np.pi/180)))
    z3_2 = z3

    p1 = x1_1 - 20 * np.cos(ang4 * np.pi/180)
    p2 = x1_1 + 120 * np.cos(ang4 * np.pi/180)

    q1 = z1_2 - 20 * np.sin(ang4 * np.pi/180)
    q2 = z1_2 + 120 * np.sin(ang4 * np.pi / 180)

    ax2.cla()
    ax2.plot([x1_1, x2_1], [-z1_2, -z2_2], '-o', color="Blue", label="Robot Arm 1")
    ax2.plot([x2_1, -50], [-z2_2, -z3_2], '-o', color="Green", label="Robot Arm 2")
    ax2.plot([x1_2, x2_2], [-z1_2, -z2_2], '-o', color="Blue", label="Robot Arm 1")
    ax2.plot([x2_2, 50], [-z2_2, -z3_2], '-o', color="Green", label="Robot Arm 2")
    # ax2.plot([p1, p2], [-q1, -q2], '-o', color="Brown")
    ax2.set_xlim(-200, 200)
    ax2.set_ylim(-200, 200)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.axvline(0, color='gray', linestyle='--')
    ax2.grid()
    ax2.legend()
    ax2.set_aspect('equal')

    plt.pause(0.001)

    # print(px, py, pz)
    print(ang1, ang2, ang3, ang4)
    # print(time_now)
    # print(np.atan2(pz, py) * 180/np.pi)
    # print("")
    # break

plt.ioff()
plt.show()