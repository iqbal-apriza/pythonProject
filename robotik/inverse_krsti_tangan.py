import numpy as np
import time
import matplotlib.pyplot as plt

a1 = 75
a2 = 85

start_time = time.time()

ang1 = 0;   ang3 = 0;   ang5 = 0
ang2 = 0;   ang4 = 0;   ang6 = 0

ang1_2 = 0; ang3_2 = 0; ang5_2 = 0
ang2_2 = 0; ang4_2 = 0; ang6_2 = 0

# track_tka = [
#     [70, 0, -120, 90, 0, 80],
#     [50, 20, -80, 70, 0, 80],
#     [-30, 50, -30, 45, 0, 80],
#     [70, 0, -120, 90, 0, 80]
# ]

# track_tki = [
#     [70, 0, -120, 90, 0, 80],
#     [50, 20, -80, 70, 0, 80],
#     [-30, 50, -30, 45, 0, 80],
#     [70, 0, -120, 90, 0, 80]
# ]

idx = 0
speed = 0
prev_time = 0
time_now = 0

def constrain(value, min_val, max_val):
    if value > max_val:
        return max_val
    elif value < min_val:
        return min_val
    else:
        return value

def easeInCubic(x):
    return x * x * x

def easeOutCubic(x):
    return 1 - np.pow(1 - x, 3)

def ready_t1_fix():
    global px, py, pz, px_2, py_2, pz_2
    global ang1, ang2, ang3, ang4, ang5, ang6
    global ang1_2, ang2_2, ang3_2, ang4_2, ang5_2, ang6_2

    px = 70
    py = 0
    pz = -120
    ang1 = 90
    ang5 = 0
    ang6 = 80

    pxr = px
    pyr = py * np.cos(ang1 * np.pi/180) - pz * np.sin(ang1 * np.pi/180)
    pzr = py * np.sin(ang1 * np.pi/180) + pz * np.cos(ang1 * np.pi/180)

    leng = np.sqrt(np.pow(pxr, 2) + np.pow(pyr, 2) - np.pow(pzr, 2))
    ang4 = np.acos(constrain((np.pow(leng, 2) - np.pow(a1, 2) - np.pow(a2, 2)) / (2 * a1 * a2), -1, 1)) * 180/np.pi

    beta = np.acos(constrain((np.pow(a1, 2) + np.pow(leng, 2) - np.pow(a2, 2)) / (2 * a1 * leng), -1, 1)) * 180 / np.pi
    leng = np.sqrt(np.pow(pxr, 2) + np.pow(pyr, 2))
    alpha = np.atan2(pzr, leng) * 180 / np.pi
    ang2ElUp = (np.atan2(pxr, pyr) + np.acos(constrain(np.cos(beta * np.pi / 180) / np.cos(alpha * np.pi / 180), -1, 1))) * 180 / np.pi
    ang2ElDo = (np.atan2(pxr, pyr) - np.acos(constrain(np.cos(beta * np.pi / 180) / np.cos(alpha * np.pi / 180), -1, 1))) * 180 / np.pi
    ang2 = (np.atan2(pxr, pyr) + np.acos(constrain(np.cos(beta * np.pi / 180) / np.cos(alpha * np.pi / 180), -1, 1))) * 180 / np.pi

    xr = pxr * np.cos(ang2 * np.pi / 180) - pyr * np.sin(ang2 * np.pi / 180)
    yr = pxr * np.sin(ang2 * np.pi / 180) + pyr * np.cos(ang2 * np.pi / 180)
    ang3 = np.atan2(xr, pzr) * 180 / np.pi

    ang1 = 90 - ang1
    ang2 = 90 - ang2
    print(f"{float(ang1):.2f}, {float(ang2):.2f}, {float(ang3):.2f}, {float(ang4):.2f}, {float(ang5):.2f}, {float(ang6):.2f} | {float(ang1):.2f}, {float(ang2):.2f}, {float(ang3):.2f}, {float(ang4):.2f}, {float(ang5):.2f}, {float(ang6):.2f}")

def theta1_fix(tracks_tka, tracks_tki, interval, kec):
    global px, py, pz, px_2, py_2, pz_2
    global ang1, ang2, ang3, ang4, ang5, ang6
    global ang1_2, ang2_2, ang3_2, ang4_2, ang5_2, ang6_2
    global prev_time, idx, time_now

    millis = time.time() - start_time
    time_now = millis - prev_time

    px = tracks_tka[idx][0] + ((tracks_tka[idx + 1][0] - tracks_tka[idx][0]) * interval[0][0])
    py = tracks_tka[idx][1] + ((tracks_tka[idx + 1][1] - tracks_tka[idx][1]) * interval[0][1])
    pz = tracks_tka[idx][2] + ((tracks_tka[idx + 1][2] - tracks_tka[idx][2]) * interval[0][2])
    ang1 = tracks_tka[idx][3] + ((tracks_tka[idx + 1][3] - tracks_tka[idx][3]) * interval[0][3])
    ang5 = tracks_tka[idx][4] + ((tracks_tka[idx + 1][4] - tracks_tka[idx][4]) * interval[0][4])
    ang6 = tracks_tka[idx][5] + ((tracks_tka[idx + 1][5] - tracks_tka[idx][5]) * interval[0][5])

    px_2 = tracks_tki[idx][0] + ((tracks_tki[idx + 1][0] - tracks_tki[idx][0]) * interval[1][0])
    py_2 = tracks_tki[idx][1] + ((tracks_tki[idx + 1][1] - tracks_tki[idx][1]) * interval[1][1])
    pz_2 = tracks_tki[idx][2] + ((tracks_tki[idx + 1][2] - tracks_tki[idx][2]) * interval[1][2])
    ang1_2 = tracks_tki[idx][3] + ((tracks_tki[idx + 1][3] - tracks_tki[idx][3]) * interval[1][3])
    ang5_2 = tracks_tki[idx][4] + ((tracks_tki[idx + 1][4] - tracks_tki[idx][4]) * interval[1][4])
    ang6_2 = tracks_tki[idx][5] + ((tracks_tki[idx + 1][5] - tracks_tki[idx][5]) * interval[1][5])

    if time_now >= kec:
        idx = idx + 1
        prev_time = millis

    if idx >= len(tracks_tka) - 1:
        idx = 0

    pxr = px
    pyr = py * np.cos(ang1 * np.pi/180) - pz * np.sin(ang1 * np.pi/180)
    pzr = py * np.sin(ang1 * np.pi/180) + pz * np.cos(ang1 * np.pi/180)

    leng = np.sqrt(np.pow(pxr, 2) + np.pow(pyr, 2) - np.pow(pzr, 2))
    ang4 = np.acos(constrain((np.pow(leng, 2) - np.pow(a1, 2) - np.pow(a2, 2)) / (2 * a1 * a2), -1, 1)) * 180/np.pi

    beta = np.acos(constrain((np.pow(a1, 2) + np.pow(leng, 2) - np.pow(a2, 2)) / (2 * a1 * leng), -1, 1)) * 180 / np.pi
    leng = np.sqrt(np.pow(pxr, 2) + np.pow(pyr, 2))
    alpha = np.atan2(pzr, leng) * 180 / np.pi
    ang2 = (np.atan2(pxr, pyr) + np.acos(constrain(np.cos(beta * np.pi / 180) / np.cos(alpha * np.pi / 180), -1, 1))) * 180 / np.pi

    xr = pxr * np.cos(ang2 * np.pi / 180) - pyr * np.sin(ang2 * np.pi / 180)
    yr = pxr * np.sin(ang2 * np.pi / 180) + pyr * np.cos(ang2 * np.pi / 180)
    ang3 = np.atan2(xr, pzr) * 180 / np.pi

    ang1 = 90 - ang1
    ang2 = 90 - ang2  

    pxr = px_2
    pyr = py_2 * np.cos(ang1_2 * np.pi/180) - pz_2 * np.sin(ang1_2 * np.pi/180)
    pzr = py_2 * np.sin(ang1_2 * np.pi/180) + pz_2 * np.cos(ang1_2 * np.pi/180)

    leng = np.sqrt(np.pow(pxr, 2) + np.pow(pyr, 2) - np.pow(pzr, 2))
    ang4_2 = np.acos(constrain((np.pow(leng, 2) - np.pow(a1, 2) - np.pow(a2, 2)) / (2 * a1 * a2), -1, 1)) * 180/np.pi

    beta = np.acos(constrain((np.pow(a1, 2) + np.pow(leng, 2) - np.pow(a2, 2)) / (2 * a1 * leng), -1, 1)) * 180 / np.pi
    leng = np.sqrt(np.pow(pxr, 2) + np.pow(pyr, 2))
    alpha = np.atan2(pzr, leng) * 180 / np.pi
    ang2_2 = (np.atan2(pxr, pyr) + np.acos(constrain(np.cos(beta * np.pi / 180) / np.cos(alpha * np.pi / 180), -1, 1))) * 180 / np.pi

    xr = pxr * np.cos(ang2 * np.pi / 180) - pyr * np.sin(ang2 * np.pi / 180)
    yr = pxr * np.sin(ang2 * np.pi / 180) + pyr * np.cos(ang2 * np.pi / 180)
    ang3_2 = np.atan2(xr, pzr) * 180 / np.pi

    ang1_2 = 90 - ang1_2
    ang2_2 = 90 - ang2_2  

def manual(tracks_tka, tracks_tki, interval, kec):
    global ang1, ang2, ang3, ang4, ang5, ang6
    global ang1_2, ang2_2, ang3_2, ang4_2, ang5_2, ang6_2
    global prev_time, idx

    millis = time.time() - start_time
    time_now = millis - prev_time

    ang1 = tracks_tka[idx][0] + ((tracks_tka[idx + 1][0] - tracks_tka[idx][0]) * interval[0][0])
    ang2 = tracks_tka[idx][1] + ((tracks_tka[idx + 1][1] - tracks_tka[idx][1]) * interval[0][1])
    ang3 = tracks_tka[idx][2] + ((tracks_tka[idx + 1][2] - tracks_tka[idx][2]) * interval[0][2])
    ang4 = tracks_tka[idx][3] + ((tracks_tka[idx + 1][3] - tracks_tka[idx][3]) * interval[0][3])
    ang5 = tracks_tka[idx][4] + ((tracks_tka[idx + 1][4] - tracks_tka[idx][4]) * interval[0][4])
    ang6 = tracks_tka[idx][5] + ((tracks_tka[idx + 1][5] - tracks_tka[idx][5]) * interval[0][5])
    
    ang1_2 = tracks_tki[idx][0] + ((tracks_tki[idx + 1][0] - tracks_tki[idx][0]) * interval[1][0])
    ang2_2 = tracks_tki[idx][1] + ((tracks_tki[idx + 1][1] - tracks_tki[idx][1]) * interval[1][1])
    ang3_2 = tracks_tki[idx][2] + ((tracks_tki[idx + 1][2] - tracks_tki[idx][2]) * interval[1][2])
    ang4_2 = tracks_tki[idx][3] + ((tracks_tki[idx + 1][3] - tracks_tki[idx][3]) * interval[1][3])
    ang5_2 = tracks_tki[idx][4] + ((tracks_tki[idx + 1][4] - tracks_tki[idx][4]) * interval[1][4])
    ang6_2 = tracks_tki[idx][5] + ((tracks_tki[idx + 1][5] - tracks_tki[idx][5]) * interval[1][5])

    if time_now >= kec:
        idx = idx + 1
        prev_time = millis

    if idx == len(tracks_tka) - 1:
        idx = 0

def gerakan1():
    global ang1, ang2, ang3, ang4, ang5, ang6
    global ang1_2, ang2_2, ang3_2, ang4_2, ang5_2, ang6_2
    global prev_time, idx, time_now

    interval = np.zeros([2, 6])

    millis = time.time() - start_time
    time_now = millis - prev_time

    track_tka = [
        [70, 0, -120, 90, 0, 80],
        [50, 20, -80, 70, 0, 80],
        [-30, 50, -30, 45, 0, 80],
        [70, 0, -120, 90, 0, 80]
    ]

    track_tki = [
        [70, 0, -120, 90, 0, 80],
        [50, 20, -80, 70, 0, 80],
        [-30, 50, -30, 45, 0, 80],
        [70, 0, -120, 90, 0, 80]
    ]

    match idx:
        case 2:
            speed = 5
            interval[0][0] = easeOutCubic(time_now / speed)
            interval[0][1] = easeInCubic(time_now / speed)
            interval[0][2] = easeOutCubic(time_now / speed)

            interval[1][0] = easeOutCubic(time_now / speed)
            interval[1][1] = easeInCubic(time_now / speed)
            interval[1][2] = easeOutCubic(time_now / speed)

            for i in range(2):
                for j in range(3):
                    interval[i][j+3] = time_now / speed
        case default:
            speed = 5
            for i in range(2):
                for j in range(6):
                    interval[i][j] = time_now / speed
        
    theta1_fix(track_tka, track_tki, interval, speed)
    print(f"{float(ang1):.2f}, {float(ang2):.2f}, {float(ang3):.2f}, {float(ang4):.2f}, {float(ang5):.2f}, {float(ang6):.2f} | {float(ang1_2):.2f}, {float(ang2_2):.2f}, {float(ang3_2):.2f}, {float(ang4_2):.2f}, {float(ang5_2):.2f}, {float(ang6_2):.2f}")
    # print(px, py, pz)


ready_t1_fix()
time.sleep(2)
prev_time = time.time() - start_time
while 1:
    # gerakan1()
    break


    # millis = time.time() - start_time
    # time_now = millis - prev_time
    # time_now = 0

    # px = (16 * np.pow(np.sin(time_now), 3)) * 5
    # py = 120
    # pz = (13 * np.cos(time_now) - 5 * np.cos(2 * time_now) - 2 * np.cos(3 * time_now) - np.cos(4 * time_now)) * 5

    # px = np.sin(2 * time_now) * 30 + 30
    # py = 100
    # pz = ((np.cos(4 * time_now) / 2) + 0.5) * 30

    # px = 2 * np.sin(time_now) * 10 + 20
    # py = 120
    # pz = np.sin(2 * time_now) * 10

    # match idx:
    #     case 2:
    #         speed = 5
    #         interval[0][0] = easeOutCubic(time_now / speed)
    #         interval[0][1] = easeInCubic(time_now / speed)
    #         interval[0][2] = easeOutCubic(time_now / speed)

    #         interval[1][0] = easeOutCubic(time_now / speed)
    #         interval[1][1] = easeInCubic(time_now / speed)
    #         interval[1][2] = easeOutCubic(time_now / speed)

    #         for i in range(2):
    #             for j in range(3):
    #                 interval[i][j+3] = time_now / speed
    #     case default:
    #         speed = 5
    #         for i in range(2):
    #             for j in range(6):
    #                 interval[i][j] = time_now / speed

    # px = track_tka[idx][0] + (track_tka[idx+1][0] - track_tka[idx][0]) * interval[0][0]
    # py = track_tka[idx][1] + (track_tka[idx+1][1] - track_tka[idx][1]) * interval[0][1]
    # pz = track_tka[idx][2] + (track_tka[idx+1][2] - track_tka[idx][2]) * interval[0][2]
    # ang1 = track_tka[idx][3] + (track_tka[idx+1][3] - track_tka[idx][3]) * interval[0][3]
    # ang5 = track_tka[idx][4] + (track_tka[idx+1][4] - track_tka[idx][4]) * interval[0][4]
    # ang6 = track_tka[idx][5] + (track_tka[idx+1][5] - track_tka[idx][5]) * interval[0][5]

    # if time_now >= speed:
    #     idx = idx + 1
    #     prev_time = millis

    # if idx == len(track_tka) - 1:
    #     break

    # # ang1 = 90

    # print(px, py, pz)

    # pxr = px
    # pyr = py * np.cos(ang1 * np.pi/180) - pz * np.sin(ang1 * np.pi/180)
    # pzr = py * np.sin(ang1 * np.pi/180) + pz * np.cos(ang1 * np.pi/180)

    # # print(pxr, pyr, pzr)

    # leng = np.sqrt(np.pow(pxr, 2) + np.pow(pyr, 2) + np.pow(pzr, 2))
    # # print(leng)
    # ang4 = np.acos(constrain((np.pow(leng, 2) - np.pow(a1, 2) - np.pow(a2, 2)) / (2 * a1 * a2), -1, 1)) * 180/np.pi

    # beta = np.acos(constrain((np.pow(a1, 2) + np.pow(leng, 2) - np.pow(a2, 2)) / (2 * a1 * leng), -1, 1)) * 180/np.pi
    # leng = np.sqrt(np.pow(pxr, 2) + np.pow(pyr, 2))
    # alpha = np.atan2(pzr, leng) * 180/np.pi
    # ang2 = (np.atan2(pxr, pyr) + np.acos(constrain(np.cos(beta * np.pi/180) / np.cos(alpha * np.pi/180), -1, 1))) * 180/np.pi

    # xr = pxr * np.cos(ang2 * np.pi/180) - pyr * np.sin(ang2 * np.pi/180)
    # yr = pxr * np.sin(ang2 * np.pi/180) + pyr * np.cos(ang2 * np.pi/180)
    # ang3 = np.atan2(xr, pzr) * 180/np.pi

    # ang1 = 90 - ang1
    # ang2 = 90 - ang2

    # # print(xr, yr)
    # print(ang1, ang2, ang3, ang4, idx)
    # print("")
    # # print(beta - alpha)
    # # print(np.atan2(px, py) * 180/np.pi)
    # break









    # leng = np.sqrt(np.pow(px, 2) + np.pow(py, 2) + np.pow(pz, 2))
    # ang2 = 75
    # ang1 = (np.atan2(np.sqrt(np.pow(px, 2) + np.pow(py, 2)), -pz) - np.acos(constrain((np.pow(a1 * np.sin(ang2 * np.pi/180), 2) + np.pow(leng, 2) - np.pow(a2 * np.sin(ang2 * np.pi/180), 2)) / (2 * (a1 * np.sin(ang2 * np.pi/180)) * leng + 0.0001), -1, 1))) * 180/np.pi
    # # ang1 = 90
    # ang2 = 90 - ang2
    #
    # pxr1 = px
    # pyr1 = py * np.cos(ang1 * np.pi/180) + pz * np.sin(ang1 * np.pi/180)
    # pzr1 = -py * np.sin(ang1 * np.pi/180) + pz * np.cos(ang1 * np.pi/180)
    #
    # pxr2 = pxr1 * np.cos(ang2 * np.pi/180) + pzr1 * np.sin(ang2 * np.pi/180)
    # pyr2 = pyr1
    # pzr2 = -pxr1 * np.sin(ang2 * np.pi/180) + pzr1 * np.cos(ang2 * np.pi/180)
    #
    # # pxr2 = px * np.cos(ang2 * np.pi/180) - pz * np.sin(ang2 * np.pi/180)
    # # pyr2 = px * np.sin(ang2 * np.pi/180) * np.sin(ang1 * np.pi/180) + py * np.cos(ang1 * np.pi/180) + pz * np.sin(ang1 * np.pi/180) * np.cos(ang2 * np.pi/180)
    #
    #
    # ang3 = np.atan2(pxr2, pyr2) * 180/np.pi
    #
    # # ang3 = np.atan2(px * np.sin(ang2 * np.pi/180) + pz * np.cos(ang2 * np.pi/180), py * np.cos(ang1 * np.pi/180) + pz * np.sin(ang1 * np.pi/180)) * 180/np.pi
    # leng = np.sqrt(np.pow((a1 * np.cos(ang2 * np.pi/180)) - px, 2) + np.pow(py, 2) + np.pow(pz, 2))
    # # ang4 = np.acos(constrain((np.pow(leng, 2) - np.pow(a1, 2) - np.pow(a2, 2)) / (2 * a1 * a2), -1, 1)) * 180/np.pi
    # if -pzr2 - a1 >= 0:
    #     ang4 = np.asin(constrain(np.sqrt(np.pow(pxr2, 2) + np.pow(pyr2, 2)) / a2, -1, 1)) * 180/np.pi
    # elif -pzr2 - a1 < 0:
    #     ang4 = 180 - (np.asin(constrain(np.sqrt(np.pow(pxr2, 2) + np.pow(pyr2, 2)) / a2, -1, 1)) * 180 / np.pi)
    # print(ang1, ang2, ang3, ang4)
    # print(px, py, pz)
    # print((np.pow(a1 * np.sin(ang2 * np.pi/180), 2) + np.pow(leng, 2) - np.pow(a2 * np.sin(ang2 * np.pi/180), 2)) / (2 * (a1 * np.sin(ang2 * np.pi/180)) * leng + 0.0001))
    # print(np.sqrt(np.pow(pxr2, 2) + np.pow(pyr2, 2)))
    # print(np.pow(pxr2, 2), np.pow(pyr2, 2))
    # print(np.sqrt(np.pow(pxr2, 2) + np.pow(pyr2, 2)) / a2)
    # print(pxr2, pyr2, pzr2)

    time.sleep(0.05)
    # print(time_now)
    # if time_now >= np.pi:
    # break







# leng = np.sqrt(np.pow(px, 2) + np.pow(py, 2) + np.pow(pz, 2))
#
# theta2 = 90
# theta4 = np.acos((np.pow(leng, 2) - np.pow(a1, 2) - np.pow(a2, 2)) / (2 * a1 * a2)) * 180/np.pi
# print(np.atan2(130, 50) * 180/np.pi)
# theta1 = (np.atan2(np.sqrt(np.pow(px, 2) + np.pow(py, 2)), -pz) - np.atan2(a2 * np.sin(theta4 * np.pi/180), a1 + (a2 * np.cos(theta4 * np.pi/180)))) * 180/np.pi
# acos_val = ((py * np.cos(theta1 * np.pi/180)) + (pz * np.sin(theta1 * np.pi/180))) / (a2 * np.sin(theta4 * np.pi/180))
# if acos_val > 1:
#     acos_val = 1
# elif acos_val < -1:
#     acos_val = -1
# else:
#     acos_val = acos_val
# theta3 = np.acos(((py * np.cos(theta1 * np.pi/180)) + (pz * np.sin(theta1 * np.pi/180))) / (a2 * np.sin(theta4 * np.pi/180))) * 180/np.pi
# theta3 = np.acos(acos_val) * 180/np.pi
# theta3 = np.atan2((px * np.sin(theta2 * np.pi/180)) + (pz * np.cos(theta2 * np.pi/180)), (py * np.cos(theta1 * np.pi/180)) + (pz * np.sin(theta1 * np.pi/180))) * 180/np.pi
# print(acos_val)

# leng2 = np.sqrt(np.pow(a1 * np.cos(theta4 * np.pi/180), 2) + np.pow(a1 * np.sin(theta4 * np.pi/180) * np.sin(theta3 * np.pi/180), 2))
# rad = np.sqrt(np.pow(px, 2) + np.pow((pz * np.cos(theta1 * np.pi/180)) + (py * np.sin(theta1 * np.pi/180)), 2))
# acos_val = (np.pow(a1, 2) + np.pow(rad, 2) - np.pow(leng2, 2)) / (2 * a1 * rad)
# if acos_val > 1:
#     acos_val = 1
# elif acos_val < -1:
#     acos_val = -1
# else:
#     acos_val = acos_val
# theta2 = (np.acos((np.pow(a1, 2) + np.pow(rad, 2) - np.pow(leng2, 2)) / (2 * a1 * rad)) - np.atan2(px, (pz * np.cos(theta1 * np.pi/180)) + (py * np.sin(theta1 * np.pi/180)))) * 180/np.pi
# theta2 = (np.acos(acos_val) - np.atan2(px, (pz * np.cos(theta1 * np.pi/180)) + (py * np.sin(theta1 * np.pi/180)))) * 180/np.pi
# print(theta1, theta2, theta3, theta4)
# print(acos_val, leng2, rad)
# print(np.atan2(px, (pz * np.cos(theta1 * np.pi/180)) + (py * np.sin(theta1 * np.pi/180))) * 180/np.pi)



# leng = np.sqrt(np.pow(px, 2) + np.pow(py, 2))
# rad = np.sqrt(np.pow(leng, 2) + np.pow(pz, 2))
#
# ang3 = np.acos((np.pow(rad, 2) - np.pow(a1, 2) - np.pow(a2, 2)) / (2 * a1 * a2)) * 180/np.pi
# ang2 = (np.atan2(np.sqrt(np.pow(py, 2) + np.pow(pz, 2)), px) - np.atan2(a2 * np.sin(ang3 * np.pi/180), a1 + (a2 * np.cos(ang3 * np.pi/180)))) * 180/np.pi
# ang1 = np.atan2(pz, leng) * 180/np.pi
# print(ang1, ang2, ang3)
#
# fig, axes = plt.subplots(2, 2, figsize=(12, 6))
# ax1, ax2, ax3, ax4 = axes.flatten()
#
# x1_1 = a1 * np.cos(ang2 * np.pi / 180) * np.cos(ang1 * np.pi/180) + 5
# x2_1 = x1_1 + a2 * np.cos((ang2 + ang3) * np.pi / 180) * np.cos(ang1 * np.pi/180)
#
# y1_1 = (a1 * np.cos(ang1 * np.pi/180)) * np.sin(ang2 * np.pi/180)
# y2_1 = y1_1 + (a2 * np.cos(ang1 * np.pi/180) * np.sin((ang2 + ang3) * np.pi/180))

# ax1.plot([0, 5], [0, 0], '-o', color="Green")
# ax1.plot([5, x1_1], [0, y1_1], '-o', color="Blue")
# ax1.plot([x1_1, x2_1], [y1_1, y2_1], '-o', color="Red")
# ax1.set_xlim(-50, 50)
# ax1.set_ylim(-50, 50)
# ax1.axhline(0, color='gray', linestyle='--')
# ax1.axvline(0, color='gray', linestyle='--')
# ax1.set_aspect('equal')
#
# y1_2 = (a1 * np.cos(ang1 * np.pi/180)) * np.sin(ang2 * np.pi/180)
# y2_2 = y1_2 + (a2 * np.cos(ang1 * np.pi/180) * np.sin((ang2 + ang3) * np.pi/180))
#
# z1 = a1 * np.sin(ang1 * np.pi/180)
# z2 = z1 + a2 * np.sin(ang1 * np.pi/180)
#
# ax2.plot([0, y1_2], [0, z1], '-o', color="Blue")
# ax2.plot([y1_2, y2_2], [z1, z2], '-o', color="Red")
# ax2.set_xlim(-50, 50)
# ax2.set_ylim(-50, 50)
# ax2.axhline(0, color='gray', linestyle='--')
# ax2.axvline(0, color='gray', linestyle='--')
# ax2.set_aspect('equal')
#
# x1_2 = a1 * np.cos(ang2 * np.pi/180) * np.cos(ang1 * np.pi/180) + 5
# x2_2 = x1_2 + a2 * np.cos((ang2 + ang3) * np.pi/180) * np.cos(ang1 * np.pi/180)
#
# ax3.plot([0, 5], [0, 0], '-o', color="Green")
# ax3.plot([5, x1_1], [0, z1], '-o', color="Blue")
# ax3.plot([x1_1, x2_2], [z1, z2], '-o', color="Red")
# ax3.set_xlim(-50, 50)
# ax3.set_ylim(-50, 50)
# ax3.axhline(0, color='gray', linestyle='--')
# ax3.axvline(0, color='gray', linestyle='--')
# ax3.set_aspect('equal')
#
# ax4.plot([0, 5], [0, 0], '-o', color="Green")
# ax4.plot([5, x1_1], [0, z1], '-o', color="Blue")
# ax4.plot([x1_1, x2_2], [z1, z2], '-o', color="Red")
# ax4.set_xlim(-50, 50)
# ax4.set_ylim(-50, 50)
# ax4.axhline(0, color='gray', linestyle='--')
# ax4.axvline(0, color='gray', linestyle='--')
# ax4.set_aspect('equal')
#
# plt.show()