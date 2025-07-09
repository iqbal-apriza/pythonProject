# ----------------------------Libraries Import----------------------------
import numpy as np
import matplotlib.pyplot as plt
import time
import ease as es


# -------------------------------Parameters-------------------------------
lengths = [20, 20]

time_start = time.time()
curr_time = 0
prev_time = 0
trgt_time = 0
idx = 0
timing = np.zeros(2)


# --------------------------------Functions--------------------------------
def millis():
    return int((time.time() - time_start) * 1000)

def constrain(val, min_val, max_val):
    return min(max(min_val, val), max_val)

def ik_elup(lengs, coords):
    leng = np.sqrt((coords[0] ** 2) + (coords[1] ** 2))

    the2 = np.acos(constrain(((leng ** 2) - (lengs[0] ** 2) - (lengs[1] ** 2)) / (2 * lengs[0] * lengs[1]), -1, 1))
    the1 = (np.atan2(coords[1], coords[0]) - np.atan2(lengs[1] * np.sin(the2), lengs[0] + lengs[1] * np.cos(the2)))

    return the1 * 180/np.pi, the2 * 180/np.pi

def ik_eldo(lengs, coords):
    leng = np.sqrt((coords[0] ** 2) + (coords[1] ** 2))

    the2 = -np.acos(constrain(((leng ** 2) - (lengs[0] ** 2) - (lengs[1] ** 2)) / (2 * lengs[0] * lengs[1]), -1, 1))
    the1 = (np.atan2(coords[1], coords[0]) - np.atan2(lengs[1] * np.sin(the2), lengs[0] + lengs[1] * np.cos(the2)))

    return the1 * 180/np.pi, the2 * 180/np.pi

def trajectory(start_pos, end_pos, tm):
    if len(start_pos) != len(end_pos):
        raise Exception("The dimension of start and end coordinate must be same")
    
    res = np.zeros(len(start_pos))
    for i in range(len(res)):
        res[i] = start_pos[i] + (end_pos[i] - start_pos[i]) * tm[i]

    return res

def interpolate(start_ang, end_ang, ct):
    the1 = start_ang[0] * (1 - ct) + end_ang[0] * ct
    the2 = start_ang[1] * (1 - ct) + end_ang[1] * ct

    return the1, the2


# ----------------------------Movement Function----------------------------
def motion1():
    global prev_time, curr_time, trgt_time, idx, timing

    tracks = [
        [0, 20],
        [0, 0],
        [10, 0],
        [10, 10],
        [20, 10],
        [20, 20],
        [0, 20]
    ]

    speed = 3
    diplacement = np.sqrt(((tracks[idx + 1][0] - tracks[idx][0]) ** 2) + ((tracks[idx + 1][1] - tracks[idx][1]) ** 2))
    trgt_time = (diplacement / speed) * 1000

    curr_time = millis() - prev_time

    match(idx):
        case 0:
            timing[0] = curr_time / trgt_time
            timing[1] = es.easeInOutQuad(timing[0])
        case 1:
            timing[0] = es.easeOutExpo(curr_time / trgt_time)
            timing[1] = es.easeInOutQuad(curr_time / trgt_time)
        case 2:
            timing[0] = es.easeInOutQuad(curr_time / trgt_time)
            timing[1] = es.easeOutExpo(curr_time / trgt_time)
        case _:
            timing[0] = curr_time / trgt_time
            timing[1] = curr_time / trgt_time

    coords = trajectory(tracks[idx], tracks[idx + 1], timing)
    angles = ik_elup(lengths, coords)

    if curr_time >= trgt_time:
        prev_time = millis()
        idx += 1
        if idx >= len(tracks) - 1:
            idx = 0

    print(curr_time, trgt_time, idx)

    return angles

def motion2():
    global prev_time, curr_time, trgt_time, idx, timing

    tracks = [
        [0, 20],
        [0, 0],
        [10, 0],
        [10, 10],
        [20, 10],
        [20, 20],
        [0, 20]
    ]

    curr_time = millis() - prev_time

    speed = 5
    diplacement = np.sqrt(((tracks[idx + 1][0] - tracks[idx][0]) ** 2) + ((tracks[idx + 1][1] - tracks[idx][1]) ** 2))
    trgt_time = (diplacement / speed) * 1000

    timing[0] = es.easeOutElastic(curr_time / trgt_time)
    timing[1] = es.easeOutElastic(curr_time / trgt_time)
    
    coords = trajectory(tracks[idx], tracks[idx + 1], timing)
    angles = ik_elup(lengths, coords)

    if curr_time >= trgt_time:
        prev_time = millis()
        idx += 1
        if idx >= len(tracks) - 1:
            idx = 0

    return angles

def motion3():
    global prev_time, curr_time, trgt_time, idx, timing

    # Define the trajectory points in angles
    tracks = [
        [0, 0],
        [30, 90],
        [90, 45],
        [90, 120],
        [0, 0]
    ]

    # Get the current time in milliseconds
    curr_time = millis() - prev_time

    # Define the speed of the movement
    speed = 30
    # Calculate the distance between the current and next points
    diplacement = np.sqrt(((tracks[idx + 1][0] - tracks[idx][0]) ** 2) + ((tracks[idx + 1][1] - tracks[idx][1]) ** 2))
    # Calculate the target time based on speed and distance
    trgt_time = (diplacement / speed) * 1000

    timing[0] = es.easeOutBounce(curr_time / trgt_time)
    timing[1] = es.easeOutBounce(curr_time / trgt_time)

    angles = trajectory(tracks[idx], tracks[idx + 1], timing)

    if curr_time >= trgt_time:
        prev_time = millis()
        idx += 1
        if idx >= len(tracks) - 1:
            idx = 0

    return angles
        

# ------------------------------Main Function------------------------------
if __name__ == "__main__":
    plt.ion()
    fig, ax = plt.subplots()
    while 1:
        angles = motion3()

        x1 = lengths[0] * np.cos(angles[0] * np.pi/180)
        y1 = lengths[0] * np.sin(angles[0] * np.pi/180)
        x2 = lengths[1] * np.cos((angles[0] + angles[1]) * np.pi/180) + x1
        y2 = lengths[1] * np.sin((angles[0] + angles[1]) * np.pi/180) + y1

        ax.cla()
        ax.plot([0, x1], [0, y1], "-o", color="Blue")
        ax.plot([x1, x2], [y1, y2], "-o", color="Green")
        ax.axhline(0, color='gray', linestyle='--')
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")
        ax.grid()

        plt.pause(0.01)

    plt.ioff()
    plt.show()