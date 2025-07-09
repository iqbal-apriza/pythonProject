import numpy as np
import matplotlib.pyplot as plt
import time

M = np.array([1, 1, -1, -1])
spd = np.array([1.41, 1.41, 0])

c = s = np.sqrt(2) / 2
A = np.array([
    [ c,  c, -c, -c],
    [ s, -s,  s, -s],
    [ 1,  1,  1,  1]
])

def motor_fk(motor):
    return np.dot(A, motor)

def motor_ik(speed):
    pseudo = A.T @ np.linalg.inv(A @ A.T)
    return np.dot(pseudo, speed)

x_now = 0
y_now = 0
w_now = 0
odo = 0

prev_tm = 0
prev_x = 0
prev_y = 0

start_tm = time.time()
while 1:
    tm = time.time() - start_tm
    dt = tm - prev_tm
    
    x = 16 * np.pow(np.sin(tm), 3)
    y = 13 * np.cos(tm) - 5 * np.cos(2 * tm) - 2 * np.cos(3 * tm) - np.cos(4 * tm)

    dx = x - prev_x
    dy = y - prev_y

    mtr = motor_ik(np.array([dx/dt, dy/dt, 0]))
    odo += np.sqrt(dx**2 + dy**2) * dt

    print(mtr, odo)

    prev_tm = tm
    prev_x = x
    prev_y = y
    time.sleep(1/24)