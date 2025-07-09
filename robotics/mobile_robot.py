import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import time

W = 10

inp = np.array([10, 30])
last_tm = 0
x = 0
y = 0
o = 0

def rad(deg):
    return deg * np.pi/180

def deg(rad):
    return rad * 180/np.pi

def robot_ik(v, w):
    vL = v - (w * W / 2)
    vR = v + (w * W / 2)
    return np.array([vL, vR])

def robot_fk(vL, vR):
    v = (vL + vR) / 2
    w = (vL - vR) / W
    return np.array([v, w])

def position(v, w, dt):
    global x, y, o
    o += w * dt
    x += v * np.cos(o) * dt
    y += v * np.sin(o) * dt

plt.ion()
fig, ax = plt.subplots()
dot, = ax.plot([], [], 'ro')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)

ax_speed = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_steer = plt.axes([0.25, 0.1, 0.65, 0.03])
speed_val = Slider(ax_speed, 'Speed', valmin=0.0, valmax=10.0, valinit=0.0, valstep=0.5)
steer_val = Slider(ax_steer, 'Steer', valmin=-70.0, valmax=70.0, valinit=0.0, valstep=0.5)

start_time = time.time()
while 1:
    tm = time.time() - start_time
    dt = tm - last_tm

    wheel_vel = robot_ik(speed_val.val, rad(steer_val.val))
    vel = robot_fk(wheel_vel[0], wheel_vel[1])
    position(vel[0], vel[1], dt)

    dot.set_data([x], [y])
    plt.draw()

    last_tm = time.time() - start_time
    plt.pause(0.0416667)