import numpy as np
import matplotlib.pyplot as plt
import time

V = 220
I = 3.8

mass = 2
water_cap = 4184

thermal_init = 30
Ta = 28

simulation_speed = 1.0

P = V * I

plt.ion()

thermal = thermal_init
prev_time = 0
start_tm = time.time()
thermal_acc = []
tm_acc = []
while 1:
    tm = (time.time() - start_tm) * simulation_speed
    dt = tm - prev_time
    W = P - ((thermal - Ta) / 0.2)

    dT = W * dt / (mass * water_cap)
    thermal += dT
    thermal_acc.append(thermal)
    tm_acc.append(np.round(tm, 2))

    plt.cla()
    plt.plot(tm_acc, thermal_acc)

    plt.pause(0.1)
    prev_time = tm

    print(f"{thermal} || {tm / 60}", end="\r")

plt.show()