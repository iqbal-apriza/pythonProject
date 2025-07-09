import numpy as np
import time

L = 241
W = 195

r = 30

def motor_speed(vel):
    rpm = 60 * vel / r
    
    M1 = 1.99994888e+01 - 1.69258088e-01 * rpm[0] + 4.45025694e-03 * rpm[0] * rpm[0] - 2.53212149e-05 * rpm[0] * rpm[0] * rpm[0] + 6.60967928e-08 * rpm[0] * rpm[0] * rpm[0] * rpm[0] - 8.05610782e-11 * rpm[0] * rpm[0] * rpm[0] * rpm[0] * rpm[0] + 3.78146120e-14 * rpm[0] * rpm[0] * rpm[0] * rpm[0] * rpm[0] * rpm[0]
    M2 = 2.47902241e+01 + 1.24643629e+00 * rpm[1] - 1.98232598e-02 * rpm[1] * rpm[1] + 1.49540830e-04 * rpm[1] * rpm[1] * rpm[1] - 5.30177527e-07 * rpm[1] * rpm[1] * rpm[1] * rpm[1] + 8.80986607e-10 * rpm[1] * rpm[1] * rpm[1] * rpm[1] * rpm[1] - 5.47667107e-13 * rpm[1] * rpm[1] * rpm[1] * rpm[1] * rpm[1] * rpm[1]
    M3 = 2.49274422e+01 + 1.37439903e+00 * rpm[2] - 2.48801750e-02 * rpm[2] * rpm[2] + 1.88792015e-04 * rpm[2] * rpm[2] * rpm[2] - 6.49764910e-07 * rpm[2] * rpm[2] * rpm[2] * rpm[2] + 1.02954524e-09 * rpm[2] * rpm[2] * rpm[2] * rpm[2] * rpm[2] - 6.03217067e-13 * rpm[2] * rpm[2] * rpm[2] * rpm[2] * rpm[2] * rpm[2]
    M4 = 2.49350155e+01 + 1.00867920e+00 * rpm[3] - 1.18025090e-02 * rpm[3] * rpm[3] + 5.97195713e-05 * rpm[3] * rpm[3] * rpm[3] - 1.40941692e-07 * rpm[3] * rpm[3] * rpm[3] * rpm[3] + 1.55925430e-10 * rpm[3] * rpm[3] * rpm[3] * rpm[3] * rpm[3] - 6.44959740e-14 * rpm[3] * rpm[3] * rpm[3] * rpm[3] * rpm[3] * rpm[3]

    return np.array([M1, M2, M3, M4], dtype=int)

def robot_kinematics(v, o):
    R = L / np.tan(o * np.pi/180)
    vRL = (R - W/2) * v/R
    vRR = (R + W/2) * v/R
    vFL = np.sqrt(L**2 + (R - W/2)**2) * v/R
    vFR = np.sqrt(L**2 + (R + W/2)**2) * v/R

    oL = np.atan(L / (R - W/2)) * 180/np.pi
    oR = np.atan(L / (R + W/2)) * 180/np.pi

    pwm_motor = motor_speed(np.array([vFR, vRR, vFL, vRL]))
    print(vRL, vRR, vFL, vFR)

    return pwm_motor, np.array([oL, oR])

start_time = time.time()

print(robot_kinematics(120, 40))
# while 1:
#     tm = time.time() - start_time
#     o = 50 * np.sin(tm / 2)
#     v = 100

#     result = robot_kinematics(v, o)

#     print(result[0])
#     print(result[1])

#     time.sleep(0.01)