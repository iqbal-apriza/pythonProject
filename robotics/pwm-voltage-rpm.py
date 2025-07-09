import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# Data
pwm = np.array([
    255, 250, 240, 230, 220, 210, 200, 190, 180, 
    170, 160, 150, 140, 130, 120, 110, 100,  90, 
     80,  70,  60,  50,  40,  30,  20,  10
])

voltage = np.array([
    10.56, 10.36, 10.26, 10.18,  10.1,  10.0,  9.93,  9.83,   9.7, 
     9.59,  9.42,  9.27,  9.05,  8.82,  8.54,  8.24,  7.89,  7.48, 
     6.94,  6.23,  5.47,  4.37,  2.95,  1.48,  0.15,  0.046
])

M1 = np.array([
    760, 756, 750, 742, 735, 727, 722, 715, 707, 
    700, 680, 668, 652, 630, 600, 576, 553, 512, 
    473, 425, 355, 266, 169,   0,   0,   0
])

M2 = np.array([
    558, 533, 512, 500, 495, 485, 479, 475, 470, 
    465, 447, 432, 415, 395, 373, 350, 312, 265, 
    225, 169, 100,  60,   0,   0,   0,   0
])

M3 = np.array([
    520, 505, 500, 498, 498, 490, 483, 476, 472, 
    465, 459, 449, 445, 429, 394, 347, 330, 320, 
    265, 200, 180,  60,   0,   0,   0,   0
])

M4 = np.array([
    750, 730, 725, 710, 706, 700, 690, 682, 676, 
    663, 652, 635, 620, 600, 570, 523, 480, 440, 
    390, 320, 220, 135,   0,   0,   0,   0
])

# Invert: fit PWM = f(voltage)
p = Polynomial.fit(voltage, pwm, deg=5)
p1 = Polynomial.fit(M1, pwm, deg=5)
p2 = Polynomial.fit(M2, pwm, deg=5)
p3 = Polynomial.fit(M3, pwm, deg=5)
p4 = Polynomial.fit(M4, pwm, deg=5)

# Create a function to evaluate inverse
def inverse_pwm(v):
    return p(v)

# Example usage
# v_test = 7.5
# print("PWM for voltage", v_test, "=", inverse_pwm(v_test))

# Plot
# v_space = np.linspace(0, 11, 200)
m1_space = np.linspace(0, 760, 200)
m2_space = np.linspace(0, 558, 200)
m3_space = np.linspace(0, 520, 200)
m4_space = np.linspace(0, 750, 200)

# plt.plot(voltage, pwm, 'ro', label='Data')
# plt.plot(v_space, inverse_pwm(v_space), 'b-', label='Fit')
plt.plot(M1, pwm, 'ro', color="green")
plt.plot(m1_space, p1(m1_space), 'b-', color="green")
plt.plot(M2, pwm, 'ro', color="red")
plt.plot(m2_space, p2(m2_space), 'b-', color="red")
plt.plot(M3, pwm, 'ro', color="blue")
plt.plot(m3_space, p3(m3_space), 'b-', color="blue")
plt.plot(M4, pwm, 'ro', color="black")
plt.plot(m4_space, p4(m4_space), 'b-', color="black")
# plt.plot(M2, pwm)
# plt.plot(M3, pwm)
# plt.plot(M4, pwm)
plt.xlabel("Voltage (V)")
plt.ylabel("PWM")
plt.title("Inverse Function: Voltage to PWM")
plt.grid()
plt.legend()
plt.show()

# p1 = p1.convert()
# p2 = p2.convert()
# p3 = p3.convert()
# p4 = p4.convert()

# print(f"M1 : {p1.coef}")
# print(f"M2 : {p2.coef}")
# print(f"M3 : {p3.coef}")
# print(f"M4 : {p4.coef}")