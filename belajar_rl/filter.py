import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# Filter Properties
# LPF
L_l = 0.5e-3
R_l = 8

# HPF
C_h = 2.2e-6
L_h = 0.3e-3
R_h = 8

# Frequencies
f = np.linspace(20, 20000, 500)
w = 2 * np.pi * f


# Filter Function
def lpf(R, L):
    H = R / ((1j * w * L) + R)
    mag = 20 * np.log(np.abs(H))
    return mag

def hpf(R, L, C):
    H = R / ((1 / (1j * w * C)) + (R_h / (L * C * np.pow(1j * w, 2))) + R)
    mag = 20 * np.log(np.abs(H))
    return mag


# Plotting
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
plt.ylim(-60, 0)
plt.xscale("log")
plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'])
plt.grid(True, which="both")

hpf_plt, = plt.plot(f, hpf(R_h, L_h, C_h))

ax_c = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_l = plt.axes([0.25, 0.1, 0.65, 0.03])
c_val = Slider(ax_c, 'Capacitor', valmin=1e-6, valmax=10e-6, valinit=C_h, valstep=0.2e-6)
l_val = Slider(ax_l, 'Inductor', valmin=0.1e-3, valmax=10e-3, valinit=L_h, valstep=0.1e-3)

def update(val):
    global R_h
    C = c_val.val
    L = l_val.val
    hpf_plt.set_ydata(hpf(R_h, L, C))

c_val.on_changed(update)
l_val.on_changed(update)

plt.show()