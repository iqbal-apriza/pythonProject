import numpy as np
import matplotlib.pyplot as plt
import dtft

tm = np.linspace(0, 1, 1000)
x = np.sin(2 * np.pi * 100 * tm)

N = np.arange(len(x))
freq = np.linspace(0, 500, 1000)

X = np.zeros(len(freq), dtype=complex)

for i in range(len(freq)):
    X[i] = dtft.dtft(x, N, 2 * np.pi * freq[i] / 1000)

fig, ax = plt.subplots(2)
ax[0].bar(freq, np.abs(X))
ax[0].set_title('Magnitude')
ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Magnitude')

ax[1].plot(freq, np.angle(X))
ax[1].set_title('Phase')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Phase (rad)')
plt.show()