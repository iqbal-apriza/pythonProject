import numpy as np
import matplotlib.pyplot as plt
import dtft

x = [4, 3, 2, 0, -1, -2, -3, -4]
N = np.arange(0, len(x) - 1)
freq = np.linspace(0, 2*np.pi, 1000)

X = np.zeros(len(freq), dtype=complex)
for i in range(len(freq)):
    X[i] = dtft.dtft(x, N, 2 * np.pi * freq[i])

figu, ax = plt.subplots(2)
ax[0].plot(freq, np.abs(X))
ax[0].set_title('Magnitude')
ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Magnitude')

ax[1].plot(freq, np.angle(X))
ax[1].set_title('Phase')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Phase (rad)')
plt.show()