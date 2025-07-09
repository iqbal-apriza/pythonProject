import numpy as np
import matplotlib.pyplot as plt
import dtft

N = np.linspace(0, 10, 100)
x = 3 * np.pow(0.9, 3) * np.ones(len(N))
freq = np.linspace(1, 20, 1000)

X = np.zeros(len(freq), dtype=complex)
for i in range(len(freq)):
    X[i] = dtft.dtft(x, N, 2 * np.pi * freq[i])
    
figure, ax = plt.subplots(2)
ax[0].plot(freq, np.abs(X))
ax[0].set_title('Magnitude')
ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Magnitude')

ax[1].plot(freq, np.angle(X))
ax[1].set_title('Phase')
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Phase (rad)')
plt.show()