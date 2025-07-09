import numpy as np
import matplotlib.pyplot as plt

def conv(x, h):
    res = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(h)):
            if i - j >= 0:
                x_val = x[i - j]
            else:
                x_val = 0
            res[i] += h[j] * x_val
    return res

N = 55
a = (N - 1) / 2
wc = 0.3 * np.pi

wd = np.hamming(N)

hd = np.zeros(N)
for n in range(N):
    if n != a:
        hd[n] = (np.sin(wc * (n - a)) / (np.pi * (n - a))) * wd[n]
    else:
        hd[n] = (wc / np.pi) * wd[n]

Hd = np.fft.fft(hd)
Hd_log = 20 * np.log(np.abs(Hd))

freq = np.zeros(N)
for i in range(N):
    freq[i] = i * 100 / N

plt.plot(freq[0:20], Hd_log[0:20])
plt.show()