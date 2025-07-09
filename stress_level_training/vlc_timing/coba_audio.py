import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wv
import time

fs, audio = wv.read(r"D:\Media\Music\Melly\13 Ketika Cinta Bertasbih.wav")
audio = audio / 32768
audio_l = audio[:, 0]
audio_r = audio[:, 1]

N = 2048

wd = np.hanning(N)

freq = np.zeros(N)
for i in range(N):
    freq[i] = i * fs / N

frange = (freq >= 20) & (freq <= 20000)
freq = freq[frange]

plt.ion()
fig, ax = plt.subplots()

lineL, = ax.plot(freq, np.zeros_like(freq), label="Left")
lineR, = ax.plot(freq, np.zeros_like(freq), label="Right")

ax.set_xlim(20, 20000)
ax.set_ylim(0, 150)
ax.set_xscale("log")
ax.set_xticks([20, 50, 100, 250, 500, 1000, 2000, 5000, 8000, 12000, 20000])
ax.set_xticklabels(["20", "50", "100", "250", "500", "1k", "2k", "5k", "8k", "12k", "20k"])
ax.grid(True, which="both")
ax.legend()

start_time = time.time()
while 1:
    playback_time = time.time() - start_time
    sample = int(playback_time * fs)

    if sample >= audio.shape[0] - N:
        break
    
    x1 = audio_l[sample : sample+N] * wd
    x2 = audio_r[sample : sample+N] * wd

    HL = np.fft.fft(x1)
    HR = np.fft.fft(x2)
    HL = HL[frange]
    HR = HR[frange]

    # ax.cla()
    # ax.plot(freq, np.abs(HL) / N)
    # ax.plot(freq, np.abs(HR) / N)
    # ax.set_xlim(20, 20000)
    # ax.set_ylim(0, 1)
    # ax.set_xscale("log")
    # ax.set_xticks([20, 50, 100, 250, 500, 1000, 2000, 5000, 8000, 12000, 20000])
    # ax.set_xticklabels(["20", "50", "100", "250", "500", "1k", "2k", "5k", "8k", "12k", "20k"])
    # ax.grid(True, which="both")

    magL = np.abs(HL)
    magR = np.abs(HR)

    lineL.set_ydata(magL)
    lineR.set_ydata(magR)
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.001)

plt.ioff()
plt.show()