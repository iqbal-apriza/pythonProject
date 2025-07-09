import numpy as np

def dtft(x, N, w):
    X = 0
    for i in range(len(N)):
        X += x[i] * np.exp(-1j * N[i] * w)
    return X