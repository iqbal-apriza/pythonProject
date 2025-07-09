import numpy as np
import pandas as pd

inputt = [
    [1, 1, 0 ,0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 0, 1, 1]
]

weight = [
    [0.2, 0.6, 0.5, 0.9],
    [0.8, 0.4, 0.7, 0.3]
]

alpha = 0.6
next_alpha = 0.5

dj = np.zeros([2])
for k in range(100):
    print(f"Epoch ke : {k+1}")
    for i in range(len(inputt)):
        for j in range(len(dj)):
            dj[j] = ((weight[j][0] - inputt[i][0]) ** 2) + ((weight[j][1] - inputt[i][1]) ** 2) + ((weight[j][2] - inputt[i][2]) ** 2) + ((weight[j][3] - inputt[i][3]) ** 2)

        for j in range(4):
            weight[np.argmin(dj)][j] = weight[np.argmin(dj)][j] + (alpha * (inputt[i][j] - weight[np.argmin(dj)][j]))
    print(pd.DataFrame(weight))
    print("")
    alpha = next_alpha * alpha

d_test = np.zeros([4, 2])
for i in range(len(inputt)):
    for j in range(2):
        d_test[i][j] = ((weight[j][0] - inputt[i][0]) ** 2) + ((weight[j][1] - inputt[i][0]) ** 2) + ((weight[j][2] - inputt[i][2]) ** 2) + ((weight[j][3] - inputt[i][3]) ** 2)

print("Data Testing")
result = pd.DataFrame({
    "Kelas 1" : d_test[:,0],
    "Kelas 2" : d_test[:,1],
    "Kategori" : np.argmin(d_test, axis=1) + 1
})
print(result)