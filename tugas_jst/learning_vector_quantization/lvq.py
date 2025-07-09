import numpy as np
import pandas as pd
from numpy.ma.core import argmin

inputt = [
    [1, 1, 0, 0, 1],
    [0, 0, 0, 1, 2],
    [0, 0, 1, 1, 2],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 0, 2]
]

weight = [
    [1, 1, 0, 0],
    [0, 0, 0, 1]
]

alpha = 0.1
next_alpha = 0.5

dj = np.zeros([2])

for a in range(100):
    print(f"Epoch ke : {a+1}")
    for i in range(3):
        for j in range(2):
            dj[j] = ((weight[j][0] - inputt[i+2][0]) ** 2) + ((weight[j][1] - inputt[i+2][1]) ** 2) + ((weight[j][2] - inputt[i+2][2]) ** 2) + ((weight[j][3] - inputt[i+2][3]) ** 2)

        min_val = dj.argmin()
        if dj.min() == inputt[i][4]:
            for j in range(4):
                weight[min_val][j] = weight[min_val][j] + (alpha * (inputt[i+2][j] - weight[min_val][j]))
        else:
            for j in range(4):
                weight[min_val][j] = weight[min_val][j] - (alpha * (inputt[i+2][j] - weight[min_val][j]))
    alpha = alpha * next_alpha
    print("Bobot Akhir:")
    print(np.array(weight))
    print("")

d_test = np.zeros([3, 2])
for i in range(3):
    for j in range(2):
        d_test[i][j] = ((weight[j][0] - inputt[i+2][0]) ** 2) + ((weight[j][1] - inputt[i+2][1]) ** 2) + ((weight[j][2] - inputt[i+2][2]) ** 2) + ((weight[j][3] - inputt[i+2][3]) ** 2)

predicted = argmin(d_test, axis=1) + 1
actual = np.array(inputt)[2:,4]
result = pd.DataFrame({
    "Input" : ["0011", "1000", "0110"],
    "Predicted" : predicted,
    "Actual" : actual,
    "Error" : predicted - actual
})

print("Hasil:")
print(result)