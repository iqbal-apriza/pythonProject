import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('dataset.csv')
df_array = df.to_numpy()

weight_v = pd.read_csv('train.csv').to_numpy()
weight_w = [-3.204145, 1.156341, -2.373211, 2.441113, -0.682233, 1.087004, 1.593776]

max_val = np.max(df_array)
min_val = np.min(df_array)

df_norm = np.zeros([12, 9])
for i in range(12):
    for j in range(9):
        df_norm[i][j] = (0.8 * (df_array[i][j] - min_val) / (max_val - min_val)) + 0.1

output_hid = np.zeros([6])
output = np.zeros([12])
for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k+1] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

num = np.zeros([12])
for i in range (12):
    num[i] = (((output[i] - 0.1) * (max_val - min_val)) + (0.8 * min_val)) / 0.8

print(num)

plt.plot(num)
plt.xlabel("Bulan")
plt.ylabel("Curah Hujan")
plt.title("Prediksi Tahun 2006")
plt.show()