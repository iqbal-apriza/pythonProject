import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("dataset.csv")
df_array = df.to_numpy()

max_val = np.max(df_array)
min_val = np.min(df_array)

df_norm = np.empty([12, 9])
for i in range(12):
    for j in range(9):
        df_norm[i][j] = (0.8 * (df_array[i][j] - min_val) / (max_val - min_val)) + 0.1

target = df_norm[:, 8]
beta = 0.7 * np.power(6, 1/8)
alpha = 0.9
mu = 0.5

np.random.seed(42)
weight_v = np.empty([9, 6])
for i in range(8):
    for j in range(6):
        weight_v[i+1][j] = np.random.uniform(-0.5, 0.5)
        weight_v[0][j] = np.random.uniform(-beta, beta)

weight_w = np.empty([7])
for i in range(7):
    weight_w[i] = np.random.uniform(-0.5, 0.5)

prev_v = np.zeros([9, 6])
prev_w = np.zeros([7])

next_v = np.empty([9, 6])
next_w = np.empty([7])

output_hid = np.empty([6])

error = np.empty([12])
error_l1 = np.empty([6])
error_l2 = np.empty([6])
del_w = np.empty([7])
del_v = np.empty([9, 6])

print(pd.DataFrame(weight_v))
print("")
print(pd.DataFrame(weight_w))
print("")

for ep in range(100000):
    print(f"Epoch ke : {ep + 1}")
    for i in range(12):
        # Perambatan Maju
        for j in range(6):
            jlh = 0
            for k in range(8):
                jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
            net = weight_v[0][j] + jlh
            output_hid[j] = 1 / (1 + np.exp(-net))

        jlh = 0
        for j in range(6):
            jlh = jlh + (output_hid[j] * weight_w[j+1])
        net = weight_w[0] + jlh
        output = 1 / (1 + np.exp(-net))

        # Perambatan Mundur
        error[i] = (target[i] - output) * output * (1 - output)
        for j in range(6):
            del_w[j+1] = alpha * error[i] * output_hid[j]
        del_w[0] = alpha * error[i]

        for j in range(6):
            error_l2[j] = error[i] * weight_w[j] * weight_w[j]
            error_l1[j] = error_l2[j] * output_hid[j] * (1 - output_hid[j])

        for j in range(6):
            for k in range(8):
                del_v[k+1][j] = alpha * error_l1[j] * df_norm[i][k]
            del_v[0][j] = alpha * error_l1[j]

        # Update Bobot
        for j in range(7):
            next_w[j] = weight_w[j] + del_w[j] + (mu * (weight_w[j] - prev_w[j]))

        for j in range(9):
            for k in range(6):
                next_v[j][k] = weight_v[j][k] + del_v[j][k] + (mu * (weight_v[j][k] - prev_v[j][k]))

        for j in range(9):
            for k in range(6):
                prev_v[j][k] = weight_v[j][k]
                weight_v[j][k] = next_v[j][k]

        for j in range(7):
            prev_w[j] = weight_w[j]
            weight_w[j] = next_w[j]

# Test Data
output = np.empty([12])
for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

mse = np.mean(error ** 2)
print("")
print(f"Mean Square Error = {mse}")
print("")
print(pd.DataFrame(weight_v))
print("")
print(pd.DataFrame(weight_w))
print("")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target
}))

plt.plot(output, label="Output")
plt.plot(target, label="Target")
plt.legend()
plt.title("Grafik Perbandingan Output dengan Target")
plt.xlabel("Bulan")
plt.ylabel("Curah Hujan")
plt.show()