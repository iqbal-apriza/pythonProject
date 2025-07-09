import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.DataFrame({
    "Data" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Fitur X" : [5, 5, 9, 1, 7, 1, 2, 9, 5, 6],
    "Fitur Y" : [8, 6, 3, 4, 8, 2, 2, 4, 10, 6],
    "Kel 1" : [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
    "Kel 2" : [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    "Kel 3" : [0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(dataset[dataset["Kel 1"] == 1]["Fitur X"], dataset[dataset["Kel 1"] == 1]["Fitur Y"], color="Red", label="Kelompok 1")
ax1.scatter(dataset[dataset["Kel 2"] == 1]["Fitur X"], dataset[dataset["Kel 2"] == 1]["Fitur Y"], color="Green", label="Kelompok 2")
ax1.scatter(dataset[dataset["Kel 3"] == 1]["Fitur X"], dataset[dataset["Kel 3"] == 1]["Fitur Y"], color="Blue", label="Kelompok 3")
ax1.legend()
ax1.set_title("Data Sebelum Clustering")

df_array = dataset.to_numpy()

prev_f = 0

centroid = np.zeros([3, 2])
ec_distance = np.zeros([10, 3])

for iterate in range(3):
    print(f"Iterasi Ke : {iterate+1}")
    for i in range(3):
        for j in range(2):
            centroid[i][j] = sum(df_array[df_array[:, i+3] == 1][:, j+1]) / len(df_array[df_array[:, i+3] == 1])

    for i in range(10):
        for j in range(3):
            ec_distance[i][j] = np.sqrt(((df_array[i][1] - centroid[j][0]) ** 2) + ((df_array[i][2] - centroid[j][1]) ** 2))

    min_group = np.argmin(ec_distance, axis=1)
    min_val = np.zeros([10, 3])
    for i in range(10):
        min_val[i][min_group[i]] = ec_distance[i][min_group[i]]
        df_array[i][3:] = 0
        df_array[i][min_group[i] + 3] = 1

    f_new = sum(sum(min_val))
    delta = np.abs(f_new - prev_f)
    prev_f = f_new

    result = pd.DataFrame({
        "Data" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Fitur X" : df_array[:, 1],
        "Fitur Y" : df_array[:, 2],
        "Kelompok 1" : df_array[:, 3],
        "Kelompok 2" : df_array[:, 4],
        "Kelompok 3" : df_array[:, 5]
    })
    print(result)
    print("")

ax2.scatter(df_array[:, 1][df_array[:, 3] == 1], df_array[:, 2][df_array[:, 3] == 1], color="Red", label="Kelompok 1")
ax2.scatter(df_array[:, 1][df_array[:, 4] == 1], df_array[:, 2][df_array[:, 4] == 1], color="Green", label="Kelompok 2")
ax2.scatter(df_array[:, 1][df_array[:, 5] == 1], df_array[:, 2][df_array[:, 5] == 1], color="Blue", label="Kelompok 3")
ax2.scatter(centroid[:, 0], centroid[:, 1], marker="*", color="Black", label="Centeroid")
ax2.legend()
ax2.set_title("Data Setelah Clustering")

plt.show()