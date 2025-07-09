import math
import matplotlib.pyplot as plt
from prettytable import PrettyTable

inputt = [
    [1, 1, 0, 0],
    [1, 0, 1, 0]
]

target = [0, 1, 1, 0]

w_l1 = [
    [0.3, 0.2, -0.3],     # Z1
    [0.2, 0.3, 0.1],      # Z2
    [-0.1, -0.2, 0.1]     # Z3
#     X1    X2    B
]

w_l2 = [-0.3, 0.3, -0.2, 0.1]   # Y
#        Z1    Z2    Z3    B

alpha = 0.1

net_l1 = [0, 0, 0, 0]
net_l2 = [0, 0, 0, 0]

out_l1 = [0, 0, 0]
out_l2 = 0

err1 = [0, 0, 0 ,0]  # δ_k
err2 = [0, 0, 0]     # δ_net_j
err3 = [0, 0, 0]     # δ_123

del_wl1 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]

del_wl2 = [0, 0, 0, 0]

table = [
    PrettyTable(["", "X1", "X2", "B"]),
    PrettyTable(["", "Z1", "Z2", "Z3", "B"]),
    PrettyTable(["", "ΔV0", "ΔV1", "ΔV2"])
]

mse = 0
epoch = 0
total_mse = []

table[0].clear_rows()
table[1].clear_rows()
table[0].add_rows([
    ["Z1", w_l1[0][0], w_l1[0][1], w_l1[0][2]],
    ["Z2", w_l1[1][0], w_l1[1][1], w_l1[1][2]],
    ["Z3", w_l1[2][0], w_l1[2][1], w_l1[2][2]]
])
table[1].add_row(["Y", w_l2[0], w_l2[1], w_l2[2], w_l2[3]])
print(table[0])
print(f"{table[1]}\n")

while 1 :
    epoch = epoch + 1
    print(f"----------------Epoch ke-{epoch}----------------")
    for i in range(4):
        print(f"-----Data ke-{i+1}-----")

        # Propagasi Maju
        print("Propagasi Maju")
        # Hitung semua keluaran di unit tersembunyi (Zj)
        for j in range(3):
            net_l1[j] = round(w_l1[j][2] + (inputt[0][i] * w_l1[j][0]) + (inputt[1][i] * w_l1[j][1]), 3)
            out_l1[j] = round(1 / (1 + math.exp(-net_l1[j])), 3)
        print(f"Z = {out_l1}")

        # Hitung semua jaringan di unit keluaran (Yk)
        net_l2[0] = round(w_l2[3] + (out_l1[0] * w_l2[0]) + (out_l1[1] * w_l2[1]) + (out_l1[2] * w_l2[2]), 3)
        out_l2 = round(1/(1 + math.exp(-net_l2[0])), 3)
        print(f"Y = {out_l2}\n")

        # Propagasi Mundur
        print("Propagasi Mundur")
        # Hitung error
        err1[i] = round((target[i] - out_l2) * out_l2 * (1 - out_l2), 3)
        print(f"δy = {err1[i]}")
        for j in range(3):
            del_wl2[j] = round((alpha * err1[i] * out_l1[j]), 3)
        del_wl2[3] = round((alpha * err1[i]), 3)
        print(f"Δw = {del_wl2}\n")

        # Hitung faktor δ unit tersembunyi berdasarkan kesalahan di setiap unit tersembunyi zj
        for j in range(3):
            err2[j] = round(err1[i] * w_l2[j], 3)
        print(f"δ_net = {err2}")

        # Faktor kesalahan unit tersembunyi
        for j in range(3):
            err3[j] = round(err2[j] * out_l1[j] * (1 - out_l1[j]), 3)
        print(f"δ = {err3}")
        print("")

        print("Nilai Perubahan Bobot")
        for j in range(3):
            del_wl1[j][2] = round(alpha * err3[j], 3)
            for k in range(2):
                del_wl1[j][k] = round(alpha * err3[j] * inputt[k][i], 3)
            table[2].add_row([f"Z{j}", del_wl1[j][2], del_wl1[j][0], del_wl1[j][1]])
        print(table[2])
        table[2].clear_rows()
        print("")

        # Perubahan Bobot
        table[0].clear_rows()
        table[1].clear_rows()
        print("Bobot Baru")
        for j in range(4):
            w_l2[j] = round(w_l2[j] + del_wl2[j], 3)
        table[1].add_row(["Y", w_l2[0], w_l2[1], w_l2[2], w_l2[3]])

        for j in range(3):
            for k in range(3):
                w_l1[j][k] = round(w_l1[j][k] + del_wl1[j][k], 3)
            table[0].add_row([f"Z{j}", w_l1[j][0], w_l1[j][1], w_l1[j][2]])
        print(table[0])
        print(table[1])
        print("")

    mse = sum([round(x ** 2, 3) for x in err1]) / len(err1)
    print(err1)
    print(f"Mean Square Error = {mse}")

    total_mse.append(mse)

    if mse <= 0.001:
        break
    print("")
    print("")
    break

print("")
print("")

print("Uji Data")
for i in range(4):
    print(f"-----Data ke-{i+1}-----")

    # Propagasi Maju
    print("Propagasi Maju")
    # Hitung semua keluaran di unit tersembunyi (Zj)
    for j in range(3):
        net_l1[j] = round(w_l1[j][2] + (inputt[0][i] * w_l1[j][0]) + (inputt[1][i] * w_l1[j][1]), 3)
        out_l1[j] = round(1 / (1 + math.exp(-net_l1[j])), 3)
    print(f"Z = {out_l1}")

    # Hitung semua jaringan di unit keluaran (Yk)
    net_l2[0] = round(w_l2[3] + (out_l1[0] * w_l2[0]) + (out_l1[1] * w_l2[1]) + (out_l1[2] * w_l2[2]), 3)
    out_l2 = round(1/(1 + math.exp(-net_l2[0])), 3)
    print(f"Y = {out_l2}\n")

plt.plot(total_mse)
plt.title("Grafik MSE vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean Square Error")
plt.grid(True)
plt.show()