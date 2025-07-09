# Nama = Muhammad Iqbal Apriza
# NIM  = 03041282227043

# Algoritma Perceptron Soal Nomor 2 dan 3

from prettytable import PrettyTable

inputt = [
    [0, 0, 1, 1],
    [0, 1, 0, 1]
]
target = [-1, -1, -1, 1]

alpha = float(input("Masukkan nilai learning rate: "))
tetha = float(input("Masukkan nilai treshold: "))

weight = [0, 0]
bias = 0
del_w = [0, 0, 0]
output = [0, 0, 0, 0]
output_net = [0, 0, 0, 0]
epoch = 0
graph = [0, 0, 0, 0]

table = PrettyTable(["X1", "X2", "Alpha", "Target", "Net", "Output", "del_w1", "del_w2", "del_b", "w1", "w2", "b"])

while output_net != target:
    epoch = epoch + 1
    print(f"Epoch ke-{epoch}")

    for i in range(4):
        output[i] = bias + (inputt[0][i] * weight[0]) + (inputt[1][i] * weight[1])
        if output[i] > tetha:
            output_net[i] = 1
        elif output[i] < -tetha:
            output_net[i] = -1
        else:
            output_net[i] = 0

        del_w = [0, 0, 0]

        if output_net[i] != target[i]:
            for j in range(2):
                del_w[j] = alpha * inputt[j][i] * target[i]
                weight[j] = weight[j] + del_w[j]
            del_w[2] = alpha * target[i]
            bias = bias + del_w[2]

        table.add_row([inputt[0][i], inputt[1][i], alpha, target[i], output[i], output_net[i], del_w[0], del_w[1], del_w[2], weight[0], weight[1], bias])
        graph[i] = (inputt[0][i] * weight[0]) + (inputt[1][i] * weight[1]) + bias

    print(table)
    print(graph)
    table.clear_rows()