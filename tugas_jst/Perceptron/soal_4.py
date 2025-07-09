# Nama = Muhammad Iqbal Apriza
# NIM  = 03041282227043

# Algoritma Perceptron Soal Nomor 4

from prettytable import PrettyTable

inputt = [
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1]
]
target = [-1, -1, -1, 1]
alpha = 1
tetha = 0.1

weight = [0, 0, 0]
bias = 0
del_w = [0, 0, 0, 0]
output = [0, 0, 0, 0]
output_net = [0, 0, 0, 0]
epoch = 0

table = PrettyTable(["X1", "X2", "X3", "Alpha", "Target", "Net", "Output", "del_w1", "del_w2", "del_w3", "del_b", "w1", "w2", "w3", "b"])

while output_net != target:
    epoch = epoch + 1
    print(f"Epoch ke-{epoch}")
    for i in range(4):
        output[i] = bias + (inputt[0][i] * weight[0]) + (inputt[1][i] * weight[1]) + (inputt[2][i] * weight[2])
        if output[i] > tetha:
            output_net[i] = 1
        elif output[i] < -tetha:
            output_net[i] = -1
        else:
            output_net[i] = 0

        del_w = [0, 0, 0, 0]

        if output_net[i] != target[i]:
            for j in range(3):
                del_w[j] = alpha * inputt[j][i] * target[i]
                weight[j] = weight[j] + del_w[j]
            del_w[3] = alpha * target[i]
            bias = bias + del_w[3]

        table.add_row([inputt[0][i], inputt[1][i], inputt[2][i], alpha, target[i], output[i], output_net[i], del_w[0], del_w[1], del_w[2], del_w[3], weight[0], weight[1], weight[2], bias])

    print(table)
    table.clear_rows()