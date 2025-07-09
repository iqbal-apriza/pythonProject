from prettytable import PrettyTable

inputt = [
    [0, 0, 1, 1],
    [0, 1, 0, 1]
]
mode = int(input("0 = AND; 1 = OR : "))
if mode == 0:
    target = [0, 0, 0, 1]
elif mode == 1:
    target = [0, 1, 1, 1]
else:
    print("Masukkan nilai 0 untuk AND dan 1 untuk OR")
    exit()

alpha = float(input("Masukkan nilai learning rate: "))
tetha = float(input("Masukkan nilai treshold: "))
weight = [float(input("Masukkan nilai awal bobot pertama: ")), float(input("Masukkan nilai awal bobot kedua: "))]

output = [0, 0, 0, 0]
output_net = [0, 0, 0, 0]

err = [0, 0, 0, 0]
del_w = [0, 0]
epoch = 0

table = PrettyTable(["X1", "X2", "Alpha", "Target", "Net", "Output", "Error", "Del-W1", "Del-W2", "W1", "W2"])

while output_net != target:
    epoch = epoch + 1
    print(f"Epoch ke-{epoch}")

    for i in range(4):
        output[i] = round((inputt[0][i] * weight[0]) + (inputt[1][i] * weight[1]), 1)
        if output[i] >= tetha:
            output_net[i] = 1
        elif output[i] < tetha:
            output_net[i] = 0

        del_w = [0, 0]
        err[i] = target[i] - output_net[i]
        if err[i] != 0:
            for j in range(2):
                del_w[j] = round((alpha * err[i] * inputt[j][i]), 1)
                weight[j] = round((weight[j] + del_w[j]), 1)

        table.add_row([inputt[0][i], inputt[1][i], alpha, target[i], output[i], output_net[i], err[i], del_w[0], del_w[1], weight[0], weight[1]])

    print(table)
    table.clear_rows()