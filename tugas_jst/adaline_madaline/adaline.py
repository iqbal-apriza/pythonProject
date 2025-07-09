import numpy as np
from prettytable import PrettyTable

inputt = [
    [1, 1, -1, -1],
    [1, -1, 1, -1]
]

target = [1, -1, -1, -1]

alpha = 0.1

# np.random.seed(42)
weight = np.zeros([3])
for i in range(3):
    weight[i] = np.random.uniform()

output = np.zeros([4])
error = np.zeros([4])

table = PrettyTable(["X1", "X2", "Output", "Target", "Error", "W1", "W2", "B"])
epoch = 0

while 1:
    epoch = epoch + 1
    print(f"Epoch ke : {epoch}")
    for i in range(4):
        net = weight[0] + (inputt[0][i] * weight[1]) + (inputt[1][i] * weight[2])
        if net >= 0:
            output[i] = 1
        elif net < 0:
            output[i] = -1

        error[i] = target[i] - output[i]

        for j in range(2):
            weight[j+1] = weight[j+1] + (alpha * error[i] * inputt[j][i])
        weight[0] = weight[0] + (alpha * error[i])

        table.add_row([inputt[0][i], inputt[1][i], np.round(output[i], 3), np.round(target[i], 3), np.round(error[i], 3), np.round(weight[1], 3), np.round(weight[2], 3), np.round(weight[0], 3)])

    print(table)
    table.clear_rows()

    mse = np.mean(error ** 2)
    print(f"Mean Squred Error = {mse}")
    print("")
    if mse <= 0.05:
        break
