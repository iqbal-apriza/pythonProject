inputt = [
    [1, -1, 1,
     -1, 1, -1,
     1, -1, 1],

    [1, 1, 1,
     -1, 1, -1,
     1, 1, 1]
]

weight = [0, 0, 0, 0, 0, 0, 0, 0, 0]
bias = 0
target = [1, -1]
output = [0, 0]
output_net = [0, 0]
total = [0, 0]

for i in range(2):
    print(f"-----Perubahan pola ke-{i+1}-----")
    for j in range(9):
        weight[j] = weight[j] + (inputt[i][j] * target[i])
        print(f"Bobot ke-{j+1} = {weight[j]}")
    bias = bias + target[i]
    print(f"Bias = {bias}")

for i in range(2):
    for j in range(9):
        total[i] = total[i] + inputt[i][j] * weight[j]
    output[i] = bias + total[i]

    if output[i] > 0:
        output_net[i] = 1
    elif output[i] < 0:
        output_net[i] = -1

print("----------Hasil Akhir----------")
print(f"Bobot = {weight}")
print(f"Bias = {bias}")
print(f"Output = {output}")
print(f"Output Net = {output_net}")
print(f"Target = {target}")