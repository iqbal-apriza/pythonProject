# Nama  = Muhammad Iqbal Apriza
# NIM   = 03041282227043
# Kelas = Indralaya

# Program Python untuk data AND menggunakan algoritma Hebb Rule

# Inisialisasi
input1      = [-1, -1, 1, 1]
input2      = [-1, 1, -1, 1]
target      = [-1, -1, -1, 1]
weight      = [0, 0]
bias        = 0
output      = [0, 0, 0, 0]
output_net  = [0, 0, 0, 0]

# Iterasi perubahan bobot (weight) dan bias
for i in range(4):
    weight[0] = weight[0] + (input1[i] * target[i])
    weight[1] = weight[1] + (input2[i] * target[i])
    bias = bias + target[i]
    print(weight, bias)

# Iterasi pengecekan data
for i in range(4):
    output[i] = (input1[i] * weight[0]) + (input2[i] * weight[1]) + bias

# Iterasi fungsi aktivasi
for i in range(4):
    if output[i] > 0:
        output_net[i] = 1
    elif output[i] <= 0:
        output_net[i] = -1

print("Hasil:")
print(f"Weight = {weight}")
print(f"Bias = {bias}")
print(f"Output = {output}")
print(f"Output Net = {output_net}")