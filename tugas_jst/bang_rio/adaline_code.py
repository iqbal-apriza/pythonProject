import numpy as np
import pandas as pd


# Fungsi aktivasi bipolar
def bipolar_activation(x):
    return 1 if x >= 0 else -1


# Data input dan target untuk fungsi logika AND
inputs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
targets = np.array([1, -1, -1, -1])

# Parameter ADALINE
learning_rate = 0.1
tolerance = 0.05
weights = np.random.rand(2) * 0.01
bias = np.random.rand() * 0.01

# Training ADALINE
epoch = 0
while True:
    total_error = 0
    epoch_data = []

    for i in range(len(inputs)):
        net_input = np.dot(inputs[i], weights) + bias
        output = bipolar_activation(net_input)
        error = targets[i] - output
        total_error += error ** 2

        # Data sebelum update
        epoch_data.append({
            'Epoch': epoch + 1,
            'x1': inputs[i][0],
            'x2': inputs[i][1],
            'Target': targets[i],
            'Net Input': net_input,
            'Output': output,
            'Error': error,
            'w1': weights[0],
            'w2': weights[1],
            'bias': bias
        })

        # Update bobot dan bias
        weights += learning_rate * error * inputs[i]
        bias += learning_rate * error

    # Tampilkan tabel untuk epoch saat ini
    epoch_df = pd.DataFrame(epoch_data)
    print(f"Epoch {epoch + 1}")
    print(epoch_df)
    print("\n" + "=" * 50 + "\n")

    epoch += 1

    # Cek kondisi toleransi
    if total_error / len(inputs) <= tolerance:
        break

print("Bobot akhir ADALINE:", weights)
print("Bias akhir ADALINE:", bias)
print("Epoch terakhir:", epoch)
