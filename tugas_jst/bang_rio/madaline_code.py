import numpy as np
import pandas as pd


# Fungsi aktivasi bipolar
def bipolar_activation(x):
    return 1 if x >= 0 else -1


# Data input dan target untuk fungsi logika XOR
inputs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
targets = np.array([-1, 1, 1, -1])

# Parameter MADALINE
learning_rate = 0.5
tolerance = 0.1

# Inisialisasi bobot dan bias acak kecil
hidden_weights = np.random.uniform(-0.5, 0.5, (2, 2))
output_weights = np.random.uniform(-0.5, 0.5, 2)
hidden_bias = np.random.uniform(-0.5, 0.5, 2)
output_bias = np.random.uniform(-0.5, 0.5)

# Training MADALINE
epoch = 0
while True:
    total_error = 0
    epoch_data = []

    for i in range(len(inputs)):
        # Forward pass
        hidden_net = np.dot(inputs[i], hidden_weights.T) + hidden_bias
        hidden_output = np.array([bipolar_activation(x) for x in hidden_net])
        final_net = np.dot(hidden_output, output_weights) + output_bias
        final_output = bipolar_activation(final_net)

        # Hitung error
        error = targets[i] - final_output
        total_error += error ** 2

        # Data sebelum update
        epoch_data.append({
            'Epoch': epoch + 1,
            'x1': inputs[i][0],
            'x2': inputs[i][1],
            'Target': targets[i],
            'Hidden Net 1': hidden_net[0],
            'Hidden Net 2': hidden_net[1],
            'Output Hidden 1': hidden_output[0],
            'Output Hidden 2': hidden_output[1],
            'Final Net': final_net,
            'Final Output': final_output,
            'Error': error,
            'w_hidden1_1': hidden_weights[0][0],
            'w_hidden1_2': hidden_weights[0][1],
            'w_hidden2_1': hidden_weights[1][0],
            'w_hidden2_2': hidden_weights[1][1],
            'w_output1': output_weights[0],
            'w_output2': output_weights[1],
            'hidden_bias1': hidden_bias[0],
            'hidden_bias2': hidden_bias[1],
            'output_bias': output_bias
        })

        # Update bobot jika ada error
        if error != 0:
            output_weights += learning_rate * error * hidden_output
            output_bias += learning_rate * error
            hidden_weights += learning_rate * error * np.outer(hidden_output, inputs[i])
            hidden_bias += learning_rate * error * hidden_output

    # Tampilkan tabel untuk epoch saat ini
    epoch_df = pd.DataFrame(epoch_data)
    print(f"Epoch {epoch + 1}")
    print(epoch_df)
    print("\n" + "=" * 50 + "\n")

    epoch += 1

    # Cek kondisi toleransi
    if total_error / len(inputs) <= tolerance:
        break

print("Bobot akhir MADALINE (hidden):", hidden_weights)
print("Bobot akhir MADALINE (output):", output_weights)
print("Bias akhir MADALINE (hidden):", hidden_bias)
print("Bias akhir MADALINE (output):", output_bias)
print("Epoch terakhir:", epoch)
