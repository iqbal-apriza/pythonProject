# -------------------Libraries Import-------------------
import numpy as np
import matplotlib.pyplot as plt


# ------------------------Dataset------------------------
inputs = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
])
target = np.array([1, 1, 1, 0])

num_rows = inputs.shape[0]
num_cols = inputs.shape[1]


# -----------------------Functions-----------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def mse(target, predicted):
    return np.average((target - predicted) ** 2)


# -----------------Weight Initialization-----------------
num_neurons_l1 = 10
num_neurons_l2 = 10

# weight_v = np.array([
#     [-0.3, 0.1, 0.1],
#     [0.3, 0.2, -0.1],
#     [0.2, 0.3, -0.2]
# ])

# weight_w = np.array([0.1, -0.3, 0.3, -0.2])

np.random.seed(42)

weight_in_l1 = np.random.uniform(-0.5, 0.5, size=(num_cols + 1, num_neurons_l1))
weight_l1_l2 = np.random.uniform(-0.5, 0.5, size=(num_neurons_l1 + 1, num_neurons_l2))
weight_l2_out = np.random.uniform(-0.5, 0.5, size=(num_neurons_l2 + 1, 1))

learn_rate = 0.1
output = np.zeros(num_rows)


# -----------------------Training-----------------------
epoch = 0
while 1:
    epoch += 1
    for i in range(num_rows):
        net_out = weight_in_l1[0] + np.dot(inputs[i], weight_in_l1[1:])
        hidden_l1_out = sigmoid(net_out)

        net_out = weight_l1_l2[0] + np.dot(net_out, weight_l1_l2[1:])
        hidden_l2_out = sigmoid(net_out)

        net_out = weight_l2_out[0] + np.dot(net_out, weight_l2_out[1:])
        output[i] = sigmoid(net_out)

        error = target[i] - hidden_l2_out[i]
        d_error = error * sigmoid_deriv(hidden_l2_out[i])

        error_hidden = d_error * weight_l1_l2[1:]
        d_hidden = error_hidden.T * sigmoid_deriv(hidden_l1_out)

        weight_l1_l2[0] += learn_rate * d_error
        weight_l1_l2[1:] += learn_rate * d_error * hidden_l1_out.reshape(-1, 1)

        weight_in_l1[0] += learn_rate * d_hidden.flatten()
        weight_in_l1[1:] += learn_rate * d_hidden * inputs[i].reshape(-1, 1)

    total_error = mse(target, output)
    print(f"Epoch : {epoch}, MSE : {total_error}")


# -----------------------Testing-----------------------
# for i in range(num_cols):
#     net_out = weight_v[0] + np.dot(inputs[i], weight_v[1:])
#     hidden_out = sigmoid(net_out)

#     net_out = weight_w[0] + np.dot(net_out, weight_w[1:])
#     hidden_l2_out[i] = sigmoid(net_out)

# print(hidden_l2_out)