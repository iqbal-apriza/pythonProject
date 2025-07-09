import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../dataset/data_kelembapan.csv")
df_array = df.to_numpy()

min_val = np.min(df_array, 0)
max_val = np.max(df_array, 0)

df_norm = np.empty([12, 9])
for i in range(12):
    for j in range(9):
        df_norm[i][j] = (0.8 * (df_array[i][j] - min_val[j]) / (max_val[j] - min_val[j])) + 0.1

target = df_norm[:, 8]

output_hid = np.empty([6])

output = np.empty([12])
error = np.empty([12])
err_tol = 0.1

fig, axs = plt.subplots(3, 4)

# Bobot random, epoch max
weight_w = [-4.74819599, 0.98053885, -2.79944548, 1.27689327, -1.14059348, 4.67779393, 3.25816066]
weight_v = [
    [ 1.4306785 , -0.36785731,  4.36724196,  0.25459546, -1.27152196, -4.47500213],
    [ 1.68125912, -0.01975304, -0.13257407,  0.08594549,  2.46024116,  1.3302206 ],
    [ 0.52489139, -0.97706111, -2.87455473,  0.01025709,  2.56228253,  2.57642619],
    [-1.8963622 , -0.57212728,  2.64073536,  0.4005749 ,  0.25546773,  4.17898026],
    [-2.1995867 , -0.10237901, -2.44254746, -1.24079179, -0.47201418, -0.77286639],
    [ 2.56195555, -0.00656057, -2.61249676, -0.16557173,  1.22907054, -4.62823979],
    [-2.71069265, -0.05742789, -1.4697366 , -0.52179475,  1.56442531,  1.19530678],
    [-0.05508266, -0.04643546, -0.84400296,  0.09881075,  2.52232453,  4.30631162],
    [-4.02074432, -0.37551556, -0.28950919, -0.84364097, -0.68246314, -2.56861667]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

print("Hasil Training dengan Data Random")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")

# axs[0, 0].plot(target, color="Red", label="Target"),
# axs[0, 0].plot(output, color="Blue", label="Output"),
# axs[0, 0].set_title("Data Random")
# axs[0, 0].label_outer()
# axs[0, 0].legend()

# Bobot random + momentum, epoch max
weight_w = [-4.04918501, 1.42288733, -2.39737942, 1.26290378, -2.02621035, 4.70971122, 2.78707219]
weight_v = [
    [-0.59396261, -0.63916815,  3.03687259,  0.18662101, -1.50123861, -6.17439383],
    [ 2.52826996,  1.61383871,  1.92179752, -0.29935337,  2.97604906,  1.20671088],
    [ 1.30898098, -1.0139359 , -2.47305373, -0.31695407,  2.5381751 ,  1.42671531],
    [-1.3219513 , -1.16247595,  1.5653589 ,  0.0909663 , -0.04296361,  5.66130356],
    [-3.19597194, -0.34798014, -4.30344787, -1.51314196, -0.8803698 , -0.73605287],
    [ 0.39822641,  0.94583387, -1.88266597,  0.21716868,  1.54717639, -2.00059485],
    [-0.9397918 ,  0.70264502, -1.29875457, -0.79393507,  1.46939314,  1.84766428],
    [ 0.78670854,  0.90801946,  0.97176179,  0.16303888,  3.67633009,  3.98501788],
    [-1.5319323 ,  0.11606724,  0.12960306, -1.05413762, -0.90490136, -2.06205367]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

print("\nHasil Training dengan Data Random + Momentum")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")

# axs[0, 1].plot(target, color="Red", label="Target"),
# axs[0, 1].plot(output, color="Blue", label="Output"),
# axs[0, 1].set_title("Data Random + Momentum")
# axs[0, 1].label_outer()
# axs[0, 1].legend()

# Bobot random + momentum + nguyen, epoch max
weight_w = [-4.03582963, 1.55348898, -2.20178175, 1.31464124, -1.97304086, 4.85815473, 2.90245969]
weight_v = [
    [-0.7003473 , -0.78387229,  3.21064036,  0.65792139, -1.48984714, -6.17996183],
    [ 2.62917208,  1.95354258,  1.40656148, -0.32152775,  2.92029663,  1.37435838],
    [ 1.37773667, -0.70698936, -2.10722867, -0.40900466,  2.65070948,  1.38394236],
    [-1.16577907, -1.36839246,  1.56350404,  0.11705475, -0.13903773,  5.57794218],
    [-2.98618239, -0.31876292, -4.31907832, -1.50898369, -0.80167532, -0.26783977],
    [ 0.30760996,  1.18881073, -2.00867369,  0.06991503,  1.48143376, -2.57746143],
    [-0.69861978,  0.89595206, -1.57202249, -0.82218169,  1.6097469 ,  1.98044257],
    [ 0.41888491,  1.07691484,  0.49917325,  0.08613224,  3.64634299,  3.97784462],
    [-1.58345928, -0.08126964, -0.18017534, -1.05109352, -1.01640807, -2.27430109]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

print("\nHasil Training dengan Data Random + Momentum + Nguyen")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")

# axs[0, 2].plot(target, color="Red", label="Target"),
# axs[0, 2].plot(output, color="Blue", label="Output"),
# axs[0, 2].set_title("Data Random + Momentum + Nguyen")
# axs[0, 2].label_outer()
# axs[0, 2].legend()

# Bobot random, mse 0.1
weight_w = [ 0.32170837, 0.02292682, -0.24965516, -0.23700446, -0.39525526, 0.24681505, -0.06804689]
weight_v = [
    [-0.17338061,  0.13771697, -0.01596711,  0.21995001,  0.07554521, -0.00406934],
    [-0.12407829,  0.23212074, -0.33368683, -0.43584189,  0.11386952, -0.47740699],
    [ 0.3335648 , -0.31807104, -0.18775804, -0.06359244,  0.12168074, -0.20631735],
    [-0.04269865, -0.3002266 ,  0.10068493,  0.11243794, -0.42466023,  0.46732422],
    [-0.19456428,  0.18436175, -0.36899142, -0.46047393, -0.23081905, -0.18685799],
    [ 0.04797761,  0.46973321,  0.45065656,  0.10442461, -0.39816357, -0.45278621],
    [-0.10998211,  0.32886064, -0.20904525, -0.35318058, -0.41314254,  0.27419088],
    [-0.49371732,  0.20696563,  0.27905038, -0.13705929,  0.3721946 , -0.16782462],
    [-0.18772358,  0.22972826,  0.39701388, -0.37469855,  0.27288376,  0.2728505 ]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Random, mse <= 0.1")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[0, 0].plot(target, color="Red", label="Target"),
axs[0, 0].plot(output, color="Blue", label="Output"),
# axs[0, 0].set_title("Data Random, mse <= 0.1")
axs[0, 0].label_outer()
axs[0, 0].legend()

# Bobot random, mse 0.01
weight_w = [ 0.32170837, 0.02292682, -0.24965516, -0.23700446, -0.39525526, 0.24681505, -0.06804689]
weight_v = [
    [-0.17338061,  0.13771697, -0.01596711,  0.21995001,  0.07554521, -0.00406934],
    [-0.12407829,  0.23212074, -0.33368683, -0.43584189,  0.11386952, -0.47740699],
    [ 0.3335648 , -0.31807104, -0.18775804, -0.06359244,  0.12168074, -0.20631735],
    [-0.04269865, -0.3002266 ,  0.10068493,  0.11243794, -0.42466023,  0.46732422],
    [-0.19456428,  0.18436175, -0.36899142, -0.46047393, -0.23081905, -0.18685799],
    [ 0.04797761,  0.46973321,  0.45065656,  0.10442461, -0.39816357, -0.45278621],
    [-0.10998211,  0.32886064, -0.20904525, -0.35318058, -0.41314254,  0.27419088],
    [-0.49371732,  0.20696563,  0.27905038, -0.13705929,  0.3721946 , -0.16782462],
    [-0.18772358,  0.22972826,  0.39701388, -0.37469855,  0.27288376,  0.2728505 ]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Random, mse <= 0.01")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[0, 1].plot(target, color="Red", label="Target"),
axs[0, 1].plot(output, color="Blue", label="Output"),
# axs[0, 1].set_title("Data Random, mse <= 0.01")
axs[0, 1].label_outer()
axs[0, 1].legend()

# Bobot random, mse 0.001
weight_w = [-0.14916599, -0.51641525, 0.98913304, 0.01890683, -2.12778146, 1.63502006, -0.35888411]
weight_v = [
    [-0.24505214,  0.10301213, -0.23640269,  0.23321843, -1.74119822, -0.27632791],
    [-0.07363235,  0.34969521,  0.04835644, -0.36556537,  1.61162019,  0.05247336],
    [ 0.34950266, -0.29831043, -0.09659478, -0.04005513,  0.03660543, -0.05550796],
    [-0.03536769, -0.23288589,  0.28671689,  0.15264797,  0.03334727,  0.74494597],
    [-0.20397039,  0.20861813, -0.35386143, -0.42831433, -0.37571639, -0.1873123 ],
    [ 0.08013852,  0.56553547,  0.71532444,  0.16846921,  0.59377549, -0.1209051 ],
    [-0.0473705 ,  0.45595131,  0.21479284, -0.28421227,  1.31528388,  0.85573752],
    [-0.51713309,  0.19675874,  0.16149946, -0.12009154, -0.48293358, -0.36287569],
    [-0.16378004,  0.30315399,  0.60045037, -0.3175004 ,  0.89376283,  0.53603525]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Random, mse <= 0.001")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[0, 2].plot(target, color="Red", label="Target"),
axs[0, 2].plot(output, color="Blue", label="Output"),
# axs[0, 2].set_title("Data Random, mse <= 0.001")
axs[0, 2].label_outer()
axs[0, 2].legend()

# Bobot random, mse 0.0001
weight_w = [-9.25282820e-01, -9.06438192e-01, 9.00225462e-01, -3.19627442e-03, -2.67600018e+00, 3.21895890e+00, 1.17015667e+00]
weight_v = [
     [-0.20151322,  0.0600649 , -0.52623949,  0.22537683, -2.24070313, -1.05827104],
     [ 0.09970463,  0.55395766,  0.35143105, -0.36629133,  2.60461295,  1.55753695],
     [ 0.27669007, -0.51448562, -0.40490702, -0.04449749,  0.43187339, -0.91065061],
     [ 0.12864609, -0.04669921,  0.56289039,  0.15105645,  1.12463293,  2.72568964],
     [-0.20470101,  0.20639551, -0.49060603, -0.43167004, -1.43088491, -1.09097478],
     [ 0.03400585,  0.55342861,  0.55006867,  0.16400376, -0.88433059, -1.79023606],
     [ 0.11774041,  0.63306309,  0.53676435, -0.28484881,  3.11116694,  3.02568888],
     [-0.42475295,  0.25905192,  0.17346761, -0.12253588, -0.04698267, -0.3810255 ],
     [-0.23502038,  0.20112547,  0.27968763, -0.32302796, -0.85298022, -2.54408137]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Random, mse <= 0.0001")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[0, 3].plot(target, color="Red", label="Target"),
axs[0, 3].plot(output, color="Blue", label="Output"),
# axs[0, 3].set_title("Data Random, mse <= 0.0001")
axs[0, 3].label_outer()
axs[0, 3].legend()

# Bobot random + momentum, mse 0.1
weight_w = [ 0.68910366, -0.00176741, -0.40442106, -0.42822215, -0.82507529, 0.47901355, -0.14982558]
weight_v = [
     [-0.33943095,  0.27564102,  0.01331227,  0.45526014,  0.2081433 ,  0.00265001],
     [-0.24088056,  0.46428073, -0.63653706, -0.86250295,  0.27110967, -0.94590227],
     [ 0.671188  , -0.63598814, -0.35573587, -0.12410683,  0.27023475, -0.40735696],
     [-0.07957941, -0.60023178,  0.23001028,  0.23542919, -0.8114228 ,  0.9422153 ],
     [-0.38405087,  0.36878697, -0.70991679, -0.91190085, -0.42545766, -0.3667767 ],
     [ 0.10295557,  0.93945543,  0.9349223 ,  0.21961395, -0.75078516, -0.89628829],
     [-0.21344576,  0.65772434, -0.39075072, -0.69825686, -0.78769966,  0.55632327],
     [-0.98354023,  0.41401294,  0.58235483, -0.26594359,  0.77606956, -0.32995602],
     [-0.36870712,  0.4594864 ,  0.82310346, -0.74107429,  0.58713121,  0.55393821]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Random + Momentum, mse <= 0.1")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[1, 0].plot(target, color="Red", label="Target"),
axs[1, 0].plot(output, color="Blue", label="Output"),
# axs[1, 0].set_title("Data Random + momentum, mse <= 0.1")
axs[1, 0].label_outer()
axs[1, 0].legend()

# Bobot random + momentum, mse 0.01
weight_w = [ 0.68910366, -0.00176741, -0.40442106, -0.42822215, -0.82507529, 0.47901355, -0.14982558]
weight_v = [
     [-0.33943095,  0.27564102,  0.01331227,  0.45526014,  0.2081433 ,  0.00265001],
     [-0.24088056,  0.46428073, -0.63653706, -0.86250295,  0.27110967, -0.94590227],
     [ 0.671188  , -0.63598814, -0.35573587, -0.12410683,  0.27023475, -0.40735696],
     [-0.07957941, -0.60023178,  0.23001028,  0.23542919, -0.8114228 ,  0.9422153 ],
     [-0.38405087,  0.36878697, -0.70991679, -0.91190085, -0.42545766, -0.3667767 ],
     [ 0.10295557,  0.93945543,  0.9349223 ,  0.21961395, -0.75078516, -0.89628829],
     [-0.21344576,  0.65772434, -0.39075072, -0.69825686, -0.78769966,  0.55632327],
     [-0.98354023,  0.41401294,  0.58235483, -0.26594359,  0.77606956, -0.32995602],
     [-0.36870712,  0.4594864 ,  0.82310346, -0.74107429,  0.58713121,  0.55393821]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Random + Momentum, mse <= 0.01")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[1, 1].plot(target, color="Red", label="Target"),
axs[1, 1].plot(output, color="Blue", label="Output"),
# axs[1, 1].set_title("Data Random + momentum, mse <= 0.1")
axs[1, 1].label_outer()
axs[1, 1].legend()

# Bobot random + momentum, mse 0.001
weight_w = [ 0.30600836, -0.5709003, 0.82720214, -0.36017972, -2.68067559, 1.09453138, -0.77661813]
weight_v = [
     [-0.66377793,  0.19375123, -0.10229607,  0.43678879, -2.05260811, -0.19732762],
     [-0.21962523,  0.52972801, -0.48322647, -0.72330214,  2.17412237, -0.86756945],
     [ 0.59147909, -0.665713  , -0.32887435, -0.10019945,  0.29918916, -0.40943688],
     [-0.1424717 , -0.55881179,  0.31693387,  0.31852818, -0.14835621,  0.97510448],
     [-0.48927562,  0.35705914, -0.74126279, -0.87391533, -0.96763779, -0.44902903],
     [ 0.0954909 ,  0.99750829,  1.03218818,  0.34202889,  0.37962212, -0.87321425],
     [-0.14222572,  0.74851347, -0.19963692, -0.54755615,  1.63495685,  0.68843781],
     [-1.1403143 ,  0.36830311,  0.47091702, -0.26751683, -0.88981141, -0.49374803],
     [-0.41989917,  0.48159198,  0.88594135, -0.64164192,  1.32252346,  0.53491414]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Random + Momentum, mse <= 0.001")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[1, 2].plot(target, color="Red", label="Target"),
axs[1, 2].plot(output, color="Blue", label="Output"),
# axs[1, 2].set_title("Data Random + momentum, mse <= 0.001")
axs[1, 2].label_outer()
axs[1, 2].legend()

# Bobot random + momentum, mse 0.0001
weight_w = [-0.40772486, -1.23679161, 1.23691534, -0.59183338, -3.31608635, 3.04014644, 0.64790943]
weight_v = [
     [-0.66938339,  0.11099484, -0.20784744,  0.43411755, -2.41074256, -1.00365799],
     [-0.20599423,  0.89629282,  0.01347181, -0.62374329,  2.54118029,  0.0812739 ],
     [ 0.57051236, -1.16970128, -0.77042954, -0.13070551,  0.40843017, -1.369541  ],
     [-0.12913146, -0.21896472,  0.89744961,  0.41972159,  2.31022981,  3.25598007],
     [-0.48587485,  0.50215973, -0.63476895, -0.84801456, -2.4404441 , -0.85575875],
     [ 0.09413176,  1.08409962,  0.99258001,  0.32279835, -1.51980919, -1.84926515],
     [-0.13156928,  0.9930339 ,  0.30648167, -0.45851703,  4.48752308,  2.64649384],
     [-1.12952766,  0.64349922,  0.96882065, -0.2101655 , -0.30125781, -0.99023996],
     [-0.42805473,  0.35264985,  0.5931272 , -0.6919024 , -1.83427908, -2.40112254]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Random + Momentum, mse <= 0.0001")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[1, 3].plot(target, color="Red", label="Target"),
axs[1, 3].plot(output, color="Blue", label="Output"),
# axs[1, 3].set_title("Data Random + momentum, mse <= 0.0001")
axs[1, 3].label_outer()
axs[1, 3].legend()

# Bobot random + momentum + nguyen, mse 0.1
weight_w = [ 0.70765142, -0.0201093, -0.3781145, -0.42429659, -0.7895435, 0.50057717, -0.14286104]
weight_v = [
     [-0.60272421,  0.48234772, -0.0284411 ,  0.78279682,  0.30278422, -0.00519706],
     [-0.24153565,  0.46428231, -0.63814127, -0.85799255,  0.27082335, -0.94491255],
     [ 0.67072533, -0.63599962, -0.35608731, -0.1200044 ,  0.27135067, -0.40650464],
     [-0.08013043, -0.60022637,  0.22926442,  0.23943173, -0.81085482,  0.94310337],
     [-0.38447082,  0.36878222, -0.7107387 , -0.90778596, -0.42486242, -0.36599944],
     [ 0.10231216,  0.93945273,  0.93337547,  0.22441703, -0.75075164, -0.89531869],
     [-0.21409269,  0.65772335, -0.39228767, -0.69423802, -0.78798754,  0.55720548],
     [-0.98388457,  0.41400514,  0.58205738, -0.26210186,  0.77683794, -0.32927455],
     [-0.3693142 ,  0.45948614,  0.82179737, -0.736531  ,  0.58714916,  0.55492309]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Nguyen + Momentum, mse <= 0.1")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[2, 0].plot(target, color="Red", label="Target"),
axs[2, 0].plot(output, color="Blue", label="Output"),
# axs[2, 0].set_title("Data Random + momentum + nguyen, mse <= 0.1")
axs[2, 0].label_outer()
axs[2, 0].legend()

# Bobot random + momentum + nguyen, mse 0.01
weight_w = [ 0.70765142, -0.0201093, -0.3781145, -0.42429659, -0.7895435, 0.50057717, -0.14286104]
weight_v = [
     [-0.60272421,  0.48234772, -0.0284411 ,  0.78279682,  0.30278422, -0.00519706],
     [-0.24153565,  0.46428231, -0.63814127, -0.85799255,  0.27082335, -0.94491255],
     [ 0.67072533, -0.63599962, -0.35608731, -0.1200044 ,  0.27135067, -0.40650464],
     [-0.08013043, -0.60022637,  0.22926442,  0.23943173, -0.81085482,  0.94310337],
     [-0.38447082,  0.36878222, -0.7107387 , -0.90778596, -0.42486242, -0.36599944],
     [ 0.10231216,  0.93945273,  0.93337547,  0.22441703, -0.75075164, -0.89531869],
     [-0.21409269,  0.65772335, -0.39228767, -0.69423802, -0.78798754,  0.55720548],
     [-0.98388457,  0.41400514,  0.58205738, -0.26210186,  0.77683794, -0.32927455],
     [-0.3693142 ,  0.45948614,  0.82179737, -0.736531  ,  0.58714916,  0.55492309]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Nguyen + Momentum, mse <= 0.01")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[2, 1].plot(target, color="Red", label="Target"),
axs[2, 1].plot(output, color="Blue", label="Output"),
# axs[2, 1].set_title("Data Random + momentum + nguyen, mse <= 0.01")
axs[2, 1].label_outer()
axs[2, 1].legend()

# Bobot random + momentum + nguyen, mse 0.001
weight_w = [ 0.50548855, -0.45075245, 0.80505081, -0.2650953, -2.80606155, 1.18100203, -0.66638881]
weight_v = [
     [-0.92575004,  0.42692644, -0.12881419,  0.83031292, -1.98030559, -0.227513  ],
     [-0.18344636,  0.50374399, -0.49151519, -0.68559388,  2.18481085, -0.82573143],
     [ 0.58652049, -0.6594743 , -0.32734089, -0.05019075,  0.18356208, -0.3937674 ],
     [-0.10773102, -0.57463426,  0.31809035,  0.3511026 , -0.02932815,  1.00751331],
     [-0.48210392,  0.3605615 , -0.73778021, -0.83763043, -0.99351265, -0.45394449],
     [ 0.12959968,  0.97554597,  1.02802093,  0.38087215,  0.44586653, -0.84338388],
     [-0.09988382,  0.71358618, -0.20987019, -0.51551911,  1.72933171,  0.74392728],
     [-1.14575808,  0.384156  ,  0.47947668, -0.22969716, -0.98667283, -0.51962986],
     [-0.40115838,  0.47032525,  0.87905124, -0.59855268,  1.22222331,  0.54746319]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Nguyen + Momentum, mse <= 0.001")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[2, 2].plot(target, color="Red", label="Target"),
axs[2, 2].plot(output, color="Blue", label="Output"),
# axs[2, 2].set_title("Data Random + momentum + nguyen, mse <= 0.001")
axs[2, 2].label_outer()
axs[2, 2].legend()

# Bobot random + momentum + nguyen, mse 0.0001
weight_w = [-0.03084528, -0.90049883, 1.06502391, -0.47768839, -3.57168053, 3.04143001, 0.52542777]
weight_v = [
     [-0.98322674,  0.37370644, -0.22723707,  0.86754698, -2.30935522, -1.23602037],
     [-0.18827082,  0.68079415, -0.08728456, -0.61242483,  2.4975473 ,  1.40092966],
     [ 0.52733868, -0.90586712, -0.64162922, -0.02006253,  0.32583215, -1.3317715 ],
     [-0.09165541, -0.41957209,  0.78372775,  0.41359547,  2.50764392,  2.96145391],
     [-0.50553798,  0.42066703, -0.65695811, -0.80705539, -2.46885079, -0.172936  ],
     [ 0.12318474,  1.00872359,  1.01209138,  0.38948404, -1.55172492, -2.09421702],
     [-0.0877728 ,  0.82746038,  0.21199158, -0.45092325,  4.60222522,  2.86800004],
     [-1.17598984,  0.50179539,  0.85172332, -0.17107485, -0.53005345,  1.77704094],
     [-0.43673386,  0.41351434,  0.66554377, -0.58950423, -2.15593844, -1.06394483]
]

for i in range(12):
    for j in range(6):
        jlh = 0
        for k in range(8):
            jlh = jlh + (df_norm[i][k] * weight_v[k+1][j])
        net = weight_v[0][j] + jlh
        output_hid[j] = 1 / (1 + np.exp(-net))

    jlh = 0
    for j in range(6):
        jlh = jlh + (output_hid[j] * weight_w[j+1])
    net = weight_w[0] + jlh
    output[i] = 1 / (1 + np.exp(-net))

    error[i] = target[i] - output[i]

accuracy = np.mean(np.abs(output - target) / target <= err_tol) * 100
print("\nHasil Training dengan Data Nguyen + Momentum, mse <= 0.0001")
print(pd.DataFrame({
    "Output" : output,
    "Target" : target,
    "Error" : output - target,
    "Akurasi" : np.abs(output - target) / target <= err_tol
}))
print(f"Mean Square Error = {np.mean(error ** 2)}")
print(f"Akurasi = {accuracy}%")

axs[2, 3].plot(target, color="Red", label="Target"),
axs[2, 3].plot(output, color="Blue", label="Output"),
# axs[2, 3].set_title("Data Random + momentum + nguyen, mse <= 0.0001")
axs[2, 3].label_outer()
axs[2, 3].legend()

plt.show()