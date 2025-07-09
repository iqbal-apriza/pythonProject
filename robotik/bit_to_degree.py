import numpy as np

degree = np.zeros([2, 6])
bit = [
    [512, 512, 512, 512, 512, 512],
    [512, 512, 512, 512, 512, 512]
]

s1_pos = 512;        s2_pos = 512
s3_pos = 512;        s4_pos = 512
s5_pos = 512;        s6_pos = 512
s23_pos = 820;       s24_pos = 208
s25_pos = 512;       s26_pos = 512
s27_pos = 512;       s28_pos = 512

degree[0][0] = ((bit[0][0] - s1_pos) * 300) / 1023
degree[0][1] = ((bit[0][1] - s3_pos) * 300) / 1023
degree[0][2] = ((bit[0][0] - s5_pos) * 300) / 1023
degree[0][3] = ((bit[0][1] - s23_pos) * 300) / 1023
degree[0][4] = ((bit[0][0] - s25_pos) * 300) / 1023
degree[0][5] = ((bit[0][1] - s27_pos) * 300) / 1023

degree[1][0] = ((bit[0][0] - s2_pos) * 300) / 1023
degree[1][1] = ((bit[0][1] - s4_pos) * 300) / 1023
degree[1][2] = ((bit[0][0] - s6_pos) * 300) / 1023
degree[1][3] = ((bit[0][1] - s24_pos) * 300) / 1023
degree[1][4] = ((bit[0][0] - s26_pos) * 300) / 1023
degree[1][5] = ((bit[0][1] - s28_pos) * 300) / 1023

print(np.round(degree, 3))