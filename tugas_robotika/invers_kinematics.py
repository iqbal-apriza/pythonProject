import numpy as np

endpoint = [0, 1, 2]  # [Px, Py, Pz]
leng = [1, 1, 1]     # [d1, a2, a3]

# Lengan Bawah
angle_3_d = np.acos(((endpoint[0] ** 2) + (endpoint[1] ** 2) + ((endpoint[2] - leng[0]) ** 2) - (leng[1] ** 2) - (leng[2] ** 2)) / (2 * leng[1] * leng[2])) * 180/np.pi
angle_2_d = (np.atan2((endpoint[2] - leng[0]) , np.sqrt((endpoint[0] ** 2) + (endpoint[1] ** 2))) - np.acos(((leng[1] ** 2) + (endpoint[0] ** 2) + (endpoint[1] ** 2) + ((endpoint[2] - leng[0]) ** 2) - (leng[2] ** 2)) / (2 * leng[1] * (np.sqrt((endpoint[0] ** 2) + (endpoint[1] ** 2) + ((endpoint[2] - leng[0]) ** 2)))))) * 180/np.pi
angle_1_d = np.atan2(endpoint[1] , endpoint[0]) * 180/np.pi
print("Elbow Down:")
print(f"Sudut 1 = {angle_1_d}")
print(f"Sudut 2 = {angle_2_d}")
print(f"Sudut 3 = {angle_3_d}")
print("")

# Lengan Atas
angle_3_u = -np.acos(((endpoint[0] ** 2) + (endpoint[1] ** 2) + ((endpoint[2] - leng[0]) ** 2) - (leng[1] ** 2) - (leng[2] ** 2)) / (2 * leng[1] * leng[2])) * 180/np.pi
angle_2_u = (np.atan2((endpoint[2] - leng[0]) , np.sqrt((endpoint[0] ** 2) + (endpoint[1] ** 2))) + np.acos(((leng[1] ** 2) + (endpoint[0] ** 2) + (endpoint[1] ** 2) + ((endpoint[2] - leng[0]) ** 2) - (leng[2] ** 2)) / (2 * leng[1] * (np.sqrt((endpoint[0] ** 2) + (endpoint[1] ** 2) + ((endpoint[2] - leng[0]) ** 2)))))) * 180/np.pi
angle_1_u = np.atan2(endpoint[1] , endpoint[0]) * 180/np.pi
print("Elbow Up:")
print(f"Sudut 1 = {angle_1_u}")
print(f"Sudut 2 = {angle_2_u}")
print(f"Sudut 3 = {angle_3_u}")