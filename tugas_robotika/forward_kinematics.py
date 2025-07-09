import numpy as np

angl = [30, 45, 60]  # [θ1, θ2, θ3]
leng = [1, 1, 1]     # [d1, a2, a3]

x_coordinate = ((leng[1] * np.cos(angl[1] * np.pi/180)) + (leng[2] * np.cos((angl[1] + angl[2]) * np.pi/180))) * np.cos(angl[0] * np.pi/180)
y_coordinate = ((leng[1] * np.cos(angl[1] * np.pi/180)) + (leng[2] * np.cos((angl[1] + angl[2]) * np.pi/180))) * np.sin(angl[0] * np.pi/180)
z_coordinate = leng[0] + (leng[1] * np.sin(angl[1] * np.pi/180)) + (leng[2] * np.sin((angl[1] + angl[2]) * np.pi/180))
print(f"Koordinat X = {x_coordinate}")
print(f"Koordinat Y = {y_coordinate}")
print(f"Koordinat Z = {z_coordinate}")