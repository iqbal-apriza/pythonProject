import numpy as np
import matplotlib.pyplot as plt

a1 = 35
a2 = 67
a3 = 70

def leg_fk(angles):
    coords = np.zeros(3)
    coords[0] = (a1 + a2 * np.cos(angles[1] * 180/np.pi) + a3 * np.cos((angles[1] + angles[2]) * 180/np.pi)) * np.cos(angles[0] * 180/np.pi)
    coords[1] = (a1 + a2 * np.cos(angles[1] * 180/np.pi) + a3 * np.cos((angles[1] + angles[2]) * 180/np.pi)) * np.sin(angles[0] * 180/np.pi)
    coords[2] = a2 * np.sin(angles[1] * 180/np.pi) + a3 * np.sin((angles[1] + angles[2]) * 180/np.pi)
    return coords

ang = np.array([60, 60, 90])
result = leg_fk(ang)

print(result)