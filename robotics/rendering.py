import numpy as np
import math as mt
import matplotlib.pyplot as plt

def deg(val):
    return val * 180/np.pi

def rad(val):
    return val * np.pi/180

def rot_x(coor, ang):
    mtx = [
        [ 1,                 0,                 0],
        [ 0,  np.cos(rad(ang)), -np.sin(rad(ang))],
        [ 0,  np.sin(rad(ang)),  np.cos(rad(ang))]
    ]
    return np.dot(mtx, coor)

def rot_y(coor, ang):
    mtx = [
        [ np.cos(rad(ang)),  0,  np.sin(rad(ang))],
        [                0,  1,                 0],
        [-np.sin(rad(ang)),  0,  np.cos(rad(ang))]
    ]
    return np.dot(mtx, coor)

def rot_z(coor, ang):
    mtx = [
        [ np.cos(rad(ang)), -np.sin(rad(ang)),  0],
        [ np.sin(rad(ang)),  np.cos(rad(ang)),  0],
        [                0,                 0,  1]
    ]
    return np.dot(mtx, coor)

points = [3, 4, 5]
projected = rot_y(points, 30)

print(projected)