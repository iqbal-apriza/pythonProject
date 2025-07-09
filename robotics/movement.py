import numpy as np
import time

start_time = time.time()
def millis():
    return int((time.time() - start_time) * 1000)

def motion1():
    tracks = [
        [0, 20],
        [20, -20],
        [-20, 20],
        [20, -20],
        [-20, 20],
        [20, -20],
        [-20, 20],
        [20, -20],
        [-20, 20],
        [0, 0]
    ]

    

    