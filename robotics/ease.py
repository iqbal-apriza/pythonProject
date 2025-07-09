import numpy as np

def easeInSine(x):
    return 1 - np.cos((x * np.pi) / 2)

def easeOutSine(x):
    return np.sin((x * np.pi) / 2)

def easeInOutSine(x):
    return -(np.cos(np.pi * x) - 1) / 2

def easeInQuad(x):
    return np.pow(x, 2)

def easeOutQuad(x):
    return 1 - np.pow(1 - x, 2)

def easeInOutQuad(x):
    if x < 0.5:
        return 2 * np.pow(x, 2)
    else:
        return 1 - np.pow(-2 * x + 2, 2) / 2
    
def easeInCubic(x):
    return np.pow(x, 3)

def easeOutCubic(x):
    return 1 - np.pow(1 - x, 3)

def easeInOutCubic(x):
    if x < 0.5:
        return 4 * np.pow(x, 3)
    else:
        return 1 - np.pow(-2 * x + 2, 3)
    
def easeInQuart(x):
    return np.pow(x, 4)

def easeOutQuart(x):
    return 1 - np.pow(1 - x, 4)

def easeInOutQuart(x):
    if x < 0.5:
        return 8 * np.pow(x, 4)
    else:
        return 1 - np.pow(-2 * x + 2, 4) / 2
    
def easeInQuint(x):
    return np.pow(x, 5)

def easeOutQuint(x):
    return 1 - np.pow(1 - x, 5)

def easeInOutQuint(x):
    if x < 0.5:
        return 16 * np.pow(x, 5)
    else:
        return 1 - np.pow(-2 * x + 2, 5) / 2
    
def easeInExpo(x):
    if x == 0:
        return 0
    else:
        return np.pow(2, 10 * x - 10)
    
def easeOutExpo(x):
    if x == 1:
        return 1
    else:
        return 1 - np.pow(2, -10 * x)
    
def easeInOutExpo(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    elif x < 0.5:
        return np.pow(2, 20 * x - 10) / 2
    else:
        return (2 - np.pow(2, -20 * x + 10)) / 2
    
def easeInCirc(x):
    return 1 - np.sqrt(1 - np.pow(x, 2))

def easeOutCirc(x):
    return np.sqrt(1 - np.pow(x - 1, 2))

def easeInOutCirc(x):
    if x < 0.5:
        return (1 - np.sqrt(1 - np.pow(2 * x, 2))) / 2
    else:
        return (np.sqrt(1 - np.pow(-2 * x + 2, 2)) + 1) / 2
    
def easeInBack(x):
    c1 = 1.70158
    c3 = c1 + 1

    return c3 * np.pow(x, 3) - c1 * np.pow(x, 2)

def easeOutBack(x):
    c1 = 1.70158
    c3 = c1 + 1

    return 1 + c3 * np.pow(x - 1, 3) + c1 * np.pow(x - 1, 2)

def easeInOutBack(x):
    c1 = 1.70158
    c2 = c1 * 1.525

    if x < 0.5:
        return (np.pow(2 * x, 2) * ((c2 + 1) * 2 * x - c2)) / 2
    else:
        return (np.pow(2 * x - 2, 2) * ((c2 + 1) * (x * 2 - 2) + c2) + 2) / 2
    
def easeInElastic(x):
    c4 = (2 * np.pi) / 3

    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        -np.pow(2, 10 * x - 10) * np.sin((x * 10 - 10.75) * c4)

def easeOutElastic(x):
    c4 = (2 * np.pi) / 3

    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return np.pow(2, -10 * x) * np.sin((x * 10 - 0.75) * c4) + 1
    
def easeInOutElastic(x):
    c5 = (2 * np.pi) / 4.5

    if x == 0:
        return 0
    elif x == 1:
        return 1
    elif x < 0.5:
        return -(np.pow(2, 20 * x - 10) * np.sin((20 * x - 11.125) * c5)) / 2
    else:
        return (np.pow(2, -20 * x + 10) * np.sin((20 * x - 11.125) * c5)) / 2 + 1
    
def easeOutBounce(x):
    n1 = 7.5625
    d1 = 2.75

    if x < 1 / d1:
        return n1 * np.pow(x, 2)
    elif x < 2 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    elif x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    else:
        x -= 2.625 / d1
        return n1 * x * x + 0.984375
    
def easeInBounce(x):
    return 1 - easeOutBounce(1 - x)

def easeInOutBounce(x):
    if x < 0.5:
        return (1 - easeOutBounce(1 - 2 * x)) / 2
    else:
        return (1 + easeOutBounce(2 * x - 1)) / 2
    
def easeCustomBounce(x, a, k, n):
    return np.clip(1 - a * np.exp(-k * x) * np.abs(np.cos(n * np.pi * x)), 0, 1)