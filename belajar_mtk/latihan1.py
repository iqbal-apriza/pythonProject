import numpy as np
import matplotlib.pyplot as plt

x_axis = np.arange(-10, 10, 0.1)
y = np.zeros(len(x_axis))
diff = np.zeros(len(y) - 1)
intl = np.zeros(len(y))
count = 0

for x in x_axis:
    y[count] = pow(x, 2)
    count = count + 1

for i in range(len(y) - 1):
    dy = y[i+1] - y[i]
    dx = 0.1

    diff[i] = dy/dx

for i in range(len(y)):
    dx = 0.1

    intl[i] = intl[i-1] + (y[i] * dx)

plt.plot(y)
plt.plot(diff)
plt.plot(intl)
plt.show()