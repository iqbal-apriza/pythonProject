from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12, 9))

image = Image.open("../kilat_8ok.jpg")

# Rotasi
rotate = image.rotate(45)

fig.add_subplot(2, 2, 1)
plt.imshow(rotate)
plt.axis("off")
plt.title("Rotasi 45 derajat")

# Translasi
translation = image.rotate(0, translate=(500, 300))

fig.add_subplot(2, 2, 2)
plt.imshow(translation)
plt.axis("off")
plt.title("Translasi")

# Zoom
width, height = image.size
left = width * 1/3
top = height * 0
right = width * 2/3
bottom = height * 1/3
zoom = image.crop((left, top, right, bottom))

fig.add_subplot(2, 2, 3)
plt.imshow(zoom)
plt.axis("off")
plt.title("Zoom")

# Shear
shear = image.transform((image.width, image.height), Image.AFFINE, (1, 0.5, -100, 0, 1, 0))

fig.add_subplot(2, 2, 4)
plt.imshow(shear)
plt.axis("off")
plt.title("Shear")

plt.show()