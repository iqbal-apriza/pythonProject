import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import hadamard

# Baca citra
img = cv2.imread('../kilat_8ok.jpg')

img_array = np.array(img)

plt.figure(figsize=(12, 6))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_resized = cv2.resize(img_gray, (128, 128))
H = hadamard(128)
wht_img = np.dot(np.dot(H, img_resized), H)

# Walsh Hadamard Image
plt.subplot(1, 2, 2)
plt.imshow(wht_img, cmap='gray')
plt.title('Walsh Hadamard Image')
plt.axis('off')

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.show()