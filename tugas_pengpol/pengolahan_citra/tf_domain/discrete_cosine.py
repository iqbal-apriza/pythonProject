import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# Load the image and convert it to grayscale
image_path = "../kilat_8ok.jpg"
img = Image.open(image_path).convert('L')  # Convert to grayscale
img_array = np.array(img)

# Function to perform a 2D DCT
def dct_2d(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

# Function to perform an inverse 2D DCT (to reconstruct the image)
def idct_2d(dct_coefficients):
    return idct(idct(dct_coefficients.T, norm='ortho').T, norm='ortho')

# Apply 2D DCT
dct_transformed = dct_2d(img_array)

# Plot the original image and its DCT
plt.figure(figsize=(12, 6))

# DCT Transformed Image (Log Scale)
plt.subplot(1, 2, 2)
plt.imshow(np.log(np.abs(dct_transformed) + 1), cmap='gray')  # Log scale for better visualization
plt.title('DCT of the Image (Log Scale)')
plt.axis('off')

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.tight_layout()
plt.show()
