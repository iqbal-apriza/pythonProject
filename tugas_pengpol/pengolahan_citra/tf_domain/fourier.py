from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image and convert it to grayscale for simplicity
image_path = "../kilat_8ok.jpg"
img = Image.open(image_path).convert('L')  # Convert to grayscale

# Convert the image to a NumPy array
img_array = np.array(img)

# Apply the 2D Fourier Transform
f_transform = np.fft.fft2(img_array)
f_shift = np.fft.fftshift(f_transform)  # Shift the zero-frequency component to the center

# Compute the magnitude spectrum (log scale for better visualization)
magnitude_spectrum = np.log(np.abs(f_shift) + 1)  # Adding 1 to avoid log(0)

# Plot the original image and its Fourier Transform magnitude spectrum
plt.figure(figsize=(12, 6))

# Magnitude Spectrum
plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Fourier Transform (Magnitude Spectrum)')
plt.axis('off')

# Original image
plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.tight_layout()
plt.show()
