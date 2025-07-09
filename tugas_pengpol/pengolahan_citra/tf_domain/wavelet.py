import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image and convert it to grayscale
image_path = "../kilat_8ok.jpg"
img = Image.open(image_path).convert('L')  # Convert to grayscale
img_array = np.array(img)


# Pad image to ensure even dimensions for rows and columns
def pad_image(image):
    rows, cols = image.shape
    if rows % 2 != 0:
        image = np.pad(image, ((0, 1), (0, 0)), mode='constant')
    if cols % 2 != 0:
        image = np.pad(image, ((0, 0), (0, 1)), mode='constant')
    return image


# Apply padding to the image
padded_img = pad_image(img_array)


def dwt_1d(signal):
    """
    Perform a 1D Haar wavelet transform on the input signal.
    """
    length = len(signal)
    approx = (signal[0:length:2] + signal[1:length:2]) / 2  # Averages
    detail = (signal[0:length:2] - signal[1:length:2]) / 2  # Differences
    return approx, detail


def dwt_2d(image):
    """
    Perform a 2D Haar wavelet transform on a 2D numpy array (image).
    """
    # Step 1: Apply 1D DWT on rows
    rows, cols = image.shape
    transformed_rows = np.zeros_like(image, dtype=float)

    for row in range(rows):
        approx, detail = dwt_1d(image[row, :])
        transformed_rows[row, 0:len(approx)] = approx
        transformed_rows[row, len(approx):len(approx) + len(detail)] = detail

    # Step 2: Apply 1D DWT on columns
    transformed_image = np.zeros_like(image, dtype=float)
    for col in range(cols):
        approx, detail = dwt_1d(transformed_rows[:, col])
        transformed_image[0:len(approx), col] = approx
        transformed_image[len(approx):len(approx) + len(detail), col] = detail

    # Extract cA (approx), cH (horizontal), cV (vertical), and cD (diagonal)
    cA = transformed_image[:rows // 2, :cols // 2]
    cH = transformed_image[:rows // 2, cols // 2:]
    cV = transformed_image[rows // 2:, :cols // 2]
    cD = transformed_image[rows // 2:, cols // 2:]

    return cA, cH, cV, cD


# Perform the 2D DWT on the padded image manually
cA, cH, cV, cD = dwt_2d(padded_img)

# Plot the results
plt.figure(figsize=(12, 8))

# Approximation (cA)
plt.subplot(2, 2, 1)
plt.imshow(cA, cmap='gray')
plt.title('Approximation Coefficients (cA)')
plt.axis('off')

# Horizontal Detail (cH)
plt.subplot(2, 2, 2)
plt.imshow(cH, cmap='gray')
plt.title('Horizontal Coefficients (cH)')
plt.axis('off')

# Vertical Detail (cV)
plt.subplot(2, 2, 3)
plt.imshow(cV, cmap='gray')
plt.title('Vertical Coefficients (cV)')
plt.axis('off')

# Diagonal Detail (cD)
plt.subplot(2, 2, 4)
plt.imshow(cD, cmap='gray')
plt.title('Diagonal Coefficients (cD)')
plt.axis('off')

plt.tight_layout()
plt.show()
