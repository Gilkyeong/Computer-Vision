import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('edgeDetectionImage.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

edge_magnitude = cv.magnitude(grad_x, grad_y)

edge_strength = cv.convertScaleAbs(edge_magnitude)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edge_strength, cmap='gray')
plt.title('Edge Strength')
plt.axis('off')

plt.tight_layout()
plt.show()
