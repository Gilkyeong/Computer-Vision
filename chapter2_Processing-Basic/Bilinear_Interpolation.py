import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("rose.png")

rows, cols = img.shape[:2]

angle = 45
scale = 1.5
M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)

new_cols, new_rows = int(cols * 1.5), int(rows * 1.5)

rotated_scaled_img = cv.warpAffine(img, M, (new_cols, new_rows), flags=cv.INTER_LINEAR)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")   
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(rotated_scaled_img, cv.COLOR_BGR2RGB))
plt.title("Result")
plt.axis("off")
    
plt.show()
