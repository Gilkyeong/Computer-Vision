import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('coffee cup.jpg')

mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (300, 300, 600, 500)  

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

img_result = img * mask2[:, :, np.newaxis]

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_result_rgb = cv.cvtColor(img_result, cv.COLOR_BGR2RGB)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask2 * 255, cmap='gray')
plt.title("Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_result_rgb)
plt.title("Background remove")
plt.axis("off")

plt.tight_layout()
plt.show()

