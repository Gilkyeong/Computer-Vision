import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('dabo.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, 100, 200)

lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=90, minLineLength=40, maxLineGap=5)

img_lines = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_lines_rgb = cv.cvtColor(img_lines, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_lines_rgb)
plt.title('Hough result')
plt.axis('off')

plt.tight_layout()
plt.show()
