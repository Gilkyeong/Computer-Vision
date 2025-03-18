import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('mistyroad.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

threshold = 127
_, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

hist = cv.calcHist([binary], [0], None, [256], [0, 256])

plt.figure(figsize=(6, 4))
plt.plot(hist, color='black')
plt.title('Histogram')
plt.xlabel('Pixel')
plt.ylabel('Frequency')
plt.show()
