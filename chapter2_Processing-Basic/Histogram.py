import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('mistyroad.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

threshold = 127
_, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

hist1 = cv.calcHist([binary], [0], None, [256], [0, 256])
hist2 = cv.calcHist([gray], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(hist1)
plt.title("Binary")   
plt.subplot(1, 2, 2)
plt.plot(hist2)
plt.title("Grayscale")
plt.show()
