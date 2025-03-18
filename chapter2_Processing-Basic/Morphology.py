import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('JohnHancocksSignature.png', cv.IMREAD_UNCHANGED)

# Otsu
_, b_image = cv.threshold(image[:, :, 3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

binary = b_image[b_image.shape[0] // 2:b_image.shape[0], 0:b_image.shape[0] // 2 + 1]

se = np.uint8([[0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0]])

# 팽창
Dilation = cv.dilate(binary, se, iterations=1)
# 침식
Erosion = cv.erode(binary, se, iterations=1)
# 닫기
Close = cv.morphologyEx(binary, cv.MORPH_CLOSE, se)
# 열기
Open = cv.morphologyEx(binary, cv.MORPH_OPEN, se)

result = np.hstack((binary, Dilation, Erosion, Close, Open))

cv.imshow('result', result)
cv.waitKey(0)
cv.destroyAllWindows()
