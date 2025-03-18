import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('JohnHancocksSignature.png', cv.IMREAD_GRAYSCALE)  

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

Dilation = cv.morphologyEx(image, cv.MORPH_DILATE, kernel)
Erosion = cv.morphologyEx(image, cv.MORPH_ERODE, kernel)
Open = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
Close = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

result = np.hstack((image, Dilation, Erosion, Open, Close))
cv.imshow('Binary | Dilation | Erosion | Opening | Closing', result)
cv.imwrite('morphology_result.png', result)
cv.waitKey(0)
cv.destroyAllWindows()