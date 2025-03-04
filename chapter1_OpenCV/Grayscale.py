import cv2 as cv
import sys
import numpy as np

img = cv.imread('soccer.jpg') 

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imwrite('soccer_gray.jpg', gray)  

gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

imgs = np.hstack((img, gray_3ch))

cv.imshow('Color and Grayscale Image', imgs)

cv.waitKey(0)
cv.destroyAllWindows()
