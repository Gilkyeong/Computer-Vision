import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

img1 = cv.imread('mot_color70.jpg')
img1 = img1[190:350, 440:560]  

img2 = cv.imread('mot_color83.jpg')

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print('Feature :', len(kp1), len(kp2))

start = time.time()
flann = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match = flann.knnMatch(des1, des2, 2)

T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)
print('Time :', time.time() - start)

img_match = cv.drawMatches(img1, kp1, img2, kp2, good_match, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img_match_rgb = cv.cvtColor(img_match, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.imshow(img_match_rgb)
plt.title('FLANN result')
plt.axis('off')
plt.show()
