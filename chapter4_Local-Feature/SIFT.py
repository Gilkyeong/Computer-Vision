import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('mot_color70.jpg')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create(nfeatures=0)  

kp, des = sift.detectAndCompute(gray, None)

img_kp = cv.drawKeypoints(img_rgb, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_kp)
plt.title('SIFT result')
plt.axis('off')

plt.tight_layout()
plt.show()
