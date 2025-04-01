import cv2 as cv
import numpy as np

img1 = cv.imread('img1.jpg')
img2 = cv.imread('img2.jpg')   


gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
ratio_thresh = 0.7
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

print("matching num :", len(good_matches))

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)

H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)

h2, w2 = img2.shape[:2]
warped_img = cv.warpPerspective(img1, H, (w2, h2))

cv.imshow("original", img2)
cv.imshow("Warped images", warped_img)

img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow("matching result", img_matches)

cv.waitKey(0)
cv.destroyAllWindows()
