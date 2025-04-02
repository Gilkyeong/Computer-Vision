## 🌀 문제 1 SIFT를 이용한 특징점 검출 및 시각화

> 이미지를 이용하여 **SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용하여 특징점을 검출**하고 시각화
---
**SIFT** <br><br>
![image](https://github.com/user-attachments/assets/9d856265-b7b2-4aa9-860c-2c1d87a9488c)

### 📄 코드 
- SIFT.py

*전체 코드*
```python
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
```
*핵심코드* <br>
**🔷 grayscale 이미지 변환**
```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```
🔹 SIFT는 Grayscale 이미지에서 특징점을 추출하므로 Grayscale 변환
<br><br>
**🔷 SIFT 객체 생성**
```python
sift = cv.SIFT_create(nfeatures=0)
```
🔹 nfeatures=0 : 추출할 특징점 수에 제한을 두지 않음
<br><br>
**🔷 특징점, 기술자 계산**
```python
kp, des = sift.detectAndCompute(gray, None)
```
🔹 detectAndCompute() : 특징점(keypoints)과 descriptors 동시에 계산
<br><br>
**🔷 특징점 시각화**
```python
img_kp = cv.drawKeypoints(img_rgb, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```
🔹 특징점을 이미지에 시각화 <br>
🔹 DRAW_RICH_KEYPOINTS : 크기와 방향 정보를 포함해 원으로 표시
<br><br>
### :octocat: 실행 결과

![Figure 2025-04-01 152956](https://github.com/user-attachments/assets/a77544fc-b252-4442-af9c-71e016998634)
<br> SIFT는 특징점을 검출할 때 단순히 x,y만 찾지 않고 특징점 scale과 방향도 함께 예측
<br> --> Scale의 크기에 따라 원의 크기가 결정됨 (큰 스케일에서는 큰 원, 작은 스케일에서는 작은 원)
<br><br>
## 🌀 문제 2 SIFT를 이용한 두 영상 간 특징점 매칭

> 주어진 이미지를 **SIFT 특징점 기반으로 매칭 수행**
---

### 📄 코드 
- SIFT_match.py

*전체 코드*
```python
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
```

*핵심 코드* <br>
**🔷 ROI를 잘라내고 비교 대상을 불러옴**
```python
img1 = cv.imread('mot_color70.jpg')
img1 = img1[190:350, 440:560]

img2 = cv.imread('mot_color83.jpg')
```
🔹 첫번째 이미지에서 특정 ROI를 잘라냄 <br>
🔹 비교 대상이 되는 두 번째 이미지를 불러옴
<br><br>
**🔷 grayscale 이미지 변환**
```python
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
```
🔹 SIFT 연산을 위한 두 이미지 grayscale 변환
<br><br>
**🔷 SIFT 수행**
```python
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
```
🔹 SIFT를 통해 특징점과 descriptors 추출
<br><br>
**🔷 매칭 연산**
```python
flann = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_match = flann.knnMatch(des1, des2, 2)
```
🔹 FLANN 기반 매칭 객체 생성 후 KNN 매칭 수행 (k=2) <br>
🔹 각 특징점에 대해 가장 가까운 2개의 후보를 찾음
<br><br>
**🔷 임계값 설정**
```python
T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)
```
🔹 두 후보의 거리 비율이 임계값보다 작을 때만 좋은 매칭으로 판단
<br><br>
### :octocat: 실행 결과

![Figure 2025-04-01 165852](https://github.com/user-attachments/assets/31428164-9548-42ba-be5f-338a7b7044e4)
<br><br>

## 🌀 문제 3 호모그래피를 이용한 이미지 정합

> SIFT 특징점을 사용하여 두 이미지 간 대응점을 찾고 **호모그래피를 계산하여 하나의 이미지 위에 정렬**
---

### 📄 코드 
- SIFT_Homography.py

*전체 코드*
```python
import cv2 as cv
import numpy as np

img1 = cv.imread('img2.jpg')
img2 = cv.imread('img3.jpg')   


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

img_matches = cv.drawMatches(img2, kp1, warped_img, kp2, good_matches, None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow("matching result", img_matches)

cv.waitKey(0)
cv.destroyAllWindows()
```

*핵심 코드* <br>
**🔷 Grayscale 이미지 변환**
```python
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
```
🔹 SIFT 연산을 위해 Grayscale 이미지 변환
<br><br>
**🔷 SIFT 수행**
```python
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
```
🔹 두 이미지에서 특징점과 desscriptor 추출 <br>
<br><br>
**🔷 KNN matching 수행**
```python
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)
```
🔹 BFMatcher(Brute-Force Matcher)를 사용하여 destriptors 간 KNN 매칭(k=2)을 수행
<br><br>
**🔷 Matching 필터링**
```python
good_matches = []
ratio_thresh = 0.7
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
```
🔹 잘못된 매칭을 제거하여 신뢰성을 높임 <br>
<br><br>
**🔷 실제 좌표 추출**
```python
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
```
🔹 매칭점에서 실제 좌표를 추출하여 호모그래피 계산을 위한 데이터로 변환 <br>
<br><br>
**🔷 호모그래피 추정**
```python
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
```
🔹 RANSAC 기반으로 호모그래피 행렬을 추정 <br>
<br><br>
**🔷 이미지 정렬 후 시각적 표시**
```python
warped_img = cv.warpPerspective(img1, H, (w2, h2))

img_matches = cv.drawMatches(img2, kp1, warped_img, kp2, good_matches, None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```
🔹 추정된 호모그래피를 이용하여 img1, img2에 정렬되도록 변형 <br>
🔹 매칭 결과를 시각화
<br><br>
### :octocat: 실행 결과

![Uploading image.png…]()

