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
🔹 BGR 이미지를 Grayscale 이미지로 변환
<br><br>
**🔷 Sobel edge 검출**
```python
grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
```
🔹 cv.Sobel() 함수로 수평(x), 수직(y) 방향의 미환
<br><br>

### :octocat: 실행 결과

![Figure 2025-04-01 152956](https://github.com/user-attachments/assets/a77544fc-b252-4442-af9c-71e016998634)
<br><br>
## 🌀 문제 2 SIFT를 이용한 두 영상 간 특징점 매칭칭

> Canny 에지 검출을 사용하여 에지 맵 생성 후 **허프 변환을 사용하여 이미지에서 직선 검출**
---

### Hough Transform (허프 변환)
![image](https://github.com/user-attachments/assets/acbb191e-a6f7-450b-b035-318c6cd15582)

### 📄 코드 
- Hough.py

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
**🔷 Grayscale 변환 후 Canny edge detection**
```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 100, 200)
```
🔹 edge 검출을 위해 이미지 Grayscale 변환 <br>
🔹 cv.Canny() 함수로 edge map 생성
<br><br>
**🔷 Hough Transform을 이용해 직선 검출**
```python
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=90, minLineLength=40, maxLineGap=5)
```
🔹 cv.HoughLinesP()을 이용해 직선 검출 <br>
🔹 rho=1 : 거리 resolution (pixel 단위) <br>
🔹 theta=np.pi/180 : 각도 resolution (1도) <br>
🔹 threshold=90 : 임곗값 <br>
🔹 minLineLength=40 : 최소 직선 길이 <br>
🔹 maxLineGap=5 : 선분 간 최대 허용 거리 (연결 조건)
<br><br>
**🔷 Hough Transform 시각화**
```python
img_lines = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2))
```
🔹 원본 이미지에 선을 직접 그리지 않기 위해 복사본 생성 <br>
🔹 검출된 모든 직선을 빨간색 직선으로 시각화  <br>
<br><br>
**🔷 matplotlib 사용을 위한 RGB image 변환**
```python
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_lines_rgb = cv.cvtColor(img_lines, cv.COLOR_BGR2RGB)
```
🔹 OpenCV는 BGR image이고 matplotlib는 RGB image이기 때문에 변환 <br>
<br><br>
### :octocat: 실행 결과

![Figure 2025-04-01 165852](https://github.com/user-attachments/assets/31428164-9548-42ba-be5f-338a7b7044e4)
<br><br>

## 🌀 문제 3 호모그래피를 이용한 이미지 정합합

> 사용자가 지정한 영역을 바탕으로 **GrabCut 알고리즘을 사용하여 객체 추출**
> 객체 추출 결과를 마스크 형태로 시각화 후 원본 이미지에서 배경을 제거하고 객체만 남은 이미지를 출력
---

### 📄 코드 
- Grabcut.py

*전체 코드*
```python
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
```

*핵심 코드* <br>
**🔷 변수 초기화**
```python
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
```
🔹 Grapcut에서 사용할 마스크와 백/포어그라운드 초기화 <br>
🔹 pixel 분포를 학습하기 위함
<br><br>
**🔷 시각화하기 위한 영역 지정**
```python
rect = (300, 300, 600, 500)
```
🔹 (300, 300, 600, 500) 범위의 사각형 <br>
<br><br>
**🔷 GrapCut 수행**
```python
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
```
🔹 반복 횟수를 5번으로 설정하여 더 정교하게 분리함
<br><br>
**🔷 Mask 적용**
```python
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
img_result = img * mask2[:, :, np.newaxis]
```
🔹 Mask를 적용하여 배경이 제거된 이미지 생성 <br>
<br><br>
**🔷 matplotlib 사용을 위한 RGB image 변환**
```python
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_result_rgb = cv.cvtColor(img_result, cv.COLOR_BGR2RGB)
```
🔹 OpenCV는 BGR image이고 matplotlib는 RGB image이기 때문에 변환 <br>
<br><br>
### :octocat: 실행 결과

![스크린샷 2025-04-01 173214](https://github.com/user-attachments/assets/601d4fa9-79e7-430a-acdb-fd6a9ad6bb6b)
