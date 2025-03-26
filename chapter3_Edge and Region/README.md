## 🌀 문제 1 소벨 에지 검출 및 결과 시각화

> 이미지 그레이 스케일로 변환 후 **Sobel 필터를 사용하여 x축과 y축 방향의 에지 검출**
> 검출된 에지 강도(edge strength) 이미지를 시각화
---
**Sobel 연산자** <br><br>
![image](https://github.com/user-attachments/assets/19b808a7-5719-43a4-98db-cc0e3bb563ac)


### 📄 코드 
- Sobel_edge.py

*전체 코드*
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('edgeDetectionImage.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grad_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

edge_magnitude = cv.magnitude(grad_x, grad_y)
edge_strength = cv.convertScaleAbs(edge_magnitude)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edge_strength, cmap='gray')
plt.title('Edge Strength')
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

![Figure 2025-03-25 153940](https://github.com/user-attachments/assets/ad7436ff-95ce-4383-9dee-fbb7bfa408f0)
<br><br>
## 🌀 문제 2 케니 에지 및 허프 변환을 이용한 직선 검출

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
import matplotlib.pyplot as plt

img = cv.imread('dabo.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, 100, 200)

lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=90, minLineLength=40, maxLineGap=5)

img_lines = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_lines_rgb = cv.cvtColor(img_lines, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_lines_rgb)
plt.title('Hough result')
plt.axis('off')

plt.tight_layout()
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

![Figure 2025-03-25 154707](https://github.com/user-attachments/assets/ccb52f88-6890-44e7-bf53-1231e55932af)
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=90, minLineLength=40, maxLineGap=10) <br><br>
![Figure 2025-03-25 154813](https://github.com/user-attachments/assets/a76c3340-c82b-4a33-9736-e0080f56f013)
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=90, minLineLength=40, maxLineGap=5)
![image](https://github.com/user-attachments/assets/1693c32a-3089-427c-bbe5-f0f314c3afe3)
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=90, minLineLength=15, maxLineGap=5)
<br><br>

## 🌀 문제 3 GrabCut을 이용한 대화식 영역 분할 및 객체 추출 

> 사용자가 지정한 영역을 바탕으로 **GrabCut 알고리즘을 사용하여 객체 추출**
> 객체 추출 결과를 마스크 형태로 시각화 후 원본 이미지에서 배경을 제거하고 객체만 남은 이미지를 출력
---

### 📄 코드 
- Grabcut.py

*전체 코드*
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('coffee cup.jpg')

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (300, 300, 600, 500)  

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
img_result = img * mask2[:, :, np.newaxis]

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_result_rgb = cv.cvtColor(img_result, cv.COLOR_BGR2RGB)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask2 * 255, cmap='gray')
plt.title("Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_result_rgb)
plt.title("Background remove")
plt.axis("off")

plt.tight_layout()
plt.show()
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

![Figure 2025-03-25 155538](https://github.com/user-attachments/assets/1307ca06-143f-45df-89a3-d3867e98a63c)

