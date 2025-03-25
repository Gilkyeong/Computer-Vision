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
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
```
🔹 이진화 처리를 위해서 BGR 이미지를 Grayscale 이미지로 변환
<br><br>
**🔷 이진화 처리**
```python
threshold = 127
_, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
```
🔹 임곗값 127을 기준으로 pixel 값을 이진화 <br>
🔹 cv.threshold(input_image, threshold, max, cv.THRESH_BINARY)
<br><br>
**🔷 히스토그램 계산**
```python
hist1 = cv.calcHist([binary], [0], None, [256], [0, 256])
hist2 = cv.calcHist([gray], [0], None, [256], [0, 256])
```
🔹 cv.calcHist() 함수로 이진화된 이미지와 grayscale 이미지의 히스토그램 계산 <br>
🔹[binary]: 입력 이미지 <br>
🔹[0]: 첫 번째 채널(Grayscale) <br>
🔹None: 마스크 사용 안 함 <br>
🔹[256]: 히스토그램의 빈(bin) 개수 <br>
🔹[0, 256]: 픽셀 값의 범위 (0~255)
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
**🔷 Otsu 알고리즘을 사용하여 이진화**
```python
_, b_image = cv.threshold(image[:, :, 3], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
```
🔹 image[:, :, 3]: **알파 채널(투명도)**을 기준으로 이진화 <br>
🔹 cv.threshold()로 Otsu 알고리즘을 사용해 자동으로 최적의 임계값 설정하여 이진화
<br><br>
**🔷 이미지 일부 선택**
```python
binary = b_image[b_image.shape[0] // 2:b_image.shape[0], 0:b_image.shape[0] // 2 + 1]
```
🔹 이미지 하단 왼쪽 부분 선택 <br>
<br><br>
**🔷 구조요소 정의**
```python
se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
```
🔹 형태학적 연산을 수행할 때 사용할 5x5 kernel 정의
<br><br>
**🔷 Dilation**
```python
Dilation = cv.dilate(binary, se, iterations=1)
```
🔹 밝은 영역(흰색) 확대 → 노이즈 제거, 선 굵게 만듦 <br>
<br><br>
**🔷 Erosion**
```python
Erosion = cv.erode(binary, se, iterations=1)
```
🔹 어두운 영역(검은색) 확대 → 얇은 선 더 얇게 만듦 <br>
<br><br>
**🔷 Close**
```python
Close = cv.morphologyEx(binary, cv.MORPH_CLOSE, se)
```
🔹 팽창 후 침식 수행 <br>
<br><br>
**🔷 Open**
```python
Open = cv.morphologyEx(binary, cv.MORPH_OPEN, se)
```
🔹 침식 후 팽창 수행 <br>
<br><br>
### :octocat: 실행 결과

![Figure 2025-03-25 154707](https://github.com/user-attachments/assets/ccb52f88-6890-44e7-bf53-1231e55932af)
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=90, minLineLength=40, maxLineGap=10) <br>
![Figure 2025-03-25 154813](https://github.com/user-attachments/assets/a76c3340-c82b-4a33-9736-e0080f56f013)
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=90, minLineLength=40, maxLineGap=5) <br>
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
**🔷 회전 변환 행렬 생성**
```python
angle = 45
scale = 1.5
M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
```
🔹 cv.getRotationMatrix2D()를 사용하여 중심점 ((cols / 2, rows / 2)): 이미지의 중심을 기준으로 이미지를 45도 회전 <br>
🔹 변환 행렬 M은 2×3 행렬
<br><br>
**🔷 이미지 확대 계산**
```python
new_cols, new_rows = int(cols * 1.5), int(rows * 1.5)
```
🔹 회전 후 이미지 크기를 원본 크기의 1.5배로 설정 <br>
<br><br>
**🔷 최종 이미지 변환**
```python
rotated_scaled_img = cv.warpAffine(img, M, (new_cols, new_rows), flags=cv.INTER_LINEAR)
```
🔹 cv.warpAffine() 함수로 이미지 회전과 확대 변환 수행 flags=cv.INTER_LINEAR <br>
🔹 flags=cv.INTER_LINEAR로 Interpolation 설정 가능 <br>
<br><br>
### :octocat: 실행 결과

![Figure 2025-03-25 155538](https://github.com/user-attachments/assets/1307ca06-143f-45df-89a3-d3867e98a63c)

