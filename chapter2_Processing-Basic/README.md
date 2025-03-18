## 🌀 문제 1 이진화 및 히스토그램 구하기

> OpenCV를 활용하여 컬러 이미지를 불러온 후 그레이스케일 변환
> **특정 임계값을 설정하여 이진화하고 이진화된 이미지의 히스토그램을 계산 후 시각화**
---
**이진화** <br><br>
![image](https://github.com/user-attachments/assets/62171667-8b89-428b-9408-4cd34b4ef341)


### 📄 코드 
- Histogram.py

*전체 코드*
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('mistyroad.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

threshold = 127
_, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

hist = cv.calcHist([binary], [0], None, [256], [0, 256])

plt.figure(figsize=(6, 4))
plt.plot(hist, color='black')
plt.title('Histogram')
plt.xlabel('Pixel')
plt.ylabel('Frequency')
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
hist = cv.calcHist([binary], [0], None, [256], [0, 256])
```
🔹 cv.calcHist() 함수로 이진화된 이미지의 히스토그램 계산 <br>
🔹[binary]: 입력 이미지 <br>
🔹[0]: 첫 번째 채널(Grayscale) <br>
🔹None: 마스크 사용 안 함 <br>
🔹[256]: 히스토그램의 빈(bin) 개수 <br>
🔹[0, 256]: 픽셀 값의 범위 (0~255)
<br><br>

### :octocat: 실행 결과

![Figure 2025-03-18 153753](https://github.com/user-attachments/assets/2bb7ba9c-b6b3-48f7-a414-c5c66cc6b4c1)
<br><br>

## 🌀 문제 2 모폴로지 연산 적용하기

> 주어진 이미지를 가지고 **Dilation, Erosion, Open, Close 모폴로지 연산을 적용**
---

### Canny edge
- gradient 크기를 구하여 임계값 설정 후 edge 검출하면 윤곽선이 두껍게 표현되는 문제점을 해결
- 비최대 억제를 사용하여 edge 검출 후 edge의 굵기를 얇게 유지 <br>
![image](https://github.com/user-attachments/assets/895d246b-c5c5-43ae-b548-48df049b97a4)

### 📄 코드 
- Morphology.py

*전체 코드*
```python
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
se = np.uint8([[0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0]])
```
🔹 형태학적 연산을 수행할 때 사용할 kernel 정의 <br>
🔹 다이아몬드 형태의 kernel을 사용하여 연산 수행
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

![image](https://github.com/user-attachments/assets/2803c11a-86b9-4e59-aa95-f7e252c12312)
<br><br>

## 🌀 문제 3 기하 연산 및 선형 보간 적용하기  

> 주어진 이미지에 **45도 회전, 회전된 이미지를 1.5배 확대하는 기하 연산 수행**
> 회전 및 확대된 이미지에 **Bilinear Interpolation을 적용**
---

### 📄 코드 
- Bilinear_Interpolation.py

*전체 코드*
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("rose.png")

rows, cols = img.shape[:2]

angle = 45
scale = 1.5
M = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)

new_cols, new_rows = int(cols * 1.5), int(rows * 1.5)

rotated_scaled_img = cv.warpAffine(img, M, (new_cols, new_rows), flags=cv.INTER_LINEAR)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")   
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(rotated_scaled_img, cv.COLOR_BGR2RGB))
plt.title("Result")
plt.axis("off")
    
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

![Figure 2025-03-18 160911](https://github.com/user-attachments/assets/74847305-37e9-4eb8-95db-578bfee899ac)



