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
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite('soccer_gray.jpg', gray)  
```
🔹 cv.cvtColor() 함수는 이미지 색상 공간을 변환 <br>
🔹 cv.COLOR_BGR2GRAY를 사용하여 BGR 이미지를 grayscale로 변환
<br><br>
**🔷 grayscale 이미지를 3채널로 변환**
```python
gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
```
🔹 cv.COLOR_GRAY2BGR를 사용하여 흑백 이미지를 BGR 3채널 형식으로 변환
<br><br>
**🔷 원본 이미지와 변환된 이미지 나란히 붙이기**
```python
imgs = np.hstack((img, gray_3ch))
```
🔹 np.hstack() 이미지를 가로로 붙이는 함수
<br><br>

### :octocat: 실행 결과

![Figure 2025-03-18 153753](https://github.com/user-attachments/assets/2bb7ba9c-b6b3-48f7-a414-c5c66cc6b4c1)
<br><br>

## 🌀 문제 2 웹캠 영상에서 에지 검출

> 웹캠을 사용하여 실시간 비디오 스트림을 가져와 **각 프레임에서 Canny Edge Detection을 적용하여 에지를 검출하여 원본 영상과 함께 출력**
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
**🔷 grayscale 이미지 변환**
```python
 gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
```
🔹 cv.cvtColor() 함수에서 cv.COLOR_BGR2GRAY를 사용하여 BGR 이미지를 grayscale로 변환 <br>
🔹 Canny edge 검출은 일반적으로 흑백 이미지에서 수행
<br><br>
**🔷 Canny edge 검출**
```python
edges = cv.Canny(gray, 100, 200)
```
🔹 cv.Canny(image, threshold1, threshold2) : Canny 알고리즘 <br>
🔹 threshold1,2 : edge를 검출할 강도 값 설정
<br><br>
**🔷 Canny edge 이미지를 3차원으로 변환 후 원본 이미지와 가로로 연결**
```python
canny_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
reslut = np.hstack((frame, canny_edges))
```
🔹 원본 이미지와 나란히 출력하기 위해 Canny edge 이미지를 3차원 변환 <br>
🔹 np.hstack()으로 두 이미지를 가로로 이어 붙여 출력

### :octocat: 실행 결과

![image](https://github.com/user-attachments/assets/2803c11a-86b9-4e59-aa95-f7e252c12312)
<br><br>

## 🌀 문제 3 마우스로 영역 선택 및 ROI 추출

> 이미지를 불러와 사용자가 마우스를 제어하여 **ROI 선택 후 선택한 영역만 따로 저장하거나 표시**
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
**🔷 변수 초기화**
```python
clone = img.copy()  
roi = None
start_x, start_y, end_x, end_y = -1, -1, -1, -1 
drawing = False
```
🔹 clone : 원본 이미지를 보존하기 위한 복사본 <br>
🔹 roi : 선택한 Region of Interest <br>
🔹 start_x,y / end_x,y : 마우스 드래그 시작 및 종료 좌표 <br>
🔹 drawing : 마우스 드래그 여부를 나타냄
<br><br>
**🔷 마우스 콜백 함수**
```python
def draw_rectangle(event, x, y, flags, param): 
```
**마우스의 이벤트를 처리** <br>
```python
if event == cv.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        drawing = True  # 드래그 시작 
```
🔹 마우스 버튼을 누를때 시작 위치 저장 후 드래그 상태를 나타냄 <br>
```python
elif event == cv.EVENT_MOUSEMOVE: 
    if drawing: 
        temp_img = img.copy()
        cv.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv.imshow('Image', temp_img)
```
🔹 실시간으로 드래그하는 영역을 표시하기 위해 화면을 갱신 <br>
🔹 cv.rectangle() : 드래그 영역을 초록색 사각형으로 표시 <br>
```python
elif event == cv.EVENT_LBUTTONUP:
    end_x, end_y = x, y
    drawing = False
```
🔹 드래그 종료 
<br><br>
**ROI 추출**
```python
roi = clone[y1:y2, x1:x2].copy()
        
cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv.imshow('Image', img)

if roi.size > 0:
    cv.imshow('ROI_image', roi)
```
🔹 드래그된 영역의 좌표를 결정한 후 선택된 영역을 roi에 저장 <br>
🔹 선택된 ROI를 화면에 표시
<br><br>

### :octocat: 실행 결과

![Figure 2025-03-18 160911](https://github.com/user-attachments/assets/74847305-37e9-4eb8-95db-578bfee899ac)



