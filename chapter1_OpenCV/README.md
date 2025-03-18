## 🌀 문제 1 이미지 불러오기 및 그레이스케일 변환

> OpenCV를 활용하여 컬러 이미지를 불러온 후 **그레이스케일 변환 및 출력**
---
**BGR --> Grayscale 변환** <br><br>
![image](https://github.com/user-attachments/assets/1c77deb3-3973-40a1-ba56-1719d4ea5eb1)


### 📄 코드 
- Grayscale.py

*전체 코드*
```python
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
```
*핵심 코드* <br><br>
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

![image](https://github.com/user-attachments/assets/233b22d6-aff2-490e-abff-1f231ca3de13)
<br><br>

## 🌀 문제 2 웹캠 영상에서 에지 검출

> 웹캠을 사용하여 실시간 비디오 스트림을 가져와 **각 프레임에서 Canny Edge Detection을 적용하여 에지를 검출하여 원본 영상과 함께 출력**
---

### Canny edge
- gradient 크기를 구하여 임계값 설정 후 edge 검출하면 윤곽선이 두껍게 표현되는 문제점을 해결
- 비최대 억제를 사용하여 edge 검출 후 edge의 굵기를 얇게 유지 <br>
![image](https://github.com/user-attachments/assets/895d246b-c5c5-43ae-b548-48df049b97a4)

### 📄 코드 
- Canny_video.py

*전체 코드*
```python
import cv2 as cv
import sys
import numpy as np

cap = cv.VideoCapture(0, cv.CAP_DSHOW)  

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    ret, frame = cap.read() 

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    edges = cv.Canny(gray, 100, 200)

    canny_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    reslut = np.hstack((frame, canny_edges))

    cv.imshow('CANNY VIDEO', reslut)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```

*핵심 코드* <br><br>
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

![image](https://github.com/user-attachments/assets/c3322dd8-424c-4fc1-8d30-c4d293a28795)
<br><br>

## 🌀 문제 3 마우스로 영역 선택 및 ROI 추출

> 이미지를 불러와 사용자가 마우스를 제어하여 **ROI 선택 후 선택한 영역만 따로 저장하거나 표시**
---

### 📄 코드 
- ROI_print.py

*전체 코드*
```python
import cv2 as cv
import sys
import numpy as np

img = cv.imread('soccer.jpg') 

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

clone = img.copy()  
roi = None
start_x, start_y, end_x, end_y = -1, -1, -1, -1 
drawing = False

def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, drawing, roi, img

    if event == cv.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        drawing = True  # 드래그 시작

    elif event == cv.EVENT_MOUSEMOVE: 
        if drawing: 
            temp_img = img.copy()
            cv.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv.imshow('Image', temp_img)

    elif event == cv.EVENT_LBUTTONUP:  #마우스 버튼을 놓았을 때
        end_x, end_y = x, y
        drawing = False  #드래그 종료

        x1, y1, x2, y2 = min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y)

        # ROI 추출
        roi = clone[y1:y2, x1:x2].copy()
        
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.imshow('Image', img)

        if roi.size > 0:
            cv.imshow('ROI_image', roi)

cv.namedWindow('Image')
cv.setMouseCallback('Image', draw_rectangle)
cv.imshow('Image', img)

while True:
    key = cv.waitKey(1)

    if key == ord('r'):
        img = clone.copy()
        roi = None
        cv.imshow('Image', img)
        cv.destroyWindow('ROI_image') 

    elif key == ord('s') and roi is not None:
        cv.imwrite('ROI_image.jpg', roi)
        print('이미지가 저장되었습니다.')

    elif key == ord('q'):  # 종료
        break

cv.destroyAllWindows()
```

*핵심 코드* <br><br>
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

![image](https://github.com/user-attachments/assets/235df943-48de-49b3-8a72-ee39967e0764)

