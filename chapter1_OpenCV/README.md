# 문제 1 이미지 불러오기 및 그레이스케일 변환
### 설명
- Opencv를 사용하여 RGB 이미지를 불러온 후 그레이스케일로 변환
- 원본 이미지와 그레이스케일로 변환된 이미지를 가로로 나란히 붙여서 화면에 출력

### 코드 
- Grayscale.py

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


**이미지 불러오기**
```python
img = cv.imread('soccer.jpg') 

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
```
기본적으로 BGR 형식으로 저장

이미지 파일의 경로를 확인하여 불러옴


**grayscale 이미지 변환**
```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite('soccer_gray.jpg', gray)  
```
cv.cvtColor() 함수는 이미지 색상 공간을 변환

cv.COLOR_BGR2GRAY를 사용하여 BGR 이미지를 grayscale로 변환


**grayscale 이미지를 3채널로 변환**
```python
gray_3ch = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
```
cv.COLOR_GRAY2BGR를 사용하여 흑백 이미지를 BGR 3채널 형식으로 변환


**원본 이미지와 변환된 이미지 나란히 붙이기**
```python
imgs = np.hstack((img, gray_3ch))
```
np.hstack() 이미지를 가로로 붙이는 함수


### 실행 결과

![image](https://github.com/user-attachments/assets/233b22d6-aff2-490e-abff-1f231ca3de13)


## 문제 2 웹캠 영상에서 에지 검출
설명
- 웹캠을 사용하여 실시간 비디오 스트림을 가져옴
- 각 프레임에서 Canny Edge Detection을 적용하여 에지를 검출하여 원본 영상과 함께 출력

코드
- Canny_video.py

실행 결과

![image](https://github.com/user-attachments/assets/c3322dd8-424c-4fc1-8d30-c4d293a28795)


## 문제 3 마우스로 영역 선택 및 ROI 추출
설명
- 이미지를 불러오고 사용자가 마우스로 클릭하고 드래그하여 ROI 선택
- 선택한 영역만 따로 저장하거나 표시

코드
- ROI_print.py

실행결과

![image](https://github.com/user-attachments/assets/235df943-48de-49b3-8a72-ee39967e0764)

