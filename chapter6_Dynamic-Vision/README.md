## 🌀 문제 1 SORT 알고리즘을 활용한 다중 객체 추적기 구현
> YOLOv3을 사용하여 Detection 후 **Sort 알고리즘을 사용하여 Tracking**
> sort.py 안에 구현된 sort 알고리즘을 사용
---
**SORT 알고리즘** <br><br>


### 📄 코드 
- Sort_Tracking.py

*전체 코드*
```python
import numpy as np
import cv2 as cv
import sys

def construct_yolo_v3():
    f=open('coco_names.txt', 'r')
    class_names=[line.strip() for line in f.readlines()]

    model=cv.dnn.readNet('yolov3.weights','yolov3.cfg')
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    
    return model,out_layers,class_names

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]
    test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True)
    
    yolo_model.setInput(test_img)
    output3=yolo_model.forward(out_layers)
    
    box,conf,id=[],[],[]		# 박스, 신뢰도, 부류 번호
    for output in output3:
        for vec85 in output:
            scores=vec85[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:	# 신뢰도가 50% 이상인 경우만 취함
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
                w,h=int(vec85[2]*width),int(vec85[3]*height)
                x,y=int(centerx-w/2),int(centery-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
            
    ind=cv.dnn.NMSBoxes(box,conf,0.5,0.4)
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects

model,out_layers,class_names=construct_yolo_v3()	# YOLO 모델 생성
colors=np.random.uniform(0,255,size=(100,3))		# 100개 색으로 트랙 구분

from sort import Sort

sort=Sort()

cap=cv.VideoCapture(0,cv.CAP_DSHOW)
if not cap.isOpened(): sys.exit('카메라 연결 실패')

while True:
    ret,frame=cap.read()
    if not ret: sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')
        
    res=yolo_detect(frame,model,out_layers)   
    persons=[res[i] for i in range(len(res)) if res[i][5]==0]

    if len(persons)==0: 
        tracks=sort.update()
    else:
        tracks=sort.update(np.array(persons))
    
    for i in range(len(tracks)):
        x1,y1,x2,y2,track_id=tracks[i].astype(int)
        cv.rectangle(frame,(x1,y1),(x2,y2),colors[track_id],2)
        cv.putText(frame,str(track_id),(x1+10,y1+40),cv.FONT_HERSHEY_PLAIN,3,colors[track_id],2)            
    
    cv.imshow('Person tracking by SORT',frame)
    
    key=cv.waitKey(1) 
    if key==ord('q'): break 
    
cap.release()		# 카메라와 연결을 끊음
cv.destroyAllWindows()
```
*핵심코드* <br>
**🔷 YOLOv3, COCO datasets 로드**
```python
def construct_yolo_v3():
    f=open('coco_names.txt', 'r')
    class_names=[line.strip() for line in f.readlines()]

    model=cv.dnn.readNet('yolov3.weights','yolov3.cfg')
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    
    return model,out_layers,class_names
```
🔹 COCO dataset에 포함된 클래스를 리스트로 저장 <br>
🔹 Train된 YOLOv3 모델의 weight와 config를 불러옴
<br><br>
**🔷 Input images Preprocessing**
```python
test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True)
```
🔹 OpenCV에서 지원하는 blob 형태로 변환 <br>
🔹 pixel 정규화 및 크기 변환, 채널 순서 변경
<br><br>
**🔷 YOLOv3로 Object Detection**
```python
yolo_model.setInput(test_img)
output3=yolo_model.forward(out_layers)
```
🔹 YOLOv3으로 Detection을 수행하여 bbox 예측값 획득
<br><br>
**🔷 신뢰도 기반 객체 필터링**
```python
 if confidence>0.5:	# 신뢰도가 50% 이상인 경우만 취함
    centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
    w,h=int(vec85[2]*width),int(vec85[3]*height)
    x,y=int(centerx-w/2),int(centery-h/2)
    box.append([x,y,x+w,y+h])
    conf.append(float(confidence))
    id.append(class_id)
```
🔹 Confidence가 50% 이상인 예측값만 표시 <br>
🔹 Detection된 객체의 위치, 확률, 클래스 정보를 리스트에 저장
<br><br>
**🔷 NMS(Non-Maximum Suppression) 적용**
```python
ind = cv.dnn.NMSBoxes(box, conf, 0.5, 0.4)
objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]
```
🔹 겹치는 예측 bbox 중 확률이 가장 높은 것만 선택하여 최종 bbox 생성
<br><br>
**🔷 Persons class만 Detection**
```python
persons=[res[i] for i in range(len(res)) if res[i][5]==0]
```
🔹 class 번호가 0인 person 객체만 Detection 후 Tracking
<br><br>
**🔷 Sort 알고리즘을 사용한 Traking**
```python
if len(persons)==0:
        tracks=sort.update()
    else:
        tracks=sort.update(np.array(persons))
```
🔹 프레임별로 감지된 사람 위치를 기반으로 Kalman Filter, Hungarian Algorithm으로 ID 부여 및 Tracking
<br><br>
**🔷 결과 출력**
```python
cv.rectangle(frame, (x1, y1), (x2, y2), colors[track_id], 2)
cv.putText(frame, str(track_id), (x1+10, y1+40), cv.FONT_HERSHEY_PLAIN, 3, colors[track_id], 2)
```
🔹 고유한 색상과 ID를 사용하여 frame 위에 시각적 표시
🔹 각 Tracking된 대상에 고유한 번호를 부여
<br><br>
### :octocat: 실행 결과
![스크린샷 2025-04-15 174213](https://github.com/user-attachments/assets/f59f3bc2-14bc-4eea-ac4e-4965b1fcf6ee)
<br><br>
## 🌀 문제 2 Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

> Mediapipe의 FaceMesh모듈을 사용하여 얼굴의 468개의 랜드마크를 추출하고 실시간 영상에 시각화하는 프로그램 구현
---
**Mediapipe** <br><br>

### 📄 코드 
- CIFAR10-CNN.py

*전체 코드*
```python
import cv2 as cv
import mediapipe as mp

mp_mesh=mp.solutions.face_mesh
mp_drawing=mp.solutions.drawing_utils
mp_styles=mp.solutions.drawing_styles

mesh=mp_mesh.FaceMesh(max_num_faces=2,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)

cap=cv.VideoCapture(0,cv.CAP_DSHOW)

while True:
    ret,frame=cap.read()
    if not ret:
      print('프레임 획득에 실패하여 루프를 나갑니다.')
      break
    
    res=mesh.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))
    
    if res.multi_face_landmarks:
        for landmarks in res.multi_face_landmarks:
          mp_drawing.draw_landmarks(image=frame,landmark_list=landmarks,connections=mp_mesh.FACEMESH_CONTOURS,landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1))
        
    cv.imshow('MediaPipe Face Mesh',cv.flip(frame,1))
    if cv.waitKey(5)==ord('q'):
      break

cap.release()
cv.destroyAllWindows()
```

*핵심 코드* <br>
**🔷 데이터셋 로드**
```python
(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
```
🔹 tensorflow.keras.datasets 모듈의 cifar10 데이터셋을 사용 <br>
🔹 CIFAR-10 : 32×32 크기의 60,000개 컬러 이미지, 10개의 클래스로 분류
<br><br>
**🔷 Normalization**
```python
mesh=mp_mesh.FaceMesh(max_num_faces=2,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)
```
🔹 최대 2명의 face detection <br>
🔹 refine_landmarks=True를 사용하여 눈동자, 입술, 홍채 등 세밀한 포인트도 포함 <br>
🔹 Detection, Tracking 신뢰도를 0.5로 설정하여 50% 이상일 때만 적용
<br><br>
**🔷 프레임 획득 및 전처리**
```python
ret, frame = cap.read()
res = mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
```
🔹 OpenCV로 웹캠에서 프레임을 읽고 BGR --> RGB로 변환 후 FaceMesh로 처리
<br><br>
**🔷 랜드마크 시각화**
```python
for landmarks in res.multi_face_landmarks:
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    )
```
🔹 Detection된 face의 랜드마크 468개를 윤곽선 형태로 그림 <br>
🔹 FACEMESH_CONTOURS를 통해 눈, 입, 얼굴 외곽선을 중심으로 선을 연결 
<br><br>
**🔷 결과 출력**
```python
cv.imshow('MediaPipe Face Mesh', cv.flip(frame, 1))
```
🔹 실시간으로 face mesh를 시각화
<br><br>
### :octocat: 실행 결과
![스크린샷 2025-04-15 185108](https://github.com/user-attachments/assets/80282387-ae19-450d-8e2f-d8a832bcdfa1)


