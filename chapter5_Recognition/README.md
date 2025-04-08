## 🌀 문제 1 간단한 이미지 분류기 구현

> MNIST datasets을 이용하여 **MLP 구현**
---
**MLP(Multi Layer Perceptron)** <br><br>
![image](https://github.com/user-attachments/assets/87b523bb-a531-4def-bab3-38eec11c9a4b)

### 📄 코드 
- MNIST_MLP.py

*전체 코드*
```python
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

(x_train,y_train),(x_test,y_test)=ds.mnist.load_data()
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

dmlp=Sequential()
dmlp.add(Dense(units=1024,activation='relu',input_shape=(784,)))
dmlp.add(Dense(units=512,activation='relu'))
dmlp.add(Dense(units=512,activation='relu'))
dmlp.add(Dense(units=10,activation='softmax'))

dmlp.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])
hist=dmlp.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)
print('acc =', dmlp.evaluate(x_test,y_test,verbose=0)[1]*100)

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.grid()
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
**🔷 정확도 측정**
```python
img_kp = cv.drawKeypoints(img_rgb, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```
🔹 특징점을 이미지에 시각화 <br>
🔹 DRAW_RICH_KEYPOINTS : 크기와 방향 정보를 포함해 원으로 표시
![스크린샷 2025-04-08 162656](https://github.com/user-attachments/assets/06d48ed4-ece3-415d-9d32-44c3b7f52a5b)
<br><br>
### :octocat: 실행 결과

![mlpacc](https://github.com/user-attachments/assets/44c99f1f-3298-4d63-84ab-726f35b49525)
![mlploss](https://github.com/user-attachments/assets/d8f61503-93e9-4e1d-bce9-52807eb33be2)
<br><br>
## 🌀 문제 2 CIFAR-10 datasets을 활용한 CNN 모델 구축

> CIFAR-10 datasets을 활용하여 **CNN을 구축하고 image classification 수행**
---
**CNN(Convolutional Neural Network)** <br><br>
![image](https://github.com/user-attachments/assets/d49fd4e0-cd3a-482e-9158-47d99863e7be)

### 📄 코드 
- CIFAR10-CNN.py

*전체 코드*
```python
import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam

(x_train,y_train),(x_test,y_test)=ds.cifar10.load_data()
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

cnn=Sequential()
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(units=512,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(units=10,activation='softmax'))

cnn.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
hist=cnn.fit(x_train,y_train,batch_size=128,epochs=50,validation_data=(x_test,y_test),verbose=2)

res=cnn.evaluate(x_test,y_test,verbose=0)
print('acc =',res[1]*100)

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
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
import matplotlib.pyplot as plt

img1 = cv.imread('img2.jpg')
img2 = cv.imread('img3.jpg')

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H_pan, mask_pan = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
warped_corners = cv.perspectiveTransform(corners_img2, H_pan)
all_corners = np.concatenate((warped_corners, np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)), axis=0)
[xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
translation = [-xmin, -ymin]
T = np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
panorama = cv.warpPerspective(img2, T @ H_pan, (xmax - xmin, ymax - ymin))
panorama[translation[1]:translation[1] + h1, translation[0]:translation[0] + w1] = img1

bf2 = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
matches2 = bf2.knnMatch(des1, des2, k=2)
good_matches2 = []
ratio_thresh = 0.7
for m, n in matches2:
    if m.distance < ratio_thresh * n.distance:
        good_matches2.append(m)
print("matching num :", len(good_matches2))
src_pts2 = np.float32([kp1[m.queryIdx].pt for m in good_matches2]).reshape(-1, 1, 2)
dst_pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches2]).reshape(-1, 1, 2)
H_match, mask_match = cv.findHomography(src_pts2, dst_pts2, cv.RANSAC)
warped_img = cv.warpPerspective(img2, H_match, (w1, h1))
img_matches = cv.drawMatches(img1, kp1, warped_img, kp2, good_matches2, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

panorama_rgb = cv.cvtColor(panorama, cv.COLOR_BGR2RGB)
img_matches_rgb = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(panorama_rgb)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_matches_rgb)
plt.axis('off')
plt.show()
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
h2, w2 = img1.shape[:2]
warped_img = cv.warpPerspective(img2, H, (w2, h2))
img_matches = cv.drawMatches(img1, kp1, warped_img, kp2, good_matches, None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```
🔹 추정된 호모그래피를 이용하여 img1, img2에 정렬되도록 변형 <br>
🔹 매칭 결과를 시각화
<br><br>
### :octocat: 실행 결과

![image](https://github.com/user-attachments/assets/7c7525dc-8db0-40aa-bf7e-2dcca67914f6)
