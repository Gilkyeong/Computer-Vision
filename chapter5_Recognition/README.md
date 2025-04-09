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
```
*핵심코드* <br>
**🔷 데이터셋 로드**
```python
(x_train,y_train),(x_test,y_test)=ds.mnist.load_data()
```
🔹 tensorflow.keras.datasets 모듈을 사용하여 MNIST 데이터셋을 불러옴 <br>
🔹 MNIST : 손글씨 숫자 이미지(28×28 크기), Train data : 60,000개 Test data : 10,000개로 구성
<br><br>
**🔷 Flattening**
```python
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
```
🔹 각 이미지는 28×28 행렬 형태 <br>
🔹 nn input으로 사용하기 위해 (60000, 784)와 (10000, 784)의 1차원 배열로 변환
<br><br>
**🔷 Normalization**
```python
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
```
🔹 0-1 사이의 값으로 정규화
<br><br>
**🔷 label One-Hot Encoding**
```python
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```
🔹 10개의 클래스 확률을 반환하도록 변환
<br><br>
**🔷 Model 구조 설계**
```python
dmlp=Sequential()
dmlp.add(Dense(units=1024,activation='relu',input_shape=(784,)))
dmlp.add(Dense(units=512,activation='relu'))
dmlp.add(Dense(units=512,activation='relu'))
dmlp.add(Dense(units=10,activation='softmax'))
```
🔹 Keras의 Sequential 모델을 사용해 layer를 쌓음 <br>
🔹 Dense layer <br>
    첫번째 layer <br>
     Unit : 1024, activation function : relu, input : 784 <br>
    두번째, 세번째 layer <br>
     Unit : 512, activation function : relu <br>
    output layer <br>
     Unit : 10 (10개 class), activation function : softmax (각 클래스에 대한 확률 값 출력)
<br><br>
**🔷 Model 컴파일**
```python
dmlp.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])
hist=dmlp.fit(x_train,y_train,batch_size=128,epochs=30,validation_data=(x_test,y_test),verbose=2)
print('acc =', dmlp.evaluate(x_test,y_test,verbose=0)[1]*100)
```
🔹 dmlp.compile : 모델 컴파일 <br>
   loss function : categorical_crossentropy, Optimizer : Adam, learning rate : 0.0001 <br>
🔹 dmlp.fit : 모델 학습 <br>
   batch size : 128, epochs : 30, verbose : 2 (각 epochs에 대한 log 출력)
<br><br>
**🔷 모델 평가**
```python
print('acc =', dmlp.evaluate(x_test, y_test, verbose=0)[1] * 100)
```
🔹 Train이 완료된 후 test data를 사용해 accuracy 평가 <br><br>
![스크린샷 2025-04-08 162656](https://github.com/user-attachments/assets/06d48ed4-ece3-415d-9d32-44c3b7f52a5b)
<br><br>
### :octocat: 실행 결과
![mlpacc](https://github.com/user-attachments/assets/44c99f1f-3298-4d63-84ab-726f35b49525)
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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

(x_train, y_train), (x_test, y_test) = ds.cifar10.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

cnn = Sequential()
cnn.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
cnn.add(Conv2D(32, (3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64, (3,3), activation='relu'))
cnn.add(Conv2D(64, (3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
hist = cnn.fit(x_train, y_train,
               batch_size=128, epochs=50,
               validation_data=(x_test, y_test),
               verbose=2)

res = cnn.evaluate(x_test, y_test, verbose=0)
print('acc =', res[1]*100)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()

test_img = load_img("dog.jpg", target_size=(32,32))
test_img_array = img_to_array(test_img)
test_img_array = test_img_array.astype(np.float32) / 255.0
test_img_array = np.expand_dims(test_img_array, axis=0)

prediction = cnn.predict(test_img_array)
predicted_class_idx = np.argmax(prediction, axis=1)[0]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
print("Index :", predicted_class_idx)
print("Class name :", class_names[predicted_class_idx])
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
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
```
🔹 0-1 사이의 값으로 정규화
<br><br>
**🔷 Label One-Hot Encoding**
```python
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```
🔹 10개의 클래스 확률을 반환하도록 변환
<br><br>
**🔷 CNN 모델 설계**
```python
cnn = Sequential()
cnn.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
cnn.add(Conv2D(32, (3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64, (3,3), activation='relu'))
cnn.add(Conv2D(64, (3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))
```
🔹 Keras의 Sequential 모델을 사용해 layer를 쌓음 <br>
```python
cnn.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
cnn.add(Conv2D(32, (3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
```
🔹 첫번째 Convolution layer <br>
    Conv2D layer <br>
     32개 fliter, kernel size : 3x3, activation function : relu, input : (32, 32, 3) → CIFAR-10 이미지 크기와 채널 수 <br>
    MaxPooling2D <br>
     2×2 풀링을 사용해 공간적 크기를 줄여 계산량 감소 및 특징 추출 <br>
    Dropout(0.25) <br>
     Overfitting을 방지하기 위해 랜덤하게 25%의 뉴런을 제거 <br>
```python
cnn.add(Conv2D(64, (3,3), activation='relu'))
cnn.add(Conv2D(64, (3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
```
🔹 두번째 Convolution layer <br>
    Conv2D layer <br>
     64개 fliter, kernel size : 3x3, activation function : relu <br>
    MaxPooling2D <br>
     2×2 풀링을 사용해 공간적 크기를 줄여 계산량 감소 및 특징 추출 <br>
    Dropout(0.25) <br>
     Overfitting을 방지하기 위해 랜덤하게 25%의 뉴런을 제거 <br>
```python
cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))
```
🔹 FC layer <br>
    Flatten <br>
     2차원 형태의 feature map을 1차원 벡터로 변환 <br>
    첫번째 Dense layer <br>
     Unit : 512, activation function : relu <br>
    Dropout(0.5) <br>
     Overfitting을 방지하기 위해 랜덤하게 50%의 뉴런을 제거 <br>
    두번째 Dense layer <br>
     Unit : 10, activation function : softmax <br>
<br><br>
**🔷 Model 컴파일**
```python
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
hist = cnn.fit(x_train, y_train,
               batch_size=128, epochs=50,
               validation_data=(x_test, y_test),
               verbose=2)
```
🔹 cnn.compile : 모델 컴파일 <br>
   loss function : categorical_crossentropy, Optimizer : Adam, learning rate : 0.001 <br>
🔹 cnn.fit : 모델 학습 <br>
   batch size : 128, epochs : 50, verbose : 2 (각 epochs에 대한 log 출력)
<br><br>
**🔷 정확도 측정**
```python
res = cnn.evaluate(x_test, y_test, verbose=0)
print('acc =', res[1]*100)
```
🔹 Train이 완료된 후 test data를 사용해 accuracy 평가 <br><br>
![image](https://github.com/user-attachments/assets/9f5199c4-ff25-44ed-8169-fe9229ab754f)
<br><br>
**🔷 이미지 classification**
```python
test_img = load_img("dog.jpg", target_size=(32,32))
test_img_array = img_to_array(test_img)
test_img_array = test_img_array.astype(np.float32) / 255.0
test_img_array = np.expand_dims(test_img_array, axis=0)

prediction = cnn.predict(test_img_array)
predicted_class_idx = np.argmax(prediction, axis=1)[0]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
print("Index :", predicted_class_idx)
print("Class name :", class_names[predicted_class_idx])
```
🔹 dog.jpg를 불러와서 학습된 cnn 모델을 사용하여 이미지 classification<br>
불러온 이미지 <br>
![dog](https://github.com/user-attachments/assets/37d0e5c0-83b9-4296-8987-22578bf6111d) <br>
성능 <br>
![스크린샷 2025-04-08 172139](https://github.com/user-attachments/assets/ac32b121-21ea-4102-8a6e-6d48c84a8112)
<br><br>
### :octocat: 실행 결과
![image](https://github.com/user-attachments/assets/d9c19ef9-f72d-4327-afc3-117acd25c23e)
<br><br>

## 🌀 문제 3 전이 학습을 활용한 이미지 분류기 개선

> 사전에 학습된 모델을 활용하여 전이 학습을 수행하고 이미지 분류기의 성능 향상
---

### 📄 코드 
- Trans_classsification.py

*전체 코드*
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10

gpus = tf.config.list_physical_devices('GPU')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def preprocess(x, y):
    x = tf.image.resize(x, [224, 224]) / 255.0
    return x, y

BATCH_SIZE = 16

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=50, validation_data=test_ds)

test_loss, test_acc = model.evaluate(test_ds)
print(f"\n acc : {test_acc:.4f}")
```

*핵심 코드* <br>
**🔷 데이터셋 로드**
```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def preprocess(x, y):
    x = tf.image.resize(x, [224, 224]) / 255.0
    return x, y
```
🔹 VGG16의 input에 맞추기 위해 224x224로 resize 후 0-1 범위로 정규화
<br><br>
**🔷 VGG16 base model 로드**
```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
```
🔹 ImageNet에 대해 사전 학습된 VGG16 모델을 불러옴 <br>
🔹 include_top=False를 통해 VGG16 모델의 마지막 부분은 제거하고 input size 224×224로 맞춤 <br>
🔹 base_model.trainable = False로 사전 학습된 weight가 학습 중에 업데이트되지 않도록 freeze <br>
<br><br>
**🔷 새로운 layer 설계**
```python
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```
🔹 Flatten layer : 2차원 feature map을 1차원 벡터로 변환 <br>
🔹 첫번째 Dense layer : Unit : 256, activation function : relu <br>
🔹 Dropout (0.5): Overfitting을 방지하기 위해 랜덤하게 50%의 뉴런을 제거 <br>
🔹 두번째 Dense layer : Unit : 10, activation function : softmax
<br><br>
**🔷 Model 컴파일**
```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_ds, epochs=50, validation_data=test_ds)
```
🔹 model.compile : 모델 컴파일 <br>
   loss function : sparse_categorical_crossentropy, Optimizer : Adam <br>
🔹 model.fit : 모델 학습 <br>
   epochs : 50
<br><br>
**🔷 정확도 측정**
```python
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n acc : {test_acc:.4f}")
```
🔹 Train이 완료된 후 test data를 사용해 accuracy 평가 <br><br>
<br><br>
### :octocat: 실행 결과

