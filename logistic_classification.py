#!/usr/bin/env python
# coding: utf-8

# 먼저 이미지를 배열화 시켜주어야 한다.
# 
# 그리고 여기 이미지에서 훈련 데이터, 테스트 데이터(사실상 검증 데이터)로 나눠준다.
# 
# 이후 numpy 형태로 저장.

# In[1]:


from PIL import Image
import os, glob, sys, numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

img_dir = './train/train'
categories = ['cat', 'dog']
np_classes = len(categories)

image_w = 64
image_h = 64


pixel = image_h * image_w * 3

X = []
y = []

for idx, cat in enumerate(categories):
    if cat == "cat":
        img_dir_detail = img_dir+"/cat*.jpg"
    else:
        img_dir_detail = img_dir+"/dog*.jpg"
    files = glob.glob(img_dir_detail)


    for i, f in enumerate(files):
        try:
            print(f)
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            X.append(data)
            y.append(idx)
            if i % 500 == 0:
                print(cat, " : ", f)
        except:
            print(cat, str(i)+" 번째에서 에러 ")
X = np.array(X)
Y = np.array(y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)
np.save("./numpy_data/logistic_image_data.npy", xy)




from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K

X_train, X_test, y_train, y_test = np.load('./numpy_data/logistic_image_data.npy')
print(X_train.shape)
print(X_train.shape[0])
print(np.bincount(y_train))
print(np.bincount(y_test))


image_w = 64
image_h = 64
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


with K.tf_ops.device('/device:GPU:0'):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model_dir = './model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = model_dir + "/logistic_classify.model"
    
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)


model.summary()


history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.15, callbacks=[checkpoint, early_stopping])

print("정확도 : %.2f " %(model.evaluate(X_test, y_test)[1]))



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss', 'acc', 'val_acc'], loc='upper left')
plt.show()
plt.savefig("data.jpg")



from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

import tensorflow as tf

seed = 5
tf.set_random_seed(seed)
np.random.seed(seed)

caltech_dir = './train/validance'

image_w = 64
image_h = 64

pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir+"/*.jpg")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)

    filenames.append(f)
    X.append(data)


X = np.array(X)
X = X.astype(float) / 255
model = load_model('./model/logistic_classify.model')

prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0
value = 0;
collect = 0;
for i in prediction:

    name = filenames[cnt].split("/")[3]
    if i>= 0.5:
        if name.find("cat") != -1:
            value += 1
            collect += 1
    else:
        if name.find("dog") != -1:
            value += 1
    cnt += 1
    print(i)
print("검증 데이터 셋 정확도 : %.2f\n" %(value/5000))
print("cat 데이터 정확도 : %.2f" %(collect/2500))




