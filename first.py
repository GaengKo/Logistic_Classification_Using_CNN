#!/usr/bin/env python
# coding: utf-8



from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import tensorflow as tf

seed = 5
tf.set_random_seed(seed)
np.random.seed(seed)

caltech_dir = './train/train'


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
value = 0
collect = 0
cat_num=0
dog_num=0
n_X = []
n_Y = []
for i in prediction:
    if i<=0.8 and i>=0.2:
        n_X.append(data)
        if files[cnt].find("cat") != -1:
            os.system("cp "+files[cnt]+" ./newDataset/cat."+str(cat_num)+".jpg")
            cat_num += 1
            #print(files[cnt],",",cat_num)
        else :
            os.system("cp "+files[cnt]+" ./newDataset/dog."+str(dog_num)+".jpg")
            dog_num += 1
            #print(files[cnt],",",dog_num)
    cnt += 1

