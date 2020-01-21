from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
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
value = 0
collect = 0
x_values = list(range(0,5000))
for i in prediction:

    name = filenames[cnt].split("/")[3]
    if i>= 0.5:
        if name.find("dog") != -1:
            value += 1
            collect += 1
            plt.scatter(cnt,i,s=1,c='b',label='dog')
        else:
            plt.scatter(cnt,i,s=1,c='r',label='cat')
    #if i >= 0.5: print("해당 " + filenames[cnt].split("/")[3]+ "  이미지는 개 로 추정됩니다.")
    #else : print("해당 " + filenames[cnt]+ "  이미지는 고양이 으로 추정됩니다.")
    else:
        if name.find("cat") != -1:
            value += 1
            plt.scatter(cnt,i,s=1,c='r',label='cat')
        else:
            plt.scatter(cnt,i,s=1,c='b',label='dog')
    cnt += 1
    #print(name," : ",i)
#print("검증 데이터 셋 정확도 : %.2f\n" %(value/1000))
print(" 데이터 정확도 : %.2f" %(collect/cnt*2))
plt.axhline(y=0.5, color='r', linewidth=1)
plt.title('Food Detection')
plt.ylabel('accuracy')
plt.xlabel('image number')
plt.savefig("result.jpg")
# In[ ]:




