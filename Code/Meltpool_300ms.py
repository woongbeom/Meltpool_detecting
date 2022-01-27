# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:18:57 2021

@author: HWB
"""

#!/usr/bin/env python
# coding: utf-8

import os, glob, numpy as np
#from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, AveragePooling2D #Dense, 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
#import keras.backend.tensorflow_backend as K
from keras import backend as K

from tensorflow.python.keras.layers import Input, Dense, Embedding, ZeroPadding2D
from tensorflow.python.keras.models import Sequential
from keras.optimizers import SGD

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

X_train, X_test, Y_train, Y_test = np.load('./2021_300ms_128.npy', allow_pickle = True)
print(X_train.shape)
print(X_train.shape[0])


# In[ ]:


categories = ["normal", "abnormal_01", "abnormal_02", "abnormal_03", "abnormal_04" ]
nb_classes = len(categories)

#일반화
X_train = X_train.astype('float') / 255.
X_test = X_test.astype('float') / 255.


# In[ ]:


# model = Sequential()
# model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))

# model.add(AveragePooling2D(pool_size = (2,2), strides=None, padding="valid", data_format=None))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.25))


# model.add(Dense(5, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(AveragePooling2D(pool_size = (2,2), strides=None, padding="valid", data_format=None))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))


model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model_dir = './model'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_path = model_dir + '/meltpool.model'
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=6)



# In[ ]:


model.summary()


# In[ ]:
    


    
tf.random.set_seed(0)

import time

start = time.time()

#데이터셋이 적어서 validation을 그냥 test 데이터로 했습니다. 
#데이터셋이 충분하시면 이렇게 하시지 마시고 validation_split=0.2 이렇게 하셔서 테스트 셋으로 나누시길 권장합니다.
#history = model.fit(X_train, Y_train, batch_size=256, epochs=30, validation_data=(X_test, Y_test), callbacks=[checkpoint, early_stopping])

with tf.device('GPU:0'):
    history = model.fit(X_train, Y_train, batch_size=128, epochs=30, validation_split=0.3, callbacks=[checkpoint, early_stopping])


# In[ ]:


print("정확도 : %.4f" %(model.evaluate(X_test, Y_test)[1] * 100), "%")

print("학습 소요 시간 :", round(time.time()-start, 2), "초")


from keras.models import load_model

model.save('./model/300ms_model.h5', overwrite=True)

# model = load_model('./model/30ms_model.h5')

# Confusion matrix

from sklearn.metrics import confusion_matrix #교차표 그리기
import seaborn as sns

prediction = model.predict(X_test)

C = confusion_matrix(Y_test.argmax(axis=1), prediction.argmax(axis=1))

norm_c_m = C / C.astype(np.float).sum(axis=1)

norm_c_m

np.set_printoptions(formatter={'float_kind':lambda x:"{0:0.3f}".format(x)})

figure = plt.figure(figsize=(5, 5))
sns.heatmap(norm_c_m, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# In[ ]:


y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()


# In[10]:


from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import time


model = load_model('./model/300ms_model.h5')
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])


caltech_dir = "D:/FILE/meltpool_img/TEST_300"
image_w = 128
image_h = 128

pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir+"/*.*")

for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

X = np.array(X)
X= X.astype('float') / 255.

start = time.time()

prediction = model.predict(X)
#cnn_score=model.score(X_test, Y_test)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0

for i in prediction:
    pre_ans = i.argmax()  # 예측 레이블
    print(i)
    print(pre_ans)
    
    pre_ans_str = ''
    if pre_ans == 0: pre_ans_str = "normal"
    elif pre_ans == 1: pre_ans_str = "abnormal_01"
    elif pre_ans == 2: pre_ans_str = "abnormal_02"
    elif pre_ans == 3: pre_ans_str = "abnormal_03"
    elif pre_ans == 4: pre_ans_str = "abnormal_04"
    else: pre_ans_str = "구분불가"
    
    if i[0] >= 0.5: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[1] >= 0.5: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[2] >= 0.5: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[3] >= 0.5: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    if i[4] >= 0.5: print("해당 "+filenames[cnt].split("\\")[1]+"이미지는 "+pre_ans_str+"로 추정됩니다.")
    cnt += 1


print("판별 소요 시간 :", round(time.time()-start, 4), "초")























