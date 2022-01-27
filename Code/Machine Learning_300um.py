

import os, glob, numpy as np
#from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Dense, Activation, ZeroPadding2D

from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
#import keras.backend.tensorflow_backend as K
from keras import backend as K

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

X_train, X_test, Y_train, Y_test = np.load('D:/FILE/meltpool_img/2021_300ms_128.npy', allow_pickle = True)

print(X_train.shape)
print(X_train.shape[0])



categories = ["normal", "abnormal_01", "abnormal_02", "abnormal_03", "abnormal_04"]
nb_classes = len(categories)

X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

tr_xe = X_train.reshape(-1, 128*128*3)
tr_ye = Y_train
te_xe = X_test.reshape(-1, 128*128*3)
te_ye = Y_test


# =============================================================================
#  KNN - 정확도 0.8333
# =============================================================================
import time

start = time.time()

from sklearn.neighbors import KNeighborsClassifier

knn_c=KNeighborsClassifier(n_neighbors=10)

knn_c.fit(tr_xe, tr_ye)
knn_pred=knn_c.predict(te_xe)

knn_c.predict_proba(te_xe)
knn_score=knn_c.score(te_xe, te_ye)


print("knn 학습 소요 시간 :", round(time.time()-start, 2), "초")
print("knn 정확도 :", round(knn_score, 3) * 100, "%")



# =============================================================================
# Decision Tree - 정확도 
# =============================================================================
import time

start2 = time.time()

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

dt=DecisionTreeClassifier()

dt=DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=5)
dt.fit(tr_xe, tr_ye)

plot_tree(dt)
dtt=export_text(dt)

dt_pred=dt.predict(te_xe)
dt_score = dt.score(te_xe, te_ye)

print("DT 학습 소요 시간 :", round(time.time()-start2, 2), "초")
print("DT 정확도 :", round(dt_score, 3) * 100, "%")

# =============================================================================
# Random Forest - 랜덤 포레스트
# =============================================================================

import time

start3 = time.time()

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=10, random_state=123) 
#매번 같은 결과값을 얻고 싶은 경우 random_state에 임의의 값을 넣어준다.
# 이 조건에 맞게 DT들이 생성됨

rf.fit(tr_xe, tr_ye)
dir(rf)
rf1_pred=rf.estimators_[0].predict(te_xe)
rf_score = rf.score(te_xe, te_ye)

print("RF 학습 소요 시간 :", round(time.time()-start3, 2), "초")
print("RF 정확도 :", round(rf_score, 3) * 100, "%")


