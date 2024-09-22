import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

pick_in=open('data.pickle','rb')
data=pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features=[]
labels=[]

for feature,label in data:
    features.append(feature)
    labels.append(label)

xtrain,xtest,ytrain,ytest=train_test_split(features,labels,test_size=0.98)

model=SVC(C=1,kernel='poly',gamma='auto')
model.fit(xtrain,ytrain)

prediction=model.predict(xtest)
accuracy=model.score(xtest,ytest)

categories=['cats','dogs']

print('Accuracy: ',accuracy)

print('Prediction is : ',categories[prediction[0]])

mypet=xtest[0].reshape(50,50)
plt.imshow(mypet,cmap='gray')
plt.show()
