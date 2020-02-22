# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sf-lN6xiAwpFiWFQhrLTSHJfjatCGLEX
"""

import csv 
import cv2
import torch
import math
import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
import torchvision.models as models
import matplotlib.image as img
from torch.utils.data import DataLoader
import PIL.Image
import random


np.random.seed(80)
 
# csv file name 
filename = "ImageID.csv"
  
# initializing the titles and rows list 
fields = [] 
rows = [] 
imdata=None
model = models.vgg16(pretrained=True)

#new_classifier = nn.Sequential(*list(model.classifier.children())[:-7])
#model.classifier = new_classifier
#model.children=nn.Sequential(*list(model.children())[:1])
#print(model.layers)
#print(*list(model.features))

image=np.zeros(shape=(1,224,224,3))
image=np.rollaxis(image,3,1)
a=torch.tensor(image,dtype=torch.float32)
print(a.shape)
print(model.features(a).shape)
dim=model.features(a).shape[3]
dim1=model.features(a).shape[1]
print(model.features(a).detach().numpy().shape)
print(dim)

num_clus=3
vlad=None
beta=0.0001

X=np.empty((2251,25088))
Y=np.empty((2251,5))

index=0
for i in range(1,217):
  j=str(i)
  while len(j)!=3:
    j='0'+j
  image=cv2.imread('classification_task/012.binoculars/012_0'+j+'.jpg',cv2.IMREAD_COLOR)
  image=cv2.resize(image,(224,224))
  image=np.array(image).reshape((1,224,224,3))
  image=np.rollaxis(image,3,1)
  a=torch.tensor(image,dtype=torch.float32)
  imdata=model.features(a).detach().numpy()
  X[index]=imdata.reshape(25088)
  Y[index]=np.array([1,0,0,0,0])
print(1)
for i in range(1,799):
  j=str(i)
  while len(j)!=3:
    j='0'+j
  image=cv2.imread('classification_task/145.motorbikes-101/145_0'+j+'.jpg',cv2.IMREAD_COLOR)
  image=cv2.resize(image,(224,224))
  image=np.array(image).reshape((1,224,224,3))
  image=np.rollaxis(image,3,1)
  a=torch.tensor(image,dtype=torch.float32)
  imdata=model.features(a).detach().numpy()
  X[index]=imdata.reshape(25088)
  Y[index]=np.array([0,1,0,0,0])
print(2)
for i in range(1,210):
  j=str(i)
  while len(j)!=3:
    j='0'+j
  image=cv2.imread('classification_task/159.people/159_0'+j+'.jpg',cv2.IMREAD_COLOR)
  image=cv2.resize(image,(224,224))
  image=np.array(image).reshape((1,224,224,3))
  image=np.rollaxis(image,3,1)
  a=torch.tensor(image,dtype=torch.float32)
  imdata=model.features(a).detach().numpy()
  X[index]=imdata.reshape(25088)
  Y[index]=np.array([0,0,1,0,0])  
print(3)
for i in range(1,202):
  j=str(i)
  while len(j)!=3:
    j='0'+j
  image=cv2.imread('classification_task/240.watch-101/240_0'+j+'.jpg',cv2.IMREAD_COLOR)
  image=cv2.resize(image,(224,224))
  image=np.array(image).reshape((1,224,224,3))
  image=np.rollaxis(image,3,1)
  a=torch.tensor(image,dtype=torch.float32)
  imdata=model.features(a).detach().numpy()
  X[index]=imdata.reshape(25088)
  Y[index]=np.array([0,0,0,1,0])
print(4)
for i in range(1,828):
  j=str(i)
  while len(j)!=3:
    j='0'+j
  image=cv2.imread('classification_task/257.clutter/257_0'+j+'.jpg',cv2.IMREAD_COLOR)
  image=cv2.resize(image,(224,224))
  image=np.array(image).reshape((1,224,224,3))
  image=np.rollaxis(image,3,1)
  a=torch.tensor(image,dtype=torch.float32)
  imdata=model.features(a).detach().numpy()
  X[index]=imdata.reshape(25088)
  Y[index]=np.array([0,0,0,0,1])
print(5)

X_train=np.empty((1575,25088))
Y_train=np.empty((1575,5))
X_test=np.empty((676,25088))
Y_test=np.empty((676,5))
l=np.zeros(2251,dtype=np.int32)
for i in range(2251):
  l[i]=i
np.random.shuffle(l)
print(l)
k1=0
k2=0
for i in range(2251):
  if i<1575:
    X_train[k1]=X[l[i]]
    Y_train[k1]=Y[l[i]]
    k1+=1
  else:
    X_test[k2]=X[l[i]]
    Y_test[k2]=Y[l[i]]
    k2+=1
#with open('7.txt') as f:
 # s=f.readlines()
  #x=np.zeros(shape=(dim*dim,dim1))
  #vlad=np.zeros(shape=(int(len(s)/5),num_clus,dim1))
  #for i in range(0,len(s),5):
   # print(i)
    #print(s[i])
    #image=np.zeros(shape=(1,224,224,3))
    #ty=s[i].split('|')[0]
    #image[0] = img.imread('content/Image_Captioning/7/train_7/'+ty)
    #print(image.shape)
    #image=np.rollaxis(image,3,1)
    #a=torch.tensor(image,dtype=torch.float32)
    #imdata=model.features(a).detach().numpy()
    #print(imdata.shape)
    
#np.save('npcap_16',feat)

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.model=nn.Sequential(nn.Linear(25088,4096),nn.ReLU(True),nn.Linear(4096,4096),nn.ReLU(True),nn.Linear(4096,5),nn.Softmax(1))
    
  def forward(self,x):
    x=self.model(x)
    return x

model1=Net()
#model1.cuda()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model1.parameters(),lr=0.001,weight_decay=1e-5)


num_epochs=300
for epoch in range(num_epochs):
  for i in range(len(X_train)):
    x=torch.tensor(X_train[i])
    y=torch.tensor(Y_train[i])
    x=x.float()
    output=model1(x)
    loss=criterion(output,y)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

cm_train=np.zeros((5,5))
cm_test=np.zeros((5,5))
for i in range(len(X_train)):
  x=torch.tensor(X_train[i])
  y=torch.tensor(Y_train[i])
  output=model1(x)
  val,ind=torch.max(output,0)
  val1,ind1=torch.max(y,0)
  cm_train[ind1][ind]+=1

for i in range(len(X_test)):
  x=torch.tensor(X_test[i])
  y=torch.tensor(Y_test[i])
  output=model1(x)
  val,ind=torch.max(output,0)
  val1,ind1=torch.max(y,0)
  cm_test[ind1][ind]+=1

print(cm_train)
print(cm_test)