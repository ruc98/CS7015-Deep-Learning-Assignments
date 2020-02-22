##########################################
#       Group 7: Task2 and Task3         #
##########################################

import os
import torch.utils.data as data_utils
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import pandas as pd
import numpy as np

### HYPERPARAMETERS ###
dataset_no = 1 # 1 for color, 2 for BnW
if dataset_no ==1:
    convergence = 1e-5
    batch_size = 4  # for dataset1, =8, for dataet2, =128
    learning_rate = 1e-3
    nodes_l1 = 200  # 80,100,150,200,500  ,best 200
    nodes_l2 = 100  # 30,60,100,150,200 ,best 100
    nodes_l3 = 30   # 32,64,128,256, best 256

if dataset_no ==2:
    convergence = 1e-5
    batch_size = 128  # for dataset1, =8, for dataet2, =128
    learning_rate = 1e-3
    nodes_l1 = 512  # 64,128,256,512  ,best 512
    nodes_l2 = 384  # 64,128,256,384 ,best 384
    nodes_l3 = 256   # 32,64,128,256, best 256

### data loading ###

torch.manual_seed(1)
np.random.seed(0)

if dataset_no ==1:
    data1=np.zeros(shape=(1596,829))
    ji=0
    labels = ['coast','forest','highway','street','tallbuilding']
    for idx, label in enumerate(labels):
        path = 'Features/'+label
        for file in os.listdir(path):
            current = os.path.join(path, file)
            in1=open(current)
            l1 = in1.read().strip().split("\n")
            l2=[]
            for i in l1:
                l2=l2+i.split(" ")
            l2.append(idx)
            l2=np.array(l2)
            l2=np.float_(l2)
            data1[ji]=l2
            ji=ji+1

if dataset_no ==2:
    datafile = 'BnW/7/data.csv'
    data1 = pd.read_csv(datafile, encoding = "UTF-8")
    data1=data1.values

np.random.shuffle(data1)
train_test_split = int(0.8*data1.shape[0])  # 80% traindata
data1 = data1[:train_test_split]            # training data
unlabeled_labeled_split = int(0.7*data1.shape[0]) # 70% of traindata as unlabeled data
data_unlabeled = data1[:unlabeled_labeled_split]
features=torch.Tensor(data_unlabeled[:,:-1])
dataloader = DataLoader(features, batch_size=batch_size, shuffle=False) # load data in batches

data_val = data1[unlabeled_labeled_split:]  # labeled train data as validation data
features_val=torch.Tensor(data_val[:,:-1])
dataloader_val = DataLoader(features_val, batch_size=1, shuffle=False)
input_dim = data1[:,:-1].shape[1]

### Model definition ###
class autoencoder1(nn.Module):                # autoencoder 1 model
    def __init__(self):
        super(autoencoder1, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, nodes_l1),
            nn.ReLU(True))
        self.decoder1 = nn.Sequential(
            nn.Linear(nodes_l1, input_dim),
            nn.Sigmoid())

    def bottle(self,x):                     # forward pass through encoder only
            x=self.encoder1(x)
            return x

        
    def forward(self, x):                   # forward pass through encoder and decoder
            x = self.encoder1(x)
            x = self.decoder1(x)
            return x

        
class autoencoder2(nn.Module):
    def __init__(self):
        super(autoencoder2, self).__init__()
        self.encoder2 = nn.Sequential(
            nn.Linear(nodes_l1,nodes_l2),
            nn.ReLU(True))
        self.decoder2 = nn.Sequential(
            nn.Linear(nodes_l2,nodes_l1),
            nn.ReLU(True))

    def bottle(self,x):
            x=self.encoder2(x)
            return x

        
    def forward(self, x):
            x = self.encoder2(x)
            x = self.decoder2(x)
            return x

        
class autoencoder3(nn.Module):
    def __init__(self):
        super(autoencoder3, self).__init__()
        self.encoder3 = nn.Sequential(
            nn.Linear(nodes_l2,nodes_l3),
            nn.ReLU(True))
        self.decoder3 = nn.Sequential(
            nn.Linear(nodes_l3,nodes_l2),
            nn.ReLU(True))

    def bottle(self,x):
            x=self.encoder3(x)
            return x

        
    def forward(self, x):
            x = self.encoder3(x)
            x = self.decoder3(x)
            return x

model1 = autoencoder1()
model2 = autoencoder2()
model3 = autoencoder3()

### training layer 1 ###
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model1.parameters(), lr=learning_rate, weight_decay=1e-5)

best_model1=model1  # to store the best model
prev_loss=10
train_cost_history1 = []
epoch=0
while(True):
    epoch+=1
    for data in dataloader:
        #print (data)
        img = data
        img = Variable(img)
        # ===================forward=====================
        output = model1(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for data_val in dataloader_val:
        img_val = data_val
        img_val = Variable(img_val)
        # ===================forward=====================
        output_val = model1(img_val)
        loss_val = criterion(output_val, img_val)

    # ===================log========================
#     print('epoch [{}], loss:{:.4f}, val_loss{:.4f}'
#           .format(epoch, loss.data,loss_val.data))
    print('{}, {:.4f}, {:.4f}'
          .format(epoch, loss.data, loss_val.data))    
    delta_loss = prev_loss - loss.data
    if delta_loss < convergence:                     # convergence criterion
        break
    best_model1=model1
    prev_loss=loss.data
    train_cost_history1.append(loss.data)

torch.save(best_model1.state_dict(), './autoencoder1.pth')  # saves the best model


### creating inputs for layer 2 ###

model1.load_state_dict(torch.load('./autoencoder1.pth')) # load best model of autoencoder 1
model1.eval()
f1 = Variable(features)
hl1 = model1.bottle(f1)              # extract features from 1 layer

f1_val = Variable(features_val)
hl1_val = model1.bottle(f1_val)
print(hl1.shape)

dataloader = DataLoader(hl1, batch_size=batch_size, shuffle=False)  # feed features from first layers as batches to the second layer
dataloader_val = DataLoader(hl1_val, batch_size=1, shuffle=False)


### training layer 2 ###
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model2.parameters(), lr=learning_rate, weight_decay=1e-5)

best_model2=model2
prev_loss=10
train_cost_history2 = []
epoch=0
while(True):
    epoch+=1
    for data in dataloader:
        img = data
        img = Variable(img)
        # ===================forward=====================
        output = model2(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for data_val in dataloader_val:
        img_val = data_val
        img_val = Variable(img_val)
        # ===================forward=====================
        output_val = model2(img_val)
        loss_val = criterion(output_val, img_val)

    # ===================log========================
#     print('epoch [{}], loss:{:.4f}, val_loss{:.4f}'
#           .format(epoch, loss.data,loss_val.data))
    print('{}, {:.4f}, {:.4f}'
          .format(epoch, loss.data, loss_val.data))    
 
    delta_loss = prev_loss - loss.data
    if delta_loss < convergence and epoch>20:
        break
    best_model2=model2
    prev_loss=loss.data
    train_cost_history2.append(loss.data)

torch.save(best_model2.state_dict(), './autoencoder2.pth')


### creating inputs for layer 3 ###

model2.load_state_dict(torch.load('./autoencoder2.pth'))
model2.eval()
f2 = Variable(hl1)
hl2 = model2.bottle(f2)

f2_val = Variable(hl1_val)
hl2_val = model2.bottle(f2_val)

print(hl2.shape)

dataloader = DataLoader(hl2, batch_size=batch_size, shuffle=False)
dataloader_val = DataLoader(hl2_val, batch_size=1, shuffle=False)

### training layer 3 ###
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model3.parameters(), lr=learning_rate, weight_decay=1e-5)

best_model3=model3
prev_loss=10
train_cost_history3 = []
epoch=0
while(True):
    epoch+=1
    for data in dataloader:
        img = data
        img = Variable(img)
        # ===================forward=====================
        output = model3(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for data_val in dataloader_val:
        img_val = data_val
        img_val = Variable(img_val)
        # ===================forward=====================
        output_val = model3(img_val)
        loss_val = criterion(output_val, img_val)

    # ===================log========================
#     print('epoch [{}], loss:{:.4f}, val_loss{:.4f}'
#           .format(epoch, loss.data,loss_val.data))
    print('{}, {:.4f}, {:.4f}'
          .format(epoch, loss.data, loss_val.data))    
 
    delta_loss = prev_loss - loss.data
    if delta_loss < convergence and epoch>20:
        break
    best_model3=model3
    prev_loss=loss.data
    train_cost_history3.append(loss.data)

torch.save(best_model3.state_dict(), './autoencoder3.pth')

        
