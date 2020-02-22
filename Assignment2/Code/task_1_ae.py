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
from PIL import Image
import numpy as np
import random

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')

#data1 = pd.read_csv("/home/pranav/Documents/cs7015/assignment_2/Assignment_2/BnW/7/data.csv", encoding = "UTF-8")
#data1=data1.values
random.seed(9001)
data1=np.zeros(shape=(1596,829))
ji=0
path='/home/pranav/Documents/cs7015/assignment_2/Assignment_2/Features/coast'
for file in os.listdir(path):
    #print(file)
    current = os.path.join(path, file)
    in1=open(current)
    l1 = in1.read().strip().split("\n")
    l2=[]
    for i in l1:
    	l2=l2+i.split(" ")
    l2.append(0)
    l2=np.array(l2)
    l2=np.float_(l2)
    data1[ji]=l2
    ji=ji+1
    

path='/home/pranav/Documents/cs7015/assignment_2/Assignment_2/Features/forest'
for file in os.listdir(path):
    current = os.path.join(path, file)
    in1=open(current)
    l1 = in1.read().strip().split("\n")
    l2=[]
    for i in l1:
    	l2=l2+i.split(" ")
    l2.append(1)
    l2=np.array(l2)
    l2=np.float_(l2)
    data1[ji]=l2
    ji=ji+1
    

path='/home/pranav/Documents/cs7015/assignment_2/Assignment_2/Features/highway'
for file in os.listdir(path):
    current = os.path.join(path, file)
    in1=open(current)
    l1 = in1.read().strip().split("\n")
    l2=[]
    for i in l1:
    	l2=l2+i.split(" ")
    l2.append(2)
    l2=np.array(l2)
    l2=np.float_(l2)
    data1[ji]=l2
    ji=ji+1
    

path='/home/pranav/Documents/cs7015/assignment_2/Assignment_2/Features/street'
for file in os.listdir(path):
    current = os.path.join(path, file)
    in1=open(current)
    l1 = in1.read().strip().split("\n")
    l2=[]
    for i in l1:
    	l2=l2+i.split(" ")
    l2.append(3)
    l2=np.array(l2)
    l2=np.float_(l2)
    data1[ji]=l2
    ji=ji+1
    

path='/home/pranav/Documents/cs7015/assignment_2/Assignment_2/Features/tallbuilding'
for file in os.listdir(path):
    current = os.path.join(path, file)
    in1=open(current)
    l1 = in1.read().strip().split("\n")
    l2=[]
    for i in l1:
    	l2=l2+i.split(" ")
    l2.append(4)
    l2=np.array(l2)
    l2=np.float_(l2)
    data1[ji]=l2
    ji=ji+1

num_epochs = 0
batch_size = 10
learning_rate = 1e-3
red_dim=48

def to_img(x):
    x = x.view(x.size(0), 1, 36, 23)
    return x

def plot_sample_img(img, name):
    img = img.view(1, 36, 23)
    save_image(img, './sample_{}.png'.format(name))

np.random.seed(120)
torch.manual_seed(999)
np.random.shuffle(data1)
tr=data1[:,828:]
tr=tr.reshape(1596)
tr=tr.astype(int)
targets=torch.Tensor(tr)
features=torch.Tensor(data1[:,:828])

dataset = data_utils.TensorDataset(features, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(828, 100),
            nn.ReLU(True),
            nn.Linear(100, red_dim),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(red_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 828),
            nn.ReLU(True))

    def bottle(self,x):
        x=self.encoder(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

best_model=model
prev_loss=10

for epoch in range(num_epochs):
    for data in dataloader:
        #print (data)
        img, clas = data
        #print (cla.shape)
        img = img.view(img.size(0), -1)
        img = Variable(img)
        #print (img.shape),128,784
        # ===================forward=====================
        output = model(img)
        # 128,784
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    
    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        save_image(x, './mlp_img/x_{}.png'.format(epoch))
        save_image(x_hat, './mlp_img/x_hat_{}.png'.format(epoch))

    loss=criterion(model(features),features)
    print(prev_loss)
    print(loss.data)
    if prev_loss-loss.data < 0.0001:
    	break

    if loss.data < prev_loss:
        best_model=model
        prev_loss=loss.data

    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data))
    	


#torch.save(best_model.state_dict(), './sim_autoencoder1.pth')
print ('model saved')
model.load_state_dict(torch.load('./sim_autoencoder1.pth'))
model.eval()

targets=torch.LongTensor(tr[:int(0.7*1596)])
features=torch.Tensor(data1[:int(0.7*1596),:828])
targets_t=torch.Tensor(tr[int(0.7*1596):])
features_t=torch.Tensor(data1[int(0.7*1596):,:828])

class mlffn(nn.Module):
    def __init__(self):
        super(mlffn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(red_dim, 70),
            nn.ReLU(True),
            nn.Linear(70, 5),
            nn.Softmax(1))

    def forward(self, x):
        x = self.encoder(x)
        return x

prev_loss=10
num_epochs=200
classif=mlffn()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    classif.parameters(), lr=0.001, weight_decay=1e-5)
best_model=classif

cm=np.zeros(shape=(5,5))

for epoch in range(num_epochs):
    for data in dataloader:
        #print (data)
        img, clas = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        #print (img.shape),128,784
        #print (clas)
        clas=clas.type(torch.LongTensor)
        # ===================forward=====================
        img = model.bottle(img)
        #print (img.shape)
        output = classif(img)
        #print (output[0][1].item())
        # 128,784
        loss = criterion(output, clas)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # ===================log========================
    loss=criterion(classif(model.bottle(features)),targets)
    if prev_loss-loss.data < 0.0001:
    	break

    if loss.data < prev_loss:
        best_model=classif
        prev_loss=loss.data

    feat=Variable(features_t)
    output=model.bottle(feat)
    output=classif(output)
    targ=targets_t.type(torch.LongTensor)
    tot=0
    corr=0
    #print (targ[0].item())
    cm=np.zeros(shape=(5,5))
    for i in range(targ.shape[0]):
        in1=0
        ma=0
        for j in range(output.shape[1]):
            if (output[i][j].item() > ma):
                ma=output[i][j].item()
                in1=j

        if (targ[i].item() == in1):
            corr=corr+1
        cm[targ[i].item()][in1]=cm[targ[i].item()][in1]+1
        #print (in1,targ[i].item())
        tot=tot+1

    corr=corr/tot
    print('epoch [{}/{}], loss:{:.4f} ,accuracy:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data,corr))
torch.save(best_model.state_dict(), './sim_autoencoder2.pth')
for i in range(5):
    te=0
    for j in range(5):
        te=te+cm[i][j]
    for j in range(5):
        cm[i][j]=cm[i][j]/te	

print(cm)