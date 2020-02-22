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
import random

def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

np.random.seed(120)
torch.manual_seed(999)
random.seed(9001)

num_epochs = 60
batch_size = 128
learning_rate = 1e-3

data1 = pd.read_csv("/home/pranav/Documents/cs7015/assignment_2/Assignment_2/BnW/7/data.csv", encoding = "UTF-8")
#data1 = pd.read_csv("/content/Assignment_2/BnW/7/data.csv", encoding = "UTF-8")
data1=data1.values
np.random.shuffle(data1)
features=data1[:,:784]
targets=torch.Tensor(data1[:int(0.7*34999),:784])
targets_t=torch.Tensor(data1[int(0.7*34999):,:784])
err_per=0.3
for j in range(data1.shape[0]):
    rn=random.sample(range(0, 784),int(err_per*784))
    for i in range(int(err_per*784)):
        features[j][rn[i]]=1-features[j][rn[i]]

features_t=torch.Tensor(features[int(0.7*34999):,:])
features=features[:int(0.7*34999),:]
features=torch.Tensor(features)

dataset = data_utils.TensorDataset(features, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 80),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(80, 784),
            nn.Sigmoid())

    def bottle(self,x):
        x=self.encoder(x)
        return x

    def rbottle(self,x):
        x=self.decoder(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class autoencoder_2(nn.Module):
    def __init__(self):
        super(autoencoder_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(80, 70),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(70, 80),
            nn.ReLU(True))

    def bottle(self,x):
        x=self.encoder(x)
        return x

    def rbottle(self,x):
        x=self.decoder(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class autoencoder_3(nn.Module):
    def __init__(self):
        super(autoencoder_3, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(70, 60),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(60, 70),
            nn.ReLU(True))

    def bottle(self,x):
        x=self.encoder(x)
        return x

    def rbottle(self,x):
        x=self.decoder(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model_st1 = autoencoder()
model_st1.load_state_dict(torch.load('./sim_autoencoder_st13.pth'))
model_st2 = autoencoder_2()
model_st2.load_state_dict(torch.load('./sim_autoencoder_st23.pth'))
model_st3 = autoencoder_3()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model_st3.parameters(), lr=learning_rate, weight_decay=1e-5)

best_model=model_st3
prev_loss=10
print ('training starting')
for epoch in range(num_epochs):
    for data in dataloader:
        #print (data)
        img, clas = data
        #print (cla.shape)
        img = Variable(img)
        #print (img.shape),128,784
        # ===================forward=====================
        #output = model_st1.bottle(img)
        #output=model_st2(output)
        #output=model_st1.rbottle(output)
        # 128,784
        #loss = criterion(output, clas)
        output=model_st1.rbottle(model_st2.rbottle(model_st3(model_st2.bottle(model_st1.bottle(img)))))
        #output=model_st1(img)
        loss=criterion(output,clas)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    if epoch % 7 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(output.cpu().data)
        save_image(x, './mlp_img2/x_{}.png'.format(epoch))
        save_image(x_hat, './mlp_img2/x_hat_{}.png'.format(epoch))
    loss=criterion(model_st1.rbottle(model_st2.rbottle(model_st3(model_st2.bottle(model_st1.bottle(features))))),targets)
    #loss=criterion(model_st1(features),targets)
    if prev_loss-loss.data < 0.00001:
        break

    if loss.data < prev_loss:
        best_model=model_st3
        prev_loss=loss.data

    print('epoch [{}/{}], loss:{:.4f} '
          .format(epoch + 1, num_epochs, loss.data))

torch.save(best_model.state_dict(), './sim_autoencoder_st34.pth')
print('finished training')

feat=Variable(features_t)
output=model_st1(feat)
targ=targets_t.type(torch.LongTensor)
tot=0
corr=0
    #print (targ)
    #print (targ[0].item())
for i in range(targ.shape[0]):
    for j in range(output.shape[1]):
        if(output[i][j] > 0.5 and targ[i][j].item()==1):
            corr=corr+1
        elif (output[i][j] <= 0.5 and targ[i][j].item()==0):
            corr=corr+1
        tot=tot+1            


corr=corr/tot
print (corr)
    
#model.load_state_dict(torch.load('./sim_autoencoder4.pth'))