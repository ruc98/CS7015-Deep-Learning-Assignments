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
from sklearn.metrics import confusion_matrix

### HYPERPARAMETERS ###
dataset_no = 1  # 1 for color, 2 for BnW

if dataset_no==1:
    convergence = 1e-7
    batch_size = 4
    learning_rate = 1e-3
    nodes_l1 = 200  # 64,128,256,512  ,best 512
    nodes_l2 = 100  # 64,128,256,384 ,best 384
    nodes_l3 = 30   # 32,64,128,256, best 256

if dataset_no==2:
    convergence = 1e-7
    batch_size = 128
    learning_rate = 1e-2
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
train_test_split = int(0.8*data1.shape[0]) # 80% traindata
data_train = data1[:train_test_split]
data_test  = data1[train_test_split:]

unlabeled_labeled_split = int(0.7*data1.shape[0]) # 70% of traindata as unlabeled data
data_labeled = data1[unlabeled_labeled_split:]
data_train = data_labeled
targets=torch.Tensor(np.ravel(data_train[:,-1]))
features=torch.Tensor(data_train[:,:-1])
targets_t=torch.Tensor(np.ravel(data_test[:,-1]))
features_t=torch.Tensor(data_test[:,:-1])

dataset = data_utils.TensorDataset(features, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
input_dim = data1[:,:-1].shape[1]

class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, nodes_l1),
            nn.ReLU(True))
        self.encoder2 = nn.Sequential(
            nn.Linear(nodes_l1,nodes_l2),
            nn.ReLU(True))
        self.encoder3 = nn.Sequential(
            nn.Linear(nodes_l2,nodes_l3),
            nn.ReLU(True))
        self.final_layer = nn.Sequential(
            nn.Linear(nodes_l3, 5),
            nn.Softmax(1))

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.final_layer(x)
        return x

sae_model = SAE()


### Classification ###

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    sae_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

best_model_cls=sae_model
prev_loss=10
train_cost_history = []
epoch=0

while(True):
    epoch+=1
    for data in dataloader:
        img, clas = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        clas=clas.type(torch.LongTensor)
        # ===================forward=====================
        output = sae_model(img)
        loss = criterion(output, clas)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    delta_loss = prev_loss - loss.data
    if dataset_no==1:
        if epoch==200:  # convergence criteria gives bad results for dataset 1
            break
    else:
        if delta_loss < convergence: # convergence gives good results for dataset 2
            break
    best_model_cls=sae_model
    prev_loss=loss.data
    train_cost_history.append(loss.data)

    # ===================log========================
    ## train acc ##
    feat=Variable(features)
    output=sae_model(feat)
    targ=targets.type(torch.LongTensor)
    tot=0
    corr=0
    for i in range(targ.shape[0]):
        in1=0
        ma=0
        for j in range(output.shape[1]):
            if (output[i][j].item() > ma):
                ma=output[i][j].item()
                in1=j

        if (targ[i].item() == in1):
            corr=corr+1
        tot=tot+1
    train_acc=corr/tot
    
    ## test acc ##
    feat=Variable(features_t)
    output=sae_model(feat)
    targ=targets_t.type(torch.LongTensor)
    tot=0
    corr=0
    for i in range(targ.shape[0]):
        in1=0
        ma=0
        for j in range(output.shape[1]):
            if (output[i][j].item() > ma):
                ma=output[i][j].item()
                in1=j

        if (targ[i].item() == in1):
            corr=corr+1
        tot=tot+1
    test_acc=corr/tot
    
    print('epoch {}, loss:{:.4f}, train_accuracy:{:.4f}, test_accuracy:{:.4f}'
          .format(epoch, loss.data, train_acc, test_acc))
torch.save(best_model_cls.state_dict(), './sae_model_final_not_pretrained.pth')

sae_model.load_state_dict(torch.load('./sae_model_final_not_pretrained.pth'))

feat=Variable(features)
output=sae_model(feat)
pred = torch.argmax(output, 1)
targ=targets.type(torch.LongTensor)

cm = confusion_matrix(pred.view(-1), targ.view(-1))


labels = ['Ankle boot','T-shirt/top','Bag','Coat','Trouser']

print("_ ,",end="")
for i in range(cm.shape[0]):
    print(labels[i],',',end="")
print("")
for i in range(cm.shape[0]):
    print(labels[i],',',end="")
    for j in range(cm.shape[1]):
        print('{} , '.format(cm[i][j]),end = "")
    print("")
