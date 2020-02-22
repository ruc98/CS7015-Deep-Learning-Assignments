from numpy import linalg as la
import pandas as pd
import os
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(120)
torch.manual_seed(999)

#data1 = pd.read_csv("/home/pranav/Documents/cs7015/assignment_2/Assignment_2/BnW/7/data.csv", encoding = "UTF-8")
#data1=data1.values
data1=np.zeros(shape=(1596,829))
ji=0
path='/home/pranav/Documents/cs7015/assignment_2/Assignment_2/Features/coast'
for file in os.listdir(path):
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
np.random.shuffle(data1)
tr=data1[:,828:]
tr=tr.reshape(1596)
feature=data1[:,:828]

#pca = PCA()  
#X_train = pca.fit_transform(feature)  
#explained_variance = pca.explained_variance_ratio_
#print (explained_variance)
#print (feature[0])
#print (feature[0].shape)

red_dim=180
mean=np.zeros(shape=(828))
for i in range(data1.shape[0]):
	mean=mean+feature[i]
mean=mean/feature.shape[0]

tr_feat=data1[:,:828]
for i in range(data1.shape[0]):
	#print (tr_feat[i])
	#print (mean)
	tr_feat[i]=tr_feat[i]-mean
	#print (tr_feat[i])

cov=np.zeros(shape=(828,828))
for i in range(data1.shape[0]):
	cov=cov+np.outer(tr_feat[i],tr_feat[i])

cov=cov/feature.shape[0]
w, v = la.eig(cov)
index=np.argsort(w)
index=index[::-1]

rel=0
err=np.zeros(shape=(828))
sor=np.zeros(shape=(828))
for i in range(827,-1,-1):
	err[i]=rel
	rel=rel+w[index[i]]

for i in range(828):
	sor[i]=i+1
	print(w[index[i]])

plt.plot(sor,err)
plt.plot(red_dim,err[red_dim],'-o')
plt.annotate('({},{:.4f})'.format(red_dim,err[red_dim]),(red_dim,err[red_dim]))
print(err)
#plt.plot(170,err[170],'-o')
#plt.annotate('({},{:.4f})'.format(170,err[170]),(170,err[170]))
plt.xlabel('Number of Principle Components')
plt.ylabel('Reconstruction Error')
plt.show()

for i in range(828):
	print(w[index[i]])

q=np.zeros(shape=(red_dim,828))
for i in range(red_dim):
	q[i]=v[:,int(index[i])]

feature=np.zeros(shape=(data1.shape[0],red_dim))
for i in range(feature.shape[0]):
	feature[i]=np.matmul(q,tr_feat[i])

rec_err=0
for i in range(red_dim,828):
	rec_err=rec_err+w[index[i]]
'''
red_dim=828
pca = PCA(n_components=red_dim)
feature=pca.fit_transform(data1[:,:828])
'''
#print (rec_err)
print ('rec_err')
print ('stop me')

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

tr=data1[:,828:]
tr=tr.reshape(1596)
targets=torch.LongTensor(tr[:int(0.7*1596)])
features=torch.Tensor(feature[:int(0.7*1596),:])
targets_t=torch.Tensor(tr[int(0.7*1596):])
features_t=torch.Tensor(feature[int(0.7*1596):,:])

batch_size = 10
learning_rate = 1e-3

dataset = data_utils.TensorDataset(features, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

prev_loss=10
num_epochs=1000
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
        img = Variable(img)
        #print (img.shape),128,784
        #print (clas)
        clas=clas.type(torch.LongTensor)
        # ===================forward=====================
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
    loss=criterion(classif(features),targets)
    if prev_loss-loss.data < 0.0001:
    	break

    if loss.data < prev_loss:
        best_model=classif
        prev_loss=loss.data

    feat=Variable(features_t)
    output=classif(feat)
    targ=targets_t.type(torch.LongTensor)
    tot=0
    corr=0
    #print (targ[0].item())
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
torch.save(best_model.state_dict(), './sim_autoencoder3.pth')
for i in range(5):
    te=0
    for j in range(5):
        te=te+cm[i][j]
    for j in range(5):
        cm[i][j]=cm[i][j]/te	

print(cm)