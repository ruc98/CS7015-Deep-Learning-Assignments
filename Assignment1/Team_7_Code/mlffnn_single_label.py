import activation as act_fun
import act_diff
import numpy as np
import math
import random
import csv
import update_rules as update

nn_architecture=[
				{ "input_dim":60, "output_dim":8, "activation":"sigmoid"},
				{ "input_dim":8, "output_dim":5, "activation":"sigmoid"}
			]

#paramter initialisation 
param_values=update.init_layers(nn_architecture)


#param_values={"W1":W1, "W2":W2, "b1":b1, "b2":b2}
#acquiring the data
type_of_input="unnormalised"
in1=open('/home/soham/Documents/DL/Assignment1/A1_Single_label_image_classification_dataset_features/image_data_feat_dim60.txt','r')
in2=open('/home/soham/Documents/DL/Assignment1/A1_Single_label_image_classification_dataset_features/image_data_labels.txt','r')
l1 = in1.read().strip().split("\n")
l2 = in2.read().strip().split("\n")
l2=np.array(l2)
l2=np.int_(l2)
sd=0
for i in range(len(l2)):
	if (l2[i] == 0 or l2[i]==1 or l2[i]==2 or l2[i]==6 or l2[i]==7):
		sd=sd+1
	Y=np.zeros(shape=(5,sd))
sf=0
c1=0;c2=0;c3=0;c4=0;c5=0;
for i in range(len(l2)):
   	if (l2[i] == 0 or l2[i]==1 or l2[i]==2):
   		Y[l2[i]][sf]=1;
   		sf=sf+1
   		c1=c1+1
   	elif(l2[i]==6):
   		Y[3][sf]=1;
   		sf=sf+1
   		c2=c2+1
   	elif(l2[i]==7):
   		Y[4][sf]=1;		
   		sf=sf+1
   		c3=c3+1
    #print(Y.shape)
l11=[]
for i in l1:
  	l12=[]
  	for j in i.split():
  		l12.append(j)
  	l11.append(l12)
l1=np.array(l11)
#print (l1)
l1=np.float_(l1)
l11=[]
X1=l1.T

X=np.zeros(shape=(60,sd))
sd=0
for i in range(X1.shape[1]):
   	if (l2[i] == 0 or l2[i]==1 or l2[i]==2 or l2[i]==6 or l2[i]==7):
   		for j in range(60):
   			X[j][sd]=X1[j][i]
   		sd=sd+1
idx = np.random.randint(Y.shape[1], size=int(sd*0.3))
if (type_of_input == "normalized"):
	max1=-1000000
	min1=1000000
	for i in range(X.shape[1]):
		for j in range(X.shape[0]):
			if (X[j][i] < min1):
				min1=X[j][i]
			if (X[j][i] > max1):
				max1=X[j][i]
	for i in range(X.shape[1]):
		for j in range(X.shape[0]):
			X[j][i]=(X[j][i]-min1)/(max1-min1)
X_test=X[:,idx]
Y_test=Y[:,idx]
idx1=[]
for i in range(Y.shape[1]):
	if i not in idx:
		idx1.append(i)
idx1=np.array(idx1)
np.random.shuffle(idx1)
X=X[:,idx1]
Y=Y[:,idx1]














#X=np.random.randn(10000,32)
##Y=np.random.randn(10000,6)
#X_train=np.random.randn(7000,32)
#X_test=np.random.randn(3000,32)
#Y_train=np.random.randn(7000,6)
#Y_test=np.random.randn(3000,6)
#csvfile=open("7/X.csv","r")
#csvreader=csv.reader(csvfile)
#index=0
#for row in csvreader:
#	X[index]=row
#	index=index+1
#csvfile=open("7/Y.csv","r")
#csvreader=csv.reader(csvfile)
#index=0
#for row in csvreader:
#	Y[index]=row
#	index=index+1
X_train=np.random.randn(60,824)
X_test=np.random.randn(60,354)
Y_train=np.random.randn(5,824)
Y_test=np.random.randn(5,354)
dt={}
for i in range(0,X.shape[1]):
	dt[i]=i
k1=0
k2=0
for i in range(X.shape[1]):
	j=dt.pop(random.choice(list(dt)))
	if i<math.floor(0.7*X.shape[1]):
		X_train[:,k1]=X[:,j]
		Y_train[:,k1]=Y[:,j]
		k1+=1
	else:
		X_test[:,k2]=X[:,j]
		Y_test[:,k2]=Y[:,j]
		k2+=1

def norma(X):
	for i in range(0,X.shape[0]):
		ma=np.amax(X[i])
		mi=np.amin(X[i])
		for j in range(0,X.shape[1]):
			X[i][j]=(X[i][j]-mi)/(ma-mi)
	return X



def single_layer_fp(X,W,b,activation="sigmoid"):
	l=[]
	for i in range(0,X.shape[1]):
		l.append(1)
	A=np.dot(W,X)+np.outer(b,np.array(l))
	if activation=="linear":
		S=act_fun.linear(A)
	elif activation=="sigmoid":
		S=act_fun.sigmoid(beta,A)
	elif activation=="tanh":
		S=act_fun.tanh(beta,A)
	elif activation=="relu":
		S=act_fun.relu(A)
	elif activation=="softplus":
		S=act_fun.softplus(A)
	elif activation=="elu":
		S=act_fun.elu(delta,A)
	elif activation=="softmax":
		S=act_fun.softmax(A)
	else:
		print("Activation function isn't supported")
	return (A,S)


def multi_layer_fp(param_values,X):
	(A1,S1)=single_layer_fp(X,param_values["W1"],param_values["b1"],nn_architecture[0]["activation"])
	(A2,S2)=single_layer_fp(S1,param_values["W2"],param_values["b2"],nn_architecture[1]["activation"])
	(A3,S3)=single_layer_fp(S2,param_values["W3"],param_values["b3"],nn_architecture[2]["activation"])
	memory={"A1": A1, "S1": S1, "A2": A2, "S2": S2, "A3":A3, "S3":S3}
	return (S3,memory)

def error(S,Y):
	sum=0.0
	for i in range(0,Y.shape[1]):
		sum=sum+np.linalg.norm(Y[:,i]-S[:,i])*np.linalg.norm(Y[:,i]-S[:,i])
	return sum/2

def back_prop(param_values,memory,X,Y):
	grad={}
	W3=param_values["W3"]
	W2=param_values["W2"]
	S3=memory["S3"]
	DELTA_O=np.multiply(Y-S3,act_diff.sigmoid_diff(beta,S3))
	grad["dW3"]=np.dot(DELTA_O,memory["S2"].T)
	grad["db3"]=np.sum(DELTA_O,axis=1)
	D2=act_diff.elu_diff(0.01,memory["A2"])
	DELTA_H1=np.multiply(np.dot(W3.T,DELTA_O),D2)
	grad["db2"]=np.sum(DELTA_H1,axis=1)
	grad["dW2"]=np.dot(DELTA_H1,memory["S1"].T)
	D1=act_diff.sigmoid_diff(beta,memory["S1"])
	DELTA_H2=np.multiply(np.dot(W2.T,DELTA_H1),D1)
	grad["dW1"]=np.dot(DELTA_H2,X.T)
	grad["db1"]=np.sum(DELTA_H2,axis=1)
	return grad

def train(param_values):
	(S2,mem)=multi_layer_fp(param_values,(X_train))
	error_new=error(S2,Y_train)/X_train.shape[1]
	index=1
	while 1:
		dic_t={}
		for i in range(0,len(Xbatches)):
			dic_t[i]=i
		error_old=error_new
		print(index,"  ",error_old)
		index+=1
		while len(dic_t)>0:
			l=dic_t.pop(random.choice(list(dic_t)))
			(S2,memory)=multi_layer_fp(param_values,(Xbatches[l]))
			grad_values=back_prop(param_values,memory,(Xbatches[l]),Ybatches[l])
			param_values=update.update_adam(param_values,grad_values,learning_rate,nn_architecture)
			

		(S2,mem)=multi_layer_fp(param_values,(X_train))
		error_new=error(S2,Y_train)/X_train.shape[1]
		if abs(error_old-error_new)<0.0005:
			break
	return index,param_values

def get_pr(param_values,X,Y):
	(S2,mem)=multi_layer_fp(param_values,X)
	confusion=np.zeros((Y.shape[0],4),dtype=int)
	for j in range(0,S2.shape[1]):
		for i in range(0,S2.shape[0]):
			if S2[i][j]>=0.5:
				S2[i][j]=1
			else:
				S2[i][j]=0
	for j in range(0,S2.shape[1]):
		for i in range(0,S2.shape[0]):
			if Y[i][j]==1:
				if S2[i][j]==1:
					confusion[i][0]+=1
				else:
					confusion[i][1]+=1
			else:
				if S2[i][j]==1:
					confusion[i][2]+=1
				else:
					confusion[i][3]+=1
	temp1=confusion[:,0]+confusion[:,2]
	temp2=confusion[:,0]+confusion[:,1]
	precision=np.sum(confusion[:,0],axis=0)/np.sum(temp1,axis=0)
	recall=np.sum(confusion[:,0],axis=0)/np.sum(temp2,axis=0)
	return (precision,recall,confusion)

def get_confusion(param_values,X,Y):
	(S2,mem)=multi_layer_fp(param_values,X)
	confusion=np.zeros((8,8),dtype=int)
	for j in range(0,S2.shape[1]):
		index1=0
		for i in range(1,S2.shape[0]):
			if S2[i][j]>S2[index1][j]:
				index1=i
		index2=0
		for i in range(0,Y.shape[0]):
			if Y[i][j]==1:
				index2=i
				break
		confusion[index2][index1]+=1
	correct=0
	total=0
	for i in range(0,confusion.shape[0]):
		for j in range(0,confusion.shape[1]):
			if i==j:
				correct+=confusion[i][j]
			total+=confusion[i][j]
	return ((correct/total),confusion)



batch_size=824
Xbatches=[]
Ybatches=[]
beta=1
learning_rate=0.00005
delta=0.01
bt={}
for i in range(0,X_train.shape[1]):
	bt[i]=i
#Xbatch=np.empty([batch_size,32])
#Ybatch=np.empty([batch_size,6])
#while len(bt)>0:
#	index=0
#	while index<batch_size:
#		l=bt.pop(random.choice(list(bt)))
#		Xbatch[index]=(X_train[l])
#		Ybatch[index]=(Y_train[l])
#		index+=1
#	Xbatches.append(Xbatch)
#	Ybatches.append(Ybatch)
#	Xbatch=np.empty([batch_size,32])
#	Ybatch=np.empty([batch_size,6])

Xbatch=np.empty([60,batch_size])
Ybatch=np.empty([5,batch_size])
print(len(bt))
while len(bt)>0:
	index=0
	while index<batch_size:
		l=bt.pop(random.choice(list(bt)))
		Xbatch[:,index]=(X_train[:,l])
		Ybatch[:,index]=(Y_train[:,l])
		index+=1
	Xbatches.append(Xbatch)
	Ybatches.append(Ybatch)
	Xbatch=np.empty([60,batch_size])
	Ybatch=np.empty([5,batch_size])

ind,param_values=train(param_values)
print(ind-1,"Epochs till convergence")
accu,confusion=get_confusion(param_values,X_train,Y_train)
print(accu)
print(confusion)
accu,confusion=get_confusion(param_values,X_test,Y_test)
print(accu)
print(confusion)
#(precision,recall,confusion)=get_pr(param_values,X_test.T,Y_test.T)
#print("Test data")
#f1=2*precision*recall/(precision+recall)
#print(precision,"\t",recall,"\t",f1)
#(precision,recall,confusion)=get_pr(param_values,X_train.T,Y_train.T)
#print("Train data")
#f1=2*precision*recall/(precision+recall)
#print(precision,"\t",recall,"\t",f1)