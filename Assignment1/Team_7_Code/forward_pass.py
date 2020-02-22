import numpy as np
import math

slope=1
#takes 0.2,1,10

def sigmoid(Z):
	return 1/(1+np.exp(-Z*slope))

def relu(Z):
	return np.maximum(0,Z)

def tanh(Z):
	return (np.exp(2*slope*Z)-1)/(np.exp(2*slope*Z)+1)

def elu(Z):
	s=[]
	delta=1;
	s=np.maximum(0,Z)
	s1=np.minimum(0,Z)
	s1=delta*(np.exp(s1)-1)
	return s+s1

def softplus(Z):
	return np.log(1+np.exp(Z))

def softmax(x):
	s1=[]
	for k in range(x.shape[1]):
		s=[]
		for i in range(x.shape[0]):
			sum=0
			for j in range(x.shape[0]):
				sum=sum+math.exp(x[j][k]-x[i][k])
			s.append(1/sum)
		s1.append(s)
	return np.array(s1).T