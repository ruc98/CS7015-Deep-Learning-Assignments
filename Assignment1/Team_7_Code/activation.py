import math
import numpy as np

def linear(A):
	return A

def tanh(beta,A):
	s=np.empty(A.shape)
	for i in range(0,A.shape[0]):
		for j in range(0,A.shape[1]):
			s[i][j]=math.tanh(beta*A[i][j])
	return s

def sigmoid(beta,A):
	s=np.empty(A.shape)
	for i in range(0,A.shape[0]):
		for j in range(0,A.shape[1]):
			s[i][j]=1/(1+math.exp(-beta*max(A[i][j],-100)))
	return s

def relu(A):
	s=np.empty(A.shape)
	for i in range(0,A.shape[0]):
		for j in range(0,A.shape[1]):
			s[i][j]=max(0,A[i][j])
	return s

def softplus(A):
	s=np.empty(A.shape)
	for i in range(0,A.shape[0]):
		for j in range(0,A.shape[1]):
			s[i][j]=math.log(1+math.exp(A[i][j]))
	return s	

def elu(delta,A):
	s=np.empty(A.shape)
	for i in range(0,A.shape[0]):
		for j in range(0,A.shape[1]):
			if A[i][j]>=0:
				s[i][j]=A[i][j]
			else:
				s[i][j]=delta*(math.exp(A[i][j])-1)
	return s

def softmax(A):
	s=np.empty(A.shape)
	for i in range(0,A.shape[1]):
		for j in range(0,A.shape[0]):
			sum=0
			for k in range(0,A.shape[0]):
				sum=sum+math.exp(A[k][i]-A[j][i])
			s[j][i]=1/sum
	return s