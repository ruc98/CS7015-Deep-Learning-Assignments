import numpy as np
import math

def linear_diff(S):
	l=np.empty(S.shape)
	for i in range(0,S.shape[0]):
		for j in range(0,S.shape[1]):
			l[i][j]=1
	return l

def sigmoid_diff(beta,S):
	l=np.empty(S.shape)
	for j in range(0,S.shape[1]):
		for i in range(0,S.shape[0]):
			l[i][j]=beta*S[i][j]*(1-S[i][j])
	return l

def tanh_diff(beta,S):
	l=np.empty(S.shape)
	for j in range(0,S.shape[1]):
		for i in range(0,S.shape[0]):
			l[i][j]=beta*(1-S[i][j])*(1+S[i][j])
	return l

def relu_diff(S):
	l=np.empty(S.shape)
	for j in range(0,S.shape[1]):
		for i in range(0,S.shape[0]):
			if S[i][j]>0:
				l[i][j]=1
			else:
				l[i][j]=0
	return l

def softplus_diff(A):
	l=np.empty(A.shape)
	for j in range(0,A.shape[1]):
		for i in range(0,A.shape[0]):
			l[i][j]=1/(1+math.exp(-A[i][j]))
	return l

def elu_diff(delta,A):
	l=np.empty(A.shape)
	for j in range(0,A.shape[1]):
		for i in range(0,A.shape[0]):
			if A[i][j]>0:
				l[i][j]=1
			else:
				l[i][j]=delta*math.exp(A[i][j])
	return l