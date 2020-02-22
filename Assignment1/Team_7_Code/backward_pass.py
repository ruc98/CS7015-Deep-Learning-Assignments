import numpy as np
import math
from forward_pass import *

def sigmoid_backward(dA, Z):
	sig = sigmoid(Z)
	return dA * sig * (1 - sig)*slope

def relu_backward(dA, Z):
	dZ = np.array(dA, copy = True)
	dZ[Z <= 0] = 0;
	return dZ;

def tanh_backward(dA, Z):
	tah=tanh(Z)
	return dA*(1-tah)*(1+tah)*slope;

def elu_backward(dA, Z):
	delta=1;
	l1=[]
	for k in range(Z.shape[1]):
		l=[]
		for i in range(Z.shape[0]):
			if a>0:
				l.append(1)
			else:
				l.append(delta*math.exp(Z[i][k]))
		l1.append(l)
	l1=np.array(l1).T

	return dA * l1

def softplus_backward(dA, Z):
	s1=[]
	for k in range(Z.shape[1]):
		s=[]
		for i in range(Z.shape[0]):
			s.append(1/(1+math.exp(-Z[i][k])))
		s1.append(s)
	s1=np.array(s1).T
	return dA*s1