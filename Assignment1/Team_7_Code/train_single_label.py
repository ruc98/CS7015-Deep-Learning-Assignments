import queue
from forward_pass import *
from backward_pass import *
pre_params={}
alpha=0.1
uw={}
vw={}
epsilon=0.00001
ro=0.6
rw={}
num=0
tep=0
l=10
set_of_queue={}
set_of_queue_dw={}
track_l=queue.Queue(maxsize=l+1)
import numpy as np

nn_architecture = [
    {"input_dim": 60, "output_dim": 5, "activation": "sigmoid"},
    {"input_dim": 5, "output_dim": 5, "activation": "softmax"},
]
nn_mode="pattern"
#takes: batch,pattern

type_of_input="unnormalized"
#takes: normalized or unnormalized

import queue

def update_delta(params_values, grads_values, learning_rate):
	for idx, layer in enumerate(nn_architecture):
		layer_idx=idx+1
		params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
		params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
		#print(learning_rate * grads_values["dW" + str(layer_idx)] )

	return params_values;

def update_gen_delta(params_values, grads_values, learning_rate):
    global pre_params
    global alpha
    for idx, layer in enumerate(nn_architecture):
        layer_idx=idx+1
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

        params_values["W" + str(layer_idx)] += alpha*pre_params["W" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] += alpha*pre_params["b" + str(layer_idx)]

        pre_params["W" + str(layer_idx)] = learning_rate * grads_values["dW" + str(layer_idx)]        
        pre_params["b" + str(layer_idx)] = learning_rate * grads_values["db" + str(layer_idx)]

    return params_values;

def update_adagrad(params_values, grads_values, learning_rate):
    global rw
    #global epsilon
    epsilon = 0.1
    learning_rate=0.01
    for idx, layer in enumerate(nn_architecture):
        layer_idx=idx+1
        params_values["W" + str(layer_idx)] -= (learning_rate * grads_values["dW" + str(layer_idx)])/(epsilon+rw["W" + str(layer_idx)])
        params_values["b" + str(layer_idx)] -= (learning_rate * grads_values["db" + str(layer_idx)])/(epsilon+rw["b" + str(layer_idx)])

        rw["W" + str(layer_idx)] = np.sqrt(rw["W" + str(layer_idx)]*rw["W" + str(layer_idx)]+grads_values["dW" + str(layer_idx)]*grads_values["dW" + str(layer_idx)])     
        rw["b" + str(layer_idx)] = np.sqrt(rw["b" + str(layer_idx)]*rw["b" + str(layer_idx)]+grads_values["db" + str(layer_idx)]*grads_values["db" + str(layer_idx)])

    return params_values;

def update_rmsprop(params_values, grads_values, learning_rate):
	global set_of_queue
	global ro
	global epsilon
	global num
	epsilon=0.01
	for idx, layer in enumerate(nn_architecture):
		layer_idx=idx+1
		re1=0
		re2=0
		te1=0
		te2=0
		for i in range(set_of_queue["W" + str(layer_idx)].qsize()):
			te1=set_of_queue["W" + str(layer_idx)].get()
			te2=set_of_queue["b" + str(layer_idx)].get()
			set_of_queue["W" + str(layer_idx)].put(te1)
			set_of_queue["b" + str(layer_idx)].put(te2)
			re1=re1+te1*te1
			re2=re2+te2*te2
		
		if (set_of_queue["W" + str(layer_idx)].qsize() > 0):
			re1=(re1*ro)/set_of_queue["W" + str(layer_idx)].qsize()
			re2=(re2*ro)/set_of_queue["W" + str(layer_idx)].qsize()

		re1=re1+grads_values["dW" + str(layer_idx)]*grads_values["dW" + str(layer_idx)]*(1-ro)
		re2=re2+grads_values["db" + str(layer_idx)]*grads_values["db" + str(layer_idx)]*(1-ro)
		re1=np.sqrt(re1)
		re2=np.sqrt(re2)

		params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]/(epsilon+re1)
		params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]/(epsilon+re2)

		if (set_of_queue["W" + str(layer_idx)].qsize() >= l):
			set_of_queue["W" + str(layer_idx)].get()
			set_of_queue["b" + str(layer_idx)].get()
		set_of_queue["W" + str(layer_idx)].put(grads_values["dW" + str(layer_idx)])
		set_of_queue["b" + str(layer_idx)].put(grads_values["db" + str(layer_idx)])

	num=num+1;

	return params_values;

def update_adadelta(params_values, grads_values, learning_rate):
	global set_of_queue
	global set_of_queue_dw
	global epsilon
	global ro
	global num
	for idx, layer in enumerate(nn_architecture):
		layer_idx=idx+1
		re1=0
		re2=0
		te1=0
		te2=0
		te3=0
		te4=0
		re3=0
		re4=0

		for i in range(set_of_queue["W" + str(layer_idx)].qsize()):
			te1=set_of_queue["W" + str(layer_idx)].get()
			te2=set_of_queue["b" + str(layer_idx)].get()
			set_of_queue["W" + str(layer_idx)].put(te1)
			set_of_queue["b" + str(layer_idx)].put(te2)
			re1=re1+te1*te1
			re2=re2+te2*te2

		for i in range(set_of_queue_dw["W" + str(layer_idx)].qsize()-1):
			te3=set_of_queue_dw["W" + str(layer_idx)].get()
			te4=set_of_queue_dw["b" + str(layer_idx)].get()
			set_of_queue_dw["W" + str(layer_idx)].put(te3)
			set_of_queue_dw["b" + str(layer_idx)].put(te4)
			re3=re3+te3*te3
			re4=re4+te4*te4

		if (set_of_queue_dw["W" + str(layer_idx)].qsize() > 1):
			re3=(re3*ro)/(set_of_queue_dw["W" + str(layer_idx)].qsize()-1)
			re4=(re4*ro)/(set_of_queue_dw["W" + str(layer_idx)].qsize()-1)

		if (set_of_queue["W" + str(layer_idx)].qsize() > 0):
			re1=(re1*ro)/set_of_queue["W" + str(layer_idx)].qsize()
			re2=(re2*ro)/set_of_queue["W" + str(layer_idx)].qsize()

		te3=0;te4=0;
		if (set_of_queue_dw["W" + str(layer_idx)].qsize() > 0):
			te3=set_of_queue_dw["W" + str(layer_idx)].get()
			te4=set_of_queue_dw["b" + str(layer_idx)].get()
			set_of_queue_dw["W" + str(layer_idx)].put(te3)
			set_of_queue_dw["b" + str(layer_idx)].put(te4)

		re3=re3+(1-ro)*te3*te3
		re4=re4+(1-ro)*te4*te4

		re1=re1+grads_values["dW" + str(layer_idx)]*grads_values["dW" + str(layer_idx)]*(1-ro)
		re2=re2+grads_values["db" + str(layer_idx)]*grads_values["db" + str(layer_idx)]*(1-ro)
		re1=np.sqrt(re1)
		re2=np.sqrt(re2)
		re3=np.sqrt(re3)
		re4=np.sqrt(re4)
		te3=(epsilon+re3) * grads_values["dW" + str(layer_idx)]/(epsilon+re1)
		te4=(epsilon+re4) * grads_values["db" + str(layer_idx)]/(epsilon+re2)

		params_values["W" + str(layer_idx)] -= te3
		params_values["b" + str(layer_idx)] -= te4

		if (set_of_queue["W" + str(layer_idx)].qsize() >= l):
			set_of_queue["W" + str(layer_idx)].get()
			set_of_queue["b" + str(layer_idx)].get()
		set_of_queue["W" + str(layer_idx)].put(grads_values["dW" + str(layer_idx)])
		set_of_queue["b" + str(layer_idx)].put(grads_values["db" + str(layer_idx)])

		if (set_of_queue_dw["W" + str(layer_idx)].qsize() >= l):
			set_of_queue_dw["W" + str(layer_idx)].get()
			set_of_queue_dw["b" + str(layer_idx)].get()
		set_of_queue_dw["W" + str(layer_idx)].put(te3)
		set_of_queue_dw["b" + str(layer_idx)].put(te4)

	num=num+1;

	return params_values;

m_adam=0

def update_adam(params_values, grads_values, learning_rate):
    global uw
    global vw
    global m_adam
    ro1=0.92
    ro2=0.99999
    m_adam=m_adam+1;
    for idx, layer in enumerate(nn_architecture):
    	layer_idx=idx+1
    	uw['W' + str(layer_idx)]=ro1*uw['W' + str(layer_idx)]+(1-ro1)*grads_values["dW" + str(layer_idx)]
    	vw['W' + str(layer_idx)]=ro2*vw['W' + str(layer_idx)]+(1-ro2)*grads_values["dW" + str(layer_idx)]*grads_values["dW" + str(layer_idx)]

    	uw['b' + str(layer_idx)]=ro1*uw['b' + str(layer_idx)]+(1-ro1)*grads_values["db" + str(layer_idx)]
    	vw['b' + str(layer_idx)]=ro2*vw['b' + str(layer_idx)]+(1-ro2)*grads_values["db" + str(layer_idx)]*grads_values["db" + str(layer_idx)]

    	uw_hat1=uw['W' + str(layer_idx)]/(1-ro1**m_adam)
    	vw_hat1=np.sqrt(vw['W' + str(layer_idx)]/(1-ro2**m_adam))
    	uw_hat2=uw['b' + str(layer_idx)]/(1-ro1**m_adam)
    	vw_hat2=np.sqrt(vw['b' + str(layer_idx)]/(1-ro2**m_adam))
    	params_values["W" + str(layer_idx)] -= (learning_rate * uw_hat1)/(epsilon+vw_hat1)
    	params_values["b" + str(layer_idx)] -= (learning_rate * uw_hat2)/(epsilon+vw_hat2)

    return params_values;

update_rule=update_adagrad
#takes: delta,gen_delta,adagrad,rmsprop,adadelta,adam

def init_layers(seed = 99):
	global pre_params
	global uw
	global vw
	global rw
	global pre_params
	global set_of_queue
	global set_of_queue_dw
	global params_values

	np.random.seed(seed)
	number_of_layers = len(nn_architecture)
	params_values = {}

	for idx, layer in enumerate(nn_architecture):
		layer_idx = idx + 1
		layer_input_size = layer["input_dim"]
		layer_output_size = layer["output_dim"]
        
		params_values["W" + str(layer_idx)] = np.random.randn(
			layer_output_size, layer_input_size) * 0.1
		params_values["b" + str(layer_idx)] = np.random.randn(
			layer_output_size, 1) * 0.1

		if (update_rule == update_gen_delta):
			pre_params['W' + str(layer_idx)] = np.random.randn(
				layer_output_size, layer_input_size) * 0
			pre_params['b' + str(layer_idx)] = np.random.randn(
				layer_output_size, 1) * 0

		if (update_rule == update_adagrad):
			rw['W' + str(layer_idx)] = np.random.randn(
				layer_output_size, layer_input_size) * 0
			rw['b' + str(layer_idx)] = np.random.randn(
				layer_output_size, 1) * 0

		if (update_rule == update_rmsprop):
			set_of_queue['W' + str(layer_idx)]=queue.Queue(maxsize=l+1)
			set_of_queue['b' + str(layer_idx)]=queue.Queue(maxsize=l+1)

		if (update_rule == update_adadelta):
			set_of_queue['W' + str(layer_idx)]=queue.Queue(maxsize=l+1)
			set_of_queue['b' + str(layer_idx)]=queue.Queue(maxsize=l+1)

			set_of_queue_dw['W' + str(layer_idx)]=queue.Queue(maxsize=l+1)
			set_of_queue_dw['b' + str(layer_idx)]=queue.Queue(maxsize=l+1)	

		if (update_rule == update_adam):
			uw['W' + str(layer_idx)]=np.random.randn(layer_output_size, layer_input_size) * 0
			vw['W' + str(layer_idx)]=np.random.randn(layer_output_size, layer_input_size) * 0
			uw['b' + str(layer_idx)]=np.random.randn(layer_output_size, 1) * 0
			vw['b' + str(layer_idx)]=np.random.randn(layer_output_size, 1) * 0

	return params_values

def full_forward_propagation(X, params_values):
	memory = {}
	A_curr = X
    
	for idx, layer in enumerate(nn_architecture):
		layer_idx = idx + 1
		A_prev = A_curr
        
		activation = layer["activation"]
		W_curr = params_values["W" + str(layer_idx)]
		b_curr = params_values["b" + str(layer_idx)]
		Z_curr = np.dot(W_curr, A_prev) + b_curr
		if activation is "relu":
			activation_func = relu
		elif activation is "sigmoid":
			activation_func = sigmoid
		elif activation is "tanh":
			activation_func = tanh
		elif activation is "softplus":
			activation_func = softplus
		elif activation is "softmax":
			activation_func = softmax
		else:
			activation_func = elu
		A_curr=activation_func(Z_curr)
        
		memory["A" + str(idx)] = A_prev
		memory["Z" + str(layer_idx)] = Z_curr

	return A_curr, memory

def cross_entropy_error(Y_hat, Y):
    m = Y_hat.shape[1]
    #cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    cost=0
    #print(Y_hat.shape)
    #print (Y_hat)
    for i in range(Y_hat.shape[0]):
    	cost=cost-np.dot(Y[i], np.log(Y_hat[i]))
    #print (cost)
    cost=cost/m

    return np.squeeze(cost)

confusion_matrix=np.zeros(shape=(5,5))

def get_accuracy_value(Y_hat, Y):
	global confusion_matrix
	#Y_hat_ = convert_prob_into_class(Y_hat)
	Y_hat_=np.zeros(shape=(Y_hat.shape[0],Y_hat.shape[1]))
	for i in range(Y_hat.shape[1]):
		ma=-1
		cl=0
		for j in range(Y_hat.shape[0]):
			if(Y_hat[j][i] > ma):
				ma=Y_hat[j][i]
				cl=j
		Y_hat_[cl][i]=1
		#print(cl)
	#print (Y_hat_)
	#print (Y)
	te=0.0
	for i in range(Y_hat.shape[1]):
		for j in range(Y_hat.shape[0]):
			if (Y[j][i] == Y_hat_[j][i] and Y[j][i] == 1):
				te=te+1
		if (Y[0][i] == 1):
			for k in range(5):
				confusion_matrix[0][k]=confusion_matrix[0][k]+Y_hat_[k][i]				
		elif (Y[1][i] == 1):
			for k in range(5):
				confusion_matrix[1][k]=confusion_matrix[1][k]+Y_hat_[k][i]
		elif (Y[2][i] == 1):
			for k in range(5):
				confusion_matrix[2][k]=confusion_matrix[2][k]+Y_hat_[k][i]
		elif (Y[3][i] == 1):
			for k in range(5):
				confusion_matrix[3][k]=confusion_matrix[3][k]+Y_hat_[k][i]
		else:
			for k in range(5):
				confusion_matrix[4][k]=confusion_matrix[4][k]+Y_hat_[k][i]

	#print (te)
	return te/(Y_hat.shape[1])

#https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795

def full_backward_propagation(Y_hat, Y, memory, params_values):
	grads_values = {}
	delta_prev=Y - Y_hat
	m = Y_hat.shape[1]
	for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
		layer_idx_curr = layer_idx_prev + 1

		activation=layer["activation"]

		if activation is "relu":
			backward_activation_func = relu_backward
		elif activation is "sigmoid":
			backward_activation_func = sigmoid_backward
		elif activation is "tanh":
			backward_activation_func = tanh_backward
		elif activation is "softplus":
			backward_activation_func = softplus_backward
		else:
			backward_activation_func = elu_backward

		acti_prev=memory["A" + str(layer_idx_prev)]
		#print (acti_prev.shape)
		#print (delta_prev.shape)

		if (layer_idx_prev < len(nn_architecture)-1):
			derivative=memory["Z" + str(layer_idx_curr)]
			#print ("hi")
			#print (derivative.shape)
			grads_values["dW" + str(layer_idx_curr)] = -np.dot(backward_activation_func(delta_prev,derivative),acti_prev.T)/m
			grads_values["db" + str(layer_idx_curr)] = -np.sum(backward_activation_func(delta_prev,derivative), axis=1, keepdims=True) / m
		else:
			grads_values["dW" + str(layer_idx_curr)] = -np.dot(delta_prev,acti_prev.T)/m
			grads_values["db" + str(layer_idx_curr)] = -np.sum(delta_prev, axis=1, keepdims=True) / m

		w_prev=params_values["W" + str(layer_idx_curr)]
		#print ("u")
		#print (w_prev.shape)
		delta_prev=np.dot(w_prev.T,delta_prev)

	return grads_values

def train():
	#X, Y, epochs, learning_rate
    global confusion_matrix
    learning_rate=0.001
    params_values = init_layers(508)
    cost_history = []
    accuracy_history = []
    in1=open('/home/pranav/Documents/cs7015/assignment_1/A1_Single_label_image_classification_dataset_features/image_data_feat_dim60.txt','r')
    in2=open('/home/pranav/Documents/cs7015/assignment_1/A1_Single_label_image_classification_dataset_features/image_data_labels.txt','r')
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

    print (Y.shape)
    print (X.shape)
    
    prev_cost=100000000000
    diff=10
    num_epoch=0
    avg_cost=0
    avg_acc=0

    while(diff > 0.000005):
    #for i in range(1):
    	for i in range(5):
    		for j in range(5):
    			confusion_matrix[i][j]=0
    	num_epoch=num_epoch+1
    	if (nn_mode == "batch"):
        	Y_hat, cashe = full_forward_propagation(X, params_values)
        	#print (Y_hat.shape)
        	#print (Y_hat.shape)
        	cost = cross_entropy_error(Y_hat, Y)

        	print (cost)

        	accuracy = get_accuracy_value(Y_hat, Y)

        	print (accuracy)
        		        
        	grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values)
        	params_values = update_rule(params_values, grads_values, learning_rate)

        	diff=prev_cost - cost

        	prev_cost=cost

        	if (num_epoch > 10000):
        		break

    	else:
    		diff=0
    		avg_acc=0
    		for i in range(X.shape[1]):
    			l=[]
    			l1=[]
    			l.append(X[:,i])
    			l=np.array(l).T
    			l1.append(Y[:,i])
    			l1=np.array(l1).T
    			Y_hat, cashe = full_forward_propagation(l, params_values)
        		#print (Y_hat.shape)
        		#print (Y_hat.shape)
    			#print (l1)
    			#print (Y_hat)
    			#print (l1)
    			#print ("hu")

    			#diff=diff + prev_cost - cost

    			#print (cost)

    			#avg_cost=avg_cost + cost

    			#prev_cost=cost

		    	#avg_acc=avg_acc + accuracy

    			#print (accuracy)
        		        
    			grads_values = full_backward_propagation(Y_hat, l1, cashe, params_values)
    			params_values = update_rule(params_values, grads_values, learning_rate)

    		print (diff)
    		print ("kj")
    		Y_hat, cashe = full_forward_propagation(X, params_values)
    		accuracy = get_accuracy_value(Y_hat, Y)
    		cost = cross_entropy_error(Y_hat, Y)
    		diff=prev_cost - cost
    		prev_cost=cost
    		print ("cost")
    		print (cost)
    		print (accuracy)
    		#print (avg_cost/(num_epoch*X.shape[1]))
    		#print (avg_acc/(X.shape[1]))
    print (confusion_matrix)
    for i in range(5):
    	teh=0
    	for j in range(5):
    		teh=teh+confusion_matrix[i][j]
    	for j in range(5):
    		confusion_matrix[i][j]=confusion_matrix[i][j]/teh
    print (confusion_matrix)
    print (num_epoch)

    for i in range(5):
    	for j in range(5):
    		confusion_matrix[i][j]=0

    #print (params_values)
    Y_hat, cashe = full_forward_propagation(X_test, params_values)
    get_accuracy_value(Y_hat, Y_test)

    for i in range(5):
    	teh=0
    	for j in range(5):
    		teh=teh+confusion_matrix[i][j]
    	for j in range(5):
    		confusion_matrix[i][j]=confusion_matrix[i][j]/teh
    #print (X_test)
    #print (Y_test)

    print (confusion_matrix)
    #print (Y_hat)
    #print (c1)
    #print (c2)
    #print (c3)

train()
#y=np.array([[1,0]])
#z=np.array([[0.5,0.5]])
#cross_entropy_error(z.T,y.T)