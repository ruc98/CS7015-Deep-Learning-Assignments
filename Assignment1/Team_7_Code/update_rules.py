import queue
import numpy as np
import math
pre_params={}
alpha=0.2
uw={}
vw={}
epsilon=0.00001
ro=0.3
rw={}
num=0
tep=0
l=10
set_of_queue={}
set_of_queue_dw={}
track_l=queue.Queue(maxsize=l+1)
import queue

def update_delta(params_values, grads_values, learning_rate,nn_architecture):
	for idx, layer in enumerate(nn_architecture):
		layer_idx=idx+1
		params_values["W" + str(layer_idx)] += learning_rate * grads_values["dW" + str(layer_idx)]        
		params_values["b" + str(layer_idx)] += learning_rate * grads_values["db" + str(layer_idx)]
		#print(learning_rate * grads_values["dW" + str(layer_idx)] )

	return params_values;

def update_gen_delta(params_values, grads_values, learning_rate,nn_architecture):
    global pre_params
    global alpha
    for idx, layer in enumerate(nn_architecture):
        layer_idx=idx+1
        params_values["W" + str(layer_idx)] += learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] += learning_rate * grads_values["db" + str(layer_idx)]

        params_values["W" + str(layer_idx)] += alpha*pre_params["W" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] += alpha*pre_params["b" + str(layer_idx)]

        pre_params["W" + str(layer_idx)] = learning_rate * grads_values["dW" + str(layer_idx)]        
        pre_params["b" + str(layer_idx)] = learning_rate * grads_values["db" + str(layer_idx)]

    return params_values;

def update_adagrad(params_values, grads_values, learning_rate,nn_architecture):
    global rw
    #global epsilon
    epsilon = 0.1
    learning_rate=0.00005
    for idx, layer in enumerate(nn_architecture):
        layer_idx=idx+1
        params_values["W" + str(layer_idx)] += (learning_rate * grads_values["dW" + str(layer_idx)])/(epsilon+rw["W" + str(layer_idx)])
        params_values["b" + str(layer_idx)] += (learning_rate * grads_values["db" + str(layer_idx)])/(epsilon+rw["b" + str(layer_idx)])

        rw["W" + str(layer_idx)] = np.sqrt(rw["W" + str(layer_idx)]**2+grads_values["dW" + str(layer_idx)]**2)    
        rw["b" + str(layer_idx)] = np.sqrt(rw["b" + str(layer_idx)]**2+grads_values["db" + str(layer_idx)]**2)

    return params_values;

def update_rmsprop(params_values, grads_values, learning_rate,nn_architecture):
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

		params_values["W" + str(layer_idx)] += learning_rate * grads_values["dW" + str(layer_idx)]/(epsilon+re1)
		params_values["b" + str(layer_idx)] += learning_rate * grads_values["db" + str(layer_idx)]/(epsilon+re2)

		if (set_of_queue["W" + str(layer_idx)].qsize() >= l):
			set_of_queue["W" + str(layer_idx)].get()
			set_of_queue["b" + str(layer_idx)].get()
		set_of_queue["W" + str(layer_idx)].put(grads_values["dW" + str(layer_idx)])
		set_of_queue["b" + str(layer_idx)].put(grads_values["db" + str(layer_idx)])

	num=num+1;

	return params_values;

def update_adadelta(params_values, grads_values, learning_rate,nn_architecture):
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

		params_values["W" + str(layer_idx)] += te3
		params_values["b" + str(layer_idx)] += te4

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

def update_adam(params_values, grads_values, learning_rate,nn_architecture):
    global uw
    global vw
    global m_adam
    ro1=0.9
    ro2=0.99
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
    	params_values["W" + str(layer_idx)] += (learning_rate * uw_hat1)/(epsilon+vw_hat1)
    	params_values["b" + str(layer_idx)] += (learning_rate * uw_hat2)/(epsilon+vw_hat2)

    return params_values;

update_rule=update_adam
#takes: delta,gen_delta,adagrad,rmsprop,adadelta,adam

def init_layers(nn_architecture,seed=99):
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
			layer_output_size, layer_input_size) * math.sqrt(1/layer_input_size)
		params_values["b" + str(layer_idx)] = np.random.randn(
			layer_output_size) * 0.1

		if (update_rule == update_gen_delta):
			pre_params['W' + str(layer_idx)] = np.random.randn(
				layer_output_size, layer_input_size) * 0
			pre_params['b' + str(layer_idx)] = np.random.randn(
				layer_output_size) * 0

		if (update_rule == update_adagrad):
			rw['W' + str(layer_idx)] = np.random.randn(
				layer_output_size, layer_input_size) * 0
			rw['b' + str(layer_idx)] = np.random.randn(
				layer_output_size) * 0

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
			uw['b' + str(layer_idx)]=np.random.randn(layer_output_size) * 0
			vw['b' + str(layer_idx)]=np.random.randn(layer_output_size) * 0

	return params_values