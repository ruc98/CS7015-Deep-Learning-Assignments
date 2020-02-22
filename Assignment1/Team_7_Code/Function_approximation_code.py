import numpy as np
import os
import math
import queue
import matplotlib.pyplot as plt
import sys

### global variables ###
m_adam=0
uw = {}
vw = {}
pre_params = {}
rw = {}
set_of_queue = {}
set_of_queue_dw = {}
l = 32
ro = 0.9

### initialization ###
def init_layers(seed=0):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        if (update_rule_fn == "gen_delta"):
            pre_params['W' + str(layer_idx)] = np.random.randn(
                    layer_output_size, layer_input_size) * 0
            pre_params['b' + str(layer_idx)] = np.random.randn(
                    layer_output_size, 1) * 0

        if (update_rule_fn == "adagrad"):
            rw['W' + str(layer_idx)] = np.random.randn(
                    layer_output_size, layer_input_size) * 0
            rw['b' + str(layer_idx)] = np.random.randn(
                    layer_output_size, 1) * 0

        if (update_rule_fn == "rmsprop"):
            global l
            set_of_queue['W' + str(layer_idx)]=queue.Queue(maxsize=l+1)
            set_of_queue['b' + str(layer_idx)]=queue.Queue(maxsize=l+1)

        if (update_rule_fn == "adadelta"):
            set_of_queue['W' + str(layer_idx)]=queue.Queue(maxsize=l+1)
            set_of_queue['b' + str(layer_idx)]=queue.Queue(maxsize=l+1)

            set_of_queue_dw['W' + str(layer_idx)]=queue.Queue(maxsize=l+1)
            set_of_queue_dw['b' + str(layer_idx)]=queue.Queue(maxsize=l+1)	

        if (update_rule_fn == "adam"):
            global uw
            global vw
            uw['W' + str(layer_idx)]=np.random.randn(layer_output_size, layer_input_size) * 0
            vw['W' + str(layer_idx)]=np.random.randn(layer_output_size, layer_input_size) * 0
            uw['b' + str(layer_idx)]=np.random.randn(layer_output_size, 1) * 0
            vw['b' + str(layer_idx)]=np.random.randn(layer_output_size, 1) * 0

    return params_values

### activation functions ###
def sigmoid(Z):
    return 1/(1+np.exp(-slope*Z))




def relu(Z):
    return np.maximum(0, Z)


##def elu(Z):
##    delta = 1
##    if (Z > 0):
##        return Z
##    else:
##        return delta*(np.exp(Z)-1)

def elu(Z):
    delta = 1
    Z1 = np.where(np.greater(Z,0), Z, delta*(np.exp(Z)-1))
    return Z1

##def elu(Z):
##        s=[]
##        delta=1;
##        s=np.maximum(0,Z)
##        s1=np.minimum(0,Z)
##        s1=delta*(np.exp(s1)-1)
##        return s+s1

def tanh(Z):
	return (np.exp(2*slope*Z)-1)/(np.exp(2*slope*Z)+1)



def softplus(Z):
	return np.log(1+np.exp(Z))

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig) * slope

def relu_backward(dA, Z):
	dZ = np.array(dA, copy = True)
	dZ[Z <= 0] = 0
	return dZ

def tanh_backward(dA, Z):
	tah=tanh(Z)
	return dA*(1-tah)*(1+tah)*slope

def elu_backward(dA, Z):
    delta = 1
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ
##def elu_backward(dA, Z):
##	delta=1;
##	l1=[]
##        for i in range(Z.shape[0]):
##            if Z>0:
##                    l1.append(1)
##            else:
##                    l1.append(delta*np.exp(Z[i]))
##        l1.append(l)
##	l1=np.array(l1).T
##
##	return dA * l1

def softplus_backward(dA, Z):
	s1=[]
	for k in range(Z.shape[1]):
		s=[]
		for i in range(Z.shape[0]):
			s.append(1/(1+math.exp(-Z[i][k])))
		s1.append(s)
	s1=np.array(s1).T
	return dA*s1


### forward propagation ###

def full_forward_propagation(X, params_values):
    memory = {}
    A_curr = X
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        activation = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        Z_curr = np.dot(W_curr, A_prev.T).T + b_curr.T    # Z_curr means activation value, a_j in notes
        if activation is "relu":
            activation_func = relu
        elif activation is "sigmoid":
            activation_func = sigmoid
        elif activation is "tanh":
            activation_func = tanh
        elif activation is "softplus":
            activation_func = softplus
        else:
            activation_func = elu
        A_curr = activation_func(Z_curr)            # A_curr means output of node,  s_j in notes

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    return A_curr, memory

### back propagation ###

def full_backward_propagation(Y_hat, Y, memory, params_values):
    grads_values = {}
    m = Y.shape[0]              # number of classes
    Y = Y.reshape(Y_hat.shape)

    #dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    dA_prev = sse_derivative(Y_hat, Y)

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activation = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        m = A_prev.shape[0]

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
                                                                # Z_curr means a_j,  activation value
                                                                # A_curr means s_j in notes,  output of node
                                                                # dZ_curr means delta
                                                                # dA_curr means sumation delta*w of next layer
                                                                # curr means l layer
                                                                # prev means i layer

        dZ_curr = backward_activation_func(dA_curr, Z_curr)     # for eqn. dW_il = eta * delta_l * s_i,
        dW_curr = np.dot(dZ_curr.T, A_prev) / m                 # dW_curr = dW_il   
        db_curr = np.sum(dZ_curr, axis=0, keepdims=True).T / m    # A_prev = s_i
        dA_prev = np.dot(W_curr.T, dZ_curr.T).T                     # dZ_curr = delta_l
                                                                # dA_curr = sumation delta_j * w_j
                                                                # backward(Z_curr) = d phi_l / d a_l
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
        
    return grads_values

### update rules ###

def update_delta(params_values, grads_values, learning_rate):
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values


def update_gen_delta(params_values, grads_values, learning_rate):
    alpha = 0.9
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

        params_values["W" + str(layer_idx)] += alpha*pre_params["W" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] += alpha*pre_params["b" + str(layer_idx)]

        pre_params["W" + str(layer_idx)] = learning_rate * grads_values["dW" + str(layer_idx)]        
        pre_params["b" + str(layer_idx)] = learning_rate * grads_values["db" + str(layer_idx)]

    return params_values;

def update_adagrad(params_values, grads_values, learning_rate):
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        rw["W" + str(layer_idx)] = np.sqrt(rw["W" + str(layer_idx)]*rw["W" + str(layer_idx)]+grads_values["dW" + str(layer_idx)]*grads_values["dW" + str(layer_idx)])     
        rw["b" + str(layer_idx)] = np.sqrt(rw["b" + str(layer_idx)]*rw["b" + str(layer_idx)]+grads_values["db" + str(layer_idx)]*grads_values["db" + str(layer_idx)])

        params_values["W" + str(layer_idx)] -= (learning_rate * grads_values["dW" + str(layer_idx)])/(epsilon+rw["W" + str(layer_idx)])
        params_values["b" + str(layer_idx)] -= (learning_rate * grads_values["db" + str(layer_idx)])/(epsilon+rw["b" + str(layer_idx)])


    return params_values;

def update_rmsprop(params_values, grads_values, learning_rate):
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
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

    #num=num+1;

    return params_values;

def update_adadelta(params_values, grads_values, learning_rate):
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
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

    #num=num+1;

    return params_values;


def update_adam(params_values, grads_values, learning_rate):
    ro1=0.9
    ro2=0.999
    global uw
    global vw
    global m_adam
    m_adam+=1;
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
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

### errors ###
def cross_entropy_error(Y_hat, Y):
    m = Y_hat.shape[0]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def cross_entropy_derivative(Y_hat, Y):
    return 0


def sse(Y_hat, Y):
    m = Y.shape[0]
    cost = 1/m * np.sum((Y - np.squeeze(Y_hat))**2, axis=0)
    return np.squeeze(cost)

def sse_derivative(Y_hat, Y):
    return -2 * (Y - Y_hat)

### training the neural network ###
### pattern mode ###
def train_pattern(X, Y, learning_rate):
    params_values = init_layers(2)
    train_cost_history = []
    test_cost_history = []
    split = int(X.shape[0] * 0.7)
    Y_train_pred = np.empty((split))
    Y_test_pred = np.empty((X.shape[0] - split))
    epoch = 0

    while (True):
        epoch += 1
        train_cost = 0
        test_cost = 0
        
        for i in range(split):              # train loop
            Xi = np.expand_dims(X[i], axis = 0)
            Yi = np.expand_dims(Y[i], axis = 0)
            Y_hat, cashe = full_forward_propagation(Xi, params_values)
            Y_train_pred[i] = Y_hat
            inst_cost = sse(Y_hat, Yi)
            train_cost += inst_cost
            grads_values = full_backward_propagation(Y_hat, Yi, cashe, params_values)
            params_values = update_rule(params_values, grads_values, learning_rate)
            
        for j in range(split, X.shape[0]):  # test loop
            Xi = np.expand_dims(X[j], axis = 0)
            Yi = np.expand_dims(Y[j], axis = 0)
            Y_hat, cashe2 = full_forward_propagation(Xi, params_values)
            Y_test_pred[j-split] = Y_hat
            inst_cost = sse(Y_hat, Yi)
            test_cost += inst_cost
            
        train_cost /= split  # mean train_cost
        test_cost /= (X.shape[0]-split) # mean test_cost
        if epoch > 1:
            delta_train_cost = train_cost_history[-1] - train_cost
            #print('epoch', epoch, 'delta_train_cost', delta_train_cost)
            if delta_train_cost < convergence:
                break
        if epoch >= 10000:
            print('too long to converge')
            break

        train_cost_history.append(train_cost)
        test_cost_history.append(test_cost)
        #print('epoch', epoch, 'train cost', train_cost, 'test cost', test_cost)
    return epoch, train_cost_history, test_cost_history, Y_train_pred, Y_test_pred

### batch mode ###
def train_batch(X, Y, learning_rate):
    params_values = init_layers(2)
    train_cost_history = []
    test_cost_history = []
    split = int(X.shape[0] * 0.7)
    X_train = X[:split]
    Y_train = Y[:split]
    X_test = X[split:]
    Y_test = Y[split:]
    Y_train_pred = np.empty((Y_train.shape))
    Y_test_pred = np.empty((Y_test.shape))
    
    epoch = 0
    while (True):
        epoch += 1
        Y_hat, cashe = full_forward_propagation(X_train, params_values)
        Y_train_pred = Y_hat
        train_cost = sse(Y_hat, Y_train)
        grads_values = full_backward_propagation(Y_hat, Y_train, cashe, params_values)
        params_values = update_rule(params_values, grads_values, learning_rate)
        
        Y_test_hat, cashe2 = full_forward_propagation(X_test, params_values)
        Y_test_pred = Y_test_hat
        test_cost = sse(Y_test_hat, Y_test)
        #print('epoch', epoch, 'train_cost', train_cost, 'test_cost', test_cost)
        if epoch > 1:
            delta_train_cost = train_cost_history[-1] - train_cost
            #print('epoch', epoch, 'delta_train_cost', delta_train_cost)
            if delta_train_cost < convergence:
                break
        if epoch >= 100000:
            print('too long to converge')
            break
        
        train_cost_history.append(train_cost)
        test_cost_history.append(test_cost)
        #print('epoch', epoch, 'train cost', train_cost, 'test cost', test_cost)

    return epoch, train_cost_history, test_cost_history, Y_train_pred, Y_test_pred

### scatter plot ###
def scatter_plot(Y_train, Y_train_pred, Y_test, Y_test_pred, super_title, i):
    plt.figure(i, figsize = [12,5])
    plt.suptitle(super_title)
    plt.subplot(121)
    plt.scatter(Y_train, Y_train_pred)
    plt.title("Train scatter plot")
    plt.xlabel("True")
    plt.ylabel("Prediction")
    plt.axis([0,1,0,1])
    plt.subplot(122)
    plt.scatter(Y_test, Y_test_pred)
    plt.title("Test scatter plot")
    plt.xlabel("True")
    plt.ylabel("Prediction")
    plt.axis([0,1,0,1])
    super_title = super_title.replace(".","").replace(",","").replace("=","").replace(" ","")
    plt.savefig(super_title +".jpg")

### Cost v/s epoch plot ###
def cost_plot(train_cost_history, test_cost_history, epoch, super_title):
    fig1 = plt.figure(23, figsize = [14,8])
    plt.suptitle(super_title)
    ax1 = plt.subplot(121)
    ax1.plot(np.arange(1,epoch,1), train_cost_history)
    plt.title("Train cost vs epoch")
    plt.xlabel("epoch")
    plt.ylabel("Train cost")


    ax2 = plt.subplot(122)
    ax2.plot(np.arange(1,epoch,1), test_cost_history)
    plt.title("Test cost vs epoch")
    plt.xlabel("epoch")
    plt.ylabel("Test cost")

    return fig1, ax1, ax2


### __Main__ ###

datafile = 'Function_approximation/7/concrete_data.txt'
####################################################################################
### Hyperparameters ###
no_of_hidden_layers =  1
type_of_input       =  "normalized"
#activation_hlayer   =  "sigmoid"
learning_mode       =  "pattern"
update_rule_fn      =  "delta"
slope               =  3
H_node              =  8
convergence = 1e-5
lr = 0.001
####################################################################################

epsilon = 1e-6
if learning_mode is "pattern":
    train = train_pattern
else:
    train = train_batch



data = np.genfromtxt(datafile)
np.random.seed(0)
np.random.shuffle(data)
X = data[:, :-1]
Y = data[:, -1]
#Y = (Y - np.mean(Y)) / (3*np.std(Y))           # to get Y between (-1,1)
Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))   # to get Y between (0,1)

if type_of_input is "normalized":  
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

if update_rule_fn is "delta":
    update_rule = update_delta
elif update_rule_fn is "gen_delta":
    update_rule = update_gen_delta
elif update_rule_fn is "adagrad":
    update_rule = update_adagrad
elif update_rule_fn is "rmsprop":
    update_rule = update_rmsprop
elif update_rule_fn is "adadelta":
    update_rule = update_adadelta
else:
    update_rule = update_adam    


#epochs = 100
epoch_arr = []
list_of_activations = ["sigmoid", "tanh", "relu", "softplus", "elu"]

legend = []
count = 0
for activation_hlayer in list_of_activations :
    if no_of_hidden_layers is 2:
        nn_architecture = [
            {"input_dim": 8,      "output_dim": H_node, "activation": activation_hlayer},
            {"input_dim": H_node, "output_dim": H_node, "activation": activation_hlayer},
            {"input_dim": H_node, "output_dim": 1,      "activation": "sigmoid"},
        ]
    else:
        nn_architecture = [
            {"input_dim": 8,      "output_dim": H_node, "activation": activation_hlayer},
            {"input_dim": H_node, "output_dim": 1,      "activation": "sigmoid"},
        ]


    print(activation_hlayer)
    count+=1

    epoch, train_cost_history, test_cost_history, Y_train_pred, Y_test_pred = train(X, Y, lr)
    print('activation ', activation_hlayer, 'Convergence_epoch ', epoch, 'MSE Train ', train_cost_history[-1], 'MSE Test ', test_cost_history[-1])
    ##epoch_arr.append(epoch)
    ##epoch_np = np.array(epoch_arr)
    ##indx = np.argmin(epoch_np)
    ##print('best H_node config = ', hidden_nodes[indx])
    split = int(X.shape[0] * 0.7)
    Y_train = Y[:split]
    Y_test = Y[split:]

    scatter_title = "Scatter Plot for "+ activation_hlayer +" activation function"
    cost_title = "Comparison of different activation functions"
    legend.append(activation_hlayer)
    #legend.join("H_nodes = "+str(H_node))

    scatter_plot(Y_train, Y_train_pred, Y_test, Y_test_pred, scatter_title, count)
    fig1, ax1, ax2 = cost_plot(train_cost_history, test_cost_history, epoch, cost_title)
ax1.legend(legend)
ax2.legend(legend)
fig1.savefig(cost_title.replace(" ","") + ".jpg")
#plt.show()
