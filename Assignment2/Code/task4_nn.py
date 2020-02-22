import pandas as pd
import tensorflow as tf
import numpy as np
from task4_rbm import *
np.random.seed(80)
def get_data():
	csvfile=pd.read_csv("data.csv")
	data=csvfile.values
	X=data[:,:784]
	Y=data[:,784:785]
	return X,Y

def split_data(X,Y):
	l=[]
	for i in range(len(X)):
		l.append(i)
	l=np.array(l)
	np.random.shuffle(l)
	X_train=[]
	X_test=[]
	Y_train=[]
	Y_test=[]
	X_ptrain=[]
	l_train=[]
	l_test=[]
	for i in range(len(l)):
		if i<0.56*len(X):
			X_ptrain.append(X[l[i]])
		elif i<0.8*len(X):
			X_train.append(X[l[i]])
			temp=np.array([0,0,0,0,0])
			temp[Y[l[i]][0]]=1
			Y_train.append(temp)
			l_train.append(Y[l[i]][0])
		else:
			X_test.append(X[l[i]])
			temp=np.array([0,0,0,0,0])
			temp[Y[l[i]][0]]=1
			Y_test.append(temp)
			l_test.append(Y[l[i]][0])
	return np.array(X_train),np.array(X_test),np.array(Y_train),np.array(Y_test),np.array(X_ptrain),np.array(l_train),np.array(l_test)


X,Y=get_data()
X_train,X_test,Y_train,Y_test,X_ptrain,l_train,l_test=split_data(X,Y)
batch_size=50
batch_num=35000//batch_size
l=[]
for i in range(len(X_train)):
	l.append(i)
l=np.array(l)
np.random.shuffle(l)
X_batches=[]
Y_batches=[]
index=0
x_batch=np.empty([batch_size,784])
y_batch=np.empty([batch_size,5])
for i in range(batch_num):
	if index<batch_size:
		x_batch[index]=X_train[l[i]]
		y_batch[index]=Y_train[l[i]]
		index=index+1
	else:
		index=0
		X_batches.append(x_batch)
		Y_batches.append(y_batch)
		x_batch=np.empty([batch_size,784])
		y_batch=np.empty([batch_size,5])


x=tf.placeholder(dtype=tf.float32,shape=(None,784))
y=tf.placeholder(dtype=tf.float32,shape=(None,5))
	
#obj1=RBM(n_visible=784,n_hidden=500)
#w1=obj1.train_rbm(X_ptrain)
#data2=obj1.gen_data(X_ptrain)
#h1=tf.layers.dense(inputs=x,units=500,activation=tf.nn.relu,kernel_initializer=tf.constant_initializer(w1))
h1=tf.layers.dense(inputs=x,units=500,activation=tf.nn.relu)

#obj2=RBM(n_visible=500,n_hidden=200)
#w2=obj2.train_rbm(data2)
#data3=obj2.gen_data(data2)
#h2=tf.layers.dense(inputs=h1,units=200,activation=tf.nn.relu,kernel_initializer=tf.constant_initializer(w2))
h2=tf.layers.dense(inputs=h1,units=200,activation=tf.nn.relu)

#obj3=RBM(n_visible=200,n_hidden=30)
#w3=obj3.train_rbm(data3)
#h3=tf.layers.dense(inputs=h2,units=30,activation=tf.nn.relu,kernel_initializer=tf.constant_initializer(w3))
h3=tf.layers.dense(inputs=h2,units=50,activation=tf.nn.relu)

	
y_hat=tf.layers.dense(inputs=h3,units=5,activation=None)
indices=tf.argmax(y_hat,axis=1)
label=tf.placeholder(dtype=tf.int32,shape=(None))
conf=tf.math.confusion_matrix(labels=label,predictions=indices)

loss=tf.losses.softmax_cross_entropy(onehot_labels=y,logits=y_hat)

optimiser=tf.train.AdamOptimizer().minimize(loss)

epochs=0
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#sess.run(w1)
	#sess.run(data2)
	#sess.run(w2)
	#sess.run(data3)
	#sess.run(w3)
	while 1:
		epochs=epochs+1
		old_loss=sess.run(loss,feed_dict={x: X_train,y: Y_train})
		for i in range(len(X_batches)):
			sess.run(optimiser,feed_dict={x: X_batches[i],y: Y_batches[i]})
		new_loss=sess.run(loss,feed_dict={x: X_train,y: Y_train})
		if abs(new_loss-old_loss)<0.0001:
			break
	print(epochs," ",new_loss)
	print(sess.run(conf,feed_dict={x: X_train,y: Y_train,label: l_train}))
	print(sess.run(conf,feed_dict={x: X_test,y: Y_test,label: l_test}))
