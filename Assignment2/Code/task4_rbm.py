import tensorflow as tf
import numpy as np
import math
class RBM:
    def __init__(self,n_visible=784,n_hidden=500,k=30):
        np.random.seed(80)
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        self.k=k
        self.learning_rate=0.001
        self.w=np.random.randn(n_visible,n_hidden)
        self.b=np.random.randn(1,n_visible)
        self.c=np.random.randn(1,n_hidden)

    def prop_h(self,visible):
        sig_acti=np.dot(visible,self.w)+np.dot(np.ones((visible.shape[0],1)),self.c)
        sig_acti=1/(1+np.exp(-np.maximum(sig_acti,-100*np.ones(sig_acti.shape))))
        return sig_acti

    def prop_v(self,hidden):
        sig_acti=np.dot(hidden,self.w.T)+np.dot(np.ones((hidden.shape[0],1)),self.b)
        sig_acti=1/(1+np.exp(-np.maximum(sig_acti,-100*np.ones(sig_acti.shape))))
        return sig_acti

    def sample_h(self,v_sample):
        h_prop=self.prop_h(v_sample)
        h_prop=(np.random.random((h_prop.shape))<h_prop).astype(np.int)
        return h_prop

    def sample_v(self,h_sample):
        v_prop=self.prop_v(h_sample)
        v_prop=(np.random.random((v_prop.shape))<v_prop).astype(np.int)
        return v_prop

    def CD(self,visibles):
        v_sample=visibles
        for i in range(self.k):
            h_sample=self.sample_h(v_sample)
            v_sample=self.sample_v(h_sample)
        h0_prop=self.prop_h(visibles)
        hk_prop=self.prop_h(v_sample)
        w_pos_grad=np.dot(visibles.T,h0_prop)
        w_neg_grad=np.dot(v_sample.T,hk_prop)
        w_grad=(w_pos_grad-w_neg_grad)/visibles.shape[0]
        b_grad=np.sum(visibles-v_sample,axis=0)/visibles.shape[0]
        c_grad=np.sum(h0_prop-hk_prop,axis=0)/visibles.shape[0]
        return w_grad,b_grad,c_grad

    def learn(self,visibles):
        w_grad,b_grad,c_grad=self.CD(visibles)
        self.w=self.w+self.learning_rate*w_grad
        self.b=self.b+self.learning_rate*b_grad
        self.c=self.c+self.learning_rate*c_grad

    def train_rbm(self,X):
        for epochs in range(20):
            self.learn(X)
        return self.w

    def gen_data(self,X):
        return self.sample_h(X)