#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 19:25:23 2017

@author: eric
"""

import numpy as np
import matplotlib.pyplot as plt
import ada_grad

np.random.seed(1)

I = 2 # dimensionality of input
h = 4 # hidden layer size
K = 3 # number of classes

N=400



#add some hint and noise
X1 = np.random.randn(N, I) + np.array([0, -2])
X2 = np.random.randn(N, I) + np.array([2, 2])
X3 = np.random.randn(N, I) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Y = np.array([0]*N + [1]*N + [2]*N)
#encode vector
T = np.zeros((Y.shape[0], K))
for i in range(0,Y.shape[0]):
    T[i, Y[i]] = 1




def sigmoid(z):
    g=1/(np.exp(-z)+1)
    return g



def acurracy(T,pred):
    n_correct = 0
    #encode back:
    for i in range(0,T.shape[0]):
        if np.all(T[i] == pred[i]):
            n_correct += 1
    return np.around((n_correct/T.shape[0]),decimals=5)


def cost(T, pred):
    tot = T * np.log(pred)
    return tot.sum()


def forward_compute(X0, W1, b1, W2, b2):
    A1 = 2*sigmoid(2*(np.dot(X0,W1)+b1))-1#first layer is tanh
    X1= np.dot(A1,W2) + b2        
    expX1 = np.exp(X1)
    A2 = expX1 / expX1.sum(axis=1, keepdims=True) #second layer is sofmax
    return A2,A1




np.random.seed(1)

class NNet_2D():
    def __init__(self,eta,D0,D1,D2,l2reg=0,maxiter=1000,rand=1,converge=0):
        self.eta=eta
        self.l2reg=l2reg
        self.rand=rand
        self.iter=maxiter
        self.D0=D0
        self.D1=D1
        self.Nclass=D2
        self.converge=converge
        
    def fit(self,X,Y,method=None):
        self.cost_record=[]
        W1 = np.random.randn(self.D0, self.D1)*2*self.rand-self.rand
        b1 = np.random.randn(self.D1)*2*self.rand-self.rand
        W2 = np.random.randn(self.D1, self.Nclass)*2*self.rand-self.rand
        b2 = np.random.randn(self.Nclass)*2*self.rand-self.rand
        if method=="ada" :
           ada_1_w=np.zeros((W1.shape[0],W1.shape[1]))
           ada_2_w=np.zeros((W2.shape[0],W2.shape[1]))
           ada_1_b=np.zeros((b1.shape[0]))
           ada_2_b=np.zeros((b2.shape[0]))
        for i in range(0,self.iter):
            output_layer,hidden_layer=forward_compute(X, W1, b1, W2, b2)
            if i%100==0 :
                cost_result=cost(Y,output_layer)
                unencode=np.argmax(output_layer, axis=1)
                encode_pred=np.zeros((unencode.shape[0],self.Nclass))
                for i in range(0,unencode.shape[0]):
                    encode_pred[i, unencode[i]] = 1
                acurracy_result=acurracy(T,encode_pred)
                print("accuarcy:",acurracy_result,"cost:",cost_result)
                self.cost_record.append(cost(Y,output_layer))
              
            delta2=np.zeros((output_layer.shape[0],output_layer.shape[1]))
            #loss function derivative
            """
            for i in range(0,delta2.shape[0]):
                for k in range(0,delta2.shape[1]):
                    delta2[i,k]=-(output_layer[i,k]-Y[i,k])
            """
            delta2=-(output_layer-Y)
            
            """
            delta1=np.zeros((hidden_layer.shape[0],W2.T.shape[1]))
            
            
            for i in range(0,hidden_layer.shape[0]):
                for k in range(0,W2.T.shape[1]):
                   for j in range(0,W2.T.shape[0]):
                         delta1[i,k]+=(delta2[i,j]*W2.T[j,k])
                         delta1[i,k]=delta1[i,k]*(1-hidden_layer[i,k]**2)
            """
            delta1=np.dot(delta2,W2.T)*(1-hidden_layer**2)   
        
            
            #use Ascent ,because is negative maxlikelihood as cost funtion
            """
            gradient1=np.zeros((W1.shape[0],W1.shape[1]))
            #gradient2=np.zeros
            for i in range(0,W1.shape[0]):
                for j in range(0,W1.shape[1]):
                    for k in range(0,X.T.shape[1]):
                        gradient1[i,j]+=X.T[i,k]*delta1[k,j]
                        
            gradient2=np.zeros((W2.shape[0],W2.shape[1]))
            for i in range(0,W2.shape[0]):
                for j in range(0,W2.shape[1]):
                    for k in range(0,hidden_layer.T.shape[1]):
                        gradient2[i,j]+=hidden_layer.T[i,k]*delta2[k,j]
            
            
            W1=W1+self.eta*((gradient1)/X.shape[0]-self.l2reg*W1)
            W2=W2+self.eta*((gradient2)/hidden_layer.shape[0]-self.l2reg*W2)
            """
            grad_w1=self.eta*(np.dot(X.T,delta1)/X.shape[0]-self.l2reg*W1)
            grad_w2=self.eta*(np.dot(hidden_layer.T,delta2)/hidden_layer.shape[0]-self.l2reg*W2)
            
            grad_b1=self.eta*delta1.sum(axis=0)/X.shape[0]
            grad_b2=self.eta*delta2.sum(axis=0)/hidden_layer.shape[0]
            if method==None :
                W1=W1+grad_w1
                W2=W2+grad_w2
                b1=b1+grad_b1
                b2=b2+grad_b2
                
                
            elif method=="ada" :
                d_w1,ada_1_w=ada_grad.compute(self.eta,ada_1_w,grad_w1)
                d_w2,ada_2_w=ada_grad.compute(self.eta,ada_2_w,grad_w2)
                
                d_b1,ada_1_b=ada_grad.compute(self.eta,ada_1_b,grad_b1)
                d_b2,ada_2_b=ada_grad.compute(self.eta,ada_2_b,grad_b2)
                
                W1=W1+d_w1
                W2=W2+d_w2
                b1=b1+d_b1
                b2=b2+d_b2
                
            
                
            
            if self.converge==0:
               pass
            else:
                if i /self.iter > self.converge:
                  self.eta=self.eta*0.99
                
        self.w1=W1
        self.w2=W2
        
        self.b1=b1
        self.b2=b2
        
        return self
    
    def plot(self):
        plt.clf()
        plt.plot(self.cost_record)
        plt.show()
       
        return self
    
    def predict(self,test_X):
        output_layer,hidden_layer=forward_compute(test_X, self.w1, self.b1,self.w2, self.b2)
         
        predict_y = np.zeros((output_layer.shape[0], 1))
        for i in range(0,output_layer.shape[0]):
            predict_y[i]=np.argmax(output_layer[i])     
        return predict_y



ANN=NNet_2D(0.05,I,h,K,l2reg=0.005,maxiter=5000,rand=5)
ANN.fit(X,T)
ANN.plot()
