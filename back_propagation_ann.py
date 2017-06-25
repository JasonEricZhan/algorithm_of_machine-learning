#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 19:25:23 2017

@author: eric
"""

import numpy as np
import matplotlib.pyplot as plt



I = 2 # dimensionality of input
h = 4 # hidden layer size
K = 3 # number of classes

N=400


np.random.seed(1)

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
    A1 = 2*sigmoid(2*(np.dot(X,W1)+b1))-1#first layer is tanh
    X1= np.dot(A1,W2) + b2        
    expX1 = np.exp(X1)
    A2 = expX1 / expX1.sum(axis=1, keepdims=True) #second layer is sofmax
    return A2,A1

class NNet_2D():
    def __init__(self,eta,l2reg=0,D0,D1,D2,maxiter=1000):
        self.eta=eta
        self.l2reg=l2reg
        self.iter=maxiter
        self.D0=D0
        self.D1=D1
        self.Nclass=D2
    def fit(self,X,Y):
        self.cost_record=[]
        W1 = np.random.randn(self.D0, self.D1)
        b1 = np.random.randn(self.D1)
        W2 = np.random.randn(self.D1, self.Nclass)
        b2 = np.random.randn(self.Nclass)
        for i in range(0,self.iter):
            output_layer,hidden_layer=forward_compute(X, W1, b1, W2, b2)
            if(self.iter%100==0 ):
                cost_result=cost(Y,output_layer)
                unencode=np.argmax(output_layer, axis=1)
                encode_pred=np.zeros((unencode.shape[0],self.Nclass))
                for i in range(0,unencode.shape[0]):
                    encode_pred[i, unencode[i]] = 1
                acurracy_result=acurracy(T,encode_pred)
                print("accuarcy:",acurracy_result,"cost:",cost_result)
                self.cost_record.append(cost(Y,output_layer))
            """ 
            delta2=np.zeros((output_layer.shape[0],output_layer.shape[1]))
            
            for i in range(0,delta2.shape[0]):
                for k in range(0,delta2.shape[1]):
                    #loss function derivative
                    delta2[i,k]=-(output_layer[i,k]-Y[i,k])
            """
            delta2=-(output_layer-Y)
            delta1=np.dot(delta2,W2.T)*(1-hidden_layer**2)
            
        
            
            #use Acent ,because is maxlikelihood as cost funtion
            W1=W1+self.eta*(np.dot(X.T,delta1)+self.l2reg*W1)
            W2=W2+self.eta*(np.dot(hidden_layer.T,delta2)+self.l2reg*W2)
            
            b2_delta2=delta2.sum(axis=0)
            b1_delta1=delta1.sum(axis=0)
            b1=b1+self.eta*b1_delta1
            b2=b2+self.eta*b2_delta2
            
        plt.clf()
        plt.plot(self.cost_record)
        plt.show()
        
        return self

ANN=NNet_2D(10e-5,l2reg=0.0001,I,h,K,maxiter=5000)
ANN.fit(X,T)

