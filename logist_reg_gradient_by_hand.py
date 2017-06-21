#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:41:10 2017

@author: eric
"""
import numpy as np
import matplotlib.pyplot as plt




d1 = 100
d2 = 2


X = np.random.randn(d1,d2)

# center the first 50 points at (-5,-5)
X[:50,:] = X[:50,:] - 5*np.ones((50,d2))

# center the last 50 points at (5, 5)
X[50:,:] = X[50:,:] + 5*np.ones((50,d2))

Y_an=np.array([0]*50 + [1]*50)

def sigmoid(z):
    g=1/(np.exp(-z)+1)
    return g

#error measure
def cross_entropy(x,w,y,l2reg):
    loss=0
    row=np.shape(y)[0]
    
    for i in range(0,row):
      if y[i]==1:
          loss=loss-np.log(sigmoid(np.dot(x[i],w)))
      else:
          loss=loss-np.log(sigmoid(1-np.dot(x[i],w)))
    
    loss=loss/row+(l2reg/row*2)*np.dot(w,w)
    return loss

    
def gradient(x,w,y,l2reg):
    gradient_error=0
    row=np.shape(x)[0]
    
    for i in range(0,row):
        pred=sigmoid(np.dot(x[i],w))
        gradient_error=gradient_error+(pred-y[i])*x[i]+(l2reg/row)*w
    
    gradient_error=gradient_error/row
    return gradient_error
 

class logistic_regression(object):
      def __init__(self,eta,l2reg=0,maxiter=1000):
          self.eta=eta
          self.iter=maxiter
          self.l2reg=l2reg #with l2 regularizer ,lambda coefficient in lagrange  mutiplier
          
      def fit(self,X,Y):
          ones = np.ones((np.shape(X)[0], 1))
          X= np.concatenate((ones, X), axis=1)
          
          #another way to initailize by produce random number comply to N(0,1) distribution
          #w = np.random.randn(np.shape(X)[1]+ 1)
            
          #set all weight to zero
          w=np.zeros(np.shape(X)[1])
          iter_last=self.iter
          costs_record = []
          
          w=w-self.eta* gradient(X,w,Y,self.l2reg)
          for i in range(0,self.iter):
              costs_record.append(cross_entropy(X,w,Y,self.l2reg))
              if np.all(gradient(X,w,Y,self.l2reg) ==0):
                 iter_last=i
                 break
              w=w-self.eta*gradient(X,w,Y,self.l2reg)
          #show the process of gradient descent
          plt.clf()
          plt.plot(range(0,iter_last), costs_record)
          plt.show()
          self.w=w
          return self
          
      def predict(self,X):
          ones = np.ones((np.shape(X)[0], 1))
          X= np.concatenate((ones, X), axis=1)
          row=np.shape(X)[0]
          predict_y=[]
          for i in range(0,row):
              if sigmoid(np.dot(X[i],self.w))>0.5:
                 answer=1
              else:
                 answer=0
              predict_y.append(answer)
          predict_y=np.array(predict_y)
          return predict_y

model=logistic_regression(0.01,l2reg=5,maxiter=100)
model.fit(X,Y_an)

plt.clf()
plt.scatter(X[:,0], X[:,1],c=Y_an,s=100, alpha=0.5)


ones = np.ones((np.shape(X)[0], 1))
X_intercept= np.concatenate((ones, X), axis=1)
x_aix=X_intercept.dot(model.w)
y_axis=-x_aix
plt.plot(x_aix, y_axis)
plt.show()

#harder data to be linear separable

X=np.array([[-0.4, 0.3],
          [-0.3, -0.1],
          [-0.2, 0.4],
          [-0.1, 0.1],
          [0.6, -0.5],
          [0.8, 0.7],
          [0.9, -0.5],
          [0.7, -0.9],
          [0.8, 0.2],
          [0.4, -0.6]])
    
Y_an= np.array([0]*5 + [1]*5)



model=logistic_regression(0.05,l2reg=0.05,maxiter=500)
model.fit(X,Y_an)

plt.clf()
plt.scatter(X[:,0], X[:,1],c=Y_an,s=100, alpha=0.5)

ones = np.ones((np.shape(X)[0], 1))
X_intercept= np.concatenate((ones, X), axis=1)
x_aix=X_intercept.dot(model.w)
y_axis=-x_aix
plt.plot(x_aix, y_axis)
plt.show()

