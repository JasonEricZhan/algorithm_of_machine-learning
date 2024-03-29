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

#produce number under N(0,1) 
X = np.random.randn(d1,d2)

# center the first 50 points at (-5,-5)
X[:50,:] = X[:50,:] - 5*np.ones((50,d2))

# center the last 50 points at (5, 5)
X[50:,:] = X[50:,:] + 5*np.ones((50,d2))

Y_an=np.array([0]*50 + [1]*50)

def sigmoid(z):
    g=1/(np.exp(-z)+1)
    return g


def cross_entropy(x,w,y):
    loss=0
    row=np.shape(y)[0]
    
    for i in range(0,row):
      if y[i]==1:
          loss=loss-np.log(sigmoid(np.dot(x[i],w)))
      else:
          loss=loss-np.log(sigmoid(1-np.dot(x[i],w)))
    
    loss=loss/row
    return loss

    
def gradient(x,w,y,l2reg,size,count):
    gradient_error=0
    row=np.shape(x)[0]
    
    if(size==row): # batch
        for i in range(0,row):
             pred=sigmoid(np.dot(x[i],w))
             gradient_error=gradient_error-(pred-y[i])*x[i]+(l2reg/row)*w
        gradient_error=gradient_error/row
    
    else:  # minibatch
        range_array=np.arange(0,row,size)
        for i in range(range_array[count],size+range_array[count]):
            if i >= row:
                size=(i-range_array[count])+1
                break
            else:
                pred=sigmoid(np.dot(x[i],w))
                gradient_error=gradient_error-(pred-y[i])*x[i]+(l2reg/row)*w
        gradient_error=gradient_error/size
    
    return gradient_error
 

class logistic_regression(object):
      def __init__(self,eta,l2reg=0,maxiter=1000,size=0):
          self.eta=eta
          self.iter=maxiter
          self.l2reg=l2reg #with l2 regularizer ,lambda coefficient in lagrange  mutiplier
          self.batch_size=size
          
      def fit(self,X,Y):
          ones = np.ones((np.shape(X)[0], 1))
          X= np.concatenate((ones, X), axis=1)
          row=np.shape(X)[0]
          #w = np.random.randn(np.shape(X)[1]+ 1)
          w=np.zeros(np.shape(X)[1])
          iter_last=self.iter
          costs_record = []
          if(self.batch_size==0):
            self.batch_size=row
         
          count=0
          w=w-self.eta* gradient(X,w,Y,self.l2reg,self.batch_size,count)
          for i in range(0,self.iter):
              count+=1
              if count*self.batch_size >= row:
                 count=0
              costs_record.append(cross_entropy(X,w,Y))
              #loss function is separate to regularizer but include in gradient(augmenting function)
              if np.all(gradient(X,w,Y,self.l2reg,self.batch_size,count) ==0):
                 iter_last=i
                 break
              w=w-self.eta*gradient(X,w,Y,self.l2reg,self.batch_size,count)
          
          plt.clf()
          plt.plot(range(0,iter_last), costs_record)
          plt.show()
          self.w=w
          return self
          
      def predict(self,test_X):
          ones = np.ones((np.shape(test_X)[0], 1))
          test_X= np.concatenate((ones, test_X), axis=1)
          row=np.shape(test_X)[0]
          predict_y=[]
          #probability threshold :0.5
          for i in range(0,row):
              if sigmoid(np.dot(test_X[i],self.w))>0.5:
                 answer=1
              else:
                 answer=0
              predict_y.append(answer)
          predict_y=np.array(predict_y)
          return predict_y

#model=logistic_regression(0.01,l2reg=5,maxiter=100) ,same performance as batch size is 10
model=logistic_regression(0.01,l2reg=5,maxiter=100,size=10)

model.fit(X,Y_an)


"""
Decision boundary inference:

1/(1+exp(w0+w1x1+w2x2))=0.5 , at this probability threshold :0.5

and after reduce(move terms and get log), we can acquire w0+w1x1+w2x2=0

after that, we can see x2=-w0/w2-(w1/w2)*x1

plot as below:
"""

plt.clf()
plt.scatter(X[:,0], X[:,1],c=Y_an,s=100, alpha=0.5)

x_aix=X[:,0]
y_axis=-(model.w[0]/model.w[2])-(model.w[1]/model.w[2])*x_aix
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
x_aix=X[:,0]
y_axis=-(model.w[0]/model.w[2])-(model.w[1]/model.w[2])*x_aix
plt.plot(x_aix, y_axis)
plt.show()


"""
ps:

It has accuracy near (or even better) to the scikit learn package logistic regression at the eta is 0.008,l2reg is 5
,maxiter is 3000,batch size is 200

on the kaggle titanic tutorial data

"""
