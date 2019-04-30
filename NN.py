# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:46:54 2019

@author: jianfal
"""

from gen_data import gen_data
import torch as pt
import numpy as np
from sklearn.model_selection import train_test_split


class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = pt.nn.Linear(1005,100)
        self.fc2 = pt.nn.Linear(100,10)
        self.fc3 = pt.nn.Linear(10,1)
    def forward(self,din):
        dout = self.fc1(din)
        dout = self.fc2(dout)
        dout = self.fc3(dout)
        return dout

def AccuarcyCompute(pred,label):
    pred = pred.data.numpy()
    label = label.data.numpy()
    var=np.mean((pred-label)**2)
    print(np.mean(pred))
    error=np.mean(abs(pred-label))
    error_percentage=round(error/np.mean(abs(label))*100,2)
    return var,error,error_percentage

def NN(y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    model = MLP()
    optimizer = pt.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    lossfunc = pt.nn.MSELoss()
    #Training
    for i in range(1000):
        optimizer.zero_grad()
        (inputs,labels) = x_train,y_train
        inputs=pt.from_numpy(inputs)
        labels=pt.from_numpy(labels)
        inputs = pt.autograd.Variable(inputs)
        labels = pt.autograd.Variable(labels)
        outputs = model(inputs.float())
        loss = lossfunc(outputs,labels.float())
        loss.backward()
        optimizer.step()
    print("training:",AccuarcyCompute(outputs,labels))
    
    #Get the result from the test set
    inputs=pt.from_numpy(x_test)
    labels=pt.from_numpy(y_test)
    inputs = pt.autograd.Variable(inputs)
    labels = pt.autograd.Variable(labels)
    outputs = model(inputs.float())
    print("Testing",AccuarcyCompute(outputs,labels.float()))
if __name__ == '__main__':
    X,y=gen_data(5,size=1000)
    X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1).astype(float)
    for i in range(len(y)):
        NN(y[i])
        
'''

Testing (0.003020943, 0.041308008, 5.76)
Testing (0.009212663, 0.07754674, 10.51)
Testing (0.0563008, 0.20312157, 25.26)
Testing (0.18667619, 0.32013768, 54.92)
Testing (0.11577501, 0.28392354, 28.3)
Testing (0.3284864, 0.46218166, 55.99)
Testing (0.19920696, 0.37214163, 76.83)
Testing (1.542784, 1.0467106, 92.92)
Testing (0.64398646, 0.6662241, 60.66)
Testing (0.4427868, 0.5533382, 52.69)
'''



