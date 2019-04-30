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
        #if i % 10 == 0:
    print("training:",AccuarcyCompute(outputs,labels))
    inputs=pt.from_numpy(x_test)
    labels=pt.from_numpy(y_test)
    inputs = pt.autograd.Variable(inputs)
    labels = pt.autograd.Variable(labels)
    outputs = model(inputs.float())
    print("Testing",AccuarcyCompute(outputs,labels.float()))
if __name__ == '__main__':
    X,y=gen_data(3,size=10000)
    X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1).astype(float)
    for i in range(len(y)):
        NN(y[i])
        
'''
training: (0.002873892208346031, 0.040181060697249464, 5.86)
Testing (0.003020943, 0.041308008, 5.76)
training: (0.008888514014013381, 0.07725185674662287, 10.35)
Testing (0.009212663, 0.07754674, 10.51)
training: (0.035802332285103156, 0.15156452387189825, 19.88)
Testing (0.0563008, 0.20312157, 25.26)
training: (0.09500131265184882, 0.23994424095768566, 41.12)
Testing (0.18667619, 0.32013768, 54.92)
training: (0.12206790803325551, 0.28525926446121, 32.01)
Testing (0.11577501, 0.28392354, 28.3)
training: (0.28153795607035725, 0.4140206074452672, 47.34)
Testing (0.3284864, 0.46218166, 55.99)
training: (0.17565253833554031, 0.3522277866956774, 68.22)
Testing (0.19920696, 0.37214163, 76.83)
training: (0.655890475119132, 0.6153195928179533, 69.52)
Testing (1.542784, 1.0467106, 92.92)
training: (0.569558326320242, 0.6049503288030024, 70.58)
Testing (0.64398646, 0.6662241, 60.66)
training: (0.25037810116888204, 0.4095860223797141, 31.58)
Testing (0.4427868, 0.5533382, 52.69)
'''



