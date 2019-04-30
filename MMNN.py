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
        self.fc1 = pt.nn.Linear(5,2)
        self.fc2 = pt.nn.Linear(2,1)
    def forward(self,din,random_effect):
        dout = self.fc1(din)
        dout = self.fc2(dout)
        return dout+random_effect
def AccuarcyCompute(pred,label):
    pred = pred.data.numpy()
    label = label.data.numpy()
    var=np.mean((pred-label)**2)
    error=np.mean(abs(pred-label))
    error_percentage=round(error/np.mean(abs(label))*100,2)
    return var,error,error_percentage
    
def E_step(sigma_u,sigma_e,Y,Z):
    K=Z.shape[1] 
    Sigma=(1/sigma_u)*(np.ones(K))
    Sigma=Sigma+np.dot(Z.T,Z).diagonal()/sigma_e
    Sigma1=1/(Sigma)
    mu=np.zeros(K)
    for i in range(len(Z)):
        mu=mu+Sigma1*Z[i,]*Y[i,]/sigma_e
    return np.reshape(mu,(K,1)),np.reshape(Sigma1,(K,1))

def M_step(mu,Sigma,Y,Z):
    K=Z.shape[1] 
    sigma_u=(np.dot(mu.T,mu)+np.sum(Sigma))/K
    sigma_e=np.sum((Y)**2)
    for i in range(len(Z)):
        sigma_e=sigma_e+np.dot(Z[i,],mu*mu+Sigma)-2*np.dot(Y[i,],np.dot(Z[i,],mu))
    sigma_e=sigma_e/len(Z)
    return sigma_u,sigma_e
def EMNN(y):
    #pre_set
    sigma_u=1
    sigma_e=1
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(Xdata[:,:5], y,Xdata[:,5:], test_size = 0.3)
    model = MLP()
    optimizer = pt.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    lossfunc = pt.nn.MSELoss()
    #training data
    X=x_train
    Y=y_train
    Z=z_train
    random_effect=np.zeros((len(X),1))
    
    def NN(x,y,random_effect):
        for i in range(100):
            optimizer.zero_grad()
            (inputs,labels) = x,y
            inputs=pt.from_numpy(inputs)
            labels=pt.from_numpy(labels)
            inputs = pt.autograd.Variable(inputs)
            labels = pt.autograd.Variable(labels)
            random_e=pt.from_numpy(random_effect).float()
            outputs = model(inputs.float(),random_e)
            loss = lossfunc(outputs,labels.float())
            loss.backward()
            optimizer.step()
        return model(inputs.float(),0)
    
    #last and repeat is to check whether it shoud stop
    last=0
    repeat=0
    for i in range(50):
        outputs=NN(X,Y,random_effect)
        epochs=5
        for e in range(epochs):
            Random_effect_true=Y-outputs.data.numpy()
            mu,Sigma1=E_step(sigma_u,sigma_e,Random_effect_true,Z)
            sigma_u,sigma_e=M_step(mu,Sigma1,Random_effect_true,Z)
        
        random_effect=np.zeros((len(X),1))
        for j in range(len(Z)):
            z=Z[j]
            p=np.dot(z,mu)
            random_effect[j]=p
        
        
        random_effect_test=np.zeros((len(z_test),1))
        for j in range(len(z_test)):
            z=z_test[j]
            p=np.dot(z,mu)
            random_effect_test[j]=p
        inputs_test=pt.from_numpy(x_test)
        labels_test=pt.from_numpy(y_test)
        inputs_test = pt.autograd.Variable(inputs_test)
        labels_test = pt.autograd.Variable(labels_test)
        random_e_test=pt.from_numpy(random_effect_test).float()
        outputs_test = model(inputs_test.float(),random_e_test)
        error_percentage=AccuarcyCompute(outputs_test,labels_test.float())
        if error_percentage[2]==last:
            repeat+=1
        else:
            last=error_percentage[2]
        if repeat>5:
            break
    print("Testing",error_percentage)
if __name__ == '__main__':
    Xdata,y=gen_data(5,size=1000)
    Xdata=np.concatenate((np.ones((Xdata.shape[0],1)),Xdata),axis=1).astype(float)
    for i in range(len(y)):
        EMNN(y[i])
        
'''
Testing (0.0015800932, 0.029433494, 4.1)
Testing (0.0029981958, 0.044504516, 4.91)
Testing (0.002603048, 0.0364661, 4.62)
Testing (0.002724695, 0.042232957, 6.26)
Testing (0.002901133, 0.038665723, 4.27)
Testing (0.0019755429, 0.036214557, 3.75)
Testing (0.002415244, 0.040518273, 6.68)
Testing (0.0023834745, 0.035962757, 3.54)
Testing (0.0021507577, 0.035938308, 3.04)
Testing (0.0031414144, 0.044444475, 3.79)

N=10000
Testing (0.0024926742, 0.040062033, 5.71)
Testing (0.0029507317, 0.043557882, 6.11)
Testing (0.003065372, 0.04415746, 5.99)
Testing (0.0030026636, 0.04395273, 5.34)
Testing (0.0030103307, 0.043855403, 4.83)

N=1000
Testing (0.002531604, 0.038599867, 5.47)
Testing (0.0030081624, 0.043093387, 6.01)
Testing (0.0031148489, 0.044860844, 6.06)
Testing (0.0031499858, 0.044847127, 5.64)
Testing (0.002765486, 0.042059153, 4.28)

N=3000
Testing (0.0025788886, 0.040610652, 5.65)
Testing (0.0029482488, 0.04301046, 6.08)
Testing (0.0030424101, 0.043871272, 6.07)
Testing (0.0027394677, 0.04155968, 5.01)
Testing (0.0029647427, 0.043070924, 5.02)

N=5000
Testing (0.0025914374, 0.040200982, 5.57)
Testing (0.002770738, 0.04197966, 5.88)
Testing (0.0029406706, 0.04297577, 5.7)
Testing (0.0030163485, 0.043473613, 5.17)
Testing (0.0031185846, 0.04446688, 4.84)

N=8000
Testing (0.0024359503, 0.03923105, 5.52)
Testing (0.0028627987, 0.042770088, 5.95)
Testing (0.0029198667, 0.04259068, 5.69)
Testing (0.0030189105, 0.043778256, 5.22)
Testing (0.0030523143, 0.043963313, 5.03)


'''



