# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:57:34 2019

@author: jianfal
"""

from gen_data import gen_data
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def AccuarcyCompute(pred,label):
    var=np.mean((pred-label)**2)
    error=np.mean(abs(pred-label))
    error_percentage=round(error/np.mean(abs(label))*100,2)
    return var,error,error_percentage
def prediction(X,beta,p,Sigma,Z):
    pred=np.zeros((len(X),1))
    for j in range(len(X)):
        x,z=X[j],Z[j]
        mu=np.dot(z,p)
        sigma=np.dot(z,Sigma)
        def w(u):#pi
            b=np.dot(x,beta)+u
            return b
        W=np.zeros(2000)
        for i in range(2000):
            u=np.random.normal(mu, sigma, 1)# Monte carlo method
            W[i]=w(u)
        pred[j]=np.mean(W)
    return pred

def E_step(beta_old,sigma_u,sigma_e,X,Y,Z):
    K=Z.shape[1] 
    Sigma=(1/sigma_u)*(np.ones(K))
    Sigma=Sigma+np.dot(Z.T,Z).diagonal()/sigma_e
    Sigma1=1/(Sigma)
    mu=np.zeros(K)
    for i in range(len(Z)):
        mu=mu+Sigma1*Z[i,]*(Y[i,]-np.dot(X[i,],beta_old))/sigma_e
    return np.reshape(mu,(K,1)),np.reshape(Sigma1,(K,1))

def M_step(mu,Sigma,X,Y,Z):
    K=Z.shape[1] 
    beta_new=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y-np.dot(Z,mu)))
    
    sigma_u=(np.dot(mu.T,mu)+np.sum(Sigma))/K
    sigma_e=np.sum((Y-np.dot(X,beta_new))**2)
    for i in range(len(Z)):
        sigma_e=sigma_e+np.dot(Z[i,],mu*mu+Sigma)-2*np.dot(Y[i,]-np.dot(X[i,],beta_new),np.dot(Z[i,],mu))
    sigma_e=sigma_e/len(Z)
    return beta_new,sigma_u,sigma_e

def EM(y):
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(Xdata[:,:5], y,Xdata[:,5:], test_size = 0.3)
    #initial_parameter
    beta_old=np.random.random(size=np.shape(Xdata[:,:5])[1])
    sigma_u=1
    sigma_e=1
    epochs=30
    X=x_train
    Y=y_train
    Z=z_train
    for e in range(epochs):
        mu,Sigma1=E_step(beta_old,sigma_u,sigma_e,X,Y,Z)
        beta_old,sigma_u,sigma_e=M_step(mu,Sigma1,X,Y,Z)
        if e%10==0:
            pred=prediction(X,beta_old,mu,Sigma1,Z)
    pred=prediction(x_test,beta_old,mu,Sigma1,z_test)
    plt.scatter(x_test[:,1],pred , marker='x')
    print(AccuarcyCompute(pred,y_test))
if __name__ == '__main__':
    Xdata,y=gen_data(10)
    Xdata=np.concatenate((np.ones((Xdata.shape[0],1)),Xdata),axis=1).astype(float)
    for i in range(len(y)):
        EM(y[i])

'''
(0.002460019249207814, 0.03566158009767635, 4.97)
(0.002691816505975828, 0.03861964516235123, 4.6)
(0.003364610734050656, 0.04351645944560852, 5.58)
(0.002092487267656577, 0.03565267468930575, 5.53)
(0.0017462519262057027, 0.034892449525360095, 3.7)
(0.00247624675649647, 0.03753371018468406, 4.35)
(0.003737213958193634, 0.04866151945218468, 8.92)
(0.0026849614351548, 0.03899822956705998, 4.29)
(0.0013618905932646977, 0.03088042985437457, 3.09)
(0.00444138987523014, 0.05103304217010265, 4.18)
'''
