# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:07:13 2019

@author: jianfal
"""

import matplotlib.pyplot as plt
import numpy as np


def gen_data(k=2,size=1000):
    np.random.seed(1)
    x=np.random.random(size)*2
    x=np.reshape(x,(len(x),1))
    b=np.random.normal(size=size)*0.05
    b=np.reshape(b,(len(b),1))
    y=[]
    num_people=int(size/10)
    for i in range(k):
    #y_small_noise=np.sin(x)+b
    #plt.scatter(x,y_small_noise , marker='x')
    
        mu=np.random.normal(loc=0,scale=i/k,size=num_people)
    
        y_noise=np.sin(x[:10])+mu[0]+b[:10]
        z=np.zeros((size,num_people),dtype=int)
        for i in range(10):
            z[i,0]=1
        for i in range(1,num_people):
            y_noise_1=np.sin(x[10*i:10*i+10])+mu[i]+b[10*i:10*i+10]
            for j in range(10):
                z[i*10+j,i]=1
            y_noise=np.concatenate((y_noise,y_noise_1),axis=0).astype(float)
        X=np.concatenate((x,x**2,x**3,x**4,z),axis=1).astype(float)
        y.append(y_noise)
    return X,y
#######################################
if __name__ == '__main__':
    Xdata,[y_small_noise,y_big_noise]=gen_data()

