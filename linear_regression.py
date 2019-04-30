# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:27:40 2019

@author: jianfal
"""
from gen_data import gen_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



def LR(y):
    x_train, x_test, y_train, y_test = train_test_split(X[:,:4], y, test_size = 0.3)
    
    model=linreg.fit(x_train, y_train)
    #print(model.coef_,model.intercept_)
    n_predic_y_test = model.predict(x_test)
    
    plt.scatter(x_test[:,1],n_predic_y_test , marker='x')
    var=np.mean((n_predic_y_test-y_test)**2)
    error=np.mean(abs(n_predic_y_test-y_test))
    error_percentage=round(error/np.mean(abs(y_test))*100,2)
    
    return var,error,error_percentage
if __name__ == '__main__':
    X,y=gen_data(10)
    linreg = LinearRegression()
    for i in range(len(y)):
    
        print(LR(y[i]))
    #print(LR(y_big_noise))
'''
(0.0015366263968832226, 0.029558440011940437, 4.12)
(0.007524175125879509, 0.06941106394372618, 9.41)
(0.051449122033343286, 0.191328963800147, 23.79)
(0.20463488066853824, 0.33023352718617116, 56.65)
(0.11767823522992713, 0.28898224042626236, 28.81)
(0.32396914319907705, 0.4598516734961317, 55.71)
(0.2232187293220335, 0.39778403290574266, 82.12)
(1.578167663561671, 1.0617364678404835, 94.25)
(0.639188444611845, 0.6624016711869223, 60.31)
(0.4300697392308613, 0.5422248373416173, 51.63)




'''


