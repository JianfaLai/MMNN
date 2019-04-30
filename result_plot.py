# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:54:35 2019

@author: jianfal
"""

import matplotlib.pyplot as plt
import numpy as np
a=[[0,4.12,5.76,4.97,4.1],
[1,9.41,10.51,4.6,4.91],
[2,23.79,25.26,5.58,4.62],
[3,56.65,54.92,5.53,6.26],
[4,28.81,28.3,3.7,4.27],
[5,55.71,55.99,4.35,3.75],
[6,82.12,76.83,8.92,6.68],
[7,94.25,92.92,4.29,3.54],
[8,60.31,60.66,3.09,3.04],
[9,51.63,52.69,4.18,3.79]]
a=np.array(a).T
b=[[6.01,6.06,6.31,6.31],
[6.08,6.07,9.66,17.13],
[5.88,5.7,16.84,31.99],
[5.95,5.69,20.17,39.56],
[6.11,5.99,21.19,41.98]
]
b=np.array(b).T
#plt.style.use('fivethirtyeight')
plt.plot(a[0],a[1],label="Linear Regression")

plt.plot(a[0],a[2],label="Neural Network")
plt.plot(a[0],a[3],label="Mixed Model with EM")
plt.plot(a[0],a[4],label="MMNN with EM")
plt.legend()
plt.xlabel('Random Effect')
plt.ylabel('Error percentage(%)')
plt.show()