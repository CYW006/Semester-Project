#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 09:09:57 2018

@author: root
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

'''
parameter space is the same as that in end_time_saffa_j_c.py
'''
bounds = np.array([[0.5, 50] , [0.5, 50]])
n_params = bounds.shape[1]
steps = 100

grid_1d = []
for i in range(0, n_params):
    tmp = np.linspace(bounds[i,0], bounds[i,1], steps)
    grid_1d.append(tmp)
    
grid = np.meshgrid(*grid_1d)

grid_points = np.zeros((steps**n_params, n_params))
for i in range(0, len(grid[0].ravel())):
    for ith_param in range(0, n_params):
        grid_points[i, ith_param] = grid[ith_param].ravel()[i]

pkl_file = open('saffa_j_c.pkl', 'rb')
endtime = pickle .load(pkl_file)
print(endtime.shape)
pkl_file.close()

num, boundary,tmp3 = plt.hist(endtime, bins = 100)
plt.show()

print(num)
print(boundary.shape)
#print(tmp2.shape)
#print(tmp3)

#prob of each 
tmp = np.array(np.nonzero(num))
density_hist = 1.0/len(tmp.ravel())/num
density = np.zeros_like(endtime)
for i in range(len(endtime)):
    for j in range(len(num)):
        if endtime[i] >= boundary[j] and endtime[i] < boundary[j+1]:
            density[i] = density_hist[j]
    if endtime[i] == boundary[len(num)]:
        density[i] = density_hist[len(num)-1]

print(np.sum(density))
samples = np.unique(np.random.choice(len(endtime), 1000, p = density.reshape(len(density))))
print(samples.shape)

plt.hist(endtime[samples], bins = 100)
plt.show()

pkl_file = open('saffa_sample.pkl', 'wb')
pickle.dump(grid_points[samples,:], pkl_file)
pkl_file.close()