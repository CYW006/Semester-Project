#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 19:28:59 2018

@author: root
"""

import numpy as np
import pickle
import h5py
from QuadCameraContourTrajectory import QuadCameraContourTrajectory
from generate_kml_file import generate_kml_file

'''
total simulation space: steps**n_params
'''
steps = 100
n_params = 2                                # num of weights to be optimized
tra_h5 = "saffa.h5"                         #trajectory file
bounds = np.array([[0.5, 50] , [0.5, 50]])  #bounds for weights

def load_trajectory(traj_name):
    with h5py.File('./data/' + traj_name, 'r') as hf:
        keytimes = np.array(hf.get('keytimes'))
        keyframes = np.transpose(np.array(hf.get('keyframes')))
        keyorientations = np.transpose(np.array(hf.get('keyorientations')))
        trajectory = np.transpose(np.array(hf.get('trajectory')))
        return keyframes, keyorientations, keytimes, trajectory
    

grid_1d = []
for i in range(0, n_params):
    tmp = np.linspace(bounds[i,0], bounds[i,1], steps)
    grid_1d.append(tmp)
    
grid = np.meshgrid(*grid_1d)

'''
grid_points: weights in discretized parameter space
'''
grid_points = np.zeros((steps**n_params, n_params))
for i in range(0, len(grid[0].ravel())):
    for ith_param in range(0, n_params):
        grid_points[i, ith_param] = grid[ith_param].ravel()[i]
 
print(grid_points.shape)
print(grid_points)
    
endtime = np.zeros((len(grid_points), 1))       #save endtime for each group of weights
       
for i in range(len(grid_points)):
    new_sample = grid_points[i,:]
    #select trajectory
    trajectory_name = tra_h5
    
    print('iteration ', i)
    
    j_w_new, c_w_new = new_sample
    # weights                                       term in paper
    lag_weight = 2                                  # weight on e^l
    contour_weight = c_w_new                              # weight on e^c
    angular_weight = 1                              # w_phi and w_psi  
    # learn jerk	
    jerk_weight = j_w_new                                # w_j on position
    angular_jerk_weight = 1                         # w_j on angles
    theta_input_weight = 1                          # weight on v_i
    # not necessary
    min_time_weight = 1                             # w_end
    end_time_weight = 0                             # w_len
    timing_weight = 0                               # w_t
        
    # load trajectory
    keyframes, keyorientations, keytimes, gps_trajectory = load_trajectory(trajectory_name)
    if keyframes.shape[1] > 3:
        keyframes = keyframes[:,0:3]
        
    # specifying weights for optimizer
    options = dict()
    options["keytimes"] = keytimes
    options["lag_weight"] = lag_weight
    options["contour_weight"] = contour_weight
    options["gimbal_weight"] = angular_weight
    options["smoothness_weight"] = jerk_weight
    options["angular_smoothness_weight"] = angular_jerk_weight
    options["theta_input_weight"] = theta_input_weight
    options["min_time_weight"] = min_time_weight
    options["end_time_weight"] = end_time_weight
    options["timing_weight"] = timing_weight
    
    trajGen = QuadCameraContourTrajectory(keyframes, keyorientations, options)
    trajectory, keytimes = trajGen.generate_trajectory()
    end_time = trajectory[trajGen.endTimeStateIndex, -1];
    endtime[i] = end_time
 
print(endtime) 

pkl_file = open(tra_h5.split('.')[0] + '_j_c.pkl', 'wb')   #save endtime to      saffa_j_c.pkl
pickle.dump(endtime, pkl_file)
pkl_file.close()