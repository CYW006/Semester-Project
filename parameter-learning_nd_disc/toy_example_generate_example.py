#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 13:47:57 2018

@author: chenyuw
"""

import numpy as np
import h5py
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt

from QuadCameraContourTrajectory import QuadCameraContourTrajectory
from plot_quad_camera_trajectory import create_quad_camera_contour_plots
from generate_kml_file import generate_kml_file


def main():
    # weights                                       term in paper
    lag_weight = 1                                  # weight on e^l


    angular_weight = 1                              # w_phi and w_psi  
    # learn jerk	
    angular_jerk_weight = 1                         # w_j on angles
    theta_input_weight = 1                          # weight on v_i
    # not necessary
    min_time_weight = 1                             # w_end
    end_time_weight = 0                             # w_len
    timing_weight = 0                               # w_t
    
    end_time = np.empty((10,1))
    abs_time = np.zeros((10,1))
    
    weights = np.array([[1, 10],[10, 10],[10, 1],[10, 20],[0.2, 10],[1, 0.1],[30, 10], [10, 30], [20, 40], [40,20]])     #[[1, 10],[10, 10],[10, 1],[10, 20]]
    
    for ith_param in range(0,5):
        # parameters
        trajectory_name = 'saffa.h5'
        
        #learn jerk, contour
        contour_weight = weights[ith_param,0]                              # weight on e^c
        jerk_weight = weights[ith_param,1]                                # w_j on position
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

        if trajectory is None:
            return

        #toy_goal
        end_time[ith_param] = trajectory[trajGen.endTimeStateIndex, -1];
        abs_time[ith_param] = np.absolute(end_time[ith_param] - 25)
        # create plot and save to directory "plots"
#        create_quad_camera_contour_plots(trajGen, trajectory, file=trajectory_name + '_mpcc')

        # create kml-file and save to directory "kmls"
#        generate_kml_file(trajGen, trajectory, gps_trajectory, trajectory_name, ith_param)

    output = open('data_trajectory.pkl','wb')
    pickle.dump(end_time, output)
    output.close()
    print(abs_time)
    
'''
    #generate prior
    #target_time T = 25s
    err_t = np.absolute(end_time -25)
    kernel = C(1.0, (1e-2, 1e2)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(jerk_w_prior, err_t)
''' 
    
    

def load_trajectory(traj_name):
    with h5py.File('./data/' + traj_name, 'r') as hf:
        keytimes = np.array(hf.get('keytimes'))
        keyframes = np.transpose(np.array(hf.get('keyframes')))
        keyorientations = np.transpose(np.array(hf.get('keyorientations')))
        trajectory = np.transpose(np.array(hf.get('trajectory')))
        return keyframes, keyorientations, keytimes, trajectory


if __name__ == '__main__':
    main()
