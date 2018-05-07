#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:24:34 2018

@author: chenyuw
"""
import numpy as np
import h5py
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization
from QuadCameraContourTrajectory import QuadCameraContourTrajectory

'''
jerk_w_prior = np.atleast_2d([5, 20, 35, 60, 90]).T
end_time = np.array([20.54273295, 24.70780309, 26.60373073, 28.55604608, 30.18930673])

#target_time T = 25s
err_t = np.absolute(end_time -25)
kernel = C(1.0, (1e-2, 1e2)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
gp.fit(jerk_w_prior, err_t)
    
# Make the prediction on the meshed x-axis (ask for MSE as well)
X = jerk_w_prior
x = np.atleast_2d(np.linspace(0, 100, 100)).T
y = err_t.ravel()
y_pred, sigma = gp.predict(x, return_std=True)
'''

def load_trajectory(traj_name):
    with h5py.File('./data/' + traj_name, 'r') as hf:
        keytimes = np.array(hf.get('keytimes'))
        keyframes = np.transpose(np.array(hf.get('keyframes')))
        keyorientations = np.transpose(np.array(hf.get('keyorientations')))
        trajectory = np.transpose(np.array(hf.get('trajectory')))
        return keyframes, keyorientations, keytimes, trajectory
    
def target_func(j_w):
    trajectory_name = 'saffa.h5'

    # weights                                       term in paper
    lag_weight = 1                                  # weight on e^l
    contour_weight = 1                              # weight on e^c
    angular_weight = 1                              # w_phi and w_psi  
    # learn jerk	
    jerk_weight = j_w                                # w_j on position
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
    err_t = np.absolute(end_time -25)
    return -err_t  #goal: maximize -err_t

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
def plot_bo(f, bo):
    mean, sigma = bo.gp.predict(np.arange(len(f)).reshape(-1, 1), return_std=True)
    
    plt.figure(figsize=(16, 9))
    plt.plot(f)
    plt.plot(np.arange(len(f)), mean)
    plt.fill_between(np.arange(len(f)), mean+sigma, mean-sigma, alpha=0.1, facecolor='green')
    plt.scatter(bo.X.flatten(), bo.Y, c="red", s=50, zorder=10)
    plt.xlim(0, len(f))
    plt.ylim(f.min()-0.1*(f.max()-f.min()), f.max()+0.1*(f.max()-f.min()))
    plt.show()


bo = BayesianOptimization(f=lambda var: target_func(var),
                          pbounds={"var": (0, 100)},
                          verbose=0)

gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 5}
bo.maximize(init_points=5, n_iter=20, acq="ei", kappa=1, **gp_params)

infile = open('data_trajectory.pkl', 'r')
f = -np.absolute(pickle.load(infile) - 25)
plot_bo(f, bo)
