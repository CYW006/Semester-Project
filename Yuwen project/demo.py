"""
demo.py

Just a little script to demonstrate how to call the interactive Bayesian optimization (IBO/EGO) code.

"""
import numpy as np
import h5py
from pref_GP import PrefGaussianProcess
from kernel import GaussianKernel_ard
from QuadCameraContourTrajectory import QuadCameraContourTrajectory

'''
Initialization
'''
trajectory_name = 'saffa.h5'    #name of trajectory
goal_time = 25                  #toy fixed-time

def load_trajectory(traj_name):
    with h5py.File('./data/' + traj_name, 'r') as hf:
        keytimes = np.array(hf.get('keytimes'))
        keyframes = np.transpose(np.array(hf.get('keyframes')))
        keyorientations = np.transpose(np.array(hf.get('keyorientations')))
        trajectory = np.transpose(np.array(hf.get('trajectory')))
        return keyframes, keyorientations, keytimes, trajectory

def demoPrefLearning():
    #write averaged EI at each iteration
    f = open('./EI.txt', 'a')
    f.write('\n')
    f.close()
    
    kernel = GaussianKernel_ard([5,5])
    
    #form of preference
    '''
    left = array([[1, 1]])
    right = array([[2, 10]])
    pref = zeros((2,left.shape[0],left.shape[1]))
    pref[0,:,:] = left
    pref[1,:,:] = right
    '''
    
    #boundary of weights
    p_bounds = np.array([[0.1, 50], [0.1, 50]]) #jerk and contour
    #GP = PrefGaussianProcess(kernel, pref, bounds = p_bounds)
    GP = PrefGaussianProcess(kernel, bounds = p_bounds)

    for i in range(0,30):
        print('iteration:', i)
        #build_preference
        pref = GP.build_preference_ge()
        #if no preference, abort
        if (pref.size == 0):
            break
        else:
            GP.addPreferences(pref)
    
    #show final preference
    prefs = GP.preferences
    for i in range(0,prefs.shape[1]):
        left = prefs[0,i,:]
        right = prefs[1,i,:]
        print ('%s preferred to %s' % (left, right))
    
    jerk_weight, contour_weight = GP.X[np.argmax(GP.Y)] 
    
    prefer_param = []
    #find most prefered weights
    for i in range(GP.preferences.shape[1]):
        r = tuple(GP.preferences[0,i,:])
        most_prefer = 0
        for j in range(0,GP.preferences.shape[1]):
            if r == tuple(GP.preferences[1,j,:]):
                most_prefer = 1
        if most_prefer == 0:
            if r not in prefer_param:
                prefer_param.append(r)
        
    #best result   
    # weights                                       term in paper
    lag_weight = 2                                  # weight on e^l
    angular_weight = 1                              # w_phi and w_psi  
    angular_jerk_weight = 1                         # w_j on angles
    theta_input_weight = 1                          # weight on v_i
    # not necessary
    min_time_weight = 1
    end_time_weight = 0                             # w_end
    timing_weight = 0                               # w_t
    
    # load trajectory
    keyframes, keyorientations, keytimes, gps_trajectory = load_trajectory(trajectory_name)
    if keyframes.shape[1] > 3:
        keyframes = keyframes[:,0:3]

    # generate the video with best weights we have found
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

    print(np.abs(end_time - goal_time))     #for toy example evaluation
    print(GP.X[np.argmax(GP.Y)])            #print best weights
    print(prefer_param)
    

if __name__ == '__main__':
    demoPrefLearning()