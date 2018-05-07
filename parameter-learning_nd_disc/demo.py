"""
demo.py

Just a little script to demonstrate how to call the interactive Bayesian optimization (IBO/EGO) code.

"""
import h5py
from numpy import array, arange, zeros
from matplotlib.pylab import *

from pref_GP import GaussianProcess, PrefGaussianProcess
from kernel import GaussianKernel_iso
from sklearn.metrics.pairwise import rbf_kernel
from QuadCameraContourTrajectory import QuadCameraContourTrajectory

#from ego.acquisition import EI, UCB, maximizeEI, maximizeUCB
#from ego.acquisition.prefutil import query2prefs
#from ego.acquisition.gallery import fastUCBGallery
#from ego.utils.testfunctions import Hartman6
def load_trajectory(traj_name):
    with h5py.File('./data/' + traj_name, 'r') as hf:
        keytimes = np.array(hf.get('keytimes'))
        keyframes = np.transpose(np.array(hf.get('keyframes')))
        keyorientations = np.transpose(np.array(hf.get('keyorientations')))
        trajectory = np.transpose(np.array(hf.get('trajectory')))
        return keyframes, keyorientations, keytimes, trajectory

def demoPrefGallery():
    
    kernel = GaussianKernel_iso([2])
    
    # set up a Preference Gaussian Process, in which the observations are
    # preferences, rather than scalars
    # assume left is prefered
#    left = array([[10, 10], [10, 1], [10, 20], [10, 30], [10, 20], [10, 30], [1, 0.1]])
#    right = array([[20, 40], [10, 10], [10, 1], [40, 20], [10, 1], [0.1, 1], [30, 10]])
    
    left = array([[1, 10], [10, 20], [1, 10], [0.2, 10]])
    right = array([[10, 10], [10, 10], [10, 20], [10, 20]])
    #left = array([[1, 10], [10, 1], [10, 20]])
    #right = array([[10, 1], [10, 10], [10, 1]])
    pref = zeros((2,left.shape[0],left.shape[1]))
    pref[0,:,:] = left
    pref[1,:,:] = right
    GP = PrefGaussianProcess(kernel,prefs = pref)
    p_bounds = array([[0.2, 10], [10, 20]])
    
    for i in range(0,20):
        print('iteration:', i)
        new_x = GP.find_newpoint(p_bounds)
        
        #only need to modify build_preference
        pref = GP.build_preference_ge(new_x.reshape((1,-1)))
        print(pref)
        #no preference
        if (pref.size == 0):
            break
        GP.addPreferences(pref)
    
    #show preference
    prefs = GP.preferences
    for i in range(0,prefs.shape[1]):
        left = prefs[0,i,:]
        right = prefs[1,i,:]
        print ('%s preferred to %s' % (left, right))
    
    trajectory_name = 'saffa.h5'
    
    contour_weight, jerk_weight = GP.X[np.argmax(GP.Y)] 
    
    # weights                                       term in paper
    lag_weight = 1                                  # weight on e^l
    angular_weight = 1                              # w_phi and w_psi  
    # learn jerk	
                               # w_j on position
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

    print(GP.loss)
    print(np.abs(end_time - 25))
    print(GP.X[np.argmax(GP.Y)])
    

if __name__ == '__main__':
    demoPrefGallery()

