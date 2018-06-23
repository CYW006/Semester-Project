import numpy as np
import h5py
from QuadCameraContourTrajectory import QuadCameraContourTrajectory
from plot_quad_camera_trajectory import create_quad_camera_contour_plots
from generate_kml_file import generate_kml_file


def main():
    # parameters
    trajectory_name = 'saffa.h5'

    # weights                                       term in paper
    lag_weight = 2                                  # weight on e^l
    contour_weight = 1                              # weight on e^c
    angular_weight = 1                              # w_phi and w_psi  
    # learn jerk	
    jerk_weight = 6                                # w_j on position
    angular_jerk_weight = 1                         # w_j on angles
    theta_input_weight = 1                          # weight on v_i
    # not necessary
    min_time_weight = 0.5                             # w_end
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
    
    
    if trajectory is None:
        return

	# trajectory[trajGen.endTimeStateIndex, 0]

    # create plot and save to directory "plots"
#    create_quad_camera_contour_plots(trajGen, trajectory, file=trajectory_name + '_mpcc')

    # create kml-file and save to directory "kmls"
    generate_kml_file(trajGen, trajectory, gps_trajectory, trajectory_name, '_test')


def load_trajectory(traj_name):
    with h5py.File('./data/' + traj_name, 'r') as hf:
        keytimes = np.array(hf.get('keytimes'))
        keyframes = np.transpose(np.array(hf.get('keyframes')))
        keyorientations = np.transpose(np.array(hf.get('keyorientations')))
        trajectory = np.transpose(np.array(hf.get('trajectory')))
        return keyframes, keyorientations, keytimes, trajectory


if __name__ == '__main__':
    main()