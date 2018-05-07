import numpy as np
from scipy.interpolate import CubicSpline, pchip_interpolate
import FORCESNLPsolver_py


class QuadCameraContourTrajectory:

    # Parameters
    nStates = 18
    nInputs = 6
    nVars = nStates + nInputs  # number of variables
    nPars = 41  # number of runtime parameters
    nStages = 60  # horizon length
    poly_order = 2  # order for local polynomial fitting

    def __init__(self, keyframes, keyorientations, options):

        self.keyframes = keyframes
        self.keyorientations = keyorientations
        self.nKeyframes = self.keyframes.shape[0]

        # Initialize state vector indices
        self.inputIndices = np.arange(0, self.nInputs)
        # Position and orientation state indices quadrotor
        self.posStateIndices = np.arange(self.nInputs, self.nInputs + 3)
        self.velStateIndices = np.arange(self.posStateIndices[-1] + 1, self.posStateIndices[-1] + 4)
        self.accStateIndices = np.arange(self.velStateIndices[-1] + 1, self.velStateIndices[-1] + 4)
        # Gimbal state indices
        self.yawStateIndex = self.accStateIndices[-1] + 1
        self.pitchStateIndex = self.yawStateIndex + 1
        self.yawVelStateIndex = self.yawStateIndex + 2
        self.pitchVelStateIndex = self.yawStateIndex + 3
        self.yawAccStateIndex = self.yawStateIndex + 4
        self.pitchAccStateIndex = self.yawStateIndex + 5
        # Theta state indices
        self.thetaStateIndex = self.pitchAccStateIndex + 1
        self.thetaVelStateIndex = self.thetaStateIndex + 1
        # End time state index
        self.endTimeStateIndex = self.thetaVelStateIndex + 1

        # Initialize options
        self.parse_options(options)

    def parse_options(self, options):
        # Initialize(default) parameters
        if 'contour_weight' not in options:
            options["contour_weight"] = 0

        if 'lag_weight' not in options:
            options["lag_weight"] = 0

        if 'position_weight' not in options:
            options["position_weight"] = 0

        if 'progress_weight' not in options:
            options["progress_weight"] = 0

        if 'smoothness_weight' not in options:
            options["smoothness_weight"] = 1

        if 'angular_smoothness_weight' not in options:
            options["angular_smoothness_weight"] = 1

        if 'theta_input_weight' not in options:
            options["theta_input_weight"] = 1

        if 'min_time_weight' not in options:
            options["min_time_weight"] = 1

        if 'velocity_weight' not in options:
            options["velocity_weight"] = 0

        if 'gimbal_weight' not in options:
            options["gimbal_weight"] = 0

        if 'timing_weight' not in options:
            options["timing_weight"] = 0

        if 'end_time_weight' not in options:
            options["end_time_weight"] = 0

        if 'velocity_profile' not in options:
            velocities = np.zeros(self.nKeyframes)
            options["velocity_profile"] = velocities

        if 'keytimes' not in options:
            times = np.arange(0, 2 * self.nKeyframes, 2)
            options["keytimes"] = times

        if 'fitlength' not in options:
            options['fitlength'] = 10

        # Extract options
        self.contour_weight = options["contour_weight"]
        self.lag_weight = options["lag_weight"]
        self.position_weight = options["position_weight"]
        self.progress_weight = options["progress_weight"]
        self.angular_smoothness_weight = options["angular_smoothness_weight"]
        self.smoothness_weight = options["smoothness_weight"]
        self.theta_input_weight = options["theta_input_weight"]
        self.min_time_weight = options["min_time_weight"]
        self.velocity_weight = options["velocity_weight"]
        self.gimbal_weight = options["gimbal_weight"]
        self.timing_weight = options["timing_weight"]
        self.end_time_weight = options["end_time_weight"]
        self.velocity_profile = options["velocity_profile"]
        self.keytimes = options["keytimes"]
        self.fitlength = options['fitlength']

    def generate_trajectory(self):
        # Interpolate keyframes
        self.interpolate_keyframes()
        # Solve problem
        [trajectory, _] = self.solve_system()
        [keytimes, self.keyframe_time_idxs, self.times, self.T] = self.get_timing(trajectory)
        return trajectory, keytimes

    def interpolate_keyframes(self):
        # Get theta from keyframes (calculate chord lengths between keyframes)
        self.theta_of_keyframes = np.append([0], np.cumsum(np.power(np.dot(np.power(np.diff(self.keyframes, axis=0), 2),
                                                                           np.ones((3, 1))), 0.25)))
        self.theta = np.linspace(self.theta_of_keyframes[0], self.theta_of_keyframes[-1],
                                 self.nStages + 1)
        # Construct spline functions for position which are parameterized by theta
        # self.pos_spline = CubicSpline(self.theta_of_keyframes, self.keyframes, axis=0)
        # Compute reference positions with spline functions
        # self.ref_positions = self.pos_spline(self.theta)
        # Compute reference positions with pchip_interpolate parameterized by theta
        self.ref_positions = np.zeros((self.nStages + 1, 3))
        self.ref_positions[:,0] = pchip_interpolate(self.theta_of_keyframes, self.keyframes[:, 0], self.theta)
        self.ref_positions[:,1] = pchip_interpolate(self.theta_of_keyframes, self.keyframes[:, 1], self.theta)
        self.ref_positions[:,2] = pchip_interpolate(self.theta_of_keyframes, self.keyframes[:, 2], self.theta)

        # pitch and yaw spline (parameterized by theta)
        self.ref_yaw = pchip_interpolate(self.theta_of_keyframes, self.keyorientations[:, 0], self.theta)
        self.ref_pitch = pchip_interpolate(self.theta_of_keyframes, self.keyorientations[:, 1], self.theta)

        # time spline (parameterized by theta)
        self.ref_timings = pchip_interpolate(self.theta_of_keyframes, self.keytimes, self.theta)
        # velocity spline (parameterized by theta)
        if self.velocity_weight != 0:
            self.ref_velocities = pchip_interpolate(self.theta_of_keyframes, self.velocity_profile, self.theta)

    def solve_system(self):
        # Initialize information for solver
        progress_id = np.zeros(self.nStages + 1)
        solution = 1e5
        old_solution = 1e6
        exitflag = 0
        # initial conditions
        Xout = np.zeros((self.nInputs + self.nStates, self.nStages + 1))
        problem = FORCESNLPsolver_py.FORCESNLPsolver_params
        problem['x0'] = np.zeros((self.nStages + 1) * self.nVars)
        problem['xinit'] = np.zeros(self.nStates - 2)  # remove end time and z-acc
        problem['xinit'][0:3] = self.ref_positions[0, :]
        problem['xinit'][3] = self.ref_yaw[0]
        problem['xinit'][4] = self.ref_pitch[0]
        problem['xfinal'] = np.zeros(self.nStates - 7)  # remove end time, z-acc, pos, yaw and pitch
        problem['xfinal'][0] = self.theta_of_keyframes[-1]
        opt_count = 1

        # Solve until we get a valid trajectory
        while exitflag != 1 or (old_solution > solution and solution > 2) or opt_count < 3:
            # Setup the problem
            old_solution = solution
            self.old_progress_id = np.copy(progress_id)

            # Compute polynomial coefficients and parameters for every time-step
            self.param = np.zeros((self.nPars, self.nStages + 1))
            end_time = self.keytimes[self.keyframes.shape[0] - 1]
            for i in range(0, progress_id.shape[0]):
                k = progress_id[i]
                if opt_count == 1:
                    fitrange = np.arange(0, self.nStages + 1, dtype=int)
                elif k >= self.nStages - (self.fitlength / 2):
                    fitrange = np.arange(self.nStages + 1 - self.fitlength, self.nStages + 1, dtype=int)
                elif k > (self.fitlength / 2):
                    fitrange = np.arange(k - (self.fitlength / 2), k + (self.fitlength / 2), dtype=int)
                else:
                    fitrange = np.arange(0, self.fitlength, dtype=int)

                px = np.polyfit(self.theta[fitrange], self.ref_positions[fitrange, 0], self.poly_order)
                py = np.polyfit(self.theta[fitrange], self.ref_positions[fitrange, 1], self.poly_order)
                pz = np.polyfit(self.theta[fitrange], self.ref_positions[fitrange, 2], self.poly_order)
                dpx = np.polyder(px)
                dpy = np.polyder(py)
                dpz = np.polyder(pz)
                gy = np.polyfit(self.theta[fitrange], self.ref_yaw[fitrange], self.poly_order)
                gp = np.polyfit(self.theta[fitrange], self.ref_pitch[fitrange], self.poly_order)
                t = np.polyfit(self.theta[fitrange], self.ref_timings[fitrange], self.poly_order)
                if self.velocity_weight != 0:
                    ref_vel = np.polyfit(self.theta[fitrange], self.ref_velocities[fitrange], self.poly_order)
                else:
                    ref_vel = [0, 0, 0]

                self.param[:, i] = np.concatenate((px, py, pz, dpx, dpy, dpz, [self.min_time_weight,
                                                                               self.contour_weight, self.lag_weight,
                                                                               self.position_weight,
                                                                               self.smoothness_weight,
                                                                               self.progress_weight,
                                                                               self.theta_input_weight], gy, gp,
                                                   [self.gimbal_weight], t,
                                                   [self.timing_weight, end_time, self.end_time_weight, i], ref_vel,
                                                   [self.velocity_weight], [self.angular_smoothness_weight]))
            # Stack up parameters
            problem['all_parameters'] = self.param.reshape((self.nStages + 1) * (self.nPars), order='F')

            # Solve the problem
            [solverout, exitflag, info] = FORCESNLPsolver_py.FORCESNLPsolver_solve(problem)
            print("exitflag = {}".format(exitflag))
            # print("f-value = {}".format(info.pself))

            # Convert solver output
            for i in range(0, self.nStages + 1):
                index = 'x' + str(i + 1).zfill(2)
                Xout[:, i] = solverout[index][0:self.nInputs + self.nStates]

            # Determine progress index with respect to reference theta for next iteration
            for pind in range(0, self.nStages + 1):
                idx = np.argmin(np.abs(self.theta - Xout[self.thetaStateIndex, pind]))  # index of closest value
                progress_id[pind] = idx

            # Compute solution to decide when to stop iterating
            solution = np.mean(np.abs(progress_id - self.old_progress_id))
            print("solution = {}".format(solution))
            # Stop optimizing in case solver reached max iteractions
            if exitflag == 0:
                break
            opt_count += 1

        return [Xout, exitflag]

    def get_timing(self, trajectory):
        no_keyframes = self.keyframes.shape[0]
        keyframe_time_idxs = np.zeros(no_keyframes, dtype=int)
        for i in range(no_keyframes-1):
            tmp = np.sum(np.square(np.abs((trajectory[self.posStateIndices, :].transpose()
                                           - self.keyframes[i, :]).transpose())), axis=0)
            idx = np.argmin(tmp)  # index of closest value
            keyframe_time_idxs[i] = idx

        keyframe_time_idxs[-1] = self.nStages
        T = trajectory[self.endTimeStateIndex, -1] / self.nStages
        times = np.arange(0, self.nStages + 1) * T
        return [times[keyframe_time_idxs], keyframe_time_idxs, times, T]