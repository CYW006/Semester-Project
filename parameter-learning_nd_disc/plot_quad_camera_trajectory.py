import numpy as np
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_quad_camera_contour_plots(trajGen, trajectory, **kwargs):

    file = None
    for key, value in kwargs.iteritems():
        if key == "file":
            file = value

    end_time = trajectory[trajGen.endTimeStateIndex, -1]
    print("end time = {}".format(end_time))
    print("T = {}".format(trajGen.T))
    theta = trajectory[trajGen.thetaStateIndex, :]
    time = np.arange(0, trajGen.nStages + 1) * trajGen.T
    # ref_pos = trajGen.pos_spline(theta)
    ref_pos = np.zeros((trajGen.nStages + 1, 3))
    ref_pos[:, 0] = pchip_interpolate(trajGen.theta, trajGen.ref_positions[:, 0], theta)
    ref_pos[:, 1] = pchip_interpolate(trajGen.theta, trajGen.ref_positions[:, 1], theta)
    ref_pos[:, 2] = pchip_interpolate(trajGen.theta, trajGen.ref_positions[:, 2], theta)
    ref_yaw = pchip_interpolate(trajGen.theta, trajGen.ref_yaw, theta)
    ref_pitch = pchip_interpolate(trajGen.theta, trajGen.ref_pitch, theta)
    if trajGen.velocity_weight != 0:
        ref_vel = pchip_interpolate(trajGen.theta, trajGen.ref_velocities, theta)
    no_keyframes = trajGen.keyframes.shape[0]

    keyframe_theta_times = np.zeros(no_keyframes, dtype=int)
    for i in range(0, no_keyframes):
        tmp = np.abs(theta - trajGen.theta_of_keyframes[i])
        idx = np.argmin(tmp) # index of closest value
        keyframe_theta_times[i] = idx

    keyframe_times = np.zeros(no_keyframes, dtype=int)
    for i in range(no_keyframes):
        tmp = np.sum(np.square(np.abs((trajectory[trajGen.posStateIndices, :].transpose() - trajGen.keyframes[i, :]).transpose())), axis=0)
        idx = np.argmin(tmp) # index of closest value
        keyframe_times[i] = idx

    posPlot = trajectory[trajGen.posStateIndices, :]
    velVectors = trajectory[trajGen.velStateIndices, :]
    velProfile = np.linalg.norm(velVectors, axis=0)
    accVectors = trajectory[trajGen.accStateIndices, :]
    inputVectors = trajectory[trajGen.inputIndices, :]

    yawAngle = trajectory[trajGen.yawStateIndex, :]
    pitchAngle = trajectory[trajGen.pitchStateIndex, :]

    keyorientation_times = np.zeros(no_keyframes, dtype=int)
    orientations = np.vstack((yawAngle, pitchAngle))
    for i in range(no_keyframes):
        tmp = np.sum(np.square(np.abs((orientations.transpose() - trajGen.keyorientations[i, :]).transpose())), axis=0)
        idx = np.argmin(tmp)
        keyorientation_times[i] = idx

    yawAngleVelocity = trajectory[trajGen.yawVelStateIndex, :]
    pitchAngleVelocity = trajectory[trajGen.pitchVelStateIndex, :]
    yawAngleAcc = trajectory[trajGen.yawAccStateIndex, :]
    pitchAngleAcc = trajectory[trajGen.pitchAccStateIndex, :]

    keytimes = trajGen.keytimes
    velocity_profile = trajGen.velocity_profile

    keyframes = np.squeeze(np.asarray(trajGen.keyframes.T))
    keyorientations = np.squeeze(np.asarray(trajGen.keyorientations.T))

    fig = plt.figure()
    ax = fig.add_subplot(331, projection='3d')
    ax.plot(posPlot[0, :], posPlot[1, :], posPlot[2, :])
    ax.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2], 'b--')
    ax.scatter(keyframes[0, :], keyframes[1, :], keyframes[2, :], c="blue")
    ax.set_title("3D Position")

    ax = fig.add_subplot(332)
    thetaTime, = ax.plot(time, theta)
    thetaTimeRef, = ax.plot(trajGen.ref_timings, trajGen.theta, 'b--')
    ax.scatter(keytimes, trajGen.theta_of_keyframes, c="blue")
    # ax.legend([thetaTime, thetaTimeRef], ["theta", "theta-ref"])
    # ax.legend([thetaTime, thetaTimeRef], ["t", "tr"])
    ax.set_title("Theta")

    ax = fig.add_subplot(333)
    x, = ax.plot(time, posPlot[0, :])
    y, = ax.plot(time, posPlot[1, :])
    z, = ax.plot(time, posPlot[2, :])
    xref, = ax.plot(time, ref_pos[:, 0], 'b--')
    yref, = ax.plot(time, ref_pos[:, 1], 'g--')
    zref, = ax.plot(time, ref_pos[:, 2], 'r--')
    ax.scatter(time[keyframe_theta_times], keyframes[0, :], c="blue")
    ax.scatter(time[keyframe_theta_times], keyframes[1, :], c="green")
    ax.scatter(time[keyframe_theta_times], keyframes[2, :], c="red")
    # ax.scatter(steps[keyframe_times], posPlot[0, keyframe_times], c="blue", marker="s")
    # ax.scatter(steps[keyframe_times], posPlot[1, keyframe_times], c="green", marker="s")
    # ax.scatter(steps[keyframe_times], posPlot[2, keyframe_times], c="red", marker="s")
    # ax.legend([x, y, z, xref, yref, zref], ["x", "y", "z", "x-ref", "y-ref", "z-ref"])
    # ax.legend([x, y, z, xref, yref, zref], ["x", "y", "z", "xr", "yr", "zr"])
    ax.set_title("Position")

    ax = fig.add_subplot(334)
    quadYaw, = ax.plot(time, yawAngle[:], 'r')
    pitch, = ax.plot(time, pitchAngle[:], 'c')
    refYaw, = ax.plot(time, ref_yaw, 'r--')
    refPitch, = ax.plot(time, ref_pitch, 'c--')
    ax.scatter(time[keyframe_theta_times], keyorientations[0, :], c="red")
    ax.scatter(time[keyframe_theta_times], keyorientations[1, :], c="cyan")
    # ax.scatter(steps[keyorientation_times], yawAngle[keyorientation_times], c="red", marker="s")
    # ax.scatter(steps[keyorientation_times], pitchAngle[keyorientation_times], c="cyan", marker="s")
    # ax.legend([gimbalYaw, quadYaw, overallYaw, pitch, refYaw, refPitch], ["g. yaw", "q. yaw", "yaw", "g. pitch", "yaw-ref", "pitch-ref"])
    # ax.legend([gimbalYaw, quadYaw, overallYaw, pitch, refYaw, refPitch], ["gy", "qy", "y", "gp", "yr", "pr"])
    ax.set_title("Camera Orientation")

    ax = fig.add_subplot(335)
    vx, = ax.plot(time, velVectors[0, :])
    vy, = ax.plot(time, velVectors[1, :])
    vz, = ax.plot(time, velVectors[2, :])
    pv, = ax.plot(time, velProfile)
    ax.plot(time, yawAngleVelocity)
    ax.plot(time, pitchAngleVelocity)
    if trajGen.velocity_weight != 0:
        vref, = ax.plot(time, ref_vel, 'c--')
        ax.scatter(time[keyframe_theta_times], velocity_profile, c="cyan")
    # ax.legend([vx, vy, vz, pv, vref], ["vx", "vy", "vz", "vel-prof", "v-ref"])
    # ax.legend([vx, vy, vz, pv, vref], ["vx", "vy", "vz", "vp", "vr"])
    ax.set_title("Velocity")

    ax = fig.add_subplot(336)
    ax.plot(time, accVectors[0, :])
    ax.plot(time, accVectors[1, :])
    ax.plot(time, accVectors[2, :])
    ax.plot(time, yawAngleAcc)
    ax.plot(time, pitchAngleAcc)
    ax.set_title("Acceleration")

    # do not print last input as it is meaningless
    ax = fig.add_subplot(337)
    ax.plot(time, inputVectors[0, :])
    ax.plot(time, inputVectors[1, :])
    ax.plot(time, inputVectors[2, :])
    ax.plot(time, inputVectors[3, :])
    ax.plot(time, inputVectors[4, :])
    ax.plot(time, inputVectors[5, :])
    ax.set_title("Input")

    ax = fig.add_subplot(338)
    ax.plot(trajGen.theta, trajGen.ref_positions[:, 0], 'b--')
    ax.plot(trajGen.theta, trajGen.ref_positions[:, 1], 'g--')
    ax.plot(trajGen.theta, trajGen.ref_positions[:, 2], 'r--')
    for i in range(0, trajGen.old_progress_id.shape[0]):
        k = trajGen.old_progress_id[i]
        if k >= trajGen.nStages - (trajGen.fitlength / 2):
            fitrange = np.arange(trajGen.nStages + 1 - trajGen.fitlength, trajGen.nStages + 1, dtype=int)
        elif k > (trajGen.fitlength / 2):
            fitrange = np.arange(k - (trajGen.fitlength / 2), k + (trajGen.fitlength / 2), dtype=int)
        else:
            fitrange = np.arange(0, trajGen.fitlength, dtype=int)

        quad_fit_x = np.polyval(trajGen.param[0:3, i],trajGen.theta[fitrange])
        ax.plot(trajGen.theta[fitrange],quad_fit_x,'b')
        quad_fit_y = np.polyval(trajGen.param[3:6, i],trajGen.theta[fitrange])
        ax.plot(trajGen.theta[fitrange],quad_fit_y,'g')
        quad_fit_z = np.polyval(trajGen.param[6:9, i],trajGen.theta[fitrange])
        ax.plot(trajGen.theta[fitrange],quad_fit_z,'r')
    ax.set_title("Local Quadratic Fit Plot")

    ax = fig.add_subplot(339)
    ax.plot(trajGen.theta, trajGen.ref_yaw, 'b--')
    ax.plot(trajGen.theta, trajGen.ref_pitch, 'g--')
    for i in range(0, trajGen.old_progress_id.shape[0]):
        k = trajGen.old_progress_id[i]
        if k >= trajGen.nStages - (trajGen.fitlength / 2):
            fitrange = np.arange(trajGen.nStages + 1 - trajGen.fitlength, trajGen.nStages + 1, dtype=int)
        elif k > (trajGen.fitlength / 2):
            fitrange = np.arange(k - (trajGen.fitlength / 2), k + (trajGen.fitlength / 2), dtype=int)
        else:
            fitrange = np.arange(0, trajGen.fitlength, dtype=int)

        quad_fit_yaw = np.polyval(trajGen.param[22:25, i],trajGen.theta[fitrange])
        ax.plot(trajGen.theta[fitrange],quad_fit_yaw,'b')
        quad_fit_pitch = np.polyval(trajGen.param[25:28, i],trajGen.theta[fitrange])
        ax.plot(trajGen.theta[fitrange],quad_fit_pitch,'g')
    ax.set_title('Gimbal Local Quadratic Fit')

    if file is None:
        plt.show()
    else:
        f = plt.gcf()
        default_size = f.get_size_inches()
        f.set_size_inches((default_size[0] * 4, default_size[1] * 3))
        plt.savefig('./plots/' + file +'.png')
        plt.close(fig)