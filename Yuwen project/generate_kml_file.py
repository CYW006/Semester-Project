from GPSMeterConverter import GPSMeterConverter
import numpy as np
from scipy.interpolate import pchip_interpolate


def generate_kml_file(trajGen, trajectory, gps_trajectory, trajectory_name,ith):
    kml_traj = np.zeros((5, trajGen.nStages + 1))
    if gps_trajectory.shape[0] == 5:
        gpsMeter = GPSMeterConverter(gps_trajectory[1, 0], gps_trajectory[0, 0], gps_trajectory[2, 0])
    else:
        gpsMeter = GPSMeterConverter(gps_trajectory[7, 0], gps_trajectory[6, 0], gps_trajectory[8, 0])

    [kml_traj[0,:], kml_traj[1,:], kml_traj[2,:]] = gpsMeter.meter_to_gps(trajectory[trajGen.posStateIndices[0],:],
                                                                          trajectory[trajGen.posStateIndices[1],:],
                                                                          trajectory[trajGen.posStateIndices[2],:])
    kml_traj[3,:] = np.rad2deg(trajectory[trajGen.yawStateIndex, :])
    kml_traj[4,:] = np.rad2deg(trajectory[trajGen.pitchStateIndex, :])

    # interpolate trajectory
    time_step = 0.01
    end_time = trajectory[trajGen.endTimeStateIndex, 0]
    dt = end_time / trajGen.nStages
    times_mpcc = np.arange(0,trajGen.nStages+1) * dt
    interp_mpcc = np.linspace(times_mpcc[0], times_mpcc[-1], int(end_time / time_step))

    longitude = pchip_interpolate(times_mpcc, kml_traj[0, :], interp_mpcc)
    latitude = pchip_interpolate(times_mpcc, kml_traj[1, :], interp_mpcc)
    altitude = pchip_interpolate(times_mpcc, kml_traj[2, :], interp_mpcc)
    heading = pchip_interpolate(times_mpcc, kml_traj[3, :], interp_mpcc)
    tilt = pchip_interpolate(times_mpcc, kml_traj[4, :], interp_mpcc)

    # generate kml file
#    filepath = './kmls/' + trajectory_name.replace('.h5', str(ith) + '.kml')
    filepath = './kmls/' + 'test' + str(ith) + '.kml'
    write_to_kml_file(filepath, time_step, longitude, latitude, altitude, heading, tilt)


def write_to_kml_file(filename, timeStep, longitude, latitude, altitude, heading, tilt):
    file = open(filename, 'w')
    file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    file.write('<kml xmlns="http://www.opengis.net/kml/2.2"\n')
    file.write(' xmlns:gx="http://www.google.com/kml/ext/2.2">\n')
    file.write('\n')
    file.write('<Document>\n')
    file.write('<gx:Tour>\n')
    file.write('<name>Tour</name>\n')
    file.write('<gx:Playlist>\n')

    for i in range(len(longitude)):
        file.write('<gx:FlyTo>\n')
        file.write('<gx:duration>' + str(timeStep) + '</gx:duration>\n')
        file.write('<gx:flyToMode>smooth</gx:flyToMode>\n')
        # file.write('<gx:flyToMode>bounce</gx:flyToMode>\n')
        file.write('<Camera>\n')
        file.write('<longitude>' + str(longitude[i]) + '</longitude>\n')
        file.write('<latitude>' + str(latitude[i]) + '</latitude>\n')
        file.write('<altitude>' + str(altitude[i]) + '</altitude>\n')
        file.write('<heading>' + str(heading[i]) + '</heading>\n')
        file.write('<tilt>' + str(tilt[i]) + '</tilt>\n')
        file.write('<roll>0</roll>\n')
        file.write('<gx:horizFov>120</gx:horizFov>\n')
        file.write('<altitudeMode>absolute</altitudeMode>\n')
        file.write('</Camera>\n')
        file.write('</gx:FlyTo>\n')

    file.write('</gx:Playlist>\n')
    file.write('</gx:Tour>\n')
    file.write('</Document>\n')
    file.write('</kml>\n')

    file.close()

