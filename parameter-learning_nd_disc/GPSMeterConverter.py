import numpy as np


class GPSMeterConverter:

    R = 6371 * 1000 # radius of earth in meter

    def __init__(self, longitude, latitude, altitude):
        self.cos_origin_lat = np.cos(np.deg2rad(latitude))
        self.origin_long = self.R * np.deg2rad(longitude) * self.cos_origin_lat
        self.origin_lat = self.R * np.deg2rad(latitude)
        self.origin_alt = altitude

    def gps_to_meter(self, longitude, latitude, altitude):
        x = self.R * np.deg2rad(longitude) * self.cos_origin_lat - self.origin_long
        y = self.R * np.deg2rad(latitude) - self.origin_lat
        z = altitude - self.origin_alt
        return [x, y, z]

    def meter_to_gps(self, x, y, z):
        longitude = np.rad2deg((x + self.origin_long) / (self.R * self.cos_origin_lat))
        latitude = np.rad2deg((y + self.origin_lat) / self.R)
        altitude = z + self.origin_alt
        return [longitude, latitude, altitude]