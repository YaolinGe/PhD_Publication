import matplotlib.pyplot as plt

from usr_func import *
import netCDF4
import time


class SINMOD:

    max_depth_layer = 6 #

    def __init__(self, sinmod_path):
        sinmod = netCDF4.Dataset(sinmod_path)
        self.salinity_sinmod = np.mean(sinmod['salinity'][:, :self.max_depth_layer, :, :], axis=0)
        self.temperature_sinmod = np.mean(sinmod['temperature'][:, :self.max_depth_layer, :, :], axis=0) - 273.15 # TODO add temperature eibv
        self.depth_sinmod = np.array(sinmod['zc'][:self.max_depth_layer])
        self.lat_sinmod = np.array(sinmod['gridLats'][:, :])
        self.lon_sinmod = np.array(sinmod['gridLons'][:, :])
        self.rearrangeSINMOD()
        pass

    def rearrangeSINMOD(self):
        self.sinmod_coordinates = []
        for i in range(self.lat_sinmod.shape[0]):
            for j in range(self.lon_sinmod.shape[1]):
                for k in range(len(self.depth_sinmod)):
                    self.sinmod_coordinates.append([self.lat_sinmod[i, j], self.lon_sinmod[i, j], self.depth_sinmod[k], self.salinity_sinmod[k, i, j]])
        self.sinmod_coordinates = np.array(self.sinmod_coordinates)

    def getSINMODOnCoordinates(self, coordinates):
        self.lat_coordinates = coordinates[:, 0]
        self.lon_coordinates = coordinates[:, 1]
        self.depth_coordinates = coordinates[:, 2]
        # print(self.lat_coordinates, self.lon_coordinates, self.depth_coordinates)
        t1 = time.time()
        x_coordinates, y_coordinates = latlon2xy(self.lat_coordinates, self.lon_coordinates, 0, 0)
        x_sinmod, y_sinmod = latlon2xy(self.sinmod_coordinates[:, 0], self.sinmod_coordinates[:, 1], 0, 0)
        x_coordinates, y_coordinates, depth_coordinates, x_sinmod, y_sinmod, depth_sinmod = \
            map(vectorise, [x_coordinates, y_coordinates, self.depth_coordinates, x_sinmod, y_sinmod, self.sinmod_coordinates[:, 2]])

        self.DistanceMatrix_x = x_coordinates @ np.ones([1, len(x_sinmod)]) - np.ones([len(x_coordinates), 1]) @ x_sinmod.T
        self.DistanceMatrix_y = y_coordinates @ np.ones([1, len(y_sinmod)]) - np.ones([len(y_coordinates), 1]) @ y_sinmod.T
        self.DistanceMatrix_depth = depth_coordinates @ np.ones([1, len(depth_sinmod)]) - np.ones([len(depth_coordinates), 1]) @ depth_sinmod.T
        # self.DistanceMatrix = self.DistanceMatrix_x ** 2 + self.DistanceMatrix_y ** 2 + self.DistanceMatrix_depth ** 2
        self.DistanceMatrix = self.DistanceMatrix_x ** 2 + self.DistanceMatrix_y ** 2 + self.DistanceMatrix_depth ** 2
        self.ind_interpolated = np.argmin(self.DistanceMatrix, axis = 1) # interpolated vectorised indices
        t2 = time.time()
        print("Interpolation takes ", t2 - t1)
        self.salinity_interpolated = vectorise(self.sinmod_coordinates[self.ind_interpolated, 3])
        return self.salinity_interpolated






