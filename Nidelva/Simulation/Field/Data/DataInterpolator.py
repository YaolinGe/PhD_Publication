"""
This script interpolates the data given a specific grid
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""

"""
Usage:
coordinates = [lat, lon, depth] # can be array
dataset_interpolated = DataInterpolator(coordinates).dataset_interpolated
"""
# TODO I have found that this is not universal, needs to be more thought of, not just for one case, for many cases in the future

import pandas as pd
import time
from usr_func import *


class DataInterpolator:

    def __init__(self, coordinates=None, data_source="SINMOD", interpolation_method="nearest_neighbour"):
        if coordinates is None:
            raise ValueError("Coordinates are not valid, please enter valid 3D coordinates consist of [lat, lon, depth]")
        # if data_source != "SINMOD" or "Delft3D":
        #     raise ModuleNotFoundError("Data source "+data_source+" cannot be found, "
        #                                                         "data source must be either SINMOD or Delft3D")
        # if interpolation_method != "nearest_neighbour":
        #     raise NotImplementedError("Interpolation method"+interpolation_method+" is not implemented yet")
        self.coordinates = coordinates
        self.data_source = data_source
        self.interpolation_method = interpolation_method
        self.interpolate_data_for_coordinates()

    def interpolate_data_for_coordinates(self):
        if self.data_source == "SINMOD":
            self.load_sinmod()
            self.interpolate_sinmod_data_for_coordinates()
        if self.data_source == "Delft3D":
            self.load_delft3d()
            self.interpolate_delft3d_data_for_coordinates()

    def load_sinmod(self):
        t1 = time.time()
        self.sinmod_datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/data_sinmod.csv"
        self.sinmod = pd.read_csv(self.sinmod_datapath)
        self.lat_sinmod = self.sinmod["lat"]
        self.lon_sinmod = self.sinmod["lon"]
        self.depth_sinmod = self.sinmod["depth"]
        self.salinity_sinmod = self.sinmod["salinity"]
        self.lat_sinmod, self.lon_sinmod, self.depth_sinmod, self.salinity_sinmod = \
            map(vectorise, [self.lat_sinmod, self.lon_sinmod, self.depth_sinmod, self.salinity_sinmod])
        t2 = time.time()
        print("Loading time consumed: ", t2 - t1)
        print("SINMOD data is loaded successfully!")

    def load_delft3d(self):
        raise NotImplementedError("load delft3d not implemented")
        # TODO add load delft3d

    def interpolate_sinmod_data_for_coordinates(self):
        self.lat_coordinates = self.coordinates[:, 0]
        self.lon_coordinates = self.coordinates[:, 1]
        self.depth_coordinates = self.coordinates[:, 2]
        # print(self.lat_coordinates, self.lon_coordinates, self.depth_coordinates)
        t1 = time.time()
        x_coordinates, y_coordinates = latlon2xy(self.lat_coordinates, self.lon_coordinates, 0, 0)
        x_sinmod, y_sinmod = latlon2xy(self.lat_sinmod, self.lon_sinmod, 0, 0)
        x_coordinates, y_coordinates, depth_coordinates, x_sinmod, y_sinmod, depth_sinmod = \
            map(vectorise, [x_coordinates, y_coordinates, self.depth_coordinates, x_sinmod, y_sinmod, self.depth_sinmod])

        self.DistanceMatrix_x = x_coordinates @ np.ones([1, len(x_sinmod)]) - np.ones([len(x_coordinates), 1]) @ x_sinmod.T
        self.DistanceMatrix_y = y_coordinates @ np.ones([1, len(y_sinmod)]) - np.ones([len(y_coordinates), 1]) @ y_sinmod.T
        self.DistanceMatrix_depth = depth_coordinates @ np.ones([1, len(depth_sinmod)]) - np.ones([len(depth_coordinates), 1]) @ depth_sinmod.T
        # self.DistanceMatrix = self.DistanceMatrix_x ** 2 + self.DistanceMatrix_y ** 2 + self.DistanceMatrix_depth ** 2
        self.DistanceMatrix = self.DistanceMatrix_x ** 2 + self.DistanceMatrix_y ** 2 + self.DistanceMatrix_depth ** 2
        self.ind_interpolated = np.argmin(self.DistanceMatrix, axis = 1) # interpolated vectorised indices
        self.salinity_interpolated = vectorise(self.salinity_sinmod[self.ind_interpolated])
        self.dataset_interpolated = pd.DataFrame(np.hstack((self.coordinates, self.salinity_interpolated)), columns = ["lat", "lon", "depth", "salinity"])
        self.dataset_interpolated.to_csv(self.sinmod_datapath + "data_sinmod_interpolated.csv")
        t2 = time.time()
        print("Data is interpolated successfully! Time consumed: ", t2 - t1)

    def interpolate_delft3d_data_for_coordinates(self):
        raise NotImplementedError("interpolation for delft3d not implemented yet")
        # TODO add interpolation for delft3d data


# if __name__ == "__main__":
#     a = DataInterpolator(coordinates=np.array([[63.42, 10.39, 0],
#                                                [63.43, 10.40, 2]]))


