"""
This script prepares data for the forthcoming operations
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""

import os
import time
import netCDF4
import pandas as pd
from datetime import datetime
from usr_func import *


class DataHandler:

    def __init__(self, data_source="SINMOD"):
        print("h")

    def load_all_sinmod_data_from_folder(self):
        self.data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/"
        files = os.listdir(self.data_folder)
        files.sort()
        print(files)
        self.counter = 0
        for file in files:
            if file.endswith(".nc"):
                print(file)
                os.system("say new data")
                t1 = time.time()
                self.sinmod = netCDF4.Dataset(self.data_folder + file)
                ref_timestamp = datetime.strptime(file[8:18], "%Y.%m.%d").timestamp()
                self.timestamp = np.array(self.sinmod["time"]) * 24 * 3600 + ref_timestamp #change ref timestamp
                self.lat = np.array(self.sinmod['gridLats'])
                self.lon = np.array(self.sinmod['gridLons'])
                self.depth = np.array(self.sinmod['zc'])[:self.layers]
                self.salinity = np.array(self.sinmod['salinity'])[:, :self.layers, :, :]
                t2 = time.time()
                print("Time consumed: ", t2 - t1)

    def average_all_sinmod_from_folder(self):
        self.data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/"
        files = os.listdir(self.data_folder)
        files.sort()
        self.sal_ave = []
        for file in files:
            if file.endswith(".nc"):
                print(file)
                t1 = time.time()
                self.sinmod = netCDF4.Dataset(self.data_folder + file)
                ref_timestamp = datetime.strptime(file[8:18], "%Y.%m.%d").timestamp()
                self.timestamp = np.array(self.sinmod["time"]) * 24 * 3600 + ref_timestamp  # change ref timestamp
                self.lat = np.array(self.sinmod['gridLats'])
                self.lon = np.array(self.sinmod['gridLons'])
                self.depth = np.array(self.sinmod['zc'])
                self.salinity = np.mean(self.sinmod['salinity'], axis = 0)
                self.sal_ave.append(self.salinity)
                t2 = time.time()
                print("Time consumed: ", t2 - t1)
        self.sal_ave = np.array(self.sal_ave)
        self.sal_ave = np.mean(self.sal_ave, axis = 0)

        lat_reorganised = []
        lon_reorganised = []
        depth_reorganised = []
        salinity_reorganised = []
        for i in range(self.lat.shape[0]):
            for j in range(self.lat.shape[1]):
                for k in range(len(self.depth)):
                    lat_reorganised.append(self.lat[i, j])
                    lon_reorganised.append(self.lon[i, j])
                    depth_reorganised.append(self.depth[k])
                    salinity_reorganised.append(self.sal_ave[k, i, j])
        vectorise = lambda x: np.array(x).reshape(-1, 1)
        lat_reorganised, lon_reorganised, depth_reorganised, salinity_reorganised = map(vectorise, [lat_reorganised,
                                                                                                    lon_reorganised,
                                                                                                    depth_reorganised,
                                                                                                    salinity_reorganised])

        print(lat_reorganised.shape)
        print(lon_reorganised.shape)
        print(depth_reorganised.shape)
        print(salinity_reorganised.shape)
        dataset = np.hstack((lat_reorganised, lon_reorganised, depth_reorganised, salinity_reorganised))
        dataset = pd.DataFrame(dataset, columns=["lat", "lon", "depth", "salinity"])

        dataset.to_csv(self.data_folder + "data_sinmod.csv", index=False)

        # file_average = h5py.File(self.data_folder + "sinmod_ave.h5", 'w')
        # file_average.create_dataset("lat", data = self.lat)
        # file_average.create_dataset("lon", data = self.lon)
        # file_average.create_dataset("depth", data = self.depth)
        # file_average.create_dataset("salinity", data = self.sal_ave)
        print("Data is created successfully")

a = DataHandler()
a.average_all_sinmod_from_folder()


