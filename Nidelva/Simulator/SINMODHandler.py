#!/usr/bin/env python3
__author__ = "Yaolin Ge"
__copyright__ = "Copyright 2021, The MASCOT Project, NTNU (https://wiki.math.ntnu.no/mascot)"
__credits__ = ["Yaolin Ge"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Yaolin Ge"
__email__ = "yaolin.ge@ntnu.no"
__status__ = "UnderDevelopment"

'''
Generate actual data
'''

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
from datetime import datetime
from usr_func import *
import os
import time
import netCDF4

class SINMODHandler:

    data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/"
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/SINMOD/"
    layers = 1

    def __init__(self):
        print("hello world")

    def load_sinmod(self):
        files = os.listdir(self.data_folder)
        files.sort()
        for file in files:
            if file.endswith(".nc"):
                print(file)
                t1 = time.time()
                self.sinmod = netCDF4.Dataset(self.data_folder + file)
                self.timestamp = np.array(self.sinmod["time"]) * 24 * 3600 + datetime(2020, 5, 1).timestamp()
                self.lat = np.array(self.sinmod['gridLats'])
                self.lon = np.array(self.sinmod['gridLons'])
                self.depth = np.array(self.sinmod['zc'])[:self.layers]
                self.salinity = np.array(self.sinmod['salinity'])[:, :self.layers, :, :]
                t2 = time.time()
                print("Time consumed: ", t2 - t1)
                # self.visualise_data()
                break


    def visualise_data(self):
        print("Here comes the data visualisation")
        counter = 0
        for i in range(self.salinity.shape[0]):
            print(i)
            plt.figure(figsize=(13, 10), tight_layout = False)
            plt.scatter(self.lon, self.lat, c=self.salinity[i, 0, :, :], vmin=0, vmax=30, cmap="Paired")
            cbar = plt.colorbar()
            cbar.ax.set_title("Salinity")  # horizontal colorbar
            plt.title("Surface salinity on " + datetime.fromtimestamp(self.timestamp[i]).strftime("%Y-%m-%d %H:%M"))
            plt.xlabel("Longitude [deg]")
            plt.ylabel("Latitude [deg]")
            plt.grid(linestyle = "--", alpha = .4)
            plt.gcf().text(.01, .01, r'SINMOD: $\copyright$SINTEF', fontsize = 14)
            # plt.show()
            plt.savefig(self.figpath + "/P_{:05d}.png".format(counter))
            plt.close("all")
            counter = counter + 1
            # break

if __name__ == "__main__":
    a = SINMODHandler()
    a.load_sinmod()
    a.visualise_data()

#%%
self = a
def visualise_data(self):
    counter = 0
    for i in range(self.salinity.shape[0]):
        print(i)
        plt.figure(figsize=(13, 10), tight_layout = False)
        plt.scatter(self.lon, self.lat, c=self.salinity[i, 0, :, :], vmin=0, vmax=30, cmap="Paired")
        cbar = plt.colorbar()
        cbar.ax.set_title("Salinity")  # horizontal colorbar
        plt.title("Surface salinity on " + datetime.fromtimestamp(self.timestamp[i]).strftime("%Y-%m-%d %H:%M"))
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.grid(linestyle = "--", alpha = .4)
        plt.gcf().text(.01, .01, r'SINMOD: $\copyright$SINTEF', fontsize = 14)
        # plt.show()
        plt.savefig(self.figpath + "/P_{:05d}.png".format(counter))
        plt.close("all")
        counter = counter + 1
        # break

visualise_data(self)

#%%

datetime.fromtimestamp(a.timestamp[-1])

#%%



