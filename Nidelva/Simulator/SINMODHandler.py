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
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Nidelva.Simulator.MetHandler import MetHandler


class SINMODHandler(MetHandler):

    data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/"
    figpath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/SINMOD/"
    layers = 1

    def __init__(self):
        MetHandler.__init__(self)
        self.load_met()
        self.load_sinmod()
        print("hello world")

    def load_sinmod(self):
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
                self.index_element()
                self.sort_data_before_plotting()
                self.visualise_data()
                # break

    def load_met(self):
        self.load_tide()
        self.load_wind()

    def index_element(self):
        self.DM_wind = np.abs(self.timestamp.reshape(-1, 1) @ np.ones([1, len(self.wind_data["timestamp"])]) - \
                       np.ones([len(self.timestamp), 1]) @ np.array(self.wind_data["timestamp"]).reshape(1, -1))
        self.DM_tide = np.abs(self.timestamp.reshape(-1, 1) @ np.ones([1, len(self.tide_data["timestamp"])]) - \
                       np.ones([len(self.timestamp), 1]) @ np.array(self.tide_data["timestamp"]).reshape(1, -1))
        self.indices_wind = np.argmin(self.DM_wind, axis = 1)
        self.indices_tide = np.argmin(self.DM_tide, axis = 1)
        print("index element successfully!")

    def sort_data_before_plotting(self):
        self.u, self.v = self.ws2uv(np.array(self.wind_data["wind_speed"])[self.indices_wind],
                                    np.array(self.wind_data["wind_angle"])[self.indices_wind])

    def visualise_data(self):

        for i in range(self.salinity.shape[0]):
            print(self.counter)
            print(i)
            fig = plt.figure(figsize=(10, 10))
            gs = GridSpec(ncols=14, nrows=10, figure=fig)
            ax = fig.add_subplot(gs[:, :-1])
            im = ax.scatter(self.lon, self.lat, c=self.salinity[i, 0, :, :], vmin=0, vmax=30, cmap="Paired")
            # cbar = plt.colorbar()

            ax.quiver(10.38, 63.45, self.u[i], self.v[i], scale=30)
            ax.text(10.33, 63.42, "Wind speed: {:.1f} [m/s]".format(np.sqrt(self.u[i] ** 2 + self.v[i] ** 2)),
                    fontsize=14)

            divider = make_axes_locatable(ax)
            cax = divider.new_vertical(size="3%", pad=0.05, pack_start=True)
            fig.add_axes(cax)
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

            cbar.ax.set_xlabel("Salinity [ppt]")
            ax.set_title("Surface salinity on " + datetime.fromtimestamp(self.timestamp[i]).strftime("%Y-%m-%d %H:%M"))
            ax.set_xlabel("Longitude [deg]")
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            ax.set_ylabel("Latitude [deg]")
            ax.grid(linestyle="--", alpha=.4)

            ax = fig.add_subplot(gs[:, -1])
            ax.bar(0, np.array(self.tide_data["tide_measured(mm)"])[self.indices_tide[i]], width=.1, align="edge")
            # ax.text(.015, self.water_discharge[ind_water_discharge, 1] + 10,
            #         str(self.water_discharge[ind_water_discharge, 1]))
            ax.set_ylim([0, 300])
            ax.set_ylabel(r'Tide height [mm]')
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            plt.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                right=False,  # ticks along the bottom edge are off
                left=False,  # ticks along the top edge are off
                labelright=False)  # labels along the bottom edge are off

            plt.gcf().text(.01, .01, r'SINMOD: $\copyright$SINTEF', fontsize=14)
            # plt.show()
            plt.savefig(self.figpath + "/P_{:05d}.png".format(self.counter))
            plt.close("all")
            self.counter = self.counter + 1
            # break

if __name__ == "__main__":
    a = SINMODHandler()
    # a.load_sinmod()
    # a.index_element()
    # a.visualise_data()

#%%
plt.plot(np.argmin(a.DM_wind, axis = 1))
plt.show()
#%%
self = a
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualise_data(self):
    counter = 0
    for i in range(self.salinity.shape[0]):
        print(i)
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(ncols=14, nrows=10, figure=fig)
        ax = fig.add_subplot(gs[:, :-1])
        im = ax.scatter(self.lon, self.lat, c=self.salinity[i, 0, :, :], vmin=0, vmax=30, cmap="Paired")
        # cbar = plt.colorbar()

        ax.quiver(10.38, 63.45, self.u[i], self.v[i], scale = 30)
        ax.text(10.33, 63.42, "Wind speed: " + str(np.sqrt(self.u[i] ** 2 + self.v[i] ** 2)) + " [m/s]", fontsize=14)

        divider = make_axes_locatable(ax)
        cax = divider.new_vertical(size="3%", pad=0.05, pack_start=True)
        fig.add_axes(cax)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

        cbar.ax.set_xlabel("Salinity [ppt]")
        ax.set_title("Surface salinity on " + datetime.fromtimestamp(self.timestamp[i]).strftime("%Y-%m-%d %H:%M"))
        ax.set_xlabel("Longitude [deg]")
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.set_ylabel("Latitude [deg]")
        ax.grid(linestyle = "--", alpha = .4)

        ax = fig.add_subplot(gs[:, -1])
        ax.bar(0, np.array(self.tide_data["tide_measured(mm)"])[self.indices_tide[i]], width=.1, align="edge")
        # ax.text(.015, self.water_discharge[ind_water_discharge, 1] + 10,
        #         str(self.water_discharge[ind_water_discharge, 1]))
        ax.set_ylim([0, 300])
        ax.set_ylabel(r'Tide height [mm]')
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            right=False,  # ticks along the bottom edge are off
            left=False,  # ticks along the top edge are off
            labelright=False)  # labels along the bottom edge are off

        plt.gcf().text(.01, .01, r'SINMOD: $\copyright$SINTEF', fontsize = 14)
        plt.show()
        plt.savefig(self.figpath + "/P_{:05d}.png".format(counter))
        plt.close("all")
        counter = counter + 1
        break

visualise_data(self)

#%%

t = a.tide_data
plt.plot(a.tide_data["tide_measured(mm)"])
plt.show()

#%%
folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/"
files = os.listdir(folder)
for file in files:
    if file.endswith(".nc"):
        print(file)


        break

