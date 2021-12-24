import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd

from usr_func import *

class Simulator:

    sinmod_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/sinmod_ave.h5"
    box_path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Config/box.txt'
    grid_path = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Config/grid_not_tilted.txt'
    lat_origin, lon_origin = 63.448, 10.4  # origin location
    depth_limit = 5 # depth == 5 [m]


    def __init__(self):
        print("Hello world")
        self.load_sinmod()
        self.load_grid()
        self.extract_data_on_grid()

    def load_sinmod(self):
        self.sinmod = h5py.File(self.sinmod_path, 'r')
        self.lat = np.array(self.sinmod.get("lat"))
        self.lon = np.array(self.sinmod.get("lon"))
        self.depth = np.array(self.sinmod.get("depth"))
        self.salinity = np.array(self.sinmod.get("salinity"))

    def load_grid(self):
        self.grid = np.loadtxt(self.grid_path, delimiter=", ")
        print("grid is loaded successfully!")

    def get_ind_for_grid(self):
        xgrid, ygrid = latlon2xy(self.grid[:, 0], self.grid[:, 1], self.lat_origin, self.lon_origin)
        xsinmod, ysinmod = latlon2xy(self.lat.reshape(-1, 1), self.lon.reshape(-1, 1), self.lat_origin, self.lon_origin)

        DM_x = xgrid.reshape(-1, 1) @ np.ones([1, len(xsinmod)]) - np.ones([len(xgrid), 1]) @ xsinmod.reshape(1, -1)
        DM_y = ygrid.reshape(-1, 1) @ np.ones([1, len(ysinmod)]) - np.ones([len(ygrid), 1]) @ ysinmod.reshape(1, -1)
        DM = DM_x ** 2 + DM_y ** 2
        ind = np.argmin(DM, axis = 1)

        return ind

    def extract_data_on_grid(self):
        ind = self.get_ind_for_grid()
        self.salinity_extracted = []
        for i in range(len(self.depth)):
            self.salinity_extracted.append(self.salinity[i, :, :].reshape(-1, 1)[ind])

        self.salinity_extracted = np.array(self.salinity_extracted)
        print("Data is successfully extracted on the grid")

    # def setup_gaussian_process(self):
        









if __name__ == "__main__":
    a = Simulator()


#%%
plt.scatter(a.lon, a.lat, c = a.salinity[0, :, :], cmap = "Paired", vmin = 0, vmax = 30)

plt.grid()
plt.xlim([10.40, 10.42])
plt.ylim([63.448, 63.46])
plt.colorbar()
plt.show()

plt.scatter(a.grid[:, 1], a.grid[:, 0], c = a.salinity_extracted[0, :, :], cmap = "Paired", vmin = 0, vmax = 30)
plt.grid()
plt.xlim([10.40, 10.42])
plt.ylim([63.448, 63.46])
plt.colorbar()
plt.show()
# box = np.array([[63.448, 10.4],
#                 [63.448, 10.42],
#                 [63.46, 10.42],
#                 [63.46, 10.4]])
# plt.plot(box[:, 1], box[:, 0])
# np.savetxt("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Config/box.txt", box, delimiter=", ")
# plt.show()

#%%


