import numpy as np
import pandas as pd

from Nidelva.Experiment.Field.Grid.GridGenerator import GridGenerator
from Nidelva.Experiment.Field.Grid.GridConfig import GridConfig
from Nidelva.Experiment.Data.AUVDataIntegrator import AUVDataIntegrator
from Nidelva.Experiment.Data.SINMOD import SINMOD
from Nidelva.Experiment.Coef.Coef import Coef
from Nidelva.Experiment.GP_Kernel.Matern_kernel import MaternKernel
from usr_func import *
import time


AUV_DATAPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/July06/Data/Adaptive/"
SINMOD_PATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2020.05.01.nc"
PIVOT = 63.446905, 10.419426  # right bottom corner
ANGLE_ROTATION = deg2rad(60)
NX = NY = 25
NZ = 5
XLIM = [0, 1000]
YLIM = [0, 1000]
ZLIM = [0.5, 2.5]

SILL = 4
RANGE_LATERAL = 400
RANGE_VERTICAL = 4.8
NUGGET = .3


class KrigingPlot:

    def __init__(self):
        t1 = time.time()
        gridConfig = GridConfig(PIVOT, ANGLE_ROTATION, NX, NY, NZ, XLIM, YLIM, ZLIM)
        gridGenerator = GridGenerator(gridConfig)
        self.grid_xyz = gridGenerator.xyz
        self.grid_coordinates = gridGenerator.coordinates
        t2 = time.time()
        print(t2 - t1)
        auvDataIntegrator = AUVDataIntegrator(AUV_DATAPATH, "AdaptiveData")
        self.auv_data = auvDataIntegrator.data
        self.sinmod = SINMOD("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2020.05.01.nc")
        self.coef = Coef()
        t2 = time.time()
        print(t2 - t1)
        t1 = time.time()
        self.prepareAUVData()
        self.Kriging()
        t2 = time.time()
        print("Kriging takes ", t2 - t1)

        pass

    def prepareAUVData(self):
        lat_auv, lon_auv, depth_auv = map(vectorise, [self.auv_data["lat"], self.auv_data["lon"], self.auv_data["depth"]])
        self.auv_coordinates = np.hstack((lat_auv, lon_auv, depth_auv))
        self.salinity_auv = vectorise(self.auv_data["salinity"])

    def Kriging(self):
        self.mu_sinmod = self.sinmod.getSINMODOnCoordinates(self.grid_coordinates)
        self.mu_sinmod_at_auv_loc = self.sinmod.getSINMODOnCoordinates(self.auv_coordinates)
        beta0 = np.kron(np.ones([1, NX * NY]), self.coef.beta0[:, 0])
        beta1 = np.kron(np.ones([1, NX * NY]), self.coef.beta1[:, 0])
        self.mu_prior = beta0 + beta1 * self.mu_sinmod
        self.mu_estimated = beta0 + beta1 * self.mu_sinmod_at_auv_loc

        # Grid
        Sigma_grid = MaternKernel(self.grid_coordinates, self.grid_coordinates, SILL, RANGE_LATERAL, RANGE_VERTICAL, NUGGET).Sigma
        # Grid-Obs
        Sigma_grid_obs = MaternKernel(self.grid_coordinates, self.auv_coordinates, SILL, RANGE_LATERAL, RANGE_VERTICAL,
                                  NUGGET).Sigma
        # Obs
        Sigma_obs = MaternKernel(self.auv_coordinates, self.auv_coordinates, SILL, RANGE_LATERAL, RANGE_VERTICAL, NUGGET).Sigma

        self.mu_posterior = self.mu_prior + Sigma_grid_obs @ np.linalg.solve(Sigma_obs, (self.salinity_auv - self.mu_sinmod_at_auv_loc))
        self.Sigma_posterior = Sigma_grid - Sigma_grid_obs @ np.linalg.solve(Sigma_obs, Sigma_grid_obs.T)


a = KrigingPlot()

#%%



#%%

import matplotlib.pyplot as plt
plt.scatter(a.grid_coordinates[:, 1], a.grid_coordinates[:, 0], c = a.mu_sinmod, cmap = "Paired", vmin = 16, vmax = 28)
plt.colorbar()
plt.show()


#%%
import matplotlib.pyplot as plt
ind_non_zero = np.where((a.auv_data["lon"] > 0) * (a.auv_data['lat'] > 0))[0]
im = plt.scatter(a.auv_data["lon"][ind_non_zero], a.auv_data["lat"][ind_non_zero], c = a.auv_data["salinity"][ind_non_zero], vmin = 26, vmax = 30, cmap = "Paired")
# plt.xlim([10.41, 10.43])
# plt.ylim([63.45, 63.46])
plt.colorbar(im)
# plt.show()
# plt.plot(a.auv_data["depth"])
plt.plot(a.grid_coordinates[:,1], a.grid_coordinates[:, 0], 'k.')
plt.grid()
plt.show()
# data = a.auv_data
#%%
datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/mu_prior_sal.txt"
data = np.loadtxt(datapath, delimiter=", ")

ds = np.hstack((a.grid_coordinates, data.reshape(-1, 1)))
import plotly.graph_objects as go
import plotly
import os

class Scatter3DPlot:

    def __init__(self, coordinates, filename):
        self.coordinates = coordinates
        self.filename = filename
        self.plot()

    def plot(self):
        fig = go.Figure(data=[go.Scatter3d(
            x=self.coordinates[:, 1],
            y=self.coordinates[:, 0],
            z=-self.coordinates[:, 2],
            mode="markers",
            # mode='markers+lines',
            marker=dict(
                size=12,
                color=self.coordinates[:, 3],
            ),
            # line = dict(
            #     width=3,
            #     color="yellow",
            # )
        )])
        plotly.offline.plot(fig,
                            filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/" + self.filename + ".html",
                            auto_open=False)
        os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/"+self.filename+".html")

Scatter3DPlot(np.hstack((a.grid_coordinates, a.mu_posterior)), "mu_posterior")
#%%
a.mu_sinmod.shape

