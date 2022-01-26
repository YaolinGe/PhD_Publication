"""
This script prepares experiment Setup for GIS
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-26
"""
import matplotlib.pyplot as plt
import pandas as pd

from Nidelva.Experiment.Field.Grid.GridGenerator import GridGenerator
from Nidelva.Experiment.Field.Grid.GridConfig import GridConfig
from Nidelva.Experiment.Data.AUVDataIntegrator import AUVDataIntegrator
from Nidelva.Experiment.Data.SINMOD import SINMOD
from Nidelva.Experiment.Coef.Coef import Coef
from Nidelva.Experiment.GP_Kernel.Matern_kernel import MaternKernel
from usr_func import *
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots


AUV_DATAPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/May27/"
AUV_DATAPATH_ADAPTIVE = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/July06/Data/Adaptive/"
SINMOD_PATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2020.05.01.nc"
STARTING_INDEX = 760 # 850 user-defined
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

THRESHOLD_SALINITY = 28


class ExperimentSetup:

    def __init__(self):
        t1 = time.time()
        self.gridConfig = GridConfig(PIVOT, ANGLE_ROTATION, NX, NY, NZ, XLIM, YLIM, ZLIM)
        self.gridGenerator = GridGenerator(self.gridConfig)
        self.grid_xyz = self.gridGenerator.xyz
        self.grid_wgs84 = self.gridGenerator.coordinates
        t2 = time.time()
        print(t2 - t1)

        t1 = time.time()
        self.auvDataIntegrator = AUVDataIntegrator(AUV_DATAPATH, "AUVData_May27")
        self.data_auv = self.auvDataIntegrator.data
        self.sinmod = SINMOD(SINMOD_PATH)
        self.coef = Coef()
        t2 = time.time()
        print(t2 - t1)

        # self.save_grid()
        # self.plot_depth_salinity()



        pass

    def prepareAUVData(self):
        self.ind_mission_start = STARTING_INDEX # user-defined
        lat_auv_wgs, lon_auv_wgs, depth_auv_wgs = map(vectorise, [self.data_auv["lat"][self.ind_mission_start:],
                                                                  self.data_auv["lon"][self.ind_mission_start:],
                                                                  self.data_auv["depth"][self.ind_mission_start:]])
        x_auv_wgs, y_auv_wgs = latlon2xy(lat_auv_wgs, lon_auv_wgs, self.gridConfig.lat_pivot, self.gridConfig.lon_pivot)

        RotationalMatrix_WGS2USR = getRotationalMatrix_WGS2USR(self.gridConfig.angle_rotation)
        self.xyz_auv_wgs = np.hstack((vectorise(x_auv_wgs), vectorise(y_auv_wgs), vectorise(depth_auv_wgs)))
        self.xyz_auv_usr = (RotationalMatrix_WGS2USR @ self.xyz_auv_wgs.T).T
        self.WGScoordinates_auv = np.hstack((lat_auv_wgs, lon_auv_wgs, depth_auv_wgs))
        self.depth_auv_rounded = vectorise(round2base(depth_auv_wgs, .5))

        self.salinity_auv = vectorise(self.data_auv["salinity"][self.ind_mission_start:])
        self.mu_sinmod_at_auv_loc = self.sinmod.getSINMODOnCoordinates(self.WGScoordinates_auv)

        self.depth_layers = vectorise(np.unique(self.grid_xyz[:, 2]))
        self.DM_depth_auv_rounded = np.abs(self.depth_auv_rounded @ np.ones([1, len(self.depth_layers)]) - np.ones([len(self.depth_auv_rounded), 1]) @ self.depth_layers.T)
        self.ind_depth_layer = np.argmin(self.DM_depth_auv_rounded, axis = 1)

        self.mu_estimated = self.coef.beta0[self.ind_depth_layer, 0].reshape(-1, 1) + \
                            self.coef.beta1[self.ind_depth_layer, 0].reshape(-1, 1) * self.mu_sinmod_at_auv_loc

    def save_grid(self):
        print("grid: ", self.grid_wgs84.shape)
        ind_surface = get_indices_equal2value(self.grid_wgs84[:, 2], 0.5)
        self.grid_surface = self.grid_wgs84[ind_surface, :]
        df = pd.DataFrame(self.grid_surface, columns=["lat", "lon", "depth"])
        df.to_csv("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Config/grid.csv", index=False)

    def plot_depth_salinity(self):
        mycmp = cm.get_cmap('BrBG', 10)
        plt.figure(figsize=(5, 5))
        plt.scatter(x=self.data_auv['salinity'], y=np.abs(self.data_auv['depth']), c=self.data_auv['salinity'],
                    cmap=mycmp, label='Samples')
        plt.colorbar()
        plt.grid()
        plt.xlabel("Salinity [ppt]")
        plt.ylabel('Depth [m]')
        # plt.title("AUV samples plot")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        # plt.legend()
        plt.savefig("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/depth_salinity.pdf")
        plt.show()

    def plot_variogram(self):

        V_v = Variogram(coordinates=np.hstack((y_loc[i].reshape(-1, 1), x_loc[i].reshape(-1, 1))),
                        values=sal_residual[i].squeeze(), use_nugget=True, model="Matern", normalize=False,
                        n_lags=100)  # model = "Matern" check
        # V_v.estimator = 'cressie'
        V_v.fit_method = 'trf'  # moment method

        fig = V_v.plot(hist=False)
        # fig.suptitle("test")
        pass

    def save_auv_surface_data(self):
        ind_surface = np.where(self.data_auv['depth'] <= 0.5)[0]
        data_auv_surface = self.data_auv.iloc[ind_surface, :]
        data_auv_surface.to_csv(AUV_DATAPATH+"AUVSurfaceData_May27.csv", index=False)
        print("Finished data creation")


setup = ExperimentSetup()
setup.save_auv_surface_data()
print(setup.grid_wgs84)

#%%
# setup.prepareAUVData()
plt.figure()
plt.plot(setup.data_auv['lon'], setup.data_auv['lat'])
plt.show()






