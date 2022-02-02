"""
This script prepares experiment Setup for GIS
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-26
"""
import matplotlib.pyplot as plt
import numpy as np
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
# AUV_DATAPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/June17/"
# AUV_DATAPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/July06/"
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
        self.auvDataIntegrator = AUVDataIntegrator(AUV_DATAPATH, "AUVData")
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

        # == rearrange sinmod data ==
        depth_sinmod = []
        salinity_sinmod = []
        for i in range(len(self.sinmod.depth_sinmod)):
            # if i >= 20:
            #     break
            for j in range(self.sinmod.salinity_sinmod[i].shape[0]):
                for k in range(self.sinmod.salinity_sinmod[i].shape[1]):
                    depth_sinmod.append(self.sinmod.depth_sinmod[i])
                    salinity_sinmod.append(self.sinmod.salinity_sinmod[i, j, k])

        mycmp = cm.get_cmap('BrBG', 10)
        plt.figure(figsize=(5, 5))
        # plt.scatter(x=self.data_auv['salinity'], y=np.abs(self.data_auv['depth']), c=self.data_auv['salinity'],
        #             cmap=mycmp, label='Samples')
        plt.scatter(x=salinity_sinmod, y=depth_sinmod, c="red", label='SINMOD samples')
        plt.scatter(x=self.data_auv['salinity'], y=np.abs(self.data_auv['depth']), c="black", alpha=.25,
                    label='AUV samples')

        # plt.scatter(x=self.data_auv['salinity'], y=np.abs(self.data_auv['depth']), c="black", label='Samples')
        # plt.colorbar()
        plt.grid()
        plt.xlabel("Salinity [ppt]")
        plt.ylabel('Depth [m]')
        plt.title("Depth versus salinity plot")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.legend()
        plt.savefig("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/depth_salinity_comparison.pdf")
        plt.show()

    def separate_regions(self): # section is where the mission is split
        timestamp = self.data_auv['timestamp']
        self.ind_sections = np.where(np.abs(np.gradient(timestamp)) > 1)[0]
        self.ind_sections = np.append(np.zeros(1), self.ind_sections, axis=0)
        self.ind_section_start = self.ind_sections[np.arange(0, len(self.ind_sections), 2)]
        self.ind_section_end = self.ind_sections[np.arange(1, len(self.ind_sections), 2)]

    def get_variogram_for_each_section(self):
        for i in range(len(self.ind_section_start)):
            print(i)
            IND_S = int(self.ind_section_start[i])
            IND_E = int(self.ind_section_end[i])
            t1 = time.time()
            if np.amin(self.data_auv['depth'][IND_S:IND_E]) < 0:
                self.data_auv['depth'][IND_S:IND_E] = self.data_auv['depth'][IND_S:IND_E] + \
                                                      np.abs(np.amin(self.data_auv['depth'][IND_S:IND_E]))
            self.ind_surface = np.where((self.data_auv['depth'][IND_S:IND_E] <= 0.65) *
                                        (self.data_auv['depth'][IND_S:IND_E] >= 0.35))[0]
            if len(self.ind_surface):
                self.ind_surface = self.ind_surface[np.random.randint(0, len(self.ind_surface), 500)]  # select only a few
                self.lat_surface = self.data_auv['lat'][self.ind_surface]
                self.lon_surface = self.data_auv['lon'][self.ind_surface]
                self.x_surface, self.y_surface = latlon2xy(self.lat_surface, self.lon_surface, 0, 0)
                self.sal_surface = self.data_auv['salinity'][self.ind_surface]
                self.wgs_auv = np.hstack(
                    (vectorise(self.lat_surface), vectorise(self.lon_surface), np.ones([len(self.lat_surface), 1]) * 0.5))
                self.sal_surface_from_sinmod = self.sinmod.getSINMODOnCoordinates(self.wgs_auv)
                self.sal_estimated_from_sinmod = self.coef.beta0[0, 0] * np.ones_like(self.sal_surface_from_sinmod) + \
                                                 self.coef.beta1[0, 0] * self.sal_surface_from_sinmod
                self.sal_residual = self.sal_surface - self.sal_estimated_from_sinmod.squeeze()
                self.coordinates = np.hstack((vectorise(self.x_surface), vectorise(self.y_surface)))

                V_v = Variogram(coordinates=self.coordinates, values=self.sal_residual.squeeze(), use_nugget=True,
                                model="Matern",
                                normalize=False,
                                n_lags=10)  # model = "Matern" check

                self.range = V_v.parameters[0]
                self.sill = V_v.parameters[1]
                self.nugget = V_v.parameters[-1]

                self.variogram_x = V_v.data()[0]
                self.variogram_y = V_v.data()[1]

                t2 = time.time()

                mycmp = cm.get_cmap("BrBG", 10)
                fig, axes = plt.subplots(1, 4, figsize=(25, 5))
                axes[0].scatter(self.lon_surface, self.lat_surface, c=self.sal_surface, cmap=mycmp, vmin=16, vmax=30)
                axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axes[0].set_xlabel("Lon")
                axes[0].set_ylabel("Lat")

                axes[1].scatter(self.lon_surface, self.lat_surface, c=self.sal_surface_from_sinmod, cmap=mycmp, vmin=16,
                                vmax=30)
                axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axes[1].set_xlabel("Lon")
                axes[1].set_ylabel("Lat")

                im = axes[2].scatter(self.lon_surface, self.lat_surface, c=self.sal_residual, cmap=mycmp, vmin=-3, vmax=3)
                axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axes[2].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axes[2].set_xlabel("Lon")
                axes[2].set_ylabel("Lat")
                plt.colorbar(im)

                axes[3].plot(self.variogram_x, self.variogram_y, 'k-.')
                plt.text(0.5, 0.5, str(V_v), horizontalalignment='center',
                     verticalalignment='center', transform=axes[3].transAxes)
                axes[3].set_xlabel("Range")
                axes[3].set_ylabel("Sill")
                # axes[3].text(self.variogram_x[-1], self.variogram_y[-1], str(V_v))
                # axes[3].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                plt.tight_layout()
                plt.savefig("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Variogram/P_{:02d}.pdf".format(i))
            else:
                print("Not enough data")
                pass
            # plt.show()

            # print(V_v)
            # break

    def get_variogram_for_selected_data_samples_and_depth(self):
        # TODO: change it to automatically generated tolerance for depth
        # TODO: change index for start and end to auto
        # == Lateral variogram ==
        IND_S = 500
        IND_E = 700

        depth_data_auv = self.data_auv['depth'][IND_S:IND_E]
        t1 = time.time()
        if np.amin(depth_data_auv) < 0:
            depth_data_auv = depth_data_auv + np.abs(np.amin(depth_data_auv))
        self.ind_surface = np.where((depth_data_auv <= 0.65) * (depth_data_auv >= 0.35))[0]
        if len(self.ind_surface):
            # self.ind_surface = self.ind_surface[np.random.randint(0, len(self.ind_surface), 100)]  # select only a few
            self.lat_surface = self.data_auv['lat'][self.ind_surface]
            self.lon_surface = self.data_auv['lon'][self.ind_surface]
            self.x_surface, self.y_surface = latlon2xy(self.lat_surface, self.lon_surface, 0, 0)
            self.sal_surface = self.data_auv['salinity'][self.ind_surface]
            self.wgs_auv = np.hstack(
                (vectorise(self.lat_surface), vectorise(self.lon_surface), np.ones([len(self.lat_surface), 1]) * 0.5))
            self.sal_surface_from_sinmod = self.sinmod.getSINMODOnCoordinates(self.wgs_auv)
            self.sal_estimated_from_sinmod = self.coef.beta0[0, 0] * np.ones_like(self.sal_surface_from_sinmod) + \
                                             self.coef.beta1[0, 0] * self.sal_surface_from_sinmod
            self.sal_residual = self.sal_surface - self.sal_estimated_from_sinmod.squeeze()
            self.coordinates = np.hstack((vectorise(self.x_surface), vectorise(self.y_surface)))

            V_v = Variogram(coordinates=self.coordinates, values=self.sal_residual.squeeze(), use_nugget=True,
                            # model="Matern",
                            normalize=False,
                            n_lags=10)  # model = "Matern" check

            self.range_lateral = V_v.parameters[0]
            self.sill_lateral = V_v.parameters[1]
            self.nugget_lateral = V_v.parameters[-1]

            self.variogram_x_lateral = V_v.data()[0]
            self.variogram_y_lateral = V_v.data()[1]

            t2 = time.time()
            print("Lateral variogram analysis is completed, time consumed: ", t2 - t1)

        # == Depth variogram ==

        self.sal_ave_depth_sinmod = np.mean(np.mean(self.sinmod.salinity_sinmod, axis=1), axis=1)
        self.depth_auv_rounded = vectorise(round2base(vectorise(self.data_auv["depth"]), .5))
        self.depth_auv_rounded_unique = np.unique(self.depth_auv_rounded)
        # self.sal_depth_auv = vectorise(self.data_auv["salinity"])
        self.sal_ave_depth_auv = []
        self.sal_ave_depth_residual = []
        self.depth_variogram = []
        for i in range(len(self.depth_auv_rounded_unique)):
            ind_depth_layer = np.where(self.depth_auv_rounded == self.depth_auv_rounded_unique[i])[0]
            sal_ave_depth_auv_depth_layer = np.mean(self.data_auv['salinity'][ind_depth_layer])
            self.sal_ave_depth_auv.append(sal_ave_depth_auv_depth_layer)
            ind_sinmod = np.where(self.sinmod.depth_sinmod == self.depth_auv_rounded_unique[i])[0]
            if ind_sinmod:
                self.depth_variogram.append(self.depth_auv_rounded_unique[i])
                self.sal_ave_depth_residual.append(sal_ave_depth_auv_depth_layer - self.sal_ave_depth_sinmod[ind_sinmod])

        self.sal_ave_depth_residual = vectorise(self.sal_ave_depth_residual)
        self.depth_variogram = vectorise(self.depth_variogram)
        coordinates = np.hstack((np.ones([len(self.depth_variogram), 1]), self.depth_variogram))

        V_v = Variogram(coordinates=coordinates, values=self.sal_ave_depth_residual.squeeze(), use_nugget=True,
                        # model="Matern",
                        normalize=False,
                        n_lags=10)  # model = "Matern" check

        self.range_vertical = V_v.parameters[0]
        self.sill_vertical = V_v.parameters[1]
        self.nugget_vertical = V_v.parameters[-1]

        self.variogram_x_vertical = V_v.data()[0]
        self.variogram_y_vertical = V_v.data()[1]

        # plt.figure(figsize=(5,5))
        plt.subplot(121)
        plt.plot(self.variogram_x_lateral, self.variogram_y_lateral, 'k-.')
        plt.xlim([0, 320])
        plt.ylim([0, 3])
        plt.grid()
        plt.xlabel("Lateral range [m]")
        plt.ylabel("Sill")

        plt.subplot(122)
        plt.plot(self.variogram_x_vertical, self.variogram_y_vertical, 'k-.')
        plt.xlim([0, 10])
        plt.ylim([0, .5])
        plt.grid()
        plt.xlabel("Vertical range [m]")
        # plt.ylabel("Sill")
        plt.suptitle("Empirical variogram")
        plt.tight_layout()
        plt.savefig("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Variogram/variogram.pdf")
        plt.show()

    def save_auv_surface_data(self):
        ind_surface = np.where(self.data_auv['depth'] <= 0.5)[0]
        data_auv_surface = self.data_auv.iloc[ind_surface, :]
        data_auv_surface.to_csv(AUV_DATAPATH+"AUVSurfaceData_May27.csv", index=False)
        print("Finished data creation")


setup = ExperimentSetup()
setup.plot_depth_salinity()
# setup.get_variogram_for_selected_data_samples_and_depth()
# setup.separate_regions()
# setup.get_variogram_for_each_section()
# setup.save_auv_surface_data()
# os.system("say finished")
#%%

inds = 0
inde = -1
plt.plot(setup.data_auv['depth'][inds:inde])
plt.show()

plt.scatter(setup.data_auv['lon'][inds:inde], setup.data_auv['lat'][inds:inde], c=setup.data_auv['salinity'][inds:inde])
plt.show()
#%%
plt.subplot(211)
plt.plot(setup.data_auv['timestamp'])
plt.xlim([1020, 1025])
plt.subplot(212)
plt.plot(np.gradient(setup.data_auv['timestamp']))
plt.xlim([1020, 1025])
plt.show()

#%%
setup.separate_regions()
print("IND START: ", setup.ind_section_start)
print("IND END: ", setup.ind_section_end)

# IND START:  [    0.  1022.  2166.  4685.  5141.  7480. 10008. 11118.]
# IND END:  [ 1021.  2165.  4684.  5140.  7479. 10007. 11117.]


#%%
plt.plot(setup.data_auv['depth'][490:750])
plt.show()

#%%
location = np.array([[63.447925, 10.410633]])
df = pd.DataFrame(location, columns=['lat', 'lon'])
df.to_csv("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/GIS/location.csv", index=False)



#%%
self = setup
# Working
IND_S = 490
IND_E = 750

# IND_S = 520
# IND_E = 720
#
# IND_S = 0
# IND_E = -1

t1 = time.time()
if np.amin(self.data_auv['depth'][IND_S:IND_E]) < 0:
    self.data_auv['depth'][IND_S:IND_E] = self.data_auv['depth'][IND_S:IND_E] + \
                                          np.abs(np.amin(self.data_auv['depth'][IND_S:IND_E]))
self.ind_surface = np.where((self.data_auv['depth'][IND_S:IND_E] <= 0.55) *
                            (self.data_auv['depth'][IND_S:IND_E] >= 0.45))[0]
if len(self.ind_surface):
    self.ind_surface = self.ind_surface[np.random.randint(0, len(self.ind_surface), 1000)]  # select only a few
    self.lat_surface = self.data_auv['lat'][self.ind_surface]
    self.lon_surface = self.data_auv['lon'][self.ind_surface]
    self.x_surface, self.y_surface = latlon2xy(self.lat_surface, self.lon_surface, 0, 0)
    self.sal_surface = self.data_auv['salinity'][self.ind_surface]
    self.wgs_auv = np.hstack(
        (vectorise(self.lat_surface), vectorise(self.lon_surface), np.ones([len(self.lat_surface), 1]) * 0.5))
    self.sal_surface_from_sinmod = self.sinmod.getSINMODOnCoordinates(self.wgs_auv)
    self.sal_estimated_from_sinmod = self.coef.beta0[0, 0] * np.ones_like(self.sal_surface_from_sinmod) + \
                                     self.coef.beta1[0, 0] * self.sal_surface_from_sinmod
    self.sal_residual = self.sal_surface - self.sal_estimated_from_sinmod.squeeze()
    self.coordinates = np.hstack((vectorise(self.x_surface), vectorise(self.y_surface)))

    V_v = Variogram(coordinates=self.coordinates, values=self.sal_residual.squeeze(), use_nugget=True,
                    # model="Matern",
                    normalize=False,
                    n_lags=10)  # model = "Matern" check

    self.range = V_v.parameters[0]
    self.sill = V_v.parameters[1]
    self.nugget = V_v.parameters[-1]

    self.variogram_x = V_v.data()[0]
    self.variogram_y = V_v.data()[1]

    t2 = time.time()

    mycmp = cm.get_cmap("BrBG", 10)
    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    axes[0].scatter(self.lon_surface, self.lat_surface, c=self.sal_surface, cmap=mycmp, vmin=16, vmax=30)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[0].set_xlabel("Lon")
    axes[0].set_ylabel("Lat")

    axes[1].scatter(self.lon_surface, self.lat_surface, c=self.sal_surface_from_sinmod, cmap=mycmp, vmin=16,
                    vmax=30)
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[1].set_xlabel("Lon")
    axes[1].set_ylabel("Lat")

    im = axes[2].scatter(self.lon_surface, self.lat_surface, c=self.sal_residual, cmap=mycmp, vmin=-3, vmax=3)
    axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[2].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[2].set_xlabel("Lon")
    axes[2].set_ylabel("Lat")
    # plt.colorbar(im)

    axes[3].plot(self.variogram_x, self.variogram_y, 'k-.')
    plt.text(0.5, 0.5, str(V_v), horizontalalignment='center',
             verticalalignment='center', transform=axes[3].transAxes)
    axes[3].set_xlabel("Range")
    axes[3].set_ylabel("Sill")
    # axes[3].text(self.variogram_x[-1], self.variogram_y[-1], str(V_v))
    # axes[3].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # plt.tight_layout()
    plt.show()



#%%
# setup.prepareAUVData()
# plt.figure()
# plt.plot(setup.data_auv['lon'], setup.data_auv['lat'])
# plt.show()
x = setup.lon_surface
y = setup.lat_surface
z = setup.data_auv['depth'][setup.ind_surface]
values = setup.sal_surface

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=values,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

plotly.offline.plot(fig, filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/variogram.html", auto_open = False)
os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/variogram.html")

#%%

V_v.plot()




