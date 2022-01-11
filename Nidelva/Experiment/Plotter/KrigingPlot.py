
from Nidelva.Experiment.Field.Grid.GridGenerator import GridGenerator
from Nidelva.Experiment.Field.Grid.GridConfig import GridConfig
from Nidelva.Experiment.Data.AUVDataIntegrator import AUVDataIntegrator
from Nidelva.Experiment.Data.SINMOD import SINMOD
from Nidelva.Experiment.Coef.Coef import Coef
from Nidelva.Experiment.GP_Kernel.Matern_kernel import MaternKernel
from usr_func import *
import time, os
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots


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

THRESHOLD_SALINITY = 28


class KrigingPlot:

    def __init__(self):
        t1 = time.time()
        self.gridConfig = GridConfig(PIVOT, ANGLE_ROTATION, NX, NY, NZ, XLIM, YLIM, ZLIM)
        self.gridGenerator = GridGenerator(self.gridConfig)
        self.grid_xyz = self.gridGenerator.xyz
        self.coordinates_grid = self.gridGenerator.coordinates
        t2 = time.time()
        print(t2 - t1)

        t1 = time.time()
        self.auvDataIntegrator = AUVDataIntegrator(AUV_DATAPATH, "AdaptiveData")
        self.data_auv = self.auvDataIntegrator.data
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
        lat_auv_wgs, lon_auv_wgs, depth_auv_wgs = map(vectorise, [self.data_auv["lat"], self.data_auv["lon"], self.data_auv["depth"]])
        x_auv_wgs, y_auv_wgs = latlon2xy(lat_auv_wgs, lon_auv_wgs, self.gridConfig.lat_pivot, self.gridConfig.lon_pivot)
        RotationalMatrix_WGS2USR = getRotationalMatrix_WGS2USR(self.gridConfig.angle_rotation)
        self.xyz_auv_wgs = np.hstack((vectorise(x_auv_wgs), vectorise(y_auv_wgs), vectorise(depth_auv_wgs)))
        self.xyz_auv_usr = (RotationalMatrix_WGS2USR @ self.xyz_auv_wgs.T).T
        self.WGScoordinates_auv = np.hstack((lat_auv_wgs, lon_auv_wgs, depth_auv_wgs))
        self.depth_auv_rounded = vectorise(round2base(depth_auv_wgs, .5))
        self.salinity_auv = vectorise(self.data_auv["salinity"])
        self.mu_sinmod_at_auv_loc = self.sinmod.getSINMODOnCoordinates(self.WGScoordinates_auv)

        self.depth_layers = vectorise(np.unique(self.grid_xyz[:, 2]))
        self.DM_depth_auv_rounded = np.abs(self.depth_auv_rounded @ np.ones([1, len(self.depth_layers)]) - np.ones([len(self.depth_auv_rounded), 1]) @ self.depth_layers.T)
        self.ind_depth_layer = np.argmin(self.DM_depth_auv_rounded, axis = 1)

        self.mu_estimated = self.coef.beta0[self.ind_depth_layer, 0].reshape(-1, 1) + \
                            self.coef.beta1[self.ind_depth_layer, 0].reshape(-1, 1) * self.mu_sinmod_at_auv_loc

    def Kriging(self):
        self.mu_sinmod = self.sinmod.getSINMODOnCoordinates(self.coordinates_grid)

        self.beta0_sinmod = np.kron(np.ones([NX * NY, 1]), self.coef.beta0[:, 0].reshape(-1, 1))
        self.beta1_sinmod = np.kron(np.ones([NX * NY, 1]), self.coef.beta1[:, 0].reshape(-1, 1))
        self.mu_prior = self.beta0_sinmod + self.beta1_sinmod * self.mu_sinmod


        # Grid
        Sigma_grid = MaternKernel(self.coordinates_grid, self.coordinates_grid, SILL, RANGE_LATERAL, RANGE_VERTICAL, NUGGET).Sigma

        # Grid-Obs
        Sigma_grid_obs = MaternKernel(self.coordinates_grid, self.WGScoordinates_auv, SILL, RANGE_LATERAL, RANGE_VERTICAL, NUGGET).Sigma

        # Obs
        Sigma_obs = MaternKernel(self.WGScoordinates_auv, self.WGScoordinates_auv, SILL, RANGE_LATERAL, RANGE_VERTICAL, NUGGET).Sigma

        self.mu_posterior = self.mu_prior + Sigma_grid_obs @ np.linalg.solve(Sigma_obs, (self.salinity_auv - self.mu_sinmod_at_auv_loc))
        self.Sigma_posterior = Sigma_grid - Sigma_grid_obs @ np.linalg.solve(Sigma_obs, Sigma_grid_obs.T)
        self.excursion_prob = EP_1D(self.mu_posterior, self.Sigma_posterior, THRESHOLD_SALINITY)

    def plot(self):
        fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                            subplot_titles=("Mean", "Std", "EP"))
        fig.add_trace(go.Volume(
            x=self.grid_xyz[:, 1],
            y=self.grid_xyz[:, 0],
            z=-self.grid_xyz[:, 2],
            value=self.mu_posterior.flatten(),
            isomin=28,
            isomax=35,
            opacity=.1,
            surface_count=30,
            colorscale="rainbow",
            # coloraxis="coloraxis1",
            colorbar=dict(x=0.3, y=0.5, len=.5),
            reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ),
            row=1, col=1
        )
        fig.add_trace(go.Volume(
            x=self.grid_xyz[:, 1],
            y=self.grid_xyz[:, 0],
            z=-self.grid_xyz[:, 2],
            value=self.excursion_prob.flatten(),
            isomin=0,
            isomax=1,
            opacity=.1,
            surface_count=30,
            colorscale="gnbu",
            # coloraxis="coloraxis1",
            colorbar=dict(x=1, y=0.5, len=.5),
            reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ),
            row=1, col=3
        )
        fig.add_trace(go.Scatter3d(
            # print(trajectory),
            x=self.xyz_auv_usr[:, 1],
            y=self.xyz_auv_usr[:, 0],
            z=-self.xyz_auv_usr[:, 2],
            mode='markers+lines',
            marker=dict(
                size=5,
                color="black",
                showscale=False,
            ),
            line=dict(
                color="yellow",
                width=3,
                showscale=False,
            ),
            showlegend=False,
        ),
            row='all', col='all'
        )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2.25, y=2.25, z=2.25)
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(nticks=4, range=[-2, 0], showticklabels=False),
                xaxis_title='Lon [deg]',
                yaxis_title='Lat [deg]',
                zaxis_title='Depth [m]',
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.5),
            scene2=dict(
                zaxis=dict(nticks=4, range=[-2, 0], ),
                xaxis_title='Lon [deg]',
                yaxis_title='Lat [deg]',
                zaxis_title='Depth [m]',
            ),
            scene2_aspectmode='manual',
            scene2_aspectratio=dict(x=1, y=1, z=.5),
            scene3=dict(
                zaxis=dict(nticks=4, range=[-2, 0], ),
                xaxis_title='Lon [deg]',
                yaxis_title='Lat [deg]',
                zaxis_title='Depth [m]',
            ),
            scene3_aspectmode='manual',
            scene3_aspectratio=dict(x=1, y=1, z=.5),
            scene_camera=camera,
            scene2_camera=camera,
            scene3_camera=camera,
        )
        plotly.offline.plot(fig,
                            filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/mu_posterior.html",
                            auto_open=False)
        os.system(
            "open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment//mu_posterior.html")

        pass

a = KrigingPlot()
#%%
self.mu_sinmod.shape
#%%

#%%

self = a

def plot(self):
    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=("Mean", "Std", "EP"))
    fig.add_trace(go.Volume(
        x=self.grid_xyz[:, 1],
        y=self.grid_xyz[:, 0],
        z=-self.grid_xyz[:, 2],
        value=self.mu_posterior.flatten(),
        isomin=28,
        isomax=35,
        opacity=.1,
        surface_count=30,
        colorscale="rainbow",
        # coloraxis="coloraxis1",
        colorbar=dict(x=0.3, y=0.5, len=.5),
        reversescale=True,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ),
        row=1, col=1
    )
    fig.add_trace(go.Volume(
        x=self.grid_xyz[:, 1],
        y=self.grid_xyz[:, 0],
        z=-self.grid_xyz[:, 2],
        value=self.excursion_prob.flatten(),
        isomin=0,
        isomax=1,
        opacity=.1,
        surface_count=30,
        colorscale="gnbu",
        # coloraxis="coloraxis1",
        colorbar=dict(x=1, y=0.5, len=.5),
        reversescale=True,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ),
        row=1, col=3
    )
    fig.add_trace(go.Scatter3d(
        # print(trajectory),
        x=self.xyz_auv_usr[:, 1],
        y=self.xyz_auv_usr[:, 0],
        z=-self.xyz_auv_usr[:, 2],
        mode='markers+lines',
        marker=dict(
            size=5,
            color="black",
            showscale=False,
        ),
        line=dict(
            color="yellow",
            width=3,
            showscale=False,
        ),
        showlegend=False,
    ),
        row='all', col='all'
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2.25, y=2.25, z=2.25)
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(nticks=4, range=[-2, 0], showticklabels=False),
            xaxis_title='Lon [deg]',
            yaxis_title='Lat [deg]',
            zaxis_title='Depth [m]',
        ),
        scene_aspectmode='manual',
        scene_aspectratio=dict(x=1, y=1, z=.5),
        scene2=dict(
            zaxis=dict(nticks=4, range=[-2, 0], ),
            xaxis_title='Lon [deg]',
            yaxis_title='Lat [deg]',
            zaxis_title='Depth [m]',
        ),
        scene2_aspectmode='manual',
        scene2_aspectratio=dict(x=1, y=1, z=.5),
        scene3=dict(
            zaxis=dict(nticks=4, range=[-2, 0], ),
            xaxis_title='Lon [deg]',
            yaxis_title='Lat [deg]',
            zaxis_title='Depth [m]',
        ),
        scene3_aspectmode='manual',
        scene3_aspectratio=dict(x=1, y=1, z=.5),
        scene_camera=camera,
        scene2_camera=camera,
        scene3_camera=camera,
    )
    plotly.offline.plot(fig,
                        filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/mu_posterior.html",
                        auto_open=False)
    os.system(
        "open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment//mu_posterior.html")


plot(self)




#%%



#%%

import matplotlib.pyplot as plt
plt.scatter(a.coordinates_grid[:, 1], a.coordinates_grid[:, 0], c = a.mu_sinmod, cmap ="Paired", vmin = 16, vmax = 28)
plt.colorbar()
plt.show()


#%%
import matplotlib.pyplot as plt
ind_non_zero = np.where((a.data_auv["lon"] > 0) * (a.data_auv['lat'] > 0))[0]
im = plt.scatter(a.data_auv["lon"][ind_non_zero], a.data_auv["lat"][ind_non_zero], c = a.data_auv["salinity"][ind_non_zero], vmin = 26, vmax = 30, cmap ="Paired")
# plt.xlim([10.41, 10.43])
# plt.ylim([63.45, 63.46])
plt.colorbar(im)
# plt.show()
# plt.plot(a.auv_data["depth"])
plt.plot(a.coordinates_grid[:, 1], a.coordinates_grid[:, 0], 'k.')
plt.grid()
plt.show()
# data = a.auv_data
#%%
datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/mu_prior_sal.txt"
data = np.loadtxt(datapath, delimiter=", ")

ds = np.hstack((a.coordinates_grid, data.reshape(-1, 1)))
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

Scatter3DPlot(np.hstack((a.coordinates_grid, a.mu_posterior)), "mu_posterior")
#%%
a.mu_sinmod.shape

