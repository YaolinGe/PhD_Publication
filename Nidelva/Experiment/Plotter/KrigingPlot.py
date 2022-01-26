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
        self.plot()
        t2 = time.time()
        print("Kriging takes ", t2 - t1)

        self.save_processed_data()

        pass

    def prepareAUVData(self):
        # self.ind_mission_start = 850 # user-defined
        self.ind_mission_start = 760 # user-defined
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

    def Kriging(self):
        self.mu_sinmod = self.sinmod.getSINMODOnCoordinates(self.coordinates_grid)

        self.beta0_sinmod = np.kron(np.ones([NX * NY, 1]), self.coef.beta0[:, 0].reshape(-1, 1))
        self.beta1_sinmod = np.kron(np.ones([NX * NY, 1]), self.coef.beta1[:, 0].reshape(-1, 1))
        self.mu_prior = self.beta0_sinmod + self.beta1_sinmod * self.mu_sinmod

        # Grid
        self.Sigma_grid = MaternKernel(self.coordinates_grid, self.coordinates_grid, SILL, RANGE_LATERAL, RANGE_VERTICAL, NUGGET).Sigma

        # Grid-Obs
        self.Sigma_grid_obs = MaternKernel(self.coordinates_grid, self.WGScoordinates_auv, SILL, RANGE_LATERAL, RANGE_VERTICAL, NUGGET).Sigma

        # Obs
        self.Sigma_obs = MaternKernel(self.WGScoordinates_auv, self.WGScoordinates_auv, SILL, RANGE_LATERAL, RANGE_VERTICAL, NUGGET).Sigma

        self.mu_posterior = self.mu_prior + self.Sigma_grid_obs @ np.linalg.solve(self.Sigma_obs, (self.salinity_auv - self.mu_sinmod_at_auv_loc))
        self.Sigma_posterior = self.Sigma_grid - self.Sigma_grid_obs @ np.linalg.solve(self.Sigma_obs, self.Sigma_grid_obs.T)
        self.excursion_prob = EP_1D(self.mu_posterior, self.Sigma_posterior, THRESHOLD_SALINITY)

    def plot(self):
        fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                            subplot_titles=("Prior field", "Posterior field", "Posterior excursion probability"))
        fig.add_trace(go.Volume(
            x=self.grid_xyz[:, 1],
            y=self.grid_xyz[:, 0],
            z=-self.grid_xyz[:, 2],
            value=self.mu_prior.flatten(),
            isomin=0,
            isomax=28,
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
            value=self.mu_posterior.flatten(),
            isomin=0,
            isomax=28,
            opacity=.1,
            surface_count=30,
            colorscale="rainbow",
            # coloraxis="coloraxis1",
            colorbar=dict(x=0.65, y=0.5, len=.5),
            reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ),
            row=1, col=2
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
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(nticks=4, range=[-2, 0], showticklabels=False),
                xaxis_title='Lon [deg]',
                yaxis_title='Lat [deg]',
                zaxis_title='Depth [m]',
            ),
            scene2_aspectmode='manual',
            scene2_aspectratio=dict(x=1, y=1, z=.5),
            scene3=dict(
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(nticks=4, range=[-2, 0], showticklabels=False),
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
                            filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Field_posterior.html",
                            auto_open=False)
        os.system(
            "open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment//Field_posterior.html")

        pass

    def save_processed_data(self):
        self.pred_err = vectorise(np.sqrt(np.diag(self.Sigma_posterior)))
        self.dataframe_field = pd.DataFrame(np.hstack((self.coordinates_grid, self.mu_sinmod, self.mu_prior,
                                                       self.mu_posterior, self.excursion_prob, self.pred_err)),
                                            columns=["lat", "lon", "depth", "mean_sinmod", "mean_prior",
                                                     "mean_posterior",
                                                     "excursion_prob", "prediction_error"])
        self.dataframe_field.to_csv(
            "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/field_data.csv")

        self.dataframe_auv = pd.DataFrame(
            np.hstack((self.WGScoordinates_auv, self.mu_sinmod_at_auv_loc, self.mu_estimated)),
            columns=["lat", "lon", "depth", "mean_sinmod_at_auv_loc", "mean_estimated"])
        self.dataframe_auv.to_csv(
            "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/auv_data.csv")
        print("Data is saved successfully!")
        pass

a = KrigingPlot()


#%%
ind_surface = np.where(a.grid_xyz[:, 2] == 0.5)[0]
plt.scatter(a.grid_xyz[ind_surface, 0], a.grid_xyz[ind_surface, 1], c = a.mu_posterior[ind_surface], cmap = "Paired", vmin = 10, vmax = 30)

plt.colorbar()
plt.show()



#%%


def plot(self):
    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=("Prior field", "Posterior field", "Posterior excursion probability"))
    fig.add_trace(go.Volume(
        x=self.grid_xyz[:, 1],
        y=self.grid_xyz[:, 0],
        z=-self.grid_xyz[:, 2],
        value=self.mu_prior.flatten(),
        isomin=0,
        isomax=28,
        opacity=.1,
        surface_count=30,
        colorscale="Blues",
        # coloraxis="coloraxis1",
        colorbar=dict(x=0.3, y=0.5, len=.5),
        # reversescale=True,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ),
        row=1, col=1
    )

    fig.add_trace(go.Volume(
        x=self.grid_xyz[:, 1],
        y=self.grid_xyz[:, 0],
        z=-self.grid_xyz[:, 2],
        value=self.mu_posterior.flatten(),
        isomin=0,
        isomax=28,
        opacity=.1,
        surface_count=30,
        colorscale="Blues",
        # coloraxis="coloraxis1",
        colorbar=dict(x=0.65, y=0.5, len=.5),
        # reversescale=True,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ),
        row=1, col=2
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
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(nticks=4, range=[-2, 0], showticklabels=False),
            xaxis_title='Lon [deg]',
            yaxis_title='Lat [deg]',
            zaxis_title='Depth [m]',
        ),
        scene2_aspectmode='manual',
        scene2_aspectratio=dict(x=1, y=1, z=.5),
        scene3=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(nticks=4, range=[-2, 0], showticklabels=False),
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
                        filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Field_posterior.html",
                        auto_open=False)
    os.system(
        "open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment//Field_posterior.html")

    pass

plot(a)

#%%

def save_processed_data_slices(self):
    self.pred_err = vectorise(np.sqrt(np.diag(self.Sigma_posterior)))
    for i in range(len(self.depth_layers)):
        ind_depth_layer = np.where(self.grid_wgs84[:, 2] == self.depth_layers[i])[0]


        self.dataframe_field = pd.DataFrame(np.hstack((self.grid_wgs84[ind_depth_layer, :],
                                                       self.mu_sinmod[ind_depth_layer, :].reshape(-1, 1),
                                                       self.mu_prior[ind_depth_layer, :].reshape(-1, 1),
                                                       self.mu_posterior[ind_depth_layer, :].reshape(-1, 1),
                                                       self.excursion_prob[ind_depth_layer, :].reshape(-1, 1),
                                                       self.pred_err[ind_depth_layer, :].reshape(-1, 1))),
                                        columns=["lat", "lon", "depth", "mean_sinmod", "mean_prior",
                                                 "mean_posterior",
                                                 "excursion_prob", "prediction_error"])

        self.dataframe_field.to_csv("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/field_data_{:d}.csv".format(i), index=False)

    self.dataframe_auv = pd.DataFrame(
        np.hstack((self.WGScoordinates_auv, self.mu_sinmod_at_auv_loc, self.mu_estimated)),
        columns=["lat", "lon", "depth", "mean_sinmod_at_auv_loc", "mean_estimated"])
    self.dataframe_auv.to_csv(
        "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/auv_data.csv", index=False)
    print("Data is saved successfully!")
    pass

self = a
save_processed_data_slices(self)

#%%
path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/field_data_0.csv"
data_surface = pd.read_csv(path)
data_surface["excursion_set"] = np.ones_like(data_surface["mean_posterior"])
data_surface["excursion_set"][data_surface["mean_posterior"] > 28] = 0

grid_x, grid_y, grid_value = interpolate_2d(data_surface["lon"], data_surface["lat"], 200, 200, data_surface["excursion_prob"])

plt.scatter(grid_x, grid_y, c=grid_value, vmin=0, vmax=1)
# plt.scatter(data_surface["lon"], data_surface["lat"], c=data_surface["excursion_prob"], s=90, vmin = 0, vmax = 1)
# plt.hexbin(data_surface["lon"], data_surface["lat"], C=data_surface["excursion_prob"], gridsize=20)
# plt.pcolor([data_surface["lon"], data_surface["lat"], ], data_surface["excursion_prob"])
plt.colorbar()
plt.show()
# import seaborn as sns
#
#
# sns.distplot(data_surface, x="lon", y="lat", kind="kde")
# plt.show()

from matplotlib.gridspec import GridSpec

#%%
fig = plt.figure(figsize=(60, 10))
gs = GridSpec(nrows=1, ncols=5)
for i in range(5):
    ax = fig.add_subplot(gs[i])
    path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/field_data_{:d}.csv".format(i)
    data = pd.read_csv(path)
    grid_x, grid_y, grid_value = interpolate_2d(data["lon"], data["lat"], 200, 200, data["excursion_prob"])
    im = ax.scatter(grid_x, grid_y, c=grid_value, vmin=0, vmax=1, cmap="GnBu")
    plt.colorbar(im)
    ax.set_xlabel('Lon [deg]')
    ax.set_ylabel("Lat [deg]")
    ax.set_title("Depth {:f}m".format(self.depth_layers[i, 0]))

plt.show()

