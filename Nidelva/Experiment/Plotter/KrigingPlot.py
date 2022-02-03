import numpy as np
import pandas as pd

from Nidelva.Experiment.Field.Grid.GridGenerator import GridGenerator
from Nidelva.Experiment.Field.Grid.GridConfig import GridConfig
from Nidelva.Experiment.Data.AUVDataIntegrator import AUVDataIntegrator
from Nidelva.Experiment.Data.SINMOD import SINMOD
from Nidelva.Experiment.Coef.Coef import Coef
from Nidelva.Experiment.GP_Kernel.Matern_kernel import MaternKernel
from usr_func import *


AUV_DATAPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/July06/Adaptive/"
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
        self.xyz_grid = self.gridGenerator.xyz
        self.depth_layers = vectorise(np.unique(self.xyz_grid[:, 2]))
        self.coordinates_grid = self.gridGenerator.coordinates

        self.auvDataIntegrator = AUVDataIntegrator(AUV_DATAPATH, "AdaptiveData")
        self.data_auv = self.auvDataIntegrator.data
        self.sinmod = SINMOD(SINMOD_PATH)
        self.coef = Coef()
        t2 = time.time()
        print("Setup takes: ", t2 - t1, " seconds")

        t1 = time.time()
        self.prepareAUVData()
        self.get_prior()
        self.Kriging()
        # self.plot()
        t2 = time.time()
        print("Kriging takes ", t2 - t1)

        pass

    def prepareAUVData(self):
        print("Here comes the auv data reorganisation")
        print("length of dataset: ", len(self.data_auv))
        datapath_auv = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/data_auv.csv"
        if os.path.exists(datapath_auv):
            print("Dataset exists already, will only load...")
            t1 = time.time()
            df = pd.read_csv(datapath_auv)
            self.coordinates_auv_wgs = df.iloc[:, :3].to_numpy()
            self.xyz_auv_wgs = df.iloc[:, 3:6].to_numpy()
            self.xyz_auv_usr = df.iloc[:, 6:9].to_numpy()
            self.salinity_auv = df.iloc[:, 9].to_numpy()
            self.mu_sinmod_at_auv_loc = df.iloc[:, 10].to_numpy()
            self.mu_estimated = df.iloc[:, -1].to_numpy()
            t2 = time.time()
            print("Data is loaded successfully!, time consumed>", t2 - t1)
            pass
        else:
            print("Data does not exist, will create new!")
            # self.ind_mission_start = 850 # user-defined
            self.ind_mission_start = 760 # user-defined
            lat_auv_wgs, lon_auv_wgs, depth_auv_wgs, self.salinity_auv = map(vectorise,
                                                                             [self.data_auv["lat"][self.ind_mission_start:],
                                                                              self.data_auv["lon"][self.ind_mission_start:],
                                                                              self.data_auv["depth"][self.ind_mission_start:],
                                                                              self.data_auv['salinity'][self.ind_mission_start:]])
            print("GridSize>", lat_auv_wgs.shape)
            t1 = time.time()
            x_auv_wgs, y_auv_wgs = latlon2xy(lat_auv_wgs, lon_auv_wgs, self.gridConfig.lat_pivot, self.gridConfig.lon_pivot)
            RotationalMatrix_WGS2USR = getRotationalMatrix_WGS2USR(self.gridConfig.angle_rotation)
            self.xyz_auv_wgs = np.hstack((vectorise(x_auv_wgs), vectorise(y_auv_wgs), vectorise(depth_auv_wgs)))
            self.xyz_auv_usr = (RotationalMatrix_WGS2USR @ self.xyz_auv_wgs.T).T
            self.coordinates_auv_wgs = np.hstack((lat_auv_wgs, lon_auv_wgs, depth_auv_wgs))
            self.depth_auv_rounded = vectorise(round2base(depth_auv_wgs, .5))

            self.mu_sinmod_at_auv_loc = self.sinmod.getSINMODOnCoordinates(self.coordinates_auv_wgs)
            t2 = time.time()
            print("Finished attaching data on auv location, it takes", t2 - t1, " seconds")

            self.DM_depth_auv_rounded = np.abs(self.depth_auv_rounded @ np.ones([1, len(self.depth_layers)]) -
                                               np.ones([len(self.depth_auv_rounded), 1]) @ self.depth_layers.T)
            self.ind_depth_layer = np.argmin(self.DM_depth_auv_rounded, axis = 1)

            self.mu_estimated = self.coef.beta0[self.ind_depth_layer, 0].reshape(-1, 1) + \
                                self.coef.beta1[self.ind_depth_layer, 0].reshape(-1, 1) * self.mu_sinmod_at_auv_loc
            df = pd.DataFrame(np.hstack((self.coordinates_auv_wgs, self.xyz_auv_wgs, self.xyz_auv_usr,
                                         self.salinity_auv, self.mu_sinmod_at_auv_loc, self.mu_estimated)),
                              columns=['lat_wgs', 'lon_wgs', 'depth_wgs', 'x_wgs', 'y_wgs', 'z_wgs',
                                       'x_usr', 'y_usr', 'z_usr', 'salinity_auv', 'mu_sinmod_at_loc', 'mu_estimated'])
            df.to_csv(datapath_auv, index=False)
            print("Finished data saving: time consumed>", t2 - t1, ' seconds')

    def get_prior(self):
        prior_path = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/mu_prior.csv"
        if os.path.exists(prior_path):
            print("Prior data exists already, will load...")
            t1 = time.time()
            df = pd.read_csv(prior_path)
            self.mu_sinmod = df.iloc[:, 0].to_numpy()
            self.mu_prior = df.iloc[:, 1].to_numpy()
            t2 = time.time()
            print("Prior data is loaded successfully! Time consumed>", t2 - t1)
            pass
        else:
            print("Prior data is not ready, will create one.")
            t1 = time.time()
            self.mu_sinmod = self.sinmod.getSINMODOnCoordinates(self.coordinates_grid)
            self.mu_prior = []
            for i in range(self.mu_sinmod.shape[0]):
                ind_depth_layer = np.where(self.coordinates_grid[i, 2] == self.depth_layers)[0]
                self.mu_prior.append(self.coef.beta0[ind_depth_layer, 0] +
                                     self.coef.beta1[ind_depth_layer, 0] * self.mu_sinmod[i])
            self.mu_prior = vectorise(self.mu_prior)
            df = pd.DataFrame(np.hstack((self.mu_sinmod, self.mu_prior)), columns=['mu_sinmod', 'mu_prior'])
            df.to_csv(prior_path, index=False)
            t2 = time.time()
            print("Finished prior creation, time consumed>", t2 - t1)


    def Kriging(self):
        print("Here comes the kriging...")
        t1 = time.time()
        # Grid
        self.Sigma_grid = MaternKernel(self.coordinates_grid, self.coordinates_grid, SILL,
                                       RANGE_LATERAL, RANGE_VERTICAL, NUGGET).Sigma

        # Grid-Obs
        self.Sigma_grid_obs = MaternKernel(self.coordinates_grid, self.coordinates_auv_wgs, SILL,
                                           RANGE_LATERAL, RANGE_VERTICAL, NUGGET).Sigma

        # Obs
        self.Sigma_obs = MaternKernel(self.coordinates_auv_wgs, self.coordinates_auv_wgs, SILL,
                                      RANGE_LATERAL, RANGE_VERTICAL, NUGGET).Sigma + \
                         NUGGET * np.identity(self.coordinates_auv_wgs.shape[0]) # TODO VERY IMPORTANT

        self.mu_posterior = self.mu_prior + self.Sigma_grid_obs @ np.linalg.solve(self.Sigma_obs, (self.salinity_auv - self.mu_estimated))
        self.Sigma_posterior = self.Sigma_grid - self.Sigma_grid_obs @ np.linalg.solve(self.Sigma_obs, self.Sigma_grid_obs.T)

        self.excursion_set_prior = get_excursion_set(self.mu_prior, THRESHOLD_SALINITY)
        self.excursion_prob_prior = get_excursion_prob_1d(self.mu_prior, self.Sigma_grid, THRESHOLD_SALINITY)
        self.boundary_prior = self.get_boundary(self.excursion_prob_prior)

        self.excursion_set_posterior = get_excursion_set(self.mu_posterior, THRESHOLD_SALINITY)
        self.excursion_prob_posterior = get_excursion_prob_1d(self.mu_posterior, self.Sigma_posterior,
                                                              THRESHOLD_SALINITY)
        self.boundary_posterior = self.get_boundary(self.excursion_prob_posterior)
        t2 = time.time()
        print("Finished kriging, time consumed>", t2 - t1)

    def get_fake_rectangular_grid(self):
        lat_grid, lon_grid = xy2latlon(self.xyz_grid[:, 0], self.xyz_grid[:, 1], self.gridConfig.lat_pivot, self.gridConfig.lon_pivot)
        lat_grid, lon_grid, depth_grid = map(vectorise, [lat_grid, lon_grid, self.xyz_grid[:, 2]])
        self.coordinates_fake = np.hstack((lat_grid, lon_grid, depth_grid))
        lat_auv, lon_auv = xy2latlon(self.xyz_auv_usr[:, 0], self.xyz_auv_usr[:, 1], self.gridConfig.lat_pivot, self.gridConfig.lon_pivot)
        lat_auv, lon_auv, depth_auv = map(vectorise, [lat_auv, lon_auv, self.xyz_auv_usr[:, 2]])
        self.coordinates_fake_auv = np.hstack((lat_auv, lon_auv, depth_auv))
        print("Fake coordinates are created successfully!")

    def get_boundary(self, excursion_prob):
        boundary = np.zeros_like(excursion_prob)
        # ind_boundary = np.where((excursion_prob >= .475) * (excursion_prob <= .525))[0]
        ind_boundary = np.where(excursion_prob >= .65)[0]
        boundary[ind_boundary] = True
        return boundary

    def plot(self):
        self.get_fake_rectangular_grid()
        # == smoothing section
        lat = self.coordinates_fake[:, 0]
        lon = self.coordinates_fake[:, 1]
        depth = self.coordinates_fake[:, 2]

        coordinates_mu_prior, values_mu_prior = interpolate_3d(lon, lat, depth, self.mu_prior)
        coordinates_ep_prior, values_ep_prior = interpolate_3d(lon, lat, depth, self.excursion_prob_prior)
        coordinates_es_prior, values_es_prior = interpolate_3d(lon, lat, depth, self.excursion_set_prior)
        coordinates_boundary_prior, values_boundary_prior = interpolate_3d(lon, lat, depth, self.boundary_prior)

        coordinates_mu_posterior, values_mu_posterior = interpolate_3d(lon, lat, depth, self.mu_posterior)
        coordinates_ep_posterior, values_ep_posterior = interpolate_3d(lon, lat, depth, self.excursion_prob_posterior)
        coordinates_es_posterior, values_es_posterior = interpolate_3d(lon, lat, depth, self.excursion_set_posterior)
        coordinates_boundary_posterior, values_boundary_posterior = interpolate_3d(lon, lat, depth, self.boundary_posterior)

        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        # fig.add_trace(go.Scatter3d(
        #     x=self.coordinates_fake[:, 1],
        #     y=self.coordinates_fake[:, 0],
        #     z=-self.coordinates_fake[:, 2],
        #     # value=self.mu_sinmod.flatten(),
        #
        #     # isomin=0,
        #     # isomax=30,
        #     # opacity=.1,
        #     # surface_count=30,
        #     mode='markers',
        #         marker=dict(
        #             size=5,
        #             color=self.mu_posterior.flatten(),
        #             colorscale="brbg",
        #             showscale=True,
        #
        #         ),
        #     # colorscale="brbg",
        #     # caps=dict(x_show=False, y_show=False, z_show=False),
        # ),
        #     row=1, col=1
        # )

        posterior = True

        # == posterior ==
        if posterior:
            coordinates_mean = coordinates_mu_posterior
            values_mean = values_mu_posterior

            coordinates_es = coordinates_es_posterior
            values_es = values_es_posterior

            coordinates_ep = coordinates_ep_posterior
            values_ep = values_ep_posterior

            coordinates_b = coordinates_boundary_posterior
            values_b = values_boundary_posterior

        else:
            coordinates_mean = coordinates_mu_prior
            values_mean = values_mu_prior

            coordinates_es = coordinates_es_prior
            values_es = values_es_prior

            coordinates_ep = coordinates_ep_prior
            values_ep = values_ep_prior

            coordinates_b = coordinates_boundary_prior
            values_b = values_boundary_prior

        fig.add_trace(go.Volume(
            x=coordinates_mean[:, 0],
            y=coordinates_mean[:, 1],
            z=-coordinates_mean[:, 2],
            value=values_mean.flatten(),
            isomin=16,
            isomax=30,
            opacity=.1,
            surface_count=30,
            coloraxis="coloraxis",
            caps=dict(x_show=False, y_show=False, z_show=False),
        ),
            row=1, col=1
        )

        # zv = coordinates_ep[:, 2]
        # for j in range(len(np.unique(zv))):
        #     ind = (zv == np.unique(zv)[j])
        #     fig.add_trace(
        #         go.Isosurface(x=coordinates_ep[ind, 0],
        #                       y=coordinates_ep[ind, 1],
        #                       z=-zv[ind],
        #                       opacity=.3,
        #                       # value=mu_cond[ind], colorscale=newcmp),
        #                       value=values_ep[ind], coloraxis="coloraxis"),
        #         row=1, col=1
        #     )

        # fig.add_trace(go.Volume(
        #     x=coordinates_ep[:, 0],
        #     y=coordinates_ep[:, 1],
        #     z=-coordinates_ep[:, 2],
        #     value=values_ep.flatten(),
        #     isomin=0,
        #     isomax=1,
        #     opacity=.05,
        #     surface_count=15,
        #     coloraxis="coloraxis",
        #     caps=dict(x_show=False, y_show=False, z_show=False),
        # ),
        #     row=1, col=1
        # )
        fig.add_trace(go.Volume(
            x=coordinates_ep[:, 0],
            y=coordinates_ep[:, 1],
            z=-coordinates_ep[:, 2],
            value=values_ep.flatten(),
            isomin=.75,
            isomax=1,
            opacity=.7,
            surface_count=1,
            # colorscale="Peach",
            colorscale="turbid",
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ),
            row=1, col=1
        )

        if posterior:
            fig.add_trace(go.Scatter3d(
                x=self.coordinates_fake_auv[:, 1],
                y=self.coordinates_fake_auv[:, 0],
                z=-self.coordinates_fake_auv[:, 2],
                # mode='markers+lines',
                mode='lines',
                line=dict(
                    color="black",
                    width=5,
                    showscale=False,
                ),
                showlegend=False,
            ),
                row='all', col='all'
            )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1)
        )
        fig.update_coloraxes(colorscale="BrBG", colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                                tickfont=dict(size=18, family="Times New Roman"),
                                                                title="Salinity",
                                                                titlefont=dict(size=18, family="Times New Roman")))
        fig.update_layout(coloraxis_colorbar_x=0.75)

        fig.update_layout(
            title={
                'text': "Prior field after assilimating SINMOD and pre-survey data",
                'y': 0.75,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=30, family="Times New Roman"),
            },
            scene=dict(
                zaxis=dict(nticks=4, range=[-3, 0], ),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Longitude", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="Latitude", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="Depth [m]", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.25),
            scene_camera=camera,
        )
        if posterior:
            filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Field_posterior.html"
            filepath = "/Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Field_posterior.html"
        else:
            filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Field_prior.html"
            filepath = "/Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Field_prior.html"
        plotly.offline.plot(fig, filename=filename, auto_open=False)
        os.system("open -a \"Google Chrome\" "+filepath)
        pass

    def save_processed_data_slices(self):
        self.pred_err = vectorise(np.sqrt(np.diag(self.Sigma_posterior)))
        for i in range(len(self.depth_layers)):
            ind_depth_layer = np.where(self.coordinates_grid[:, 2] == self.depth_layers[i])[0]

            self.dataframe_field = pd.DataFrame(np.hstack((self.coordinates_grid[ind_depth_layer, :],
                                                           self.mu_sinmod[ind_depth_layer,].reshape(-1, 1),
                                                           self.mu_prior[ind_depth_layer, :].reshape(-1, 1),
                                                           self.mu_posterior[ind_depth_layer, :].reshape(-1, 1),
                                                           self.excursion_prob_prior[ind_depth_layer, :].reshape(-1, 1),
                                                           self.excursion_prob_posterior[ind_depth_layer, :].reshape(-1, 1),
                                                           self.excursion_set_prior[ind_depth_layer, :].reshape(-1, 1),
                                                           self.excursion_set_posterior[ind_depth_layer, :].reshape(-1, 1),
                                                           self.pred_err[ind_depth_layer, :].reshape(-1, 1))),
                                                columns=["lat", "lon", "depth", "mu_sinmod", "mu_prior", "mu_posterior",
                                                         "excursion_prob_prior", "excursion_prob_posterior",
                                                         "excursion_set_prior", "excursion_set_posterior",
                                                         "prediction_error"])
            datapath_layer = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/field_data_{:d}.csv".format(i)
            self.dataframe_field.to_csv(datapath_layer, index=False)
            print("Data is saved successfully!")

        pass

a = KrigingPlot()
a.plot()

#%%
def save_processed_data_slices(self):
    self.pred_err = vectorise(np.sqrt(np.diag(self.Sigma_posterior)))
    for i in range(len(self.depth_layers)):
        ind_depth_layer = np.where(self.coordinates_grid[:, 2] == self.depth_layers[i])[0]

        self.dataframe_field = pd.DataFrame(np.hstack((self.coordinates_grid[ind_depth_layer, :],
                                                       self.mu_sinmod[ind_depth_layer].reshape(-1, 1),
                                                       self.mu_prior[ind_depth_layer].reshape(-1, 1),
                                                       self.mu_posterior[ind_depth_layer].reshape(-1, 1),
                                                       self.excursion_prob_prior[ind_depth_layer].reshape(-1, 1),
                                                       self.excursion_prob_posterior[ind_depth_layer].reshape(-1, 1),
                                                       self.excursion_set_prior[ind_depth_layer].reshape(-1, 1),
                                                       self.excursion_set_posterior[ind_depth_layer].reshape(-1, 1),
                                                       self.pred_err[ind_depth_layer].reshape(-1, 1))),
                                            columns=["lat", "lon", "depth", "mu_sinmod", "mu_prior", "mu_posterior",
                                                     "excursion_prob_prior", "excursion_prob_posterior",
                                                     "excursion_set_prior", "excursion_set_posterior",
                                                     "prediction_error"])
        datapath_layer = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/field_data_{:d}.csv".format(
            i)
        self.dataframe_field.to_csv(datapath_layer, index=False)
        print("Data is saved successfully!")

    pass

save_processed_data_slices(a)

