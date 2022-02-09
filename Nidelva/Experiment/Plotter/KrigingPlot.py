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
        nochange = True
        print("Here comes the auv data reorganisation")
        print("length of dataset: ", len(self.data_auv))
        datapath_auv = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/data_auv.csv"
        if os.path.exists(datapath_auv) and nochange:
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
            self.depth_auv_rounded = vectorise(round2base(depth_auv_wgs, .5))
            self.coordinates_auv_wgs = np.hstack((lat_auv_wgs, lon_auv_wgs, depth_auv_wgs)) # TODO check rounded or not
            # self.coordinates_auv_wgs = np.hstack((lat_auv_wgs, lon_auv_wgs, self.depth_auv_rounded)) # TODO check rounded or not

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

    def get_data_on_bigger_canvas(self):
        self.box = self.get_box_region()
        self.box_region = mplPath.Path(self.box)

        # == get bigger canvas ==
        lat_min, lon_min, depth_min = map(np.amin, [self.coordinates_grid[:, 0], self.coordinates_grid[:, 1], self.coordinates_grid[:, 2]])
        lat_max, lon_max, depth_max = map(np.amax, [self.coordinates_grid[:, 0], self.coordinates_grid[:, 1], self.coordinates_grid[:, 2]])
        nlat = 100
        nlon = 100
        ndepth = 5
        lat_canvas = np.linspace(lat_min, lat_max, nlat)
        lon_canvas = np.linspace(lon_min, lon_max, nlon)
        depth_canvas = np.linspace(depth_min, depth_max, ndepth)
        self.coordinates_canvas = []
        for i in range(nlat):
            for j in range(nlon):
                for k in range(ndepth):
                    self.coordinates_canvas.append([lat_canvas[i], lon_canvas[j], depth_canvas[k]])
        self.coordinates_canvas = np.array(self.coordinates_canvas)
        self.lat_canvas = self.coordinates_canvas[:, 0]
        self.lon_canvas = self.coordinates_canvas[:, 1]
        self.depth_canvas = self.coordinates_canvas[:, 2]

        print("Here comes the data extraction")
        # == extract data ==
        t1 = time.time()
        x_canvas, y_canvas = latlon2xy(self.lat_canvas, self.lon_canvas, 0, 0)
        x_grid, y_grid = latlon2xy(self.coordinates_grid[:, 0], self.coordinates_grid[:, 1], 0, 0)
        x_canvas, y_canvas, depth_canvas, x_grid, y_grid, depth_grid = \
            map(vectorise, [x_canvas, y_canvas, self.depth_canvas, x_grid, y_grid, self.coordinates_grid[:, 2]])

        self.DistanceMatrix_x = x_canvas @ np.ones([1, len(x_grid)]) - np.ones([len(x_canvas), 1]) @ x_grid.T
        self.DistanceMatrix_y = y_canvas @ np.ones([1, len(y_grid)]) - np.ones([len(y_canvas), 1]) @ y_grid.T
        self.DistanceMatrix_depth = depth_canvas @ np.ones([1, len(depth_grid)]) - np.ones(
            [len(depth_canvas), 1]) @ depth_grid.T
        self.DistanceMatrix = self.DistanceMatrix_x ** 2 + self.DistanceMatrix_y ** 2 + self.DistanceMatrix_depth ** 2
        self.ind_extracted = np.argmin(self.DistanceMatrix, axis=1)  # interpolated vectorised indices
        t2 = time.time()
        print("Data extraction takes ", t2 - t1)
        self.mu_prior_canvas = vectorise(self.mu_prior[self.ind_extracted])
        self.mu_posterior_canvas = vectorise(self.mu_posterior[self.ind_extracted])
        self.ep_prior_canvas = vectorise(self.excursion_prob_prior[self.ind_extracted])
        self.ep_posterior_canvas = vectorise(self.excursion_prob_posterior[self.ind_extracted])

        counter = 0
        for i in range(len(self.lat_canvas)):
            if not self.box_region.contains_point((self.lat_canvas[i], self.lon_canvas[i])):
                counter = counter + 1
                # self.mu_prior_canvas[i] = float("NaN")
                # self.mu_posterior_canvas[i] = float("NaN")
                # self.ep_prior_canvas[i] = float("NaN")
                # self.ep_posterior_canvas[i] = float("NaN")
                self.mu_prior_canvas[i] = -1
                self.mu_posterior_canvas[i] = -1
                self.ep_prior_canvas[i] = -1
                self.ep_posterior_canvas[i] = -1
        print(counter, " points are not in the grid")
        # plt.plot()
        # plt.plot(self.coordinates_canvas[:, 1], self.coordinates_canvas[:, 0], 'k.')
        # plt.plot(self.coordinates_grid[:, 1], self.coordinates_grid[:, 0], 'r.')
        # plt.plot(self.box[:, 1], self.box[:, 0], 'k*', markersize=10)
        # plt.show()

    def get_box_region(self):
        loc_usr = np.array([[XLIM[0], YLIM[0], 0],
                            [XLIM[0], YLIM[1], 0],
                            [XLIM[1], YLIM[1], 0],
                            [XLIM[1], YLIM[0], 0]])
        rom_wgs = getRotationalMatrix_USR2WGS(ANGLE_ROTATION)
        loc_wgs = (rom_wgs @ loc_usr.T).T
        lat_box, lon_box = xy2latlon(loc_wgs[:, 0], loc_wgs[:, 1], PIVOT[0], PIVOT[1])
        box = np.hstack((vectorise(lat_box), vectorise(lon_box)))
        return box

    def get_boundary(self, excursion_prob):
        boundary = np.zeros_like(excursion_prob)
        # ind_boundary = np.where((excursion_prob >= .475) * (excursion_prob <= .525))[0]
        ind_boundary = np.where(excursion_prob >= .65)[0]
        boundary[ind_boundary] = True
        return boundary

    def plot(self):
        # self.get_fake_rectangular_grid()
        # == smoothing section
        lat = self.coordinates_canvas[:, 0]
        lon = self.coordinates_canvas[:, 1]
        depth = self.coordinates_canvas[:, 2]

        values_mu = self.mu_posterior_canvas
        values_ep = self.ep_posterior_canvas
        # values_mu = refill_nan_values(self.mu_posterior_canvas)
        # values_ep = refill_nan_values(self.ep_posterior_canvas)

        # coordinates_mu_prior, values_mu_prior = interpolate_3d(lon, lat, depth, self.mu_prior_canvas)
        # coordinates_ep_prior, values_ep_prior = interpolate_3d(lon, lat, depth, self.ep_prior_canvas)
        # coordinates_es_prior, values_es_prior = interpolate_3d(lon, lat, depth, self.excursion_set_prior)
        # coordinates_boundary_prior, values_boundary_prior = interpolate_3d(lon, lat, depth, self.boundary_prior)
        #
        # coordinates_mu_posterior, values_mu_posterior = interpolate_3d(lon, lat, depth, self.mu_posterior_canvas)
        # coordinates_ep_posterior, values_ep_posterior = interpolate_3d(lon, lat, depth, self.ep_posterior_canvas)
        # coordinates_es_posterior, values_es_posterior = interpolate_3d(lon, lat, depth, self.excursion_set_posterior)
        # coordinates_boundary_posterior, values_boundary_posterior = interpolate_3d(lon, lat, depth, self.boundary_posterior)

        # == remove smoothing section ==

        with open('colorscale.txt', 'r') as file:
            colorscales = file.read().splitlines()

        # for colorscale in colorscales:
        for colorscale in ["gnbu"]:
            t1 = time.time()
            fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

            xv = self.coordinates_canvas[:, 1]
            yv = self.coordinates_canvas[:, 0]
            zv = self.coordinates_canvas[:, 2]
            values = values_ep
            for j in range(len(np.unique(zv))):
                ind = (zv == np.unique(zv)[j])
                fig.add_trace(
                    go.Isosurface(x=xv[ind], y=yv[ind], z=-zv[ind],
                                      isomin=0,
                                      isomax=1,
                                  # value=mu_cond[ind], colorscale=newcmp),
                                  value=values[ind], coloraxis="coloraxis"),
                    row=1, col=1
                )

            posterior = True
            if posterior:
                fig.add_trace(go.Scatter3d(
                    x=self.coordinates_auv_wgs[:, 1],
                    y=self.coordinates_auv_wgs[:, 0],
                    z=-self.coordinates_auv_wgs[:, 2],
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
            fig.update_coloraxes(colorscale=colorscale, colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                                    tickfont=dict(size=18, family="Times New Roman"),
                                                                    title="Excursion Probability",
                                                                    titlefont=dict(size=18, family="Times New Roman")))
            fig.update_layout(coloraxis_colorbar_x=0.75)

            fig.update_layout(
                title={
                    'text': "Posterior field after AUV sampling",
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
                scene_aspectratio=dict(x=1, y=1, z=.5),
                scene_camera=camera,
            )
            if posterior:
                filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Field_posterior_"+colorscale+".html"
                filepath = "/Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Field_posterior_"+colorscale+".html"
            else:
                filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Field_prior.html"
                filepath = "/Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/Field_prior.html"
            plotly.offline.plot(fig, filename=filename, auto_open=False)
            os.system("open -a \"Google Chrome\" "+filepath)
            pass
            t2 = time.time()
            print("Time consumed: ", t2 - t1)

    def save_processed_data_slices(self):
        self.pred_err = vectorise(np.sqrt(np.diag(self.Sigma_posterior)))
        for i in range(len(self.depth_layers)):
            ind_depth_layer = np.where(self.coordinates_grid[:, 2] == self.depth_layers[i])[0]

            self.dataframe_field = pd.DataFrame(np.hstack((self.coordinates_grid[ind_depth_layer, :],
                                                           self.mu_sinmod[ind_depth_layer].reshape(-1, 1),
                                                           self.mu_prior[ind_depth_layer].reshape(-1, 1),
                                                           self.mu_posterior[ind_depth_layer].reshape(-1, 1),
                                                           self.excursion_prob_prior[ind_depth_layer].reshape(-1, 1),
                                                           self.excursion_prob_posterior[ind_depth_layer].reshape(-1,
                                                                                                                  1),
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

    def smooth_data(self):
        datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/field_data_1.csv"
        datapath_new = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/field_data_1_smooth.csv"
        df = pd.read_csv(datapath)

        # == get the coarser grid data ==
        self.lat = df['lat'].to_numpy()
        self.lon = df['lon'].to_numpy()
        self.ep_prior = df['excursion_prob_prior'].to_numpy()
        self.ep_posterior = df['excursion_prob_posterior'].to_numpy()
        self.x, self.y = latlon2xy(self.lat, self.lon, PIVOT[0], PIVOT[1])
        self.x, self.y = map(vectorise, [self.x, self.y])

        # == rotate back to normal grid ==
        self.xyz_wgs = np.hstack((self.x, self.y, np.ones_like(self.x)))
        self.rom_usr = getRotationalMatrix_WGS2USR(ANGLE_ROTATION)
        self.xyz_usr = (self.rom_usr @ self.xyz_wgs.T).T

        # == interpolate to finer grid ==
        grid_x_posterior, grid_y_posterior, grid_value_posterior = interpolate_2d(self.xyz_usr[:, 0], self.xyz_usr[:, 1], 100, 100, self.ep_posterior)
        grid_x_prior, grid_y_prior, grid_value_prior = interpolate_2d(self.xyz_usr[:, 0], self.xyz_usr[:, 1], 100, 100, self.ep_prior)
        grid_value_posterior = refill_nan_values(grid_value_posterior)
        self.grid_refined = []
        self.values_refined = []
        for j in range(grid_x_posterior.shape[0]):
            for k in range(grid_x_posterior.shape[1]):
                self.grid_refined.append([grid_x_posterior[j, k], grid_y_posterior[j, k], 0])
                self.values_refined.append([grid_value_prior[j, k], grid_value_posterior[j, k]])
        self.grid_refined = np.array(self.grid_refined)
        self.values_refined = np.array(self.values_refined)

        # == rotate finer regular grid back to wgs grid and convert from xyz to latlon ==
        rom_wgs = getRotationalMatrix_USR2WGS(ANGLE_ROTATION)
        self.grid_refined_wgs = (rom_wgs @ self.grid_refined.T).T
        lat_refined, lon_refined = xy2latlon(self.grid_refined_wgs[:, 0], self.grid_refined_wgs[:, 1], PIVOT[0], PIVOT[1])
        lat_refined, lon_refined = map(vectorise, [lat_refined, lon_refined])
        self.coordinates_refined = np.hstack((lat_refined, lon_refined, np.ones_like(lat_refined)))
        df = pd.DataFrame(np.hstack((self.coordinates_refined, self.values_refined)), columns=['lat', 'lon', 'depth', 'ep_prior', 'ep_posterior'])
        df.to_csv(datapath_new, index=False)
        print("Finished smoothing~")

a = KrigingPlot()
a.get_data_on_bigger_canvas()
a.plot()
# a.smooth_data()


