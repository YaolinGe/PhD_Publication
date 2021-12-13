import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
from usr_func import *

class SINMOD:
    layers = 5
    data_folder = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/transfer_145711_files_ee6457cf/"

    def __init__(self):
        print("Hello world")
        self.load_smaller_grid()
        self.load_data()
        self.rearrange_grid()
        self.extract_data_on_smaller_grid()

    def load_data(self):
        files = os.listdir(self.data_folder)
        files.sort()
        for file in files:
            print(file)
            t1 = time.time()
            self.sinmod = netCDF4.Dataset(self.data_folder + file)
            self.lat = np.array(self.sinmod['gridLats'])
            self.lon = np.array(self.sinmod['gridLons'])
            self.depth = np.array(self.sinmod['zc'])[:self.layers]
            self.salinity = np.mean(np.array(self.sinmod['salinity'])[:, :self.layers, :, :], axis = 0)
            t2 = time.time()
            print("Time consumed: ", t2 - t1)
            # self.visualise_data()
            break

    def load_smaller_grid(self):
        self.grid = np.loadtxt("Nidelva/Config/grid.txt", delimiter=", ")
        self.grid_plot = np.loadtxt("Nidelva/Config/grid_plot.txt", delimiter=", ")
        print("Grid is loaded successfully")

    def extract_data_on_smaller_grid(self):
        self.x_small, self.y_small = latlon2xy(self.grid[:, 0], self.grid[:, 1], self.grid[0, 0], self.grid[0, 1])
        
        self.x, self.y = latlon2xy(self.grid_3d[:, 0], self.grid_3d[:, 1], self.grid[0, 0], self.grid[0, 1])
        self.dist_x = self.x_small.reshape(-1, 1) @ np.ones([1, len(self.x)]) - np.ones([len(self.x_small), 1]) @ self.x.reshape(1, -1)
        self.dist_y = self.y_small.reshape(-1, 1) @ np.ones([1, len(self.y)]) - np.ones([len(self.y_small), 1]) @ self.y.reshape(1, -1)
        self.dist_z = self.grid[:, 2].reshape(-1, 1) @ np.ones([1, len(self.grid_3d[:, 2])]) - \
                 np.ones([len(self.grid[:, 2]), 1]) @ self.grid_3d[:, 2].reshape(1, -1)
        self.dist = self.dist_x ** 2 + self.dist_y ** 2 + self.dist_z ** 2
        self.ind_sinmod = np.argmin(self.dist, axis = 1)
        self.salinity_small_grid = self.grid_3d[self.ind_sinmod, 3]
        print("Finished data extraction")

    def rearrange_grid(self):
        print("Here I will rearrange the grid")
        self.grid_3d = []
        for i in range(self.lat.shape[0]):
            for j in range(self.lat.shape[1]):
                for k in range(len(self.depth)):
                    # print([self.lat[i, j], self.lon[i, j], self.depth[k], self.salinity[k, i, j]])
                    self.grid_3d.append([self.lat[i, j], self.lon[i, j], self.depth[k], self.salinity[k, i, j]])
        self.grid_3d = np.array(self.grid_3d)

    def visualise_data(self):
        print("Here comes the data visualisation")

        fig = make_subplots(rows=1, cols=1,
                            specs=[[{"type": "scene"}]])

        # The regular grid needs to be used for plotting, image it is a square but the data is extracted at the certain locations
        fig.add_trace(go.Scatter3d(
            x=self.grid_plot[:, 1],
            y=self.grid_plot[:, 0],
            z=-self.grid_plot[:, 2],
            mode = "markers",
            marker = dict(
                size = 12,
                color = self.salinity_small_grid.flatten(),
                colorscale = "RdBu",
                # coloraxis = "coloraxis",
                # opacity = .8,
                cmin = 10,
                cmax = 33,
            )
        ), row = 1, col = 1)
        # fig.add_trace(go.Volume(
        #     x=self.grid_3d[:, 1],
        #     y=self.grid_3d[:, 0],
        #     z=self.grid_3d[:, 2],
        #     value=self.grid_3d[:, 3],
        #     isomin=24,
        #     isomax=33.3,
        #     opacity=.1,
        #     # surface_count=30,
        #     coloraxis="coloraxis",
        #     # caps=dict(x_show=False, y_show=False, z_show=False),
        # ),
        #     row=1, col=1)

        # camera1 = dict(
        #     up=dict(x=0, y=0, z=1),
        #     center=dict(x=0, y=0, z=0),
        #     eye=dict(x=1, y=-1, z=1)
        # )
        fig.update_coloraxes(colorscale="rainbow")
        fig.update_layout(
            scene=dict(
                # zaxis=dict(nticks=4, range=[-5, 6], ),
                xaxis_title='Distance along Lon direction [m]',
                yaxis_title='Distance along Lat direction [m]',
                zaxis_title='Depth [m]',
            ),
            # scene_camera=camera1,
            # title=datetime.fromtimestamp(self.timestamp_data[i]).strftime("%Y%m%d - %H:%M"),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.4),
            coloraxis_colorbar_x=-0.05,
        )
        # fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        # if not server:
        plotly.offline.plot(fig,
                            filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/sinmod.html",
                            auto_open=True)
        # else:
        #     fig.write_image(self.figpath + "I_{:05d}.png".format(counter), width=1980, height=1080)
        # counter = counter + 1
        # print(counter)


if __name__ == "__main__":
    a = SINMOD()
    # a.load_data()
    a.visualise_data()

#%%
plt.plot(a.grid_3d[:, 3])
plt.show()

#%%
data_file = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/transfer_145711_files_ee6457cf/samples_2020.05.01.nc"
nc = netCDF4.Dataset(data_file)
salinity = np.mean(nc['salinity'], axis = 0)
lat = np.array(nc['gridLats'])
lon = np.array(nc['gridLons'])

#%%
self = a
def visualise_data(self):
    print("Here comes the data visualisation")

    fig = make_subplots(rows=1, cols=1,
                        specs=[[{"type": "scene"}]])

    # The regular grid needs to be used for plotting, image it is a square but the data is extracted at the certain locations
    # fig.add_trace(go.Scatter3d(
    #     x=self.grid_plot[:, 1],
    #     y=self.grid_plot[:, 0],
    #     z=-self.grid_plot[:, 2],
    #     mode="markers",
    #     marker=dict(
    #         size=12,
    #         color=self.salinity_small_grid.flatten(),
    #         colorscale="RdBu",
    #         # coloraxis = "coloraxis",
    #         # opacity = .8,
    #         cmin=10,
    #         cmax=33,
    #     )
    # ), row=1, col=1)
    fig.add_trace(go.Volume(
        x=self.grid_plot[:, 1],
        y=self.grid_plot[:, 0],
        z=-self.grid_plot[:, 2],
        value=self.salinity_small_grid.flatten(),
        isomin=5,
        # isomax=,
        opacity=.1,
        surface_count=15,
        coloraxis="coloraxis",
        caps=dict(x_show=False, y_show=False, z_show=False),
    ),
        row=1, col=1)

    camera1 = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1, y=-1, z=1)
    )
    fig.update_coloraxes(colorscale="rainbow")
    fig.update_layout(
        scene=dict(
            # zaxis=dict(nticks=4, range=[-5, 6], ),
            xaxis_title='Distance along Lon direction [m]',
            yaxis_title='Distance along Lat direction [m]',
            zaxis_title='Depth [m]',
        ),
        # scene_camera=camera1,
        # title=datetime.fromtimestamp(self.timestamp_data[i]).strftime("%Y%m%d - %H:%M"),
        scene_aspectmode='manual',
        scene_aspectratio=dict(x=1, y=1, z=.4),
        coloraxis_colorbar_x=-0.05,
    )
    # fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    # if not server:
    plotly.offline.plot(fig,
                        filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/sinmod.html",
                        auto_open=True)
    # else:
    #     fig.write_image(self.figpath + "I_{:05d}.png".format(counter), width=1980, height=1080)
    # counter = counter + 1
    # print(counter)


visualise_data(self)

#%%

# import plotly.graph_objects as go
# import plotly
#
# fig = go.Figure(go.Scattergeo())
# fig.update_geos(
#     visible=False, resolution=50, scope="europe",
#     showcountries=True, countrycolor="Black",
#     showsubunits=True, subunitcolor="Blue"
# )
# fig.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
# plotly.offline.plot(fig,
#                     filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/map.html",
#                     auto_open=True)

