"""
This script plots the knowledge
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-06
"""


from usr_func import *
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interpn
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import os
from scipy.interpolate import NearestNDInterpolator
# plotly.io.orca.config.executable = '/usr/local/bin/orca'
# plotly.io.orca.config.save()

from Nidelva.Simulation.Plotter.SlicerPlot import SlicerPlot


def interpolate_2d(x, y, nx, ny, value, interpolation_method="linear"):
    xmin, ymin = map(np.amin, [x, y])
    xmax, ymax = map(np.amax, [x, y])
    points = np.hstack((vectorise(x), vectorise(y)))
    xv = np.linspace(xmin, xmax, nx)
    yv = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(xv, yv)
    grid_value = griddata(points, value, (grid_x, grid_y), method=interpolation_method)
    return grid_x, grid_y, grid_value

def refill_nan_values(data):
    mask = np.where(~np.isnan(data))
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    filled_data = interp(*np.indices(data.shape))
    return filled_data

def interpolate_3d(x, y, z, value):
    z_layer = np.unique(z)
    grid = []
    values = []
    nx = 50
    ny = 50
    nz = len(z_layer)
    # values_smoothered = np.zeros([nx-2, ny-2, nz])
    for i in range(len(z_layer)):
        ind_layer = np.where(z == z_layer[i])[0]
        # print("layer: ", z_layer[i])
        grid_x, grid_y, grid_value = interpolate_2d(x[ind_layer], y[ind_layer], nx=nx, ny=ny, value=value[ind_layer], interpolation_method="cubic")
        grid_value = refill_nan_values(grid_value)
        # print("grid_x: ", grid_x.shape)
        for j in range(grid_x.shape[0]):
            for k in range(grid_x.shape[1]):
                grid.append([grid_x[j, k], grid_y[j, k], z_layer[i]])
                values.append(grid_value[j, k])
                # values_smoothered[j-1, k-1, i] = grid_value[j, k]

    grid = np.array(grid)
    values = np.array(values)
    # print(grid.shape)
    # print(values.shape)
    # print(np.any(np.isnan(values)))
    # values_smoothered = gaussian_filter(values_smoothered, .0000001)
    # vs = []
    # for i in range(nz):
    #     for j in range(nx):
    #         for k in range(ny):
    #             vs.append(values_smoothered[j, k, i])
    # vs = np.array(vs)

    return grid, values


class KnowledgePlot:

    def __init__(self, knowledge=None, vmin=28, vmax=30, filename="mean", html=False):
        if knowledge is None:
            raise ValueError("")
        self.knowledge = knowledge
        self.coordinates = self.knowledge.coordinates
        self.vmin = vmin
        self.vmax = vmax
        self.filename = filename
        self.html = html
        self.plot()

    def plot(self):
        lat = self.coordinates[:, 0]
        lon = self.coordinates[:, 1]
        depth = self.coordinates[:, 2]
        depth_layer = np.unique(depth)
        number_of_plots = len(depth_layer)

        # print(lat.shape)
        points_mean, values_mean = interpolate_3d(lon, lat, depth, self.knowledge.mu)
        points_std, values_std = interpolate_3d(lon, lat, depth, np.sqrt(np.diag(self.knowledge.Sigma)))
        points_ep, values_ep = interpolate_3d(lon, lat, depth, self.knowledge.excursion_prob)
        trajectory = np.array(self.knowledge.trajectory)

        fig = make_subplots(rows = 1, cols = 3, specs = [[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                            subplot_titles=("Mean", "Std", "EP"))
        fig.add_trace(go.Volume(
            x = points_mean[:, 0],
            y = points_mean[:, 1],
            z = -points_mean[:, 2],
            value=values_mean.flatten(),
            isomin=self.vmin,
            isomax=self.vmax,
            opacity = .1,
            surface_count = 30,
            colorscale = "rainbow",
            # coloraxis="coloraxis1",
            colorbar=dict(x=0.3,y=0.5, len=.5),
            reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show = False),
            ),
            row=1, col=1
        )
        # print(values_std)
        if len(values_std):
            fig.add_trace(go.Volume(
                x=points_std[:, 0],
                y=points_std[:, 1],
                z=-points_std[:, 2],
                value=values_std.flatten(),
                isomin=0,
                isomax=1,
                opacity=.1,
                surface_count=30,
                colorscale = "rdbu",
                colorbar=dict(x=0.65, y=0.5, len=.5),
                reversescale=True,
                caps=dict(x_show=False, y_show=False, z_show=False),
            ),
                row=1, col=2
            )

        fig.add_trace(go.Volume(
            x=points_ep[:, 0],
            y=points_ep[:, 1],
            z=-points_ep[:, 2],
            value=values_ep.flatten(),
            isomin=0,
            isomax=1,
            opacity=.1,
            surface_count=30,
            colorscale = "gnbu",
            colorbar=dict(x=1, y=0.5, len=.5),
            reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ),
            row=1, col=3
        )

        if len(self.knowledge.ind_cand):
            fig.add_trace(go.Scatter3d(
                x=self.knowledge.coordinates[self.knowledge.ind_cand, 1],
                y=self.knowledge.coordinates[self.knowledge.ind_cand, 0],
                z=-self.knowledge.coordinates[self.knowledge.ind_cand, 2],
                mode='markers',
                marker=dict(
                    size=15,
                    color="white",
                    showscale=False,
                ),
                showlegend=False,
            ),
                row='all', col='all'
            )

        if len(self.knowledge.ind_cand_filtered):
            fig.add_trace(go.Scatter3d(
                x=self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 1],
                y=self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 0],
                z=-self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 2],
                mode='markers',
                marker=dict(
                    size=10,
                    color="blue",
                    showscale=False,
                ),
                showlegend=False, # remove all unnecessary trace names
            ),
                row='all', col='all'
            )

        if self.knowledge.trajectory:
            fig.add_trace(go.Scatter3d(
                # print(trajectory),
                x=trajectory[:, 1],
                y=trajectory[:, 0],
                z=-trajectory[:, 2],
                mode='markers+lines',
                marker=dict(
                    size=5,
                    color = "black",
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

        fig.add_trace(go.Scatter3d(
            x=[self.knowledge.coordinates[self.knowledge.ind_now, 1]],
            y=[self.knowledge.coordinates[self.knowledge.ind_now, 0]],
            z=[-self.knowledge.coordinates[self.knowledge.ind_now, 2]],
            mode='markers',
            marker=dict(
                size=20,
                color="red",
                showscale=False,
            ),
            showlegend=False, # remove all unnecessary trace names
        ),
            row='all', col='all'
        )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2.25, y=2.25, z=2.25)
        )

        fig.update_layout(
            scene = dict(
                zaxis = dict(nticks=4, range=[-2,0],),
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

        # fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
        # if self.html:
        # print("Save html")
        plotly.offline.plot(fig, filename = self.filename+".html", auto_open = False)
            # os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/"+self.filename+".html")
        # fig.write_image(self.filename+".png", width=1980, height=1080, engine = "orca")







