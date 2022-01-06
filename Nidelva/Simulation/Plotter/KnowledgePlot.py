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
import os
from scipy.interpolate import NearestNDInterpolator
plotly.io.orca.config.executable = '/usr/local/bin/orca'
plotly.io.orca.config.save()

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

    def __init__(self, knowledge=None, vmin=28, vmax=30, filename="mean"):
        if knowledge is None:
            raise ValueError("")
        self.knowledge = knowledge
        self.coordinates = self.knowledge.coordinates
        self.vmin = vmin
        self.vmax = vmax
        self.filename = filename
        self.plot()

    def plot(self):
        lat = self.coordinates[:, 0]
        lon = self.coordinates[:, 1]
        depth = self.coordinates[:, 2]
        depth_layer = np.unique(depth)
        number_of_plots = len(depth_layer)

        # print(lat.shape)
        points, values = interpolate_3d(lon, lat, depth, self.knowledge.mu)
        trajectory = np.array(self.knowledge.trajectory)

        # print(points)
        # print(values)

        # SlicerPlot(points, values)
        fig = go.Figure(data = go.Volume(
            x = points[:, 0],
            y = points[:, 1],
            z = -points[:, 2],
            # value=values_smoothered.flatten(),
            value=values.flatten(),
            # x=lon,
            # y=lat,
            # z=depth,
            # value=self.knowledge.mu.flatten(),
            isomin=self.vmin,
            isomax=self.vmax,
            opacity = .1,
            # opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
            surface_count = 30,
            # colorscale = "rainbow",
            coloraxis="coloraxis",
            reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show = False),
            ),)

        # if self.knowledge.trajectory:
        #     fig.add_trace(go.Scatter3d(
        #         # print(trajectory),
        #         x=trajectory[:, 1],
        #         y=trajectory[:, 0],
        #         z=-trajectory[:, 2],
        #         mode='markers+lines',
        #         marker=dict(
        #             size=5,
        #             color = "black",
        #             showscale = False,
        #         ),
        #         line=dict(
        #             color="yellow",
        #             width=3,
        #             showscale=False,
        #         )
        #     ))

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
        ))

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
        ))

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
                    showscale = False,
                ),
                line=dict(
                    color="yellow",
                    width=3,
                    showscale=False,
                )
            ))

        # fig = go.Figure(data = go.Scatter3d(
        #     x = points[:, 0],
        #     y = points[:, 1],
        #     z = points[:, 2],
        #
        #     # x=lon,
        #     # y=lat,
        #     # z=depth,
        #     mode="markers",
        #     marker=dict(
        #         # size=self.path[:, 3] * 100,
        #         # color=self.path[:, 3] * 100 + 17,
        #         # colorscale = "RdBu",
        #         # color="black",
        #         # color=self.knowledge.mu.squeeze(),
        #         color=values_smoothered.flatten(),
        #         coloraxis="coloraxis",
        #         showscale=True,
        #         reversescale=True,
        #
        #     ),))
        fig.update_coloraxes(colorscale="rainbow")

        fig.update_layout(
            scene = dict(
                zaxis = dict(nticks=4, range=[-2,0],),
                xaxis_title='Lon [deg]',
                yaxis_title='Lat [deg]',
                zaxis_title='Depth [m]',
            ),
        )

        # fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
        fig.update_layout(scene_aspectmode='manual',
                          scene_aspectratio=dict(x=1, y=1, z=.5))

        plotly.offline.plot(fig, filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/"+self.filename+".html", auto_open = False)
        os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/"+self.filename+".html")
        # fig.write_image("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/Myopic/S_{:04d}.png".format(self.knowledge.step_no), width=1980, height=1080, engine = "orca")


# KnowledgePlot(a.knowledge)
# print("finished")






