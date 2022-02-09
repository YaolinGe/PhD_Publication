"""
This script plots the knowledge
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-06
"""
import numpy as np

from usr_func import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import os


class Plot3D:

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

        points_es = points_mean
        values_es = np.zeros_like(values_mean)
        values_es[values_mean < self.knowledge.threshold_salinity] = True
        points_es, values_es = interpolate_3d(points_es[:, 0], points_es[:, 1], points_es[:, 2], values_es.flatten())

        # points_es, values_es = interpolate_3d(lon, lat, depth, self.knowledge.excursion_set)
        # points_std, values_std = interpolate_3d(lon, lat, depth, np.sqrt(np.diag(self.knowledge.Sigma)))
        # points_ep, values_ep = interpolate_3d(lon, lat, depth, self.knowledge.excursion_prob)
        trajectory = np.array(self.knowledge.trajectory)

        fig = make_subplots(rows = 1, cols = 1, specs = [[{'type': 'scene'}]])
        # fig = make_subplots(rows = 1, cols = 3, specs = [[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        #                     subplot_titles=("Mean", "Std", "EP"))


        fig.add_trace(go.Volume(
            x = points_mean[:, 0],
            y = points_mean[:, 1],
            z = -points_mean[:, 2],
            value=values_mean.flatten(),
            isomin=self.vmin,
            isomax=self.vmax,
            opacity = .4,
            surface_count = 10,
            # colorscale = "BrBG",
            coloraxis="coloraxis",
            # colorscale=discrete_cmap,
            # coloraxis="coloraxis1",
            # colorbar=dict(x=0.75,y=0.5, len=1),
            # reversescale=True,
            caps=dict(x_show=False, y_show=False, z_show = False),
            ),
            row=1, col=1
        )

        # fig.add_trace(go.Volume(
        #     x = points_es[:, 0],
        #     y = points_es[:, 1],
        #     z = -points_es[:, 2],
        #     value=values_es.flatten(),
        #     isomin=0,
        #     isomax=1,
        #     opacity = 0.5,
        #     surface_count = 1,
        #     colorscale = "Reds",
        #     showscale=False,
        #     # colorscale=discrete_cmap,
        #     # coloraxis="coloraxis1",
        #     # colorbar=dict(x=.75,y=0.5, len=1),
        #     # reversescale=True,
        #     caps=dict(x_show=False, y_show=False, z_show = False),
        #     ),
        #     row=1, col=1
        # )

        # print(values_std)
        # if len(values_std):
        #     fig.add_trace(go.Volume(
        #         x=points_std[:, 0],
        #         y=points_std[:, 1],
        #         z=-points_std[:, 2],
        #         value=values_std.flatten(),
        #         isomin=0,
        #         isomax=1,
        #         opacity=.1,
        #         surface_count=30,
        #         colorscale = "rdbu",
        #         colorbar=dict(x=0.65, y=0.5, len=.5),
        #         reversescale=True,
        #         caps=dict(x_show=False, y_show=False, z_show=False),
        #     ),
        #         row=1, col=2
        #     )
        #
        # fig.add_trace(go.Volume(
        #     x=points_ep[:, 0],
        #     y=points_ep[:, 1],
        #     z=-points_ep[:, 2],
        #     value=values_ep.flatten(),
        #     isomin=0,
        #     isomax=1,
        #     opacity=.1,
        #     surface_count=30,
        #     colorscale = "gnbu",
        #     colorbar=dict(x=1, y=0.5, len=.5),
        #     reversescale=True,
        #     caps=dict(x_show=False, y_show=False, z_show=False),
        # ),
        #     row=1, col=3
        # )

        # if len(self.knowledge.ind_cand):
        #     fig.add_trace(go.Scatter3d(
        #         x=self.knowledge.coordinates[self.knowledge.ind_cand, 1],
        #         y=self.knowledge.coordinates[self.knowledge.ind_cand, 0],
        #         z=-self.knowledge.coordinates[self.knowledge.ind_cand, 2],
        #         mode='markers',
        #         marker=dict(
        #             size=15,
        #             color="white",
        #             showscale=False,
        #         ),
        #         showlegend=False,
        #     ),
        #         row='all', col='all'
        #     )

        # if len(self.knowledge.ind_cand_filtered):
        #     fig.add_trace(go.Scatter3d(
        #         x=self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 1],
        #         y=self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 0],
        #         z=-self.knowledge.coordinates[self.knowledge.ind_cand_filtered, 2],
        #         mode='markers',
        #         marker=dict(
        #             size=10,
        #             color="blue",
        #             showscale=False,
        #         ),
        #         showlegend=False, # remove all unnecessary trace names
        #     ),
        #         row='all', col='all'
        #     )
        #
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
        #             showscale=False,
        #         ),
        #         line=dict(
        #             color="yellow",
        #             width=3,
        #             showscale=False,
        #         ),
        #         showlegend=False,
        #     ),
        #     row='all', col='all'
        #     )
        #
        # fig.add_trace(go.Scatter3d(
        #     x=[self.knowledge.coordinates[self.knowledge.ind_now, 1]],
        #     y=[self.knowledge.coordinates[self.knowledge.ind_now, 0]],
        #     z=[-self.knowledge.coordinates[self.knowledge.ind_now, 2]],
        #     mode='markers',
        #     marker=dict(
        #         size=20,
        #         color="red",
        #         showscale=False,
        #     ),
        #     showlegend=False, # remove all unnecessary trace names
        # ),
        #     row='all', col='all'
        # )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=-1.25, z=.5)
        )
        fig.update_coloraxes(colorscale="BrBG", colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                                tickfont=dict(size=18, family="Times New Roman"),
                                                                title="Salinity",
                                                                titlefont=dict(size=18, family="Times New Roman")))
        fig.update_layout(coloraxis_colorbar_x=0.8)
        fig.update_layout(
            title={
                'text': "",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            scene = dict(
                zaxis = dict(nticks=4, range=[-2.5,-0.5],),
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

            # scene2=dict(
            #     zaxis=dict(nticks=4, range=[-2, 0], ),
            #     xaxis_title='Lon [deg]',
            #     yaxis_title='Lat [deg]',
            #     zaxis_title='Depth [m]',
            # ),
            # scene2_aspectmode='manual',
            # scene2_aspectratio=dict(x=1, y=1, z=.5),
            # scene3=dict(
            #     zaxis=dict(nticks=4, range=[-2, 0], ),
            #     xaxis_title='Lon [deg]',
            #     yaxis_title='Lat [deg]',
            #     zaxis_title='Depth [m]',
            # ),
            # scene3_aspectmode='manual',
            # scene3_aspectratio=dict(x=1, y=1, z=.5),
            # scene_camera=camera,
            # scene2_camera=camera,
            # scene3_camera=camera,
            # annotations=[
            #     dict(
            #         x=10.405,
            #         y=63.45,
            #         # z=0,
            #         text="Boundary envelope",
            #         textangle=0,
            #         ax=-50,
            #         ay=-70,
            #         font=dict(
            #             color="black",
            #             size=18,
            #             family="Times New Roman"
            #         ),
            #         arrowcolor="black",
            #         arrowsize=3,
            #         arrowwidth=1,
            #         arrowhead=1),
            # ],
        )

        # fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
        # if self.html:
        # print("Save html")
        print(self.filename)
        plotly.offline.plot(fig, filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Visualisation/"+self.filename+".html", auto_open = False)
        print('Finished plotting')
        os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Visualisation/"+self.filename+".html")
        # fig.write_image(self.filename+".png", width=1980, height=1080, engine = "orca")








