"""
This script plots the content
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-25
"""
import numpy as np

from usr_func import *
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots


class ContentPlot:

    def __init__(self, knowledge=None, trajectory=None, vmin=28, vmax=28, filename="mean", html=False):
        if knowledge is None:
            raise ValueError("")
        self.knowledge = knowledge
        self.coordinates = self.knowledge.coordinates
        self.trajectory = np.array(trajectory)
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
        points_mean, values_mean = interpolate_3d(lon, lat, depth, self.knowledge.mu)
        points_es, values_es = interpolate_3d(lon, lat, depth, self.knowledge.excursion_set)
        # trajectory = np.array(self.knowledge.trajectory)
        trajectory = self.trajectory
        # print(self.knowledge.trajectory)
        # trajectory = np.append(trajectory, np.array([[self.knowledge.coordinates[self.knowledge.ind_now, 0],
        #                                               self.knowledge.coordinates[self.knowledge.ind_now, 1],
        #                                               self.knowledge.coordinates[self.knowledge.ind_now, 2]]]).reshape(1, -1), axis=0)

        fig = make_subplots(rows = 1, cols = 1, specs = [[{'type': 'scene'}]])
        fig.add_trace(go.Volume(
            x = points_mean[:, 0],
            y = points_mean[:, 1],
            z = -points_mean[:, 2],
            value=values_mean.flatten(),
            isomin=self.vmin,
            isomax=self.vmax,
            opacity = .4,
            surface_count = 10,
            coloraxis="coloraxis",
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
        #     opacity = 0.4,
        #     surface_count = 1,
        #     colorscale = "Reds",
        #     showscale=False,
        #     caps=dict(x_show=False, y_show=False, z_show = False),
        #     ),
        #     row=1, col=1
        # )

        # if self.knowledge.trajectory:
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
                color="black",
                width=3,
                showscale=False,
            ),
            showlegend=False,
        ),
        row='all', col='all'
        )

        fig.add_trace(go.Scatter3d(
            # print(trajectory),
            x=trajectory[:, 1],
            y=trajectory[:, 0],
            z=-0.5 * np.ones_like(trajectory[:, 2]).flatten(),
            mode='lines',
            line=dict(
                color="blue",
                width=5,
                dash="dash",
                showscale=False,
            ),
            showlegend=False,
        ),
        row='all', col='all'
        )

        fig.add_trace(go.Scatter3d(
            x=[trajectory[-1, 1]],
            y=[trajectory[-1, 0]],
            z=[-trajectory[-1, 2]],
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
            eye=dict(x=0, y=-1.25, z=.5)
        )

        fig.update_coloraxes(colorscale="BrBG", colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                                tickfont=dict(size=18, family="Times New Roman"),
                                                                title="Salinity",
                                                                titlefont=dict(size=18, family="Times New Roman")))
        fig.update_layout(coloraxis_colorbar_x=0.8)

        fig.update_layout(
            title={
                'text': "Lawnmower pattern illustration",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font':dict(size=30, family="Times New Roman"),
            },
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
        )

        plotly.offline.plot(fig, filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Visualisation/Lawnmower/"+self.filename+".html", auto_open = False)
        os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Visualisation/Lawnmower/"+self.filename+".html")
        # fig.write_image("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Visualisation/Lawnmower/"+self.filename+".png", width=1980, height=1080)

