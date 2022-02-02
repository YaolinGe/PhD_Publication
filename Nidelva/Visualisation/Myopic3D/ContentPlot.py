"""
This script plots the content
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-25
"""


from usr_func import *
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots


class ContentPlot:

    def __init__(self, knowledge=None, vmin=28, vmax=28, filename="mean", html=False):
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
        points_mean, values_mean = interpolate_3d(lon, lat, depth, self.knowledge.mu)
        points_es, values_es = interpolate_3d(lon, lat, depth, self.knowledge.excursion_set)
        trajectory = np.array(self.knowledge.trajectory)
        trajectory = np.append(trajectory, np.array([[self.knowledge.coordinates[self.knowledge.ind_now, 0],
                                                      self.knowledge.coordinates[self.knowledge.ind_now, 1],
                                                      self.knowledge.coordinates[self.knowledge.ind_now, 2]]]).reshape(1, -1), axis=0)

        fig = make_subplots(rows = 1, cols = 1, specs = [[{'type': 'scene'}]])
        fig.add_trace(go.Volume(
            x = points_mean[:, 0],
            y = points_mean[:, 1],
            z = -points_mean[:, 2],
            value=values_mean.flatten(),
            isomin=self.vmin,
            isomax=self.vmax,
            opacity = .1,
            surface_count = 10,
            coloraxis="coloraxis",
            caps=dict(x_show=False, y_show=False, z_show = False),
            ),
            row=1, col=1
        )

        fig.add_trace(go.Volume(
            x = points_es[:, 0],
            y = points_es[:, 1],
            z = -points_es[:, 2],
            value=values_es.flatten(),
            isomin=0,
            isomax=1,
            opacity = 0.4,
            surface_count = 1,
            colorscale = "Reds",
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show = False),
            ),
            row=1, col=1
        )

        if len(self.knowledge.ind_cand):
            fig.add_trace(go.Scatter3d(
                x=self.knowledge.coordinates[self.knowledge.ind_cand, 1],
                y=self.knowledge.coordinates[self.knowledge.ind_cand, 0],
                z=-self.knowledge.coordinates[self.knowledge.ind_cand, 2],
                mode='markers',
                marker=dict(
                    size=15,
                    color="yellow",
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
                    color="black",
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
            eye=dict(x=1.25, y=-1.25, z=1)
        )

        fig.update_coloraxes(colorscale="BrBG", colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                                tickfont=dict(size=18, family="Times New Roman"),
                                                                title="Salinity",
                                                                titlefont=dict(size=18, family="Times New Roman")))
        fig.update_layout(coloraxis_colorbar_x=0.75)

        fig.update_layout(
            title={
                'text': "Adaptive 3D myopic illustration",
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
                zaxis_title=dict(text="Depth", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.25),
            scene_camera=camera,
        )

        plotly.offline.plot(fig, filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Visualisation/Myopic3D/"+self.filename+".html", auto_open = False)
        os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Visualisation/Myopic3D/"+self.filename+".html")
        # fig.write_image("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Visualisation/Myopic3D/"+self.filename+".png", width=1980, height=1080)

