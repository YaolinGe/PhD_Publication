"""
This script visualises the radar
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-03
"""


from Nidelva.Simulation.ES_Strategies.Knowledge import Knowledge
from Nidelva.Simulation.Field.Data.DataInterpolator import DataInterpolator
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from Nidelva.Simulation.GP_kernel.Matern_kernel import MaternKernel
from Nidelva.Simulation.ES_Strategies.Radar import Radar
from usr_func import *
import time
import plotly.graph_objects as go
import plotly
import os


# ==== Field Config ====
DEPTH = [.5, 1, 1.5, 2.0, 2.5]
DISTANCE_LATERAL = 120
DISTANCE_VERTICAL = np.abs(DEPTH[1] - DEPTH[0])
DISTANCE_TOLERANCE = 1
DISTANCE_SELF = np.abs(DEPTH[-1] - DEPTH[0])
THRESHOLD = 28
# ==== End Field Config ====

# ==== GP Config ====
SILL = .5
RANGE_LATERAL = 550
RANGE_VERTICAL = 2
NUGGET = .04
# ==== End GP Config ====

# ==== Plot Config ======
VMIN = 16
VMAX = 30
# ==== End Plot Config ==


class Test:

    knowledge = None

    def __init__(self):
        self.test_config()
        pass

    def test_config(self):
        t1 = time.time()
        self.polygon = np.array([[6.344800000000000040e+01, 1.040000000000000036e+01],
                                [6.344800000000000040e+01, 1.041999999999999993e+01],
                                [6.346000000000000085e+01, 1.041999999999999993e+01],
                                [6.346000000000000085e+01, 1.040000000000000036e+01]])
        gridGenerator = GridGenerator(polygon=self.polygon, depth=DEPTH, distance_neighbour=DISTANCE_LATERAL, no_children=6)
        # grid = gridGenerator.grid
        coordinates = gridGenerator.coordinates
        self.knowledge = Knowledge(coordinates=coordinates, polygon=self.polygon, mu=None, Sigma=None,
                                   threshold_salinity=THRESHOLD, kernel=None, ind_prev=[], ind_now=[],
                                   distance_lateral=DISTANCE_LATERAL, distance_vertical=DISTANCE_VERTICAL,
                                   distance_tolerance=DISTANCE_TOLERANCE, distance_self=DISTANCE_SELF)

        t2 = time.time()
        print("test config is done, time consumed: ", t2 - t1)

    def run(self):
        self.knowledge.ind_now = 256
        radar = Radar(self.knowledge)
        fig = go.Figure(data=[go.Scatter3d(
            x=self.knowledge.coordinates[:, 1],
            y=self.knowledge.coordinates[:, 0],
            z=-self.knowledge.coordinates[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color="black",
            ),
        )])
        fig.add_trace(go.Scatter3d(
            x=[self.knowledge.coordinates[self.knowledge.ind_now, 1]],
            y=[self.knowledge.coordinates[self.knowledge.ind_now, 0]],
            z=[-self.knowledge.coordinates[self.knowledge.ind_now, 2]],
            mode='markers',
            marker=dict(
                size=12,
                color="red",
            ),
        ))
        fig.add_trace(go.Scatter3d(
            x=self.knowledge.coordinates[self.knowledge.ind_cand, 1],
            y=self.knowledge.coordinates[self.knowledge.ind_cand, 0],
            z=-self.knowledge.coordinates[self.knowledge.ind_cand, 2],
            mode='markers',
            marker=dict(
                size=3,
                color="blue",
            ),
        ))
        fig.add_trace(go.Scatter3d(
            x=radar.lon_radar.flatten(),
            y=radar.lat_radar.flatten(),
            z=-radar.depth_radar.flatten(),
            opacity=0.1,
            # value=radar.distance_radar.flatten(),
            # isomin=10,
            # isomax=50,
            # surface_count=5,  # number of isosurfaces, 2 by default: only min and max
            # colorbar_nticks=5,  # colorbar ticks correspond to isosurface values
            # caps=dict(x_show=False, y_show=False)
            mode='markers',
            marker=dict(
                size=3,
                color="yellow",
            ),
        ))
        # fig.add_trace(go.Isosurface(
        #     x=radar.lon_radar.flatten(),
        #     y=radar.lat_radar.flatten(),
        #     z=-radar.depth_radar.flatten(),
        #     value=radar.distance_radar.flatten(),
        #     isomin=0,
        #     isomax=1,
        #
        #     surface_count=10,  # number of isosurfaces, 2 by default: only min and max
        #     # colorbar_nticks=5,  # colorbar ticks correspond to isosurface values
        # ))

        fig.update_layout(
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.25),
        )
        plotly.offline.plot(fig,
                            filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/Radar.html",
                            auto_open=False)
        os.system(
            "open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/Radar.html")

        pass


a = Test()
a.run()

#%%

x = np.linspace(-1.1, 1.1, 100)
y = np.linspace(-1.1, 1.1, 100)
z = np.linspace(-.6, .6, 30)
xx, yy, zz = np.meshgrid(x, y, z)
xv, yv, zv = map(vectorise, [xx, yy, zz])
values = xv ** 2 + yv ** 2 + zv ** 2 / .5 ** 2

fig = go.Figure(data=[go.Isosurface(
    x=xv,
    y=yv,
    z=zv,
    value=values,
    # isomin=1,
    isomax=1,

    surface_count=1,
    # colorbar_nticks=1,  # colorbar ticks correspond to isosurface values
    caps=dict(x_show=False, y_show=False, z_show=False)
    # mode='markers',
    # marker=dict(
    #     size=1,
    #     color="black",
    # ),
)])

plotly.offline.plot(fig,
                    filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/testIsosurface.html",
                    auto_open=False)
os.system(
    "open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/testIsosurface.html")

