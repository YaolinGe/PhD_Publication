"""
This script tests the grid generation for the polygon boundary
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""


from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from gmplot import GoogleMapPlotter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from usr_func import *
import plotly.graph_objects as go
import plotly
import os

polygon = np.array([[6.344800000000000040e+01, 1.040000000000000036e+01],
                   [6.344800000000000040e+01, 1.041999999999999993e+01],
                   [6.346000000000000085e+01, 1.041999999999999993e+01],
                   [6.346000000000000085e+01, 1.040000000000000036e+01]])
depth = [.5, 1., 1.5]

gridGenerator = GridGenerator(polygon = polygon, depth=depth, distance_neighbour = 120, no_children=6)
coordinates = gridGenerator.coordinates
ind_start = np.random.randint(0, coordinates.shape[0])
print(ind_start)

def find_candidates_loc(ind_now, coordinates, distance_neighbour, distance_self):
    dx, dy = latlon2xy(coordinates[:, 0], coordinates[:, 1], coordinates[ind_now, 0], coordinates[ind_now, 1])
    dz = coordinates[:, 2] - coordinates[ind_now, 2]
    dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    ind = np.where((dist <= distance_neighbour) * (dist > distance_self))[0]

    return ind

def trajectoryPlot(coordinates, trajectory, candidateLoc, filename):
    # SlicerPlot(points, values)

    # trajectory = np.array(Trajectory),
    path = np.array(trajectory)

    fig = go.Figure(data=go.Scatter3d(
        x=coordinates[:, 1],
        y=coordinates[:, 0],
        z=-coordinates[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color="white",
            showscale=False,
        ),
    ))
    fig.add_trace(go.Scatter3d(
        x=candidateLoc[:, 1],
        y=candidateLoc[:, 0],
        z=-candidateLoc[:, 2],
        mode='markers',
        marker=dict(
            size=15,
            color="blue",
            showscale=False,
        ),
    ))

    fig.add_trace(go.Scatter3d(
        x=path[:, 1],
        y=path[:, 0],
        z=-path[:, 2],
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
        )
    ), )

    plotly.offline.plot(fig, filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/"+filename+".html", auto_open = False)
    os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/"+filename+".html")
        # fig.write_image("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/Myopic/S_{:04d}.pn


ind_now = ind_start
distance_neighbour = np.sqrt(120 ** 2 + .5 ** 2)
distance_self = 1 # distance from top to bottom layer
path = []
path.append(coordinates[ind_now].tolist())

for i in range(4):

    id = find_candidates_loc(ind_now, coordinates, distance_neighbour, distance_self)
    print("candidate loc: ", id)
    print(path)
    print("ind_now: ", ind_now)
    trajectoryPlot(coordinates, path, coordinates[id, :], "Test_" + str(i))
    ind_now = id[np.random.randint(0, len(id))]
    path.append(coordinates[ind_now].tolist())




# plt.plot(grid[:, 1], grid[:, 0], 'k.')
# plt.show()

def plotGridonMap(grid):
    def color_scatter(gmap, lats, lngs, values=None, colormap='coolwarm',
                      size=None, marker=False, s=None, **kwargs):
        def rgb2hex(rgb):
            """ Convert RGBA or RGB to #RRGGBB """
            rgb = list(rgb[0:3])  # remove alpha if present
            rgb = [int(c * 255) for c in rgb]
            hexcolor = '#%02x%02x%02x' % tuple(rgb)
            return hexcolor

        if values is None:
            colors = [None for _ in lats]
        else:
            cmap = plt.get_cmap(colormap)
            norm = Normalize(vmin=min(values), vmax=max(values))
            scalar_map = ScalarMappable(norm=norm, cmap=cmap)
            colors = [rgb2hex(scalar_map.to_rgba(value)) for value in values]
        for lat, lon, c in zip(lats, lngs, colors):
            gmap.scatter(lats=[lat], lngs=[lon], c=c, size=size, marker=marker, s=s, **kwargs)

    initial_zoom = 12
    apikey = 'AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro'
    gmap = GoogleMapPlotter(grid[0, 0], grid[0, 1], initial_zoom, apikey=apikey)
    color_scatter(gmap, grid[:, 0], grid[:, 1], np.zeros_like(grid[:, 0]), size=20, colormap='hsv')
    gmap.draw("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Missions/MapPlot/map.html")

# plotGridonMap(grid)
# import os
# os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Missions/MapPlot/map.html")






