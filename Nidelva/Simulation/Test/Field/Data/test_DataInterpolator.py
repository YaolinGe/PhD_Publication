"""
This script tests data interpolation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""

from Nidelva.Simulation.Field.Data.DataInterpolator import DataInterpolator
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
import matplotlib.pyplot as plt
from usr_func import *

polygon = np.array([[6.344800000000000040e+01, 1.040000000000000036e+01],
                   [6.344800000000000040e+01, 1.041999999999999993e+01],
                   [6.346000000000000085e+01, 1.041999999999999993e+01],
                   [6.346000000000000085e+01, 1.040000000000000036e+01]])
depth = [.5, 1, 1.5]
gridGenerator = GridGenerator(polygon = polygon, depth=depth, distance_neighbour = 120, no_children=6)
coordinates = gridGenerator.coordinates
data_interpolator = DataInterpolator(coordinates = coordinates)
dataset_interpolated = data_interpolator.dataset_interpolated
dataset_sinmod = data_interpolator.sinmod

#%%
import plotly.graph_objects as go
import plotly

fig = go.Figure(data=[go.Scatter3d(
    x=dataset_interpolated["lat"],
    y=dataset_interpolated["lon"],
    z=-dataset_interpolated["depth"],
    mode='markers',
    marker=dict(
        size=12,
        color=dataset_interpolated["salinity"],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        cmin = 10,
        cmax = 28,
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/sinmod_interpolated.html", auto_open=False)
import os
os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/sinmod_interpolated.html")


#%%
'''
Only plot certain layers to make it work 
'''
depth_plot = 1.5
ind_surface = np.where(dataset_sinmod["depth"] == depth_plot)[0]
ind_surface_interpolated = np.where(dataset_interpolated["depth"] == depth_plot)[0]
import matplotlib.pyplot as plt
plt.scatter(dataset_sinmod["lon"][ind_surface], dataset_sinmod["lat"][ind_surface], c = dataset_sinmod["salinity"][ind_surface],
            vmin = 26, vmax = 30, cmap = "Paired", alpha = .1)
plt.scatter(dataset_interpolated["lon"][ind_surface_interpolated], dataset_interpolated["lat"][ind_surface_interpolated],
            c = dataset_interpolated["salinity"][ind_surface_interpolated], vmin = 26, vmax = 30, cmap = "Paired")

plt.axis([10.40,10.44, 63.44, 63.47])
plt.grid()
plt.colorbar()
plt.show()





