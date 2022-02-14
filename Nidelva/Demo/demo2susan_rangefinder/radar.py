"""
This script demonstrates the radar for finding the neighbouring locations
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-14 (happy valentines day)
"""

# == Config ==
import pandas as pd
import numpy as np
DISTANCE_LATERAL = 180 # This needs to be adjusted
DISTANCE_VERTICAL = 1.5 # This needs to be adjusted
CIRCUMFERENCE = 40075000
datapath_grid = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Demo/demo2susan_rangefinder/grid.csv"
fig_filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Demo/demo2susan_rangefinder/radar.html"
ind_current_loc = 152 # This can be any location
# == END Config ==

# == Func ==
def deg2rad(deg):
    return deg / 180 * np.pi

def latlon2xy(lat, lon, lat_origin, lon_origin):
    x = deg2rad((lat - lat_origin)) / 2 / np.pi * CIRCUMFERENCE
    y = deg2rad((lon - lon_origin)) / 2 / np.pi * CIRCUMFERENCE * np.cos(deg2rad((lat)))
    return x, y
# == End Func ==


#% Step I: load grid
coordinates_grid = pd.read_csv(datapath_grid).to_numpy()
loc_current = coordinates_grid[ind_current_loc]

#% Step II: evaluate neighbouring distance
dx, dy = latlon2xy(coordinates_grid[:, 0], coordinates_grid[:, 1], loc_current[0], loc_current[1])
dz = coordinates_grid[:, 2] - loc_current[2]
distance_vector = (dx / DISTANCE_LATERAL) ** 2 + (dy / DISTANCE_LATERAL) ** 2 + (dz / DISTANCE_VERTICAL) ** 2
ind_cand = np.where((distance_vector <= 1) * (distance_vector > .3))[0]
loc_cand = coordinates_grid[ind_cand]

#% Step III: visualise the result
import plotly.graph_objects as go
import plotly

fig = go.Figure(go.Scatter3d(
    x=coordinates_grid[:, 1],
    y=coordinates_grid[:, 0],
    z=-coordinates_grid[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color="black",
        showscale=False,
    )))

fig.add_trace(go.Scatter3d(
    x=[loc_current[1]],
    y=[loc_current[0]],
    z=[-loc_current[2]],
    mode='markers',
    marker=dict(
        size=15,
        color="red",
        showscale=False,
    ),
))

fig.add_trace(go.Scatter3d(
    x=loc_cand[:, 1],
    y=loc_cand[:, 0],
    z=-loc_cand[:, 2],
    mode='markers',
    marker=dict(
        size=10,
        color="blue",
        showscale=False,
    ),
))

plotly.offline.plot(fig, filename = fig_filename, auto_open = True)

