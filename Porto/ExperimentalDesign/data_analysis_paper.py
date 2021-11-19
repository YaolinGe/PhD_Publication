
# Plot delft3d prior data
import numpy as np
import matplotlib.pyplot as plt
import h5py
from usr_func import *

# path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Nov2016_sal_1.mat"
path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged_all/North_Moderate_all.h5"
data = h5py.File(path_data, 'r')
lat = np.array(data.get("lat"))
lon = np.array(data.get("lon"))
depth = np.array(data.get("depth"))
salinity = np.array(data.get("salinity"))
depth[depth > 0] = 0

lat_f = lat.flatten()
lon_f = lon.flatten() # flattened
depth_f = depth.flatten()
salinity_f = salinity.flatten()

lat_lc, lon_lc = 41.048, -8.829 # left bottom corner coordinate

lat_rc, lon_rc = 41.150, -8.70
nlat = 20
nlon = 15
ndepth = 4
lat_diff = 0.00833333
lon_diff = 0.00857143

X = np.linspace(0, 16000, nlat)
Y = np.linspace(0, 12000, nlon)
depth_domain = np.linspace(0, -5, ndepth)

alpha = 12
Rm = get_rotational_matrix(alpha)

def get_value_at_loc(loc, k_neighbour):
    lat_loc, lon_loc, depth_loc = loc
    x_dist, y_dist = latlon2xy(lat_f, lon_f, lat_loc, lon_loc)
    depth_dist = depth_f - depth_loc
    dist = np.sqrt(x_dist ** 2 + y_dist ** 2 + depth_dist ** 2) # cannot use lat lon since deg/rad will mix metrics
    ind_neighbour = np.argsort(dist)[:k_neighbour]  # use nearest 100 neighbours to compute the average
    value = np.nanmean(salinity_f[ind_neighbour])
    return value

grid = []
grid_r = []
grid_rd = []
grid_rf = []
values = np.zeros([nlat, nlon, ndepth])

for i in range(nlat):
    print(i)
    for j in range(nlon):
        for k in range(ndepth):
            tmp = Rm @ np.array([X[i], Y[j]])
            xnew, ynew = tmp
            lat_loc, lon_loc = xy2latlon(xnew, ynew, lat_lc, lon_lc)
            # values[i, j, k] = get_value_at_loc([lat_loc, lon_loc, depth_domain[k]], 1)
            grid.append([X[i], Y[j], depth_domain[k]])
            # grid.append([lat_domain[i], lon_domain[j], depth_domain[k]])
            grid_r.append([lat_loc, lon_loc, depth_domain[k]])
            # grid_rd.append([x_old, y_old, depth_domain[k]])
            # grid_rf.append([x_new, y_new, depth_domain[k]])
            # print(lat_domain[i], lon_domain[j], depth_domain[k])
            # print(lat_new, lon_new, depth_domain[k])
    # break
grid = np.array(grid)
grid_r = np.array(grid_r)
# grid_rd = np.array(grid_rd)
# grid_rf = np.array(grid_rf)
plt.scatter(lon[:, :, 0], lat[:, :, 0], c = salinity[:, :, 0], cmap = "Paired", vmin = 15, vmax = 35)
plt.plot(grid_r[:, 1], grid_r[:, 0], 'k.')
plt.colorbar()
plt.show()


#%%
import plotly.graph_objects as go
import plotly
plotly.io.orca.config.executable = '/usr/local/bin/orca'
plotly.io.orca.config.save()

fig = go.Figure(data=[go.Scatter3d(
    x=grid[:, 1].flatten(),
    y=grid[:, 0].flatten(),
    z=grid[:, 2].flatten(),
    mode='markers',
    marker=dict(
        size=12,
        color=values.flatten(),                # set color to an array/list of desired values
        # colorscale='jet',   # choose a colorscale
        coloraxis = "coloraxis",
        # cmin=10,
        showscale = True,
        opacity=0.8
    )
)])
fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=1, y=1, z=.5))
fig.update_coloraxes(cmin = 20, cmax = 35, colorscale = "jet")
plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_scatter.html", auto_open = True)


#%%
import plotly.graph_objects as go
import plotly
from scipy.ndimage import gaussian_filter
values_gussian_filtered = gaussian_filter(values, 1)
# values_gaussian_filtered = values

fig = go.Figure(data = go.Volume(
    x=grid[:, 0].flatten(),
    y=grid[:, 1].flatten(),
    z=grid[:, 2].flatten(),
    value=values_gussian_filtered.flatten(),
    # isomin=10,
    # isomax=33,
    opacity = .1,
    # opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
    surface_count = 20,
    # surface_fill = .4, # create texture on the surface
    colorbar_nticks = 10,  # colorbar ticks correspond to isosurface values
    # slices_z=dict(show=True, locations=[0]), # add slice
    # slices_y=dict(show=True, locations=[0]),
    # slices_x=dict(show=True, locations=[0]),
    # colorscale=px.colors.qualitative.Set1,
    # colorscale=px.colors.sequential.Aggrnyl,
    # colorscale=px.colors.diverging.Spectral,
    # colorscale = "Aggrnyl",
    # reversescale=True,
    caps=dict(x_show=False, y_show=False, z_show = False),
    ))

fig.update_layout(
    scene = dict(
        # xaxis = dict(nticks=4, range=[-0.1,0.1],),
        # yaxis = dict(nticks=4, range=[-0.5,0.5],),
        # zaxis = dict(nticks=4, range=[-5,0],),
        xaxis_title='Lon [deg]',
        yaxis_title='Lat [deg]',
        zaxis_title='Depth [m]',
    ),
)
camera1 = dict(
    # up=dict(x=0, y=0, z=1),
    # center=dict(x=0, y=0, z=0),
    # eye=dict(x=1.25, y=-1.25, z=1.25)
)

fig.update_layout(scene_camera=camera1,
                  title="Salinity field under north moderate wind condition extract from Delft 3D")
# fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
# fig.update_layout(scene_aspectmode='manual',
#                   scene_aspectratio=dict(x=nd / nd, y=nd/nd, z=ndepth/nd))

plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_delft3d.html", auto_open = True)
# fig.write_image("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/fig/delft3d/sal_north_moderate.pdf", width=1980, height=1080, engine = "orca")

#%%
# plt.plot(grid_rd[:, 1],grid_rd[:, 0], 'k.')
# plt.plot(grid_rf[:, 1],grid_rf[:, 0], 'r.')
# plt.show()
#%% extract rectangular grid
lat_lc, lon_lc = 41.075, -8.82
lat_rc, lon_rc = 41.150, -8.70
nlat = 10
nlon = 15
ndepth = 4
lat_domain = np.linspace(lat_lc, lat_rc, nlat)
lon_domain = np.linspace(lon_lc, lon_rc, nlon)
depth_domain = np.linspace(0, -5, ndepth)
values_rect = np.zeros([nlat, nlon, ndepth])

grid_rect = []
k_neighbour = 1
for i in range(nlat):
    print(i)
    for j in range(nlon):
        for k in range(ndepth):
            values_rect[i, j, k] = get_value_at_loc([lat_domain[i], lon_domain[j], depth_domain[k]], k_neighbour)
            grid_rect.append([lat_domain[i], lon_domain[j], depth_domain[k]])

grid_rect = np.array(grid_rect)

import plotly.graph_objects as go
import plotly
fig = go.Figure(data=[go.Scatter3d(
    x=grid_rect[:, 1].flatten(),
    y=grid_rect[:, 0].flatten(),
    z=grid_rect[:, 2].flatten(),
    mode='markers',
    marker=dict(
        size=12,
        color=values_rect.flatten(),                # set color to an array/list of desired values
        # colorscale='jet',   # choose a colorscale
        coloraxis = "coloraxis",
        # cmin=10,
        showscale = True,
        opacity=0.8
    )
)])
fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=1, y=1, z=.2))
fig.update_coloraxes(cmin = 20, cmax = 35, colorscale = "jet")
plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_scatter.html", auto_open = True)

#%%
from scipy.ndimage import gaussian_filter
values_gussian_filtered = gaussian_filter(values_rect, 1)
# values_gussian_filtered = values_rect

import plotly.graph_objects as go
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
plotly.io.orca.config.executable = '/usr/local/bin/orca'
plotly.io.orca.config.save()

fig = make_subplots(
    rows=1, cols=1,
    specs=[[{'type': 'scene'}]])

fig.add_trace(go.Volume(
    x=grid_rect[:, 1].flatten(),
    y=grid_rect[:, 0].flatten(),
    z=grid_rect[:, 2].flatten(),
    value=values_gussian_filtered.flatten(),
    # value=values_rect.flatten(),
    # isomin=10,
    # isomax=33,
    opacity = .1,
    # opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
    surface_count = 20,
    # surface_fill = .4, # create texture on the surface
    # colorbar_nticks = 10,  # colorbar ticks correspond to isosurface values
    # slices_z=dict(show=True, locations=[0]), # add slice
    # slices_y=dict(show=True, locations=[0]),
    # slices_x=dict(show=True, locations=[0]),
    # colorscale=px.colors.qualitative.Set1,
    # colorscale=px.colors.sequential.Aggrnyl,
    # colorscale=px.colors.diverging.Spectral,
    colorscale = "Aggrnyl",
    # reversescale=True,
    caps=dict(x_show=False, y_show=False, z_show = False),

    # name = "Salinity field under north moderate wind condition bird view"
    ),
    row = 1, col = 1)

fig.update_layout(
    scene = dict(
        # xaxis = dict(nticks=4, range=[-0.1,0.1],),
        # yaxis = dict(nticks=4, range=[-0.5,0.5],),
        zaxis = dict(nticks=4, range=[-5,0],),
        xaxis_title='Lon [deg]',
        yaxis_title='Lat [deg]',
        zaxis_title='Depth [m]',
    ),
)
camera1 = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=-1.25, z=1.25)
)

fig.update_layout(scene_camera=camera1,
                  title="Salinity field under north moderate wind condition extract from Delft 3D")
fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=nlon / nlon, y=nlat/nlon, z=ndepth/nlon))

plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_delft3d.html", auto_open = True)
# fig.write_image("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/fig/delft3d/sal_north_moderate.pdf", width=1980, height=1080, engine = "orca")

