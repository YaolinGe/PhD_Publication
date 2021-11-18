
# Plot delft3d prior data
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.path as mplPath  # used to determine whether a point is inside the grid or not
# path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Nov2016_sal_1.mat"
path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged_all/North_Moderate_all.h5"
data = h5py.File(path_data, 'r')
lat = np.array(data.get("lat"))
lon = np.array(data.get("lon"))
depth = np.array(data.get("depth"))
salinity = np.array(data.get("salinity"))
depth[depth > 0] = 0


def deg2rad(deg):
    return deg / 180 * np.pi

def rad2deg(rad):
    return rad / np.pi * 180

def latlon2xy(lat, lon, lat_origin, lon_origin):
    x = deg2rad((lat - lat_origin)) / 2 / np.pi * circumference
    y = deg2rad((lon - lon_origin)) / 2 / np.pi * circumference * np.cos(deg2rad(lat))
    return x, y

def xy2latlon(x, y, lat_origin, lon_origin):
    lat = lat_origin + rad2deg(x * np.pi * 2.0 / circumference)
    lon = lon_origin + rad2deg(y * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat))))
    return lat, lon

nd = 10
ndepth = 3
alpha = 11
depth_min = 0
depth_max = -5
lat_lc, lon_lc = 41.0427, -8.8261 # lat, lon at the left bottom corner
distance = 10000
circumference = 40075000 # [m], circumference

depth_domain = np.linspace(depth_min, depth_max, ndepth)

x = np.linspace(0, distance, nd)
y = np.linspace(0, distance, nd)

def get_rotational_matrix(alpha):
    R = np.array([[np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))],
                  [-np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])
    return R
Rm = get_rotational_matrix(alpha)

values = np.zeros([nd, nd, ndepth])
k_neighbour = 100

grid = []
grid_r = []
for i in range(nd):
    print(i)
    for j in range(nd):
        for k in range(ndepth):
            xr, yr = Rm @ [x[i], y[j]]
            lat_grid, lon_grid = xy2latlon(x[i], y[j], lat_lc, lon_lc)
            lat_grid_r, lon_grid_r = xy2latlon(xr, yr, lat_lc, lon_lc)
            grid.append([lat_grid, lon_grid, depth_domain[k]])
            grid_r.append([lat_grid_r, lon_grid_r, depth_domain[k]])
            lon_dist = lon.flatten() - lon_grid_r
            lat_dist = lat.flatten() - lat_grid_r
            depth_dist = depth.flatten() - depth_domain[k]
            dist = np.sqrt(lon_dist ** 2 + lat_dist ** 2 + depth_dist ** 2)
            ind_neighbour = np.argsort(dist)[:k_neighbour] # use nearest 100 neighbours to compute the average
            values[i, j, k] = np.nanmean(salinity.flatten()[ind_neighbour])

grid = np.array(grid)
grid_r = np.array(grid_r)
#%%

plt.scatter(lon[:, :, 0], lat[:, :, 0], c = salinity[:, :, 0], cmap = "Paired", vmin = 15, vmax = 35)
plt.plot(grid_r[:, 1], grid_r[:, 0], 'k.')
plt.plot(grid[:, 1], grid[:, 0], 'r.')
plt.colorbar()
plt.show()
#%%
import plotly.graph_objects as go
import plotly
fig = go.Figure(data=[go.Scatter3d(
    x=grid_r[:, 1].flatten(),
    y=grid_r[:, 0].flatten(),
    z=grid_r[:, 2].flatten(),
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
                  scene_aspectratio=dict(x=nd, y=nd, z=ndepth))
fig.update_coloraxes(cmin = 20, cmax = 35, colorscale = "jet")
plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_scatter.html", auto_open = True)


#%%
from scipy.ndimage import gaussian_filter
values_gussian_filtered = gaussian_filter(values, 1)

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
    x=grid_r[:, 1].flatten(),
    y=grid_r[:, 0].flatten(),
    z=grid_r[:, 2].flatten(),
    # value=values_gussian_filtered.flatten(),
    value=values.flatten(),
    # isomin=10,
    # isomax=33,
    # opacity = .1,
    # opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
    # surface_count = 20,
    # surface_fill = .4, # create texture on the surface
    # colorbar_nticks = 10,  # colorbar ticks correspond to isosurface values
    # slices_z=dict(show=True, locations=[0]), # add slice
    # slices_y=dict(show=True, locations=[0]),
    # slices_x=dict(show=True, locations=[0]),
    # colorscale=px.colors.qualitative.Set1,
    # colorscale=px.colors.sequential.Aggrnyl,
    # colorscale=px.colors.diverging.Spectral,
    # colorscale = "Aggrnyl",
    # reversescale=True,
    # caps=dict(x_show=False, y_show=False, z_show = False),

    # name = "Salinity field under north moderate wind condition bird view"
    ),
    row = 1, col = 1)

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
fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=nd / nd, y=nd/nd, z=ndepth/nd))

plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_delft3d.html", auto_open = True)
# fig.write_image("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/fig/delft3d/sal_north_moderate.pdf", width=1980, height=1080, engine = "orca")


