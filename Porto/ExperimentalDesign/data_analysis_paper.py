
# Plot delft3d prior data
import numpy as np
import matplotlib.pyplot as plt
import h5py
from usr_func import *

# path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Nov2016_sal_1.mat"
path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged_all/North_Moderate_all.h5"
path_wind = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Wind/wind_data_uv.txt"
data_wind = np.loadtxt(path_wind, delimiter=",")

path_tide = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Tide/tide.txt"
data_tide = np.loadtxt(path_tide, delimiter=", ")

path_water_discharge = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/WaterDischarge/data_water_discharge.txt"
data_waterdischarge = np.loadtxt(path_water_discharge, delimiter=", ")

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

def get_rotational_matrix(alpha):
    R = np.array([[np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))],
                  [-np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])
    return R

lat_lc, lon_lc = 41.045, -8.819 # left bottom corner coordinate
nlat = 20
nlon = 15
ndepth = 4
max_distance_lat = 18000
max_distance_lon = 14000
max_depth = -5
alpha = 12
Rm = get_rotational_matrix(alpha)

X = np.linspace(0, max_distance_lat, nlat)
Y = np.linspace(0, max_distance_lon, nlon)
depth_domain = np.linspace(0, max_depth, ndepth)

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
        tmp = Rm @ np.array([X[i], Y[j]])
        xnew, ynew = tmp
        lat_loc, lon_loc = xy2latlon(xnew, ynew, lat_lc, lon_lc)
        for k in range(ndepth):
            values[i, j, k] = get_value_at_loc([lat_loc, lon_loc, depth_domain[k]], 1)
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
from scipy.ndimage import gaussian_filter
values_gussian_filtered = gaussian_filter(values, 1)
# values_gaussian_filtered = values
from plotly.subplots import make_subplots

# The regular grid needs to be used for plotting, image it is a square but the data is extracted at the certain locations
fig = go.Figure(data = go.Volume(
    x=grid[:, 1].flatten(),
    y=grid[:, 0].flatten(),
    z=grid[:, 2].flatten(),
    value=values_gussian_filtered.flatten(),
    # isomin=10,
    # isomax=33,
    opacity = .1,
    # opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
    surface_count = 30,
    colorscale = "rainbow",
    # reversescale=True,
    caps=dict(x_show=False, y_show=False, z_show = False),
    ),)
fig = go.Figure(data = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=z,  # set color to an array/list of desired values
        colorscale='Viridis',  # choose a colorscale
        opacity=0.8
))
fig.add_trace(go.Cone(
    x=[lon_lc],
    y=[lat_lc],
    z=[0],
    u=[1],
    v=[1],
    w=[0],
    # colorscale='Blues',
    showscale = False,
    sizemode="absolute",
    sizeref=1000))

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
# fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=1, y=1, z=.5))

plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_delft3d.html", auto_open = True)
# fig.write_image("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/fig/delft3d/sal_north_moderate.pdf", width=1980, height=1080, engine = "orca")



