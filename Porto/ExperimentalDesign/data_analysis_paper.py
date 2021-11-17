
# Plot delft3d prior data
import numpy as np
import matplotlib.pyplot as plt
import h5py

path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged_all/North_Moderate_all.h5"
data = h5py.File(path_data, 'r')
lat = np.array(data.get("lat"))
lon = np.array(data.get("lon"))
depth = np.array(data.get("depth"))
salinity = np.array(data.get("salinity"))

depth[depth > 0] = 0

lat_min = np.nanmin(lat)
lat_max = np.nanmax(lat)
lon_min = np.nanmin(lon)
lon_max = np.nanmax(lon)
depth_min = np.nanmin(depth)
depth_max = np.nanmax(depth)

nlon = 25
nlat = 25
ndepth = 25
lon_domain = np.linspace(lon_min, lon_max, nlon)
lat_domain = np.linspace(lat_min, lat_max, nlat)
depth_domain = np.linspace(depth_min, depth_max, ndepth)
lon_d, lat_d, depth_d = np.meshgrid(lon_domain, lat_domain, depth_domain)
values = np.zeros([nlon, nlat, ndepth])

k_neighbour = 10

for i in range(nlon):
    print(i)
    for j in range(nlat):
        print(j)
        for k in range(ndepth):
            # print(k)
            lon_dist = lon.flatten() - lon_d[i, j, k]
            lat_dist = lat.flatten() - lat_d[i, j, k]
            depth_dist = depth.flatten() - depth_d[i, j, k]
            # x_dist = X - xd[i, j, k]
            # y_dist = Y - yd[i, j, k]
            # z_dist = Z - zd[i, j, k]
            dist = np.sqrt(lon_dist ** 2 + lat_dist ** 2 + depth_dist ** 2)
            ind_neighbour = np.argsort(dist)[:k_neighbour]
            # print(salinity.flatten()[ind_neighbour])
            # ind_x, ind_y, ind_z = np.where(dist == np.nanmin(dist))
            values[i, j, k] = np.nanmean(salinity.flatten()[ind_neighbour])

            # break
        # break
#%%
import plotly.graph_objects as go
import plotly

fig = go.Figure(data=[go.Scatter3d(
    x=lon_d.flatten(),
    y=lat_d.flatten(),
    z=depth_d.flatten(),
    mode='markers',
    marker=dict(
        size=12,
        color=values.flatten(),                # set color to an array/list of desired values
        colorscale='jet',   # choose a colorscale
        showscale = True,
        opacity=0.8
    )
)])
plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_delft3d.html", auto_open = True)


#%%
from scipy.ndimage import gaussian_filter
values_gussian_filtered = gaussian_filter(values, 1)

import plotly.graph_objects as go
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
plotly.io.orca.config.executable = '/usr/local/bin/orca'
plotly.io.orca.config.save()

# fig = make_subplots(
    # rows=1, cols=2,
    # specs=[[{'type': 'surface'}, {'type': 'surface'}]])

fig = make_subplots(
    rows=1, cols=1,
    specs=[[{'type': 'surface'}]])


fig.add_trace(go.Isosurface(
    x=lon_d.flatten(),
    y=lat_d.flatten(),
    z=depth_d.flatten(),
    value=values_gussian_filtered.flatten(),
    # isomin=10,
    isomax=34,
    opacity = .1,
    # opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
    surface_count = 50,
    # surface_fill = .4, # create texture on the surface
    # colorbar_nticks = 10,  # colorbar ticks correspond to isosurface values
    slices_z=dict(show=True, locations=[0]), # add slice
    slices_y=dict(show=True, locations=[0]),
    slices_x=dict(show=True, locations=[0]),
    # colorscale=px.colors.qualitative.Set1,
    # colorscale=px.colors.sequential.Aggrnyl,
    colorscale=px.colors.diverging.Spectral,
    reversescale=True,
    caps=dict(x_show=False, y_show=False, z_show = False),
    # name = "Salinity field under north moderate wind condition bird view"
    ),
    row = 1, col = 1)

# fig.add_trace(go.Isosurface(
#     x=lon_d.flatten(),
#     y=lat_d.flatten(),
#     z=depth_d.flatten(),
#     value=values_gussian_filtered.flatten(),
#     # isomin=10,
#     isomax=34,
#     opacity = .1,
#     # opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
#     surface_count = 50,
#     # surface_fill = .4, # create texture on the surface
#     # colorbar_nticks = 10,  # colorbar ticks correspond to isosurface values
#     slices_z=dict(show=True, locations=[0]), # add slice
#     slices_y=dict(show=True, locations=[0]),
#     slices_x=dict(show=True, locations=[0]),
#     # colorscale=px.colors.qualitative.Set1,
#     # colorscale=px.colors.sequential.Aggrnyl,
#     colorscale=px.colors.diverging.Spectral,
#     reversescale=True,
#     caps=dict(x_show=False, y_show=False, z_show = False),
#     ),
#     row = 1, col = 2)


fig.update_layout(
    scene = dict(
        # xaxis = dict(nticks=4, range=[-0.1,0.1],),
        # yaxis = dict(nticks=4, range=[-0.5,0.5],),
        zaxis = dict(nticks=4, range=[-11,0],),
        xaxis_title='Lon [deg]',
        yaxis_title='Lat [deg]',
        zaxis_title='Depth [m]',
    ),
    # scene2 = dict(
    #     # xaxis = dict(nticks=4, range=[-0.1,0.1],),
    #     # yaxis = dict(nticks=4, range=[-0.5,0.5],),
    #     zaxis=dict(nticks=4, range=[-11, 0], ),
    #     xaxis_title='Lon [deg]',
    #     yaxis_title='Lat [deg]',
    #     zaxis_title='Depth [m]',
    # )
)
camera1 = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=-1.25, z=1.25)
)

# camera2 = dict(
#     up=dict(x=0, y=0, z=1),
#     center=dict(x=0, y=0, z=0),
#     eye=dict(x=.8, y=-1.86, z=-.2)
# )

fig.update_layout(scene_camera=camera1,
                  # scene2_camera=camera2,
                  title="Salinity field under north moderate wind condition extract from Delft 3D")

fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=1, y=1, z=.5),
                  # scene2_aspectmode='manual',
                  # scene2_aspectratio=dict(x=1, y=1, z=.5)
                  )

plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_delft3d.html", auto_open = True)
fig.write_image("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Porto/fig/delft3d/sal_north_moderate.pdf", width=1980, height=1080, engine = "orca")


