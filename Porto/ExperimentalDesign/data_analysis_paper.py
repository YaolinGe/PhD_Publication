
# Plot delft3d prior data
import numpy as np
import matplotlib.pyplot as plt
import h5py

path_data = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Porto/Prior/Nov_Prior/Merged_all/North_Moderate_all.h5"
data = h5py.File(path_data, 'r')
lat = np.nan_to_num(data.get("lat"))
lon = np.nan_to_num(data.get("lon"))
depth = np.nan_to_num(data.get("depth"))
salinity = np.nan_to_num(data.get("salinity"))


#%% problem is it has nan values

layer_depth = 3
X = lon[:, :, :layer_depth]
Y = lat[:, :, :layer_depth]
Z = depth[:, :, :layer_depth]
# Z = np.zeros_like(Y)
# for i in range(len(np.nanmean(depth[:, :, :layer_depth], axis = (0, 1)))):
#     Z[:, :, i] = np.ones_like(lat[:, :, 0]) * np.nanmean(depth[:, :, :layer_depth], axis = (0, 1))[i]
S = salinity[:, :, :layer_depth]

nx = 25
ny = 25
nz = 5
x_domain = np.linspace(-8.8, -8.7, nx)
y_domain = np.linspace(41.1, 41.15, ny)
z_domain = np.linspace(-1, 0, nz)
xd, yd, zd = np.meshgrid(x_domain, y_domain, z_domain)
values = np.zeros([nx, ny, nz])
for i in range(nx):
    print(i)
    for j in range(ny):
        for k in range(nz):
            x_dist = X - xd[i, j, k]
            y_dist = Y - yd[i, j, k]
            z_dist = Z - zd[i, j, k]
            dist = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
            ind_x, ind_y, ind_z = np.where(dist == np.nanmin(dist))
            values[i, j, k] = S[ind_x, ind_y, ind_z]

# Continue working on the data ploting

#%%

Xf = X.flatten()
Yf = Y.flatten()
Zf = Z.flatten()
valuesf = values.flatten()

ind_non_zeros = np.where((Xf != 0) * (Yf != 0) * (Zf != 0) * (valuesf != 0))[0] # only non-zero values
ind_refined = np.where((Xf[ind_non_zeros] < -8.7) * (Xf[ind_non_zeros] > -8.8) * (Yf[ind_non_zeros] > 41.1) * (Yf[ind_non_zeros] < 41.15))[0]
#%%
import plotly.graph_objects as go
import plotly
fig = go.Figure(data=go.Isosurface(
    x=xd.flatten(),
    y=yd.flatten(),
    z=zd.flatten(),
    value=values.flatten(),
    isomin=20,
    # isomax=0.8,
    opacity=0.5, # needs to be small to see through all surfaces
    surface_count=25, # needs to be a large number for good volume rendering
    # caps = dict(x_show = True, y_show = False)
    ))

plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_delft3d.html", auto_open = True)
#%%

# ind_non_zeros = np.where((Xf != 0) * (Yf != 0) * (Zf != 0) * (valuesf != 0))[0] # only non-zero values

# import plotly.graph_objects as go
# import plotly
fig = go.Figure(data=go.Volume(
    x=Xf[ind_non_zeros][ind_refined].flatten(),
    y=Yf[ind_non_zeros][ind_refined].flatten(),
    z=Zf[ind_non_zeros][ind_refined].flatten(),
    value=valuesf[ind_non_zeros][ind_refined].flatten(),
    isomin=34,
    # isomax=0.8,
    # opacity=0.1, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    ))

# fig = go.Figure(data=[go.Scatter3d(
#     x=Xf[ind_non_zeros].flatten(),
#     y=Yf[ind_non_zeros].flatten(),
#     z=Zf[ind_non_zeros].flatten(),
#     mode='markers',
#     marker=dict(
#         size=12,
#         color=valuesf[ind_non_zeros].flatten(),                # set color to an array/list of desired values
#         colorscale='Viridis',   # choose a colorscale
#         opacity=0.8
#     )
# )])
plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_delft3d.html", auto_open = True)
#%%
fig = go.Figure(data=[go.Scatter3d(
    x=Xf[ind_non_zeros][ind_refined].flatten(),
    y=Yf[ind_non_zeros][ind_refined].flatten(),
    z=Zf[ind_non_zeros][ind_refined].flatten(),
    mode='markers',
    marker=dict(
        size=12,
        color=valuesf[ind_non_zeros][ind_refined].flatten(),                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_delft3d.html", auto_open = True)
#%%


import plotly.graph_objects as go
import plotly.express as px
import numpy as np

df = px.data.tips()

x1 = np.linspace(-4, 4, 9)
y1 = np.linspace(-5, 5, 11)
z1 = np.linspace(-5, 5, 11)

X, Y, Z = np.meshgrid(x1, y1, z1)

values = (np.sin(X ** 2 + Y ** 2)) / (X ** 2 + Y ** 2)

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    isomin=-0.5,
    isomax=0.5,
    value=values.flatten(),
    opacity=0.1,
    opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
    colorscale='RdBu'
))

plotly.offline.plot(fig, filename = "Porto/fig/delft3d/Sal_delft3d.html", auto_open = True)

