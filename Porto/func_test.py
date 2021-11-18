import numpy as np
import matplotlib.pyplot as plt
import plotly

x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
z = np.linspace(-1, 1, 10)
xx, yy, zz = np.meshgrid(x, y, z)
xf = xx.flatten()
yf = yy.flatten()
zf = zz.flatten()

def gaussian_func(x, y, z, sigma):
    # p = 1 / 2 / np.pi / sigma ** 2 * np.exp( - 1 / 2 / sigma ** 2 * ((x - 1) ** 2 + (y - 1) ** 2 + (z - 1) ** 2))
    p = 1 / 2 / np.pi / sigma ** 2 * np.exp(- 1 / 2 / sigma ** 2 * ((x - 0) ** 2 + (y - 0) ** 2 + (z - 0) ** 2))
    return p

sigma = 1
p = gaussian_func(xf, yf, zf, sigma)

import plotly.graph_objects as go
X, Y, Z, values = xf, yf, zf, p

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=0.1,
    isomax=.15,
    opacity = .1,
    opacityscale=[[-0.5, 1], [-0.2, 0], [0.2, 0], [0.5, 1]],
    surface_count = 5,
    # surface_fill = .4, # create texture on the surface
    colorbar_nticks = 5,  # colorbar ticks correspond to isosurface values
    # slices_z=dict(show=True, locations=[0]), # add slice
    # slices_y=dict(show=True, locations=[0]),
    # slices_x=dict(show=True, locations=[0]),
    colorscale="BlueRed",
    caps=dict(x_show=False, y_show=False),
    # color = 'rgba(244,22,100,0.6)'
    ))
# fig.update_l_eye=dict(x=1ayout(
# #     margin=dict(t=1, l=1, b=0), # tight layout
# #     scene_camera.86, y=0.61, z=0.98))

fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[-0.1,0.1],),
        yaxis = dict(nticks=4, range=[-0.5,0.5],),
        zaxis = dict(nticks=4, range=[-0.5,0.5],),),
    # )
)
fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=1, y=1, z=1))

# fig.update_layout(scene_aspectmode='cube')
    # width=700,
    # margin=dict(r=20, l=10, b=10, t=10))

plotly.offline.plot(fig, filename = "Porto/gaussian.html", auto_open = True)

#%%
#%%
import plotly
import plotly.graph_objects as go

fig = go.Figure(data = go.Scatter3d(
    x = lon.flatten(),
    y = lat.flatten(),
    z = depth.flatten(),
    mode='markers',
    marker=dict(
        size=12,
        color=salinity.flatten(),  # set color to an array/list of desired values
        colorscale='jet',  # choose a colorscale
        showscale=True,
        opacity=0.8
    )
))
plotly.offline.plot(fig, filename = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/scatter.html", auto_open = True)



