import numpy as np

x = np.arange(10)
y = np.arange(10)
z = np.arange(10)

xx, yy, zz = np.meshgrid(x, y, z)
X = xx.flatten()
Y = yy.flatten()
Z = zz.flatten()
import plotly
import plotly.graph_objects as go

# Generate nicely looking random 3D-field
np.random.seed(0)
l = 30
X, Y, Z = np.mgrid[:l, :l, :l]


vol = np.zeros((l, l, l))
pts = (l * np.random.rand(3, 15)).astype(np.int)
vol[tuple(indices for indices in pts)] = 1
from scipy import ndimage
vol = ndimage.gaussian_filter(vol, 4)
vol /= vol.max()

fig = go.Figure(data=go.Volume(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=vol.flatten(),
    # isomin=0.2,
    # isomax=0.7,
    opacity=0.1,
    surface_count=25,
    ))
fig.update_layout(scene_xaxis_showticklabels=False,
                  scene_yaxis_showticklabels=False,
                  scene_zaxis_showticklabels=False)
fig.show()
plotly.offline.plot(fig, filename = "Sample_illu.html", auto_open=True)

#%%

import numpy as np
import plotly.graph_objects as go

# Generate nicely looking random 3D-field
np.random.seed(0)
l = 30
X, Y, Z = np.mgrid[:l, :l, :l]
vol = np.zeros((l, l, l))
# pts = (l * np.random.rand(3, 15)).astype(np.int)
# vol[tuple(indices for indices in pts)] = 1
vol[0, 0, 0] = 1
vol[10, 10, 10] = 1
#%%
from scipy import ndimage
vol = ndimage.gaussian_filter(vol, 4)
vol
#%%
vol /= vol.max()

vol
