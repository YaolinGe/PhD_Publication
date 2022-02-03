import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import scipy.spatial.distance as scdist
import time
from usr_func import *
circumference = 40075000
datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/July06/Data/"
figpath = '/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/'
SINMOD_datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/"


mu_prior = np.loadtxt("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Data/mu_prior_sal.txt").reshape(-1, 1)
beta0 = np.loadtxt("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Coef/beta0.txt", delimiter = ",")
beta1 = np.loadtxt("/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/Experiment/Coef/beta1.txt", delimiter = ",")
t1 = time.time()



#% Data extraction from the raw data
rawTemp = pd.read_csv(datapath + "Temperature.csv", delimiter=', ', header=0, engine='python')
rawLoc = pd.read_csv(datapath + "EstimatedState.csv", delimiter=', ', header=0, engine='python')
rawSal = pd.read_csv(datapath + "Salinity.csv", delimiter=', ', header=0, engine='python')
rawDepth = pd.read_csv(datapath + "Depth.csv", delimiter=', ', header=0, engine='python')

# To group all the time stamp together, since only second accuracy matters
rawSal.iloc[:, 0] = np.ceil(rawSal.iloc[:, 0])
rawTemp.iloc[:, 0] = np.ceil(rawTemp.iloc[:, 0])
rawCTDTemp = rawTemp[rawTemp.iloc[:, 2] == 'SmartX']
rawLoc.iloc[:, 0] = np.ceil(rawLoc.iloc[:, 0])
rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])
rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])

depth_ctd = rawDepth[rawDepth.iloc[:, 2] == 'SmartX']["value (m)"].groupby(rawDepth["timestamp"]).mean()
depth_dvl = rawDepth[rawDepth.iloc[:, 2] == 'DVL']["value (m)"].groupby(rawDepth["timestamp"]).mean()
depth_est = rawLoc["depth (m)"].groupby(rawLoc["timestamp"]).mean()

# indices used to extract data
lat_origin = rawLoc["lat (rad)"].groupby(rawLoc["timestamp"]).mean()
lon_origin = rawLoc["lon (rad)"].groupby(rawLoc["timestamp"]).mean()
x_loc = rawLoc["x (m)"].groupby(rawLoc["timestamp"]).mean()
y_loc = rawLoc["y (m)"].groupby(rawLoc["timestamp"]).mean()
z_loc = rawLoc["z (m)"].groupby(rawLoc["timestamp"]).mean()
depth = rawLoc["depth (m)"].groupby(rawLoc["timestamp"]).mean()
time_loc = rawLoc["timestamp"].groupby(rawLoc["timestamp"]).mean()
time_sal= rawSal["timestamp"].groupby(rawSal["timestamp"]).mean()
time_temp = rawCTDTemp["timestamp"].groupby(rawCTDTemp["timestamp"]).mean()
dataSal = rawSal["value (psu)"].groupby(rawSal["timestamp"]).mean()
dataTemp = rawCTDTemp.iloc[:, -1].groupby(rawCTDTemp["timestamp"]).mean()

#% Rearrange data according to their timestamp
data = []
time_mission = []
xauv = []
yauv = []
zauv = []
dauv = []
sal_auv = []
temp_auv = []
lat_auv = []
lon_auv = []

for i in range(len(time_loc)):
    if np.any(time_sal.isin([time_loc.iloc[i]])) and np.any(time_temp.isin([time_loc.iloc[i]])):
        time_mission.append(time_loc.iloc[i])
        xauv.append(x_loc.iloc[i])
        yauv.append(y_loc.iloc[i])
        zauv.append(z_loc.iloc[i])
        dauv.append(depth.iloc[i])
        lat_temp = rad2deg(lat_origin.iloc[i]) + rad2deg(x_loc.iloc[i] * np.pi * 2.0 / circumference)
        lat_auv.append(lat_temp)
        lon_auv.append(rad2deg(lon_origin.iloc[i]) + rad2deg(y_loc.iloc[i] * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_temp)))))
        sal_auv.append(dataSal[time_sal.isin([time_loc.iloc[i]])].iloc[0])
        temp_auv.append(dataTemp[time_temp.isin([time_loc.iloc[i]])].iloc[0])
    else:
        print(datetime.fromtimestamp(time_loc.iloc[i]))
        continue

lat4, lon4 = 63.446905, 10.419426  # right bottom corner
lat_auv = np.array(lat_auv).reshape(-1, 1)
lon_auv = np.array(lon_auv).reshape(-1, 1)
Dx = deg2rad(lat_auv - lat4) / 2 / np.pi * circumference
Dy = deg2rad(lon_auv - lon4) / 2 / np.pi * circumference * np.cos(deg2rad(lat_auv))

xauv = np.array(xauv).reshape(-1, 1)
yauv = np.array(yauv).reshape(-1, 1)

alpha = deg2rad(60)
Rc = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
TT = (Rc @ np.hstack((Dx, Dy)).T).T
xauv_new = TT[:, 0].reshape(-1, 1)
yauv_new = TT[:, 1].reshape(-1, 1)

zauv = np.array(zauv).reshape(-1, 1)
dauv = np.array(dauv).reshape(-1, 1)
sal_auv = np.array(sal_auv).reshape(-1, 1)
temp_auv = np.array(temp_auv).reshape(-1, 1)
time_mission = np.array(time_mission).reshape(-1, 1)

datasheet = np.hstack((time_mission, lat_auv, lon_auv, xauv, yauv, zauv, dauv, sal_auv, temp_auv))
# np.savetxt(os.getcwd() + "data.txt", datasheet, delimiter = ",")


#%%


starting_index = 0
origin = [lat4, lon4]
distance = 1000
depth_obs = [0.5, 1.0, 1.5, 2.0, 2.5]  # planned depth to be observed
box = BBox(lat4, lon4, distance, 60)
N1 = 25 # number of grid points along north direction
N2 = 25 # number of grid points along east direction
N3 = 5 # number of layers in the depth dimension
N = N1 * N2 * N3 # total number of grid points

sigma = np.sqrt(4)
tau = np.sqrt(.3)
Threshold = 28
eta = 4.5 / 400
ksi = 1000 / 24 / 0.5


XLIM = [0, distance]
YLIM = [0, distance]
ZLIM = [0.5, 2.5]
x = np.linspace(XLIM[0], XLIM[1], N1)
y = np.linspace(YLIM[0], YLIM[1], N2)
z = np.array(depth_obs)
grid = []
for k in z:
    for i in x:
        for j in y:
            grid.append([i, j, k])
grid = np.array(grid)
xv = grid[:, 0].reshape(-1, 1)
yv = grid[:, 1].reshape(-1, 1)
zv = grid[:, 2].reshape(-1, 1)
dx = x[1] - x[0]
coordinates= getCoordinates(box, N1, N2, dx, 60)


grid = np.array(grid)
H_grid = compute_H(grid, grid, ksi)
Sigma_prior = Matern_cov(sigma, eta, H_grid)

def myround(x, base=1.):
    return base * np.round(x/base)

dauv_new = myround(dauv, base = .5)
# ind = (dauv_new > 0).squeeze()
ind = range(starting_index, len(dauv_new))
xauv_new_truncated = xauv_new[ind].reshape(-1, 1)
yauv_new_truncated = yauv_new[ind].reshape(-1, 1)
dauv_new_truncated = dauv_new[ind].reshape(-1, 1)
Xauv_new = myround(xauv_new_truncated, base = dx)
Yauv_new = myround(yauv_new_truncated, base = dx)
sal_auv_truncated = sal_auv[ind].reshape(-1, 1)
coordinates_auv_truncated = np.hstack((lat_auv[ind], lon_auv[ind]))

SINMOD_path = SINMOD_datapath + 'samples_2020.05.01.nc'
SINMOD = netCDF4.Dataset(SINMOD_path)

# #%%
# for i in range(5):
#     sal_sinmod, temp_sinmod = GetSINMODFromCoordinates(SINMOD, coordinates, depth_obs[i])
#     plt.imshow((beta0[i, 0] + beta1[i, 1] * sal_sinmod).reshape(N1, N2), vmin = 15, vmax = 30)
#     plt.colorbar()
#     plt.show()

mu_cond = mu_prior
Sigma_cond = Sigma_prior
def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z


x_eye = -1.25
y_eye = -1.25
z_eye = .5
# for i in range(10):
for i in [len(xauv_new_truncated)]:
# for i in len(xauv_new)):
    mu_cond = mu_prior
    Sigma_cond = Sigma_prior
    print(i)
    XAUV = xauv_new_truncated[:i + 1]
    YAUV = yauv_new_truncated[:i + 1]
    DAUV = dauv_new_truncated[:i + 1]
    COORDINATES = coordinates_auv_truncated[:i + 1]

    sal_sinmod, temp_sinmod = GetSINMODFromCoordinates(SINMOD, COORDINATES, DAUV)

    mu_sal_est = []
    for j in range(len(sal_sinmod)):
        print(depth_obs)
        print(dauv_new_truncated)
        k = np.where(depth_obs == dauv_new_truncated[j])[0][0]
        mu_sal_est.append(beta0[k, 0] + beta1[k, 0] * sal_sinmod[j, 0])
    mu_sal_est = np.array(mu_sal_est).reshape(-1, 1)

    obs = np.hstack((XAUV, YAUV, DAUV))
    H_obs = compute_H(obs, obs, ksi)
    Sigma_obs = Matern_cov(sigma, eta, H_obs) + tau ** 2 * np.identity(H_obs.shape[0])

    H_grid_obs = compute_H(grid, obs, ksi)
    Sigma_grid_obs = Matern_cov(sigma, eta, H_grid_obs)

    mu_cond = mu_cond + Sigma_grid_obs @ np.linalg.solve(Sigma_obs, (sal_auv[:i + 1] - mu_sal_est))
    Sigma_cond = Sigma_cond - Sigma_grid_obs @ np.linalg.solve(Sigma_obs, Sigma_grid_obs.T)
    perr = np.diag(Sigma_cond).reshape(-1, 1)
    EP = get_excursion_prob_1d(mu_cond, Sigma_cond, Threshold)
    ##
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

    for j in range(len(np.unique(zv))):
        ind = (zv == np.unique(zv)[j])
        fig.add_trace(
            go.Isosurface(x=xv[ind], y=yv[ind], z=-zv[ind],
                          # value=mu_cond[ind], colorscale=newcmp),
                          value=EP[ind], coloraxis = "coloraxis"),
            row=1, col=1
        )
    fig.add_trace(
        go.Scatter3d(
            x=XAUV.squeeze(), y=YAUV.squeeze(), z=np.array(-DAUV.squeeze()),
            marker=dict(
                size=4,
                color="black",
                showscale=False
            ),
            line=dict(
                color='darkblue',
                width=2
            )
        ),
        row=1, col=1
    )
    fig.update_coloraxes(colorscale = "gnbu")
    # fig.update_coloraxes(colorscale = newcmp)
    xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -i * .005)
    fig.update_layout(
        scene={
            'aspectmode': 'manual',
            'aspectratio': dict(x=1, y=1, z=.5),
        },
        showlegend=False,
        scene_camera_eye=dict(x=xe, y=ye, z=ze),
        title="AUV explores the field"
    )
    plotly.offline.plot(fig, filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/updated_excursion_probability.html", auto_open = False)
    os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Experiment/updated_excursion_probability.html")
    # fig.write_image(figpath + "/T_{:04d}.png".format(i), width=1980, height=1080)
t2 = time.time()

print("Time consumed: ", t2 - t1)




