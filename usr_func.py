import numpy as np
from scipy.stats import mvn, norm
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
circumference = 40075000 # [m], circumference


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


def get_rotational_matrix(alpha):
    R = np.array([[np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))],
                  [-np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])
    return R


def setLoggingFilename(filename):
    import logging
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(filename=filename, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def vectorise(value):
    return np.array(value).reshape(-1, 1)


def EIBV_1D(threshold, mu, Sig, F, R):
    Sigxi = Sig @ F.T @ np.linalg.solve(F @ Sig @ F.T + R, F @ Sig)
    V = Sig - Sigxi
    sa2 = np.diag(V).reshape(-1, 1)  # the corresponding variance term for each location
    IntA = 0.0
    for i in range(len(mu)):
        sn2 = sa2[i]
        m = mu[i]
        IntA = IntA + mvn.mvnun(-np.inf, threshold, m, sn2)[0] - mvn.mvnun(-np.inf, threshold, m, sn2)[0] ** 2
    return IntA


def EP_1D(mu, Sigma, Threshold):
    EP = np.zeros_like(mu)
    for i in range(EP.shape[0]):
        EP[i] = norm.cdf(Threshold, mu[i], Sigma[i, i])
    return EP


def GPupd(mu_cond, Sigma_cond, F, R, y_sampled):
    C = F @ Sigma_cond @ F.T + R
    mu_cond = mu_cond + Sigma_cond @ F.T @ np.linalg.solve(C,(y_sampled - F @ mu_cond))
    Sigma_cond = Sigma_cond - Sigma_cond @ F.T @ np.linalg.solve(C, F @ Sigma_cond)
    return mu_cond, Sigma_cond


def getFVector(ind, N):
    F = np.zeros([1, N])
    F[0, ind] = True
    return F


def get_grid_ind_at_nearest_loc(loc, coordinates):
    lat, lon, depth = loc
    dx, dy = latlon2xy(coordinates[:, 0], coordinates[:, 1], lat, lon)
    dz = coordinates[:, 2] - depth
    dist = dx ** 2 + dy ** 2 + dz ** 2
    ind = np.argmin(dist)
    return ind

# loc = [63.47, 10.415, 0]
# id = get_ind_nearest(loc, coordinates)
# import matplotlib.pyplot as plt
# plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
# plt.plot(loc[1], loc[0], 'r*')
# plt.plot(coordinates[id, 1], coordinates[id, 0], 'bx')
# plt.show()



