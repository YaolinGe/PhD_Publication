import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# from gmplot import GoogleMapPlotter

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import mvn
import scipy.spatial.distance as scdist
import netCDF4
import os
# import gmplot
from datetime import date
#from skgstat import Variogram, DirectionalVariogram
from sklearn.linear_model import LinearRegression
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})
circumference = 40075000 # [m]
err_bound = 0.1 # [m]


def EP_1D(mu, Sigma, Threshold):
    '''
    This function computes the excursion probability
    :param mu:
    :param Sigma:
    :param Threshold:
    :return:
    '''
    EP = np.zeros_like(mu)
    for i in range(EP.shape[0]):
        EP[i] = norm.cdf(Threshold, mu[i], Sigma[i, i])
    return EP


def EP_2D(mu, Sigma, Threshold_T, Threshold_S):
    '''
    :param mu:
    :param Sigma:
    :param Threshold_T:
    :param Threshold_S:
    :return:
    '''
    EP = []
    for i in np.arange(0, mu.shape[0], 2):
        Sigmaxi = np.array([[Sigma[i, i], Sigma[i, i + 1]], [Sigma[i + 1, i], Sigma[i + 1, i + 1]]])
        muxi = mu[i:i + 2] # contains the mu for temp and salinity
        EP.append(mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.zeros([2, 1]),
                       np.subtract(np.array([[Threshold_T], [Threshold_S]]), muxi), Sigmaxi)[0])
    EP = np.array(EP)
    return EP


def ravel_index(loc, n1, n2, n3):
    '''
    :param loc:
    :param n1:
    :param n2:
    :param n3:
    :return:
    '''
    x, y, z = loc
    ind = int(z * n1 * n2 + y * n1 + x)
    return ind


def unravel_index(ind, n1, n2, n3):
    '''
    :param ind:
    :param n1:
    :param n2:
    :param n3:
    :return:
    '''
    zind = np.floor(ind / (n1 * n2))
    residual = ind - zind * (n1 * n2)
    yind = np.floor(residual / n1)
    xind = residual - yind * n1
    loc = [int(xind), int(yind), int(zind)]
    return loc


def find_starting_loc(EP_Prior, n1, n2, n3):
    '''
    :param EP_Prior:
    :param n1:
    :param n2:
    :param n3:
    :return:
    '''
    ep_criterion = 0.5
    ind = (np.abs(EP_Prior - ep_criterion)).argmin()
    loc = unravel_index(ind, n1, n2, n3)
    return loc


def xy2latlon(x, y, origin, distance, alpha):
    '''
    :param x: index from origin along left line
    :param y: index from origin along right line
    :param origin:
    :param distance:
    :param alpha:
    :return:
    '''
    R = np.array([[np.cos(deg2rad(alpha)), -np.sin(deg2rad(alpha))],
                  [np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])
    x_loc, y_loc = R @ np.vstack((x * distance, y * distance)) # converted xp/yp with distance inside
    lat_origin, lon_origin = origin
    lat = lat_origin + rad2deg(x_loc * np.pi * 2.0 / circumference)
    lon = lon_origin + rad2deg(y_loc * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat))))
    return np.hstack((lat, lon))


def rotateXY(nx, ny, distance, alpha):
    '''
    :param nx:
    :param ny:
    :param distance:
    :param alpha:
    :return:
    '''
    R = np.array([[np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))],
                  [-np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])
    x = np.arange(nx) * distance
    y = np.arange(ny) * distance
    gridx, gridy = np.meshgrid(x, y)
    gridxnew = np.zeros_like(gridx)
    gridynew = np.zeros_like(gridy)
    for i in range(nx):
        for j in range(ny):
            gridxnew[i, j], gridynew[i, j] = R @ np.vstack((gridx[i, j], gridy[i, j]))
    return gridxnew, gridynew


def getCoefficients(data, SINMOD, depth):
    '''
    :param data:
    :param SINMOD:
    :param depth:
    :return:
    '''
    timestamp = data[:, 0].reshape(-1, 1)
    # lat_auv = rad2deg(data[:, 1].reshape(-1, 1))
    # lon_auv = rad2deg(data[:, 2].reshape(-1, 1))
    lat_auv = data[:, 1].reshape(-1, 1)
    lon_auv = data[:, 2].reshape(-1, 1)
    # print(lat_auv, lon_auv)

    xauv = data[:, 3].reshape(-1, 1)
    yauv = data[:, 4].reshape(-1, 1)
    zauv = data[:, 5].reshape(-1, 1)
    depth_auv = data[:, 6].reshape(-1, 1)
    sal_auv = data[:, 7].reshape(-1, 1)
    temp_auv = data[:, 8].reshape(-1, 1)
    lat_auv = lat_auv + rad2deg(xauv * np.pi * 2.0 / circumference)
    lon_auv = lon_auv + rad2deg(yauv * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat_auv))))

    depthl = np.array(depth) - err_bound
    depthu = np.array(depth) + err_bound

    # print(depthl, depthu)

    beta0 = np.zeros([len(depth), 2])
    beta1 = np.zeros([len(depth), 2])
    sal_residual = []
    temp_residual = []
    x_loc = []
    y_loc = []

    # print(depth_auv)
    for i in range(len(depth)):
        ind_obs = (depthl[i] <= depth_auv) & (depth_auv <= depthu[i])
        lat_obs = lat_auv[ind_obs].reshape(-1, 1)
        lon_obs = lon_auv[ind_obs].reshape(-1, 1)
        sal_obs = sal_auv[ind_obs].reshape(-1, 1)
        temp_obs = temp_auv[ind_obs].reshape(-1, 1)
        # print(lat_obs.shape)
        # print(lon_obs.shape)
        sal_SINMOD, temp_SINMOD = GetSINMODFromCoordinates(SINMOD, np.hstack((lat_obs, lon_obs)), depth[i])
        # print(sal_SINMOD.shape)
        # print(temp_SINMOD.shape)
        if sal_SINMOD.shape[0] != 0:
            model_sal = LinearRegression()
            model_sal.fit(sal_SINMOD, sal_obs)
            beta0[i, 0] = model_sal.intercept_
            beta1[i, 0] = model_sal.coef_

            model_temp = LinearRegression()
            model_temp.fit(temp_SINMOD, temp_obs)
            beta0[i, 1] = model_temp.intercept_
            beta1[i, 1] = model_temp.coef_

            sal_residual.append(sal_obs - beta0[i, 0] - beta1[i, 0] * sal_SINMOD)
            temp_residual.append(temp_obs - beta0[i, 1] - beta1[i, 1] * temp_SINMOD)
            x_loc.append(xauv[ind_obs])
            y_loc.append(yauv[ind_obs])
        model_sal = LinearRegression()
        model_sal.fit(sal_SINMOD, sal_obs)
        beta0[i, 0] = model_sal.intercept_
        beta1[i, 0] = model_sal.coef_

        model_temp = LinearRegression()
        model_temp.fit(temp_SINMOD, temp_obs)
        beta0[i, 1] = model_temp.intercept_
        beta1[i, 1] = model_temp.coef_

        sal_residual.append(sal_obs - beta0[i, 0] - beta1[i, 0] * sal_SINMOD)
        temp_residual.append(temp_obs - beta0[i, 1] - beta1[i, 1] * temp_SINMOD)
        x_loc.append(xauv[ind_obs])
        y_loc.append(yauv[ind_obs])

    return beta0, beta1, sal_residual, temp_residual, x_loc, y_loc


def deg2rad(deg):
    '''
    :param deg:
    :return:
    '''
    return deg / 180 * np.pi


def rad2deg(rad):
    '''
    :param rad:
    :return:
    '''
    return rad / np.pi * 180


def BBox(lat, lon, distance, alpha):
    '''
    :param lat:
    :param lon:
    :param distance:
    :param alpha:
    :return:
    '''
    lat4 = deg2rad(lat)
    lon4 = deg2rad(lon)

    lat2 = lat4 + distance * np.sin(deg2rad(alpha)) / circumference * 2 * np.pi
    lat1 = lat2 + distance * np.cos(deg2rad(alpha)) / circumference * 2 * np.pi
    lat3 = lat4 + distance * np.sin(np.pi / 2 - deg2rad(alpha)) / circumference * 2 * np.pi

    lon2 = lon4 + distance * np.cos(deg2rad(alpha)) / (circumference * np.cos(lat2)) * 2 * np.pi
    lon3 = lon4 - distance * np.cos(np.pi / 2 - deg2rad(alpha)) / (circumference * np.cos(lat3)) * 2 * np.pi
    lon1 = lon3 + distance * np.cos(deg2rad(alpha)) / (circumference * np.cos(lat1)) * 2 * np.pi

    box = np.vstack((np.array([lat1, lat2, lat3, lat4]), np.array([lon1, lon2, lon3, lon4]))).T

    return rad2deg(box)


def getCoordinates(box, nx, ny, distance, alpha):
    '''
    :param box:
    :param nx:
    :param ny:
    :param distance:
    :param alpha:
    :return:
    '''
    R = np.array([[np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))],
                  [-np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])

    lat_origin, lon_origin = box[-1, :]
    x = np.arange(nx) * distance
    y = np.arange(ny) * distance
    gridx, gridy = np.meshgrid(x, y)

    lat = np.zeros([nx, ny])
    lon = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            xnew, ynew = R @ np.vstack((gridx[i, j], gridy[i, j]))
            lat[i, j] = lat_origin + rad2deg(xnew * np.pi * 2.0 / circumference)
            lon[i, j] = lon_origin + rad2deg(ynew * np.pi * 2.0 / (circumference * np.cos(deg2rad(lat[i, j]))))
    coordinates = np.hstack((lat.reshape(-1, 1), lon.reshape(-1, 1)))
    return coordinates


def GetSINMODFromCoordinates(SINMOD, coordinates, depth):
    '''
    get sinmod data from coordinates with depth
    :param SINMOD:
    :param coordinates:
    :param depth:
    :return:
    '''
    salinity = np.mean(SINMOD['salinity'][:, :, :, :], axis=0)
    temperature = np.mean(SINMOD['temperature'][:, :, :, :], axis=0) - 273.15
    depth_sinmod = np.array(SINMOD['zc'])
    lat_sinmod = np.array(SINMOD['gridLats'][:, :]).reshape(-1, 1)
    lon_sinmod = np.array(SINMOD['gridLons'][:, :]).reshape(-1, 1)
    sal_sinmod = np.zeros([coordinates.shape[0], 1])
    temp_sinmod = np.zeros([coordinates.shape[0], 1])

    for i in range(coordinates.shape[0]):
        lat, lon = coordinates[i]
        ind_depth = np.where(np.array(depth_sinmod) == depth)[0][0]
        idx = np.argmin((lat_sinmod - lat) ** 2 + (lon_sinmod - lon) ** 2)
        sal_sinmod[i] = salinity[ind_depth].reshape(-1, 1)[idx]
        temp_sinmod[i] = temperature[ind_depth].reshape(-1, 1)[idx]
    return sal_sinmod, temp_sinmod


def getSIMODfromCoordinates(SINMOD, coordinates, depth):
    '''
    get sinmod purely from coordinates
    :param SINMOD:
    :param coordinates:
    :param depth:
    :return:
    '''
    salinity = np.mean(SINMOD['salinity'][:, :, :, :], axis=0)
    temperature = np.mean(SINMOD['temperature'][:, :, :, :], axis=0) - 273.15
    depth_sinmod = np.array(SINMOD['zc'])
    lat_sinmod = np.array(SINMOD['gridLats'][:, :]).reshape(-1, 1)
    lon_sinmod = np.array(SINMOD['gridLons'][:, :]).reshape(-1, 1)
    sal_sinmod = np.zeros([coordinates.shape[0], 1])
    temp_sinmod = np.zeros([coordinates.shape[0], 1])

    for i in range(coordinates.shape[0]):
        lat, lon = coordinates[i]
        ind_depth = np.where(np.array(depth_sinmod) == depth)[0][0]
        idx = np.argmin((lat_sinmod - lat) ** 2 + (lon_sinmod - lon) ** 2)
        sal_sinmod[i] = salinity[ind_depth].reshape(-1, 1)[idx]
        temp_sinmod[i] = temperature[ind_depth].reshape(-1, 1)[idx]
    return sal_sinmod, temp_sinmod


def Matern_cov(sigma, eta, t):
    '''
    :param sigma: scaling coef
    :param eta: range coef
    :param t: distance matrix
    :return: matern covariance
    '''
    return sigma ** 2 * (1 + eta * t) * np.exp(-eta * t)

import numpy as np
import matplotlib.pyplot as plt

from random import sample
from scipy.stats import norm, mvn
import scipy.spatial.distance as scdist
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
import time

## SETUP FONT ##
# plt.ioff()  # Running plt.ioff() - plots are kept in background, plt.ion() plots are shown as they are generated
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.style': 'oblique'})

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

cmap_proba = ListedColormap(sns.color_palette("RdBu_r", 30))
CMAP_EXCU = ListedColormap(sns.color_palette("Reds", 300))
## CONSTRUCT A BIG CLASS ##


def distance_matrix(sites1v, sites2v):
    '''
    :param sites1v:
    :param sites2v:
    :return:
    '''
    n = len(sites1v)
    ddE = np.abs(sites1v * np.ones([1, n]) - np.ones([n, 1]) * sites1v.T)
    dd2E = ddE * ddE
    ddN = np.abs(sites2v * np.ones([1, n]) - np.ones([n, 1]) * sites2v.T)
    dd2N = ddN * ddN
    t = np.sqrt(dd2E + dd2N)
    return t


def compute_H(grid1, grid2, ksi):
    '''
    :param grid1:
    :param grid2:
    :param ksi:
    :return:
    '''
    X1 = grid1[:, 0].reshape(-1, 1)
    Y1 = grid1[:, 1].reshape(-1, 1)
    Z1 = grid1[:, -1].reshape(-1, 1)
    X2 = grid2[:, 0].reshape(-1, 1)
    Y2 = grid2[:, 1].reshape(-1, 1)
    Z2 = grid2[:, -1].reshape(-1, 1)

    distX = X1 @ np.ones([1, X2.shape[0]]) - np.ones([X1.shape[0], 1]) @ X2.T
    distY = Y1 @ np.ones([1, Y2.shape[0]]) - np.ones([Y1.shape[0], 1]) @ Y2.T
    distXY = distX ** 2 + distY ** 2
    distZ = Z1 @ np.ones([1, Z2.shape[0]]) - np.ones([Z1.shape[0], 1]) @ Z2.T
    dist = np.sqrt(distXY + (ksi * distZ) ** 2)

    return dist


## Functions used
def Matern_cov(sigma, eta, H):
    '''
    :param sigma: scaling coef
    :param eta: range coef
    :param H: distance matrix
    :return: matern covariance
    '''
    return sigma ** 2 * (1 + eta * H) * np.exp(-eta * H)


def Exp_cov(eta, t):
    '''
    :param eta:
    :param t:
    :return:
    '''
    return np.exp(eta * t)


# def
def plotf(Y, string, xlim = None, ylim = None, vmin = None, vmax = None):
    '''
    :param Y:
    :param string:
    :return:
    '''
    # plt.figure(figsize=(5,5))
    fig = plt.gcf()
    plt.imshow(Y, vmin = vmin, vmax = vmax, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
    plt.title(string)
    plt.xlabel("s1")
    plt.ylabel("s2")
    plt.colorbar(fraction=0.045, pad=0.04)
    # plt.gca().invert_yaxis()
    # plt.show()
    # plt.savefig()
    return fig


def mu(H, beta):
    '''
    :param H: design matrix
    :param beta: regression coef
    :return: mean
    '''
    return np.dot(H, beta)


def EIBV_1D(threshold, mu, Sig, F, R):
    '''
    :param threshold:
    :param mu:
    :param Sig:
    :param F: sampling matrix
    :param R: noise matrix
    :return: EIBV evaluated at every point
    '''
    Sigxi = Sig @ F.T @ np.linalg.solve( F @ Sig @ F.T + R, F @ Sig)
    V = Sig - Sigxi
    sa2 = np.diag(V).reshape(-1, 1) # the corresponding variance term for each location
    IntA = 0.0
    for i in range(len(mu)):
        sn2 = sa2[i]
        m = mu[i]
        IntA = IntA + mvn.mvnun(-np.inf, threshold, m, sn2)[0] - mvn.mvnun(-np.inf, threshold, m, sn2)[0] ** 2
    return IntA


def EIBV_2D(Threshold_T, Threshold_S, mu, Sig, F, R):
    '''
    :param Threshold_T:
    :param Threshold_S:
    :param mu:
    :param Sig:
    :param F: sampling matrix
    :param R: noise matrix
    :return: EIBV evaluated at every point
    '''
    # Update the field variance
    a = np.dot(Sig, F.T)
    b = np.dot(np.dot(F, Sig), F.T) + R
    c = np.dot(F, Sig)
    Sigxi = np.dot(a, np.linalg.solve(b, c))  # new covariance matrix
    V = Sig - Sigxi  # Uncertainty reduction # updated covariance

    IntA = 0.0
    N = mu.shape[0]
    # integrate out all elements in the bernoulli variance term
    for i in np.arange(0, N, 2):

        # extract the corresponding variance reduction term
        SigMxi = Sigxi[np.ix_([i, i + 1], [i, i + 1])]

        # extract the corresponding mean terms
        Mxi = [mu[i], mu[i + 1]] # temp and salinity

        sn2 = V[np.ix_([i, i + 1], [i, i + 1])]
        # vv2 = np.add(sn2, SigMxi) # was originally used to make it obscure
        vv2 = Sig[np.ix_([i, i + 1], [i, i + 1])]

        # compute the first part of the integration
        Thres = np.vstack((Threshold_T, Threshold_S))
        mur = np.subtract(Thres, Mxi)
        IntB_a = mvn.mvnun(np.array([[-np.inf], [-np.inf]]), np.zeros([2, 1]), mur, vv2)[0]

        # compute the second part of the integration, which is squared
        mm = np.vstack((Mxi, Mxi))
        # SS = np.array([[vv2, SigMxi], [SigMxi, vv2]]) # thought of it as a simplier version
        SS = np.add(np.vstack((np.hstack((sn2, np.zeros((2, 2)))), np.hstack((np.zeros((2, 2)), sn2)))),
                    np.vstack((np.hstack((SigMxi, SigMxi)), np.hstack((SigMxi, SigMxi)))))
        Thres = np.vstack((Threshold_T, Threshold_S, Threshold_T, Threshold_S))
        mur = np.subtract(Thres, mm)
        IntB_b = mvn.mvnun(np.array([[-np.inf], [-np.inf], [-np.inf], [-np.inf]]), np.zeros([4, 1]), mur, SS)[0]

        # compute the total integration
        IntA = IntA + np.nansum([IntB_a, -IntB_b])

    return IntA


def GPupd(mu, Sig, R, F, y_sampled):
    '''
    :param mu:
    :param Sig:
    :param R:
    :param F:
    :param y_sampled:
    :return:
    '''
    C = F @ Sig @ F.T + R
    mu_p = mu + Sig @ F.T @ np.linalg.solve(C, (y_sampled - F @ mu))
    Sigma_p = Sig - Sig @ F.T @ np.linalg.solve(C, F @ Sig)
    return mu_p, Sigma_p


def find_candidates_loc(x_ind, y_ind, z_ind, N1, N2, N3):
    '''
    :param x_ind:
    :param y_ind:
    :param z_ind:
    :param N1: number of grid along x direction
    :param N2:
    :param N3:
    :return:
    '''

    x_ind_l = [x_ind - 1 if x_ind > 0 else x_ind]
    x_ind_u = [x_ind + 1 if x_ind < N1 - 1 else x_ind]
    y_ind_l = [y_ind - 1 if y_ind > 0 else y_ind]
    y_ind_u = [y_ind + 1 if y_ind < N2 - 1 else y_ind]
    z_ind_l = [z_ind - 1 if z_ind > 0 else z_ind]
    z_ind_u = [z_ind + 1 if z_ind < N3 - 1 else z_ind]

    x_ind_v = np.unique(np.vstack((x_ind_l, x_ind, x_ind_u)))
    y_ind_v = np.unique(np.vstack((y_ind_l, y_ind, y_ind_u)))
    z_ind_v = np.unique(np.vstack((z_ind_l, z_ind, z_ind_u)))

    x_ind, y_ind, z_ind = np.meshgrid(x_ind_v, y_ind_v, z_ind_v)

    return x_ind.reshape(-1, 1), y_ind.reshape(-1, 1), z_ind.reshape(-1, 1)


def find_next_EIBV_1D(x_cand, y_cand, z_cand, x_now, y_now, z_now,
                      x_pre, y_pre, z_pre, N1, N2, N3, Sig, mu, tau, Threshold):

    id = []
    dx1 = x_now - x_pre
    dy1 = y_now - y_pre
    dz1 = z_now - z_pre
    vec1 = np.array([dx1, dy1, dz1])
    for i in x_cand:
        for j in y_cand:
            for z in z_cand:
                if i == x_now and j == y_now and z == z_now:
                    continue
                dx2 = i - x_now
                dy2 = j - y_now
                dz2 = z - z_now
                vec2 = np.array([dx2, dy2, dz2])
                if np.dot(vec1, vec2) >= 0:
                    id.append(ravel_index([i, j, z], N1, N2, N3))
                else:
                    continue
    id = np.unique(np.array(id))

    M = len(id)
    noise = tau ** 2
    R = np.diagflat(noise)  # diag not anymore support constructing matrix from vector
    N = N1 * N2 * N3
    eibv = []
    for k in range(M):
        F = np.zeros([1, N])
        F[0, id[k]] = True
        eibv.append(EIBV_1D(Threshold, mu, Sig, F, R))
    ind_desired = np.argmin(np.array(eibv))
    x_next, y_next, z_next = unravel_index(id[ind_desired], N1, N2, N3)

    return x_next, y_next, z_next


def find_next_EIBV_2D(x_cand, y_cand, z_cand, x_now, y_now, z_now,
                      x_pre, y_pre, z_pre, N1, N2, N3, Sig, mu, tau, Threshold_T, Threshold_S):

    id = []
    dx1 = x_now - x_pre
    dy1 = y_now - y_pre
    dz1 = z_now - z_pre
    vec1 = np.array([dx1, dy1, dz1])
    for i in x_cand:
        for j in y_cand:
            for z in z_cand:
                if i == x_now and j == y_now and z == z_now:
                    continue
                dx2 = i - x_now
                dy2 = j - y_now
                dz2 = z - z_now
                vec2 = np.array([dx2, dy2, dz2])
                if np.dot(vec1, vec2) >= 0:
                    id.append(ravel_index([i, j, z], N1, N2, N3))
                else:
                    continue
    id = np.unique(np.array(id))

    M = len(id)
    noise = np.ones([2, 1]) * tau ** 2
    R = np.diagflat(noise)  # diag not anymore support constructing matrix from vector
    N = N1 * N2 * N3 * 2
    eibv = []
    for k in range(M):
        F = np.zeros([2, N])
        F[0, id[k]] = True
        F[1, id[k] + 1] = True
        eibv.append(EIBV_2D(Threshold_T, Threshold_S, mu, Sig, F, R))
    ind_desired = np.argmin(np.array(eibv))
    x_next, y_next, z_next = unravel_index(id[ind_desired], N1, N2, N3)

    return x_next, y_next, z_next

#
# class CustomGoogleMapPlotter(GoogleMapPlotter):
#     def __init__(self, center_lat, center_lng, zoom, apikey='AIzaSyAZ_VZXoJULTFQ9KSPg1ClzHEFjyPbJUro',
#                  map_type='satellite'):
#         if apikey == '':
#             try:
#                 with open('apikey.txt', 'r') as apifile:
#                     apikey = apifile.readline()
#             except FileNotFoundError:
#                 pass
#         print(apikey)
#         GoogleMapPlotter(center_lat, center_lng, zoom, apikey=apikey)
#         super().__init__(center_lat, center_lng, zoom, apikey = apikey)
#
#         self.map_type = map_type
#         assert(self.map_type in ['roadmap', 'satellite', 'hybrid', 'terrain'])
#
#     def write_map(self,  f):
#         f.write('\t\tvar centerlatlng = new google.maps.LatLng(%f, %f);\n' %
#                 (self.center[0], self.center[1]))
#         f.write('\t\tvar myOptions = {\n')
#         f.write('\t\t\tzoom: %d,\n' % (self.zoom))
#         f.write('\t\t\tcenter: centerlatlng,\n')
#
#         # Change this line to allow different map types
#         f.write('\t\t\tmapTypeId: \'{}\'\n'.format(self.map_type))
#
#         f.write('\t\t};\n')
#         f.write(
#             '\t\tvar map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);\n')
#         f.write('\n')
#
#     def color_scatter(self, lats, lngs, values=None, colormap='coolwarm',
#                       size=None, marker=False, s=None, **kwargs):
#         def rgb2hex(rgb):
#             """ Convert RGBA or RGB to #RRGGBB """
#             rgb = list(rgb[0:3]) # remove alpha if present
#             rgb = [int(c * 255) for c in rgb]
#             hexcolor = '#%02x%02x%02x' % tuple(rgb)
#             return hexcolor
#
#         if values is None:
#             colors = [None for _ in lats]
#         else:
#             cmap = plt.get_cmap(colormap)
#             norm = Normalize(vmin=min(values), vmax=max(values))
#             scalar_map = ScalarMappable(norm=norm, cmap=cmap)
#             colors = [rgb2hex(scalar_map.to_rgba(value)) for value in values]
#         for lat, lon, c in zip(lats, lngs, colors):
#             self.scatter(lats=[lat], lngs=[lon], c=c, size=size, marker=marker,
#                          s=s, **kwargs)


import plotly.graph_objects as go
import plotly
plotly.io.orca.config.executable = '/Users/yaoling/anaconda3/bin/orca/'
plotly.io.orca.config.save()
from plotly.subplots import make_subplots

COLORSCALE = 'jet'

class plot3D():
    def __init__(self, data3D):
        self.X = data3D[:, 0].flatten()
        self.Y = data3D[:, 1].flatten()
        self.Z = data3D[:, 2].flatten()
        self.val = data3D[:, -1].flatten()


    # fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    #                     subplot_titles=("Excursion Probability", "Directional vector"))
    def draw3DVolume(self, isomin = None, isomax = None, opacity = None,
                     surface_count = None, colorbar = dict(len = .5)):
        fig = make_subplots(rows = 1, cols = 1, specs = [[{'type': 'scene'}]],
                            subplot_titles=("Salinity"))

        fig.add_trace(
            go.Volume(
                x=self.X, y=self.Y, z=self.Z,
                value=self.val,
                isomin=isomin,
                isomax=isomax,
                opacity=opacity,
                surface_count=surface_count,
                colorbar=colorbar,
            ),
            row=1, col=1
        )
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.25, y=-1.25, z=1.25)
        )
        fig.update_layout(
            scene={
                'aspectmode': 'manual',
                'aspectratio': dict(x=1, y=1, z=.5),
                'xaxis': {'range': [self.X.min(), self.X.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                'yaxis': {'range': [self.Y.min(), self.Y.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                'zaxis': {'range': [self.Z.min() - 1, self.Z.max()], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            },
            coloraxis=dict(colorscale=COLORSCALE),
            showlegend=False,
            scene_camera=camera
        )
        fig.update_coloraxes(colorscale=COLORSCALE)
        plotly.offline.plot(fig)

    def draw3DSurface(self):
        fig = make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]],
                            subplot_titles=("Salinity"))

        for i in range(len(np.unique(self.Z))):
            ind = (self.Z == np.unique(self.Z)[i])
            fig.add_trace(
                go.Isosurface(x=self.X[ind], y=self.Y[ind], z=self.Z[ind], value=self.val[ind], coloraxis='coloraxis'),
                row=1, col=1
            )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-4, y=-4, z=3.5)
        )
        fig.update_layout(
            scene={
                'aspectmode': 'manual',
                'aspectratio': dict(x=1, y=1, z=1),
                # 'xaxis': {'range': [self.X.min(), self.X.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                # 'yaxis': {'range': [self.Y.min(), self.Y.max() + 90], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 100},
                # 'zaxis': {'range': [self.Z.min() - 1, self.Z.max()], 'rangemode': 'tozero', 'tickmode': "linear", 'tick0': 0, 'dtick': 0.5},
            },
            coloraxis=dict(colorscale=COLORSCALE),
            showlegend=False,
            scene_camera=camera
        )
        fig.update_coloraxes(colorscale=COLORSCALE)
        plotly.offline.plot(fig)
        # fig.write_image(figpath + "sal.png".format(j), width=1980, height=1080)



