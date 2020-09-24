#%%
print("hello world")


import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import itertools
import random
import scipy.stats
import time
import os
import scipy.spatial.distance as scdist
import datetime


def H(s, t):
    h = np.sqrt(scdist.cdist(s, t, 'sqeuclidean'))
    return h


def gp(xa, xb, yb):

    na = xa.shape[0]
    nb = xb.shape[0]

    mu = np.mean(yb)
    sig = np.std(yb)

    mu_a = mu * np.ones([na, 1])
    mu_b = mu * np.ones([nb, 1])

    sigma = 0.1
    coeff = 10

    # matern kernels used to find the covariance matrix
    Hb = H(xb, xb)
    Sb = sig ** 2 * np.multiply((1 + coeff * Hb), np.exp(-coeff * Hb)) + sigma ** 2 * np.identity(nb) # matern kernel, with 3/2 coefficients

    Hab = H(xa, xb)
    Sab = sig ** 2 * np.multiply((1 + coeff * Hab), np.exp(-coeff * Hab))

    Ha = H(xa, xa)
    Sa = sig ** 2 * np.multiply((1 + coeff * Ha), np.exp(-coeff * Ha))

    # faster inverse using np.linalg.solve() directly
    mu_updated = mu_a + np.dot(Sab, np.linalg.solve(Sb, (yb - mu_b)))
    Sigma_updated = Sa - np.dot(Sab, np.linalg.solve(Sb, Sab.T))

    return mu_updated, Sigma_updated


def EI(mu_updated, sigma_updated, ymax_current):
    unit_norm = scipy.stats.norm()
    z = (mu_updated - ymax_current) / sigma_updated
    EI = np.multiply((mu_updated - ymax_current), unit_norm.cdf(z)) + np.multiply(sigma_updated, unit_norm.pdf(z))
    return EI


def init_path_generator(nt, ns, steps):
    Gd_temp = np.zeros([nt, ns])
    Gd = np.zeros([steps, nt * ns])
    row_init = np.random.randint(0, nt, 1)
    col_init = np.random.randint(0, ns, 1)
    row_old = row_init
    col_old = col_init

    for i in range(steps):
        row_new, col_new = update_coordinate(row_old, col_old)
        row_new = regulate_dim(row_new, nt)
        col_new = regulate_dim(col_new, ns)
        Gd_temp[row_new, col_new] = True
        Gd[i, :] = Gd_temp.flatten()
        row_old, col_old = row_new, col_new

    return Gd


def update_coordinate(row, col):
    row_ind = np.random.randint(-1, 2, 1)
    col_ind = np.random.randint(-1, 2, 1)
    row_new = row + row_ind
    col_new = col + col_ind
    return row_new, col_new


def regulate_dim(row, rowlim):
    if row < 0:
        row = row + 1
    elif row >= rowlim:
        row = rowlim - 1
    return row


def pos_candidates(row, rowlim, col, collim):

    pos_row = []
    pos_col = []

    for i in range(3):
        row_temp = row + i - 1
        if (row_temp >= 0) and (row_temp < rowlim):
            pos_row.append(row_temp)

    for j in range(3):
        col_temp = col + j - 1
        if (col_temp >= 0) and (col_temp < collim):
            pos_col.append(col_temp)

    return np.array(pos_row).astype(int), np.array(pos_col).astype(int)


def create_dir(filename):
    print("It exists or not:", os.path.exists(filename))
    # print(a)
    if not (os.path.exists(filename)):
        try:
            os.mkdir(filename)
            os.mkdir(filename + "/path")
            os.mkdir(filename + "/mean")
            os.mkdir(filename + "/std")
            os.mkdir(filename + "/EI")
            print("Successfully created a new directory")
        except:
            print("Failed to create new directory")
    else:
        print("Path is already existed")


'''
Some plot functions, do not change
'''


def display_path(Gd, nt, ns):
    row, col = np.nonzero(Gd.reshape(nt, ns))
    plt.plot(col, row, 'k*', markersize = 10)


# ==================================== Main Function ========================================================


# main function workflow
# setup grids


tv = np.arange(0.02, 1.02, 0.02).reshape(-1, 1)
nt = tv.shape[0]

sv = np.arange(0.02, 1.02, 0.02).reshape(-1, 1)
ns = sv.shape[0]

# generate init paths for preliminary sampling
steps_init = 10
Gd = init_path_generator(nt, ns, steps_init) # generate the initial path for evaluation
# plt.imshow(Gd[-1].reshape(nt, ns))  # show the latest path
# plt.title("The initial path for evaluation")
# plt.show()

# define the mesh grid for function evaluations
t = tv * np.ones([1, ns])
s = np.ones([nt, 1]) * sv.T
n = ns * nt


# define true hiden function


def f(s, t):
    y = 1.5 * np.cos(15 * s) + 0.5 * np.sin(10 * t) - np.cos(15 * np.multiply(s, t))
    return y

# evaluate the true function


y = f(s, t)


fig, ax = plt.subplots()
ax.imshow(y, extent = (0, 1, 0, 1))
ax.set(title = 'true function', xlabel = 's', ylabel = 't')
# postfix_time_label = str(datetime.datetime.now().month) + "_" + str(datetime.datetime.now().day) + "_" \
#                      + str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute)
postfix_time_label = '_'
filename = os.getcwd() + "/fig_" + postfix_time_label
create_dir(filename)


fig_path = os.getcwd() + "/fig_" + postfix_time_label + "/true.png"
plt.savefig(fig_path)
# plt.show()
# plt.clf()
# plt.close()

# vectorise the input
t_vec = t.reshape(n, 1)
s_vec = s.reshape(n, 1)
y_vec = y.reshape(n, 1)

# BayesOpt
step_no = 0     # iteration counter
steps_total = 100   # maximum iteration
EI_max = np.zeros([steps_total, 1])
EI_mean = np.zeros([steps_total, 1])
Gd_total = np.zeros([steps_total + steps_init, n])
Gd_total[0:steps_init, :] = Gd

ind_old = np.nonzero(Gd[-1])
row_old, col_old = np.unravel_index(ind_old, (nt, ns))
# ind_next = []
row_old = row_old.squeeze()
col_old = col_old.squeeze()


# =======# =======# =======# =======# =======# =======# =======# =======# =======# =======# =======# =======# =======#


while step_no < steps_total:

    time_start = time.time()
    Gd_temp = np.zeros([1, n])

    # design new evaluation points
    if step_no == 0:
        # sample random locations initially
        ind_sampled = np.array(np.nonzero(Gd[-1])).squeeze()
        t_sampled = t_vec[ind_sampled]
        s_sampled = s_vec[ind_sampled]
        y_sampled = y_vec[ind_sampled]

    else:
        # x_neighbours, y_neighbours = list_all_neighbours(x_old.squeeze()[-1], ns, y_old.squeeze()[-1], nt)
        row_next, col_next = pos_candidates(row_old[-1], nt, col_old[-1], ns)

        ind_temp = []
        for i in range(row_next.shape[0]):
            for j in range(col_next.shape[0]):
                # ind_next.append(np.ravel_multi_index((y_next[j], x_next[i]), (nt, ns)))
                ind_temp.append(np.ravel_multi_index((row_next[i], col_next[j]), (nt, ns)))

        ind_temp = np.array(ind_temp)
        ind_temp = ind_temp.flatten().astype(int)

        ei_next = ei[ind_temp]
        val_EI = np.sort(-ei_next.squeeze())
        ind_EI = np.argsort(-ei_next.squeeze())
        ind_ei_max = ind_temp[ind_EI[0]]

        row_new, col_new = np.unravel_index(ind_ei_max, (nt, ns))
        ind_new = np.ravel_multi_index((row_new, col_new), (nt, ns))    # can be changed to use ind_ei_max
        ind_sampled = np.append(ind_sampled, ind_new)

        # x_new, y_new = update_coordinate(x_new, y_new)
        row_old = np.append(row_old, row_new)
        col_old = np.append(col_old, col_new)
        t_sampled = np.append(t_sampled, t_vec[ind_ei_max, np.newaxis], axis = 0)
        s_sampled = np.append(s_sampled, s_vec[ind_ei_max, np.newaxis], axis = 0)
        y_sampled = np.append(y_sampled, y_vec[ind_ei_max, np.newaxis], axis = 0)

    Gd_temp[:, ind_sampled] = True
    Gd_total[step_no - 1, :] = Gd_temp

    fig, ax = plt.subplots()
    # plt.imshow(Gd_total[step_no - 1, :].reshape(nt, ns))
    plt.spy(Gd_total[step_no - 1, :].reshape(nt, ns))
    ax.set(title = 'Updated path for the {} time run'.format(step_no))
    fig_path = os.getcwd() + "/fig_" + postfix_time_label + "/path/" + "{:03d}".format(step_no) + ".png"
    plt.savefig(fig_path)
    # plt.show()

    # apply GP
    mu_updated, Cov_updated = gp(np.concatenate((s_vec, t_vec), axis = 1),
                                np.concatenate((s_sampled, t_sampled), axis = 1),
                                y_sampled)
    sigma = np.sqrt(np.diagonal(Cov_updated)).reshape(-1, 1)  # take out the standard deviation
    # by picking out the diagonal variance terms
    fig, ax = plt.subplots()
    plt.imshow(np.array(mu_updated).reshape(nt, ns))
    ax.set(title = 'Predication of mean for the {} time run'.format(step_no))
    display_path(Gd_total[step_no - 1, :], nt, ns)
    fig_path = os.getcwd() + "/fig_" + postfix_time_label + "/mean/" + "{:03d}".format(step_no) + ".png"
    plt.savefig(fig_path)
    # plt.show()

    fig, ax = plt.subplots()
    plt.imshow(np.array(sigma).reshape(nt, ns))
    display_path(Gd_total[step_no - 1, :], nt, ns)
    ax.set(title='Predication of std dev for the {} time run'.format(step_no))
    fig_path = os.getcwd() + "/fig_" + postfix_time_label + "/std/" + "{:03d}".format(step_no) + ".png"
    plt.savefig(fig_path)
    # plt.show()

    # update the current maximum value
    # f_star = max(y_sampled.squeeze())
    f_star = max(y_sampled.squeeze())  # find the maximum current evaluated samples so far
    f_star_vec = f_star * np.ones([n, 1])  # vectorise the f_star so to feed in the EI computation

    # apply EI to find where to sample next and display the expected improvement
    ei = EI(mu_updated, sigma, f_star_vec)
    fig, ax = plt.subplots()
    plt.imshow(ei.reshape(nt, ns))
    display_path(Gd_total[step_no - 1, :], nt, ns)
    ax.set(title = 'Expected improvement for the {} time run'.format(step_no))
    fig_path = os.getcwd() + "/fig_" + postfix_time_label + "/EI/" + "{:03d}".format(step_no) + ".png"
    plt.savefig(fig_path)
    # plt.show()

    EI_max[step_no] = max(ei[ind_sampled])
    EI_mean[step_no] = np.mean(ei[ind_sampled])
    print("Iteration {} time, max_EI is {}, mean_EI is {}".format(step_no, max(ei), np.mean(ei)))

    step_no = step_no + 1
    # print(iter_no)
    # sleep()
    time_end = time.time()
    print(f"runtime is {time_end - time_start} seconds")

## %% test another method of making animation
# import matplotlib.pyplot as plt
# import os
import imageio

# png_dir = os.getcwd() + "/fig/mean/"
# fig_path = os.getcwd() + "/fig_" + postfix_time_label + "/EI/" + "{:03d}".format(step_no) + ".png"
names = ["mean", "std", "EI"]

for name in names:

    png_dir = os.getcwd() + "/fig_" + postfix_time_label + "/" + name + "/"
    name = os.getcwd() + "/fig_" + postfix_time_label + "/" + name + postfix_time_label + ".gif"
    image_file_names = []
    images = []

    # print(os.listdir(png_dir))
    for file_name in os.listdir(png_dir):
        if file_name.endswith('.png'):
            image_file_names.append(file_name)

    # sorted_files = sorted(image_file_names, key=lambda y: int(y.split('_')[0]))
    files =  os.listdir(png_dir)
    # print(files)

    sorted_files = sorted(image_file_names)
    # print(sorted_files)

    frame_length = 0.2 # seconds between frames
    end_pause = 4 # seconds to stay on last frame
    # loop through files, join them to image array, and write to GIF called 'wind_turbine_dist.gif'
    # for ii in range(0,len(sorted_files)):
    for ii in range(0,len(sorted_files)):
        file_path = os.path.join(png_dir, sorted_files[ii])
        if ii==len(sorted_files)-1:
            for jj in range(0,int(end_pause/frame_length)):
                images.append(imageio.imread(file_path))
        else:
            images.append(imageio.imread(file_path))
    # the duration is the time spent on each image (1/duration is frame rate)
    imageio.mimsave(name, images,'GIF',duration=frame_length)










