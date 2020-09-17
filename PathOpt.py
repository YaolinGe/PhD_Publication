#%%
print("hello world")

import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import random
import scipy.stats
import time
import os
import scipy.spatial.distance as scdist

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

    Hb = H(xb, xb)
    Sb = sig ** 2 * np.multiply((1 + coeff * Hb), np.exp(-coeff * Hb)) + sigma ** 2 * np.identity(nb)

    Hab = H(xa, xb)
    Sab = sig ** 2 * np.multiply((1 + coeff * Hab), np.exp(-coeff * Hab))

    Ha = H(xa, xa)
    Sa = sig ** 2 * np.multiply((1 + coeff * Ha), np.exp(-coeff * Ha))

    # cholesky decomposition
    mu_updated = mu_a + np.dot(Sab, np.linalg.solve(Sb, (yb - mu_b)))
    Sigma_updated = Sa - np.dot(Sab, np.linalg.solve(Sb, Sab.T))

    return mu_updated, Sigma_updated

def EI(mu_updated, sigma_updated, ymax_current):
    unit_norm = scipy.stats.norm()
    z = (mu_updated - ymax_current) / sigma_updated
    EI = np.multiply((mu_updated - ymax_current), unit_norm.cdf(z)) + np.multiply(sigma_updated, unit_norm.pdf(z))
    return EI

def path_generator(nt, ns, steps):
    Gd_temp = np.zeros([nt, ns])
    Gd = np.zeros([steps, nt * ns])
    x_init = np.random.randint(0, ns, 1)
    y_init = np.random.randint(0, nt, 1)
    x_old = x_init
    y_old = y_init

    for i in range(steps):
        x_new, y_new = update_coordinate(x_old, y_old)
        x_new = regulate_ind(x_new, ns)
        y_new = regulate_ind(y_new, nt)
        Gd_temp[x_new, y_new] = True
        Gd[i, :] = Gd_temp.flatten()
        x_old, y_old = x_new, y_new

    return Gd

def update_coordinate(x, y):
    x_ind = np.random.randint(-1, 2, 1)
    y_ind = np.random.randint(-1, 2, 1)
    x_new = x + x_ind
    y_new = y + y_ind
    return x_new, y_new

def list_all_neighbours(x, ns, y, nt):
    x_next = np.zeros([3, 3])
    y_next = np.zeros([3, 3])

    for i in range(3):
        for j in range(3):
            # x_next[i, j] = x + i - 1
            # y_next[i, j] = y + j - 1
            x_next[i, j] = regulate_ind(x + i - 1, ns)
            y_next[i, j] = regulate_ind(y + j - 1, nt)

    return x_next.astype(int), y_next.astype(int)

# def list_all_neighbours(x, ns, y, nt):
#     x = np.arange(-1, 2)
#     y = np.arange(-1, 2)
#     neighbours = list(itertools.product(x, y))
#     print(neighbours)
#     neighbours = np.array(neighbours)
#     print(neighbours)
#     neighbours = neighbours.squeeze() + [x, y]
#     # neighbours[neighbours > ns] =
#
#     return neighbours

# x = 3
# y = 5
# xx, yy = list_all_neighbours(x, 10, y, 10)
# print(xx)
# print(yy)

def regulate_ind(x, n):
    # if x < 0:
    #     x = x + 3
    # elif x >= n:
    #     x = n - 3
    return x

# main function workflow
# setup grids
tv = np.arange(0.02, 1.02, 0.02).reshape(-1, 1)
nt = tv.shape[0]

sv = np.arange(0.02, 1.02, 0.02).reshape(-1, 1)
ns = sv.shape[0]

steps_init = 10
Gd = path_generator(nt, ns, steps_init) # generate the initial path for evaluation
plt.imshow(Gd[-1].reshape(nt, ns))
plt.title("The initial path for evaluation")
plt.show()

# define the mesh grid for function evaluations
t = tv * np.ones([1, ns])
s = np.ones([nt, 1]) * sv.T
n = ns * nt

# define true hiden function
def f(s, t):
    y = 1.5 * np.cos(15 * s) + 0.5 * np.sin(10 * t) - np.cos(15 * np.multiply(s, t))
    return y

y = f(s, t)

fig, ax = plt.subplots()
ax.imshow(y, extent = [0, 1, 0, 1])
ax.set(title = 'true function', xlabel = 't', ylabel = 's')
fig_path = os.getcwd() + "/fig/true.png"
plt.savefig(fig_path)
plt.show()
plt.clf()
plt.close()

# vectorise the input
t_vec = t.reshape(n, 1)
s_vec = s.reshape(n, 1)
y_vec = y.reshape(n, 1)

# BayesOpt
step_no = 0     # iteration counter
steps_total = 10   # maximum iteration
EI_max = np.zeros([steps_total, 1])
EI_mean = np.zeros([steps_total, 1])
Gd_total = np.zeros([steps_total + steps_init, n])
Gd_total[0:steps_init, :] = Gd
ind_old = np.nonzero(Gd[-1])
x_old, y_old = np.unravel_index(ind_old, (ns, nt))
ind_neighbours = np.zeros([3, 3])

while(step_no < steps_total):

    time_start = time.time()
    Gd_temp = np.zeros([1, n])

    # design new evaluation points
    if step_no == 0:
        # sample random locations initially
        ind_sampled = np.nonzero(Gd[-1].reshape(n, 1).squeeze())
        t_sampled = t_vec[ind_sampled]
        s_sampled = s_vec[ind_sampled]
        y_sampled = y_vec[ind_sampled]

    else:
        x_neighbours, y_neighbours = list_all_neighbours(x_old.squeeze()[-1], ns, y_old.squeeze()[-1], nt)
        for i in range(x_neighbours.shape[0]):
            for j in range(y_neighbours.shape[0]):
                ind_neighbours[i, j] = np.ravel_multi_index((x_neighbours[i, j], y_neighbours[i, j]), (nt, ns))

        ind_next = ind_neighbours.flatten().astype(int)
        ei_neighbours = ei[ind_next]

        val_EI = np.sort(-ei_neighbours.squeeze())
        ind_EI = np.argsort(-ei_neighbours.squeeze())

        ind_sampled = ind_EI[0]
        x_new, y_new = np.unravel_index(ind_sampled, (nt, ns))
        x_new, y_new = update_coordinate(x_new, y_new)
        x_old = np.append(x_old, x_new)
        y_old = np.append(y_old, y_new)
        t_sampled = np.append(t_sampled, t_vec[ind_sampled, np.newaxis], axis = 0)
        s_sampled = np.append(s_sampled, s_vec[ind_sampled, np.newaxis], axis = 0)
        y_sampled = np.append(y_sampled, y_vec[ind_sampled, np.newaxis], axis = 0)

    Gd_temp[:, ind_sampled] = True
    Gd_total[step_no, :] = Gd_temp

    fig, ax = plt.subplots()
    plt.imshow(Gd_total[step_no, :].reshape(nt, ns))
    ax.set(title = 'Updated path for the {} time run'.format(step_no + 1))
    fig_path = os.getcwd() + "/fig/path_" + str(step_no + 1) + ".png"
    plt.savefig(fig_path)
    plt.show()

    # apply GP
    mu_updated, Cov_updated = gp(np.concatenate((s_vec, t_vec), axis = 1),
                                np.concatenate((s_sampled, t_sampled), axis = 1),
                                y_sampled)
    sigma = np.sqrt(np.diagonal(Cov_updated)).reshape(-1, 1)

    fig, ax = plt.subplots()
    plt.imshow(np.array(mu_updated).reshape(nt, ns))
    ax.set(title = 'Predication of mean for the {} time run'.format(step_no + 1))
    fig_path = os.getcwd() + "/fig/mean_" + str(step_no + 1) + ".png"
    plt.savefig(fig_path)
    plt.show()

    fig, ax = plt.subplots()
    plt.imshow(np.array(sigma).reshape(nt, ns))
    ax.set(title='Predication of std dev for the {} time run'.format(step_no + 1))
    fig_path = os.getcwd() + "/fig/std_" + str(step_no + 1) + ".png"
    plt.savefig(fig_path)
    plt.show()

    # update the current maximum value
    # f_star = max(y_sampled.squeeze())
    f_star = max(y_sampled.squeeze())
    f_star_vec = f_star * np.ones([n, 1])

    # apply EI to find where to sample next and display the expected improvement
    ei = EI(mu_updated, sigma, f_star_vec)
    fig, ax = plt.subplots()
    plt.imshow(ei.reshape(nt, ns))
    ax.set(title = 'Expected improvement for the {} time run'.format(step_no + 1))
    fig_path = os.getcwd() + "/fig/EI_" + str(step_no + 1) + ".png"
    plt.savefig(fig_path)
    plt.show()

    EI_max[step_no] = max(ei)
    EI_mean[step_no] = np.mean(ei)
    print("Iteration {} time, max_EI is {}, mean_EI is {}".format(step_no + 1, max(ei), np.mean(ei)))

    step_no = step_no + 1
    # print(iter_no)
    # sleep()
    time_end = time.time()
    print(f"runtime is {time_end - time_start} seconds")







#%%

def insert_new_waypoint(pos, grid_size, path):
    row = pos[0]
    col = pos[1]

    n_row = grid_size[0]    # number of elements in the row
    n_col = grid_size[1]    # number of elements in the column
    new_path = np.zeros([1, n_row * n_col])
    print(new_path)
    print(new_path.shape)

    new_ind = n_row * row + col
    print(new_ind)

    new_path[new_ind] = True

    path = np.vstack(path, new_path)
    return path


pos = [3, 3]
# print(pos[0])
# print(pos[0])
nt = 5
ns = 5
grid_size = [nt, ns]
p = np.zeros([1, nt * ns])
# print(p)
# print(p.shape)

p_new = insert_new_waypoint(pos, grid_size, p)






