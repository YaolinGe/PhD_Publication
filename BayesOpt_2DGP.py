print("hello world")

import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import random
import scipy.stats
import time
import os

def H(s, t):
    ns = s.shape[0]
    nt = t.shape[0]

    Hs = [np.abs(i - j) for (i, j) in itertools.product(s[:, 0], t[:, 0])]
    Hs = np.array(Hs).reshape(ns, nt)

    Ht = [np.abs(i - j) for (i, j) in itertools.product(s[:, 1], t[:, 1])]
    Ht = np.array(Ht).reshape(ns, nt)

    return np.sqrt(Hs ** 2 + Ht ** 2)

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

    # cholesky decomposition,
    mu_updated = mu_a + np.dot(Sab, np.dot(np.linalg.inv(Sb), (yb - mu_b)))
    Sigma_updated = Sa - np.dot(Sab, np.dot(np.linalg.inv(Sb), Sab.T))

    return mu_updated, Sigma_updated


def EI(mu_updated, sigma_updated, ymax_current):
    unit_norm = scipy.stats.norm()
    z = (mu_updated - ymax_current) / sigma_updated
    EI = np.multiply((mu_updated - ymax_current), unit_norm.cdf(z)) + np.multiply(sigma_updated, unit_norm.pdf(z))
    # EI = (mu - f_star) * unit_norm.cdf((mu - f_star) / sigma) + sigma * unit_norm.pdf((mu - f_star)/sigma)
    return EI

# main function workflow
# setup grids
tv = np.arange(0.02, 1.02, 0.02).reshape(-1, 1)
nt = tv.shape[0]

sv = np.arange(0.02, 1.02, 0.02).reshape(-1, 1)
ns = sv.shape[0]

# define the mesh grid for function evaluations
t = tv * np.ones([1, ns])
s = np.ones([nt, 1]) * sv.T
n = ns * nt

# define true hiden function
def f(s, t):
    # ns = s.shape[0]
    # nt = t.shape[0]
    # y = [1.5 * np.cos(15 * j) + 0.5 * np.sin(10 * i) - np.cos(15 * np.multiply(j, i))
    #      for (i, j) in itertools.product(t, s)]
    y = 1.5 * np.cos(15 * s) + 0.5 * np.sin(10 * t) - np.cos(15 * np.multiply(s, t))
    # y = np.array(y).reshape(ns, nt)
    return y

# display the true function
# y = f(sv, tv)
y = f(s, t)

 fig, ax = plt.subplots()
ax.imshow(y, extent = [0, 1, 0, 1])
ax.set(title = 'true function', xlabel = 't', ylabel = 's')
fig_path = os.getcwd() + "/fig/true.pdf"
plt.savefig(fig_path)
plt.show()
plt.clf()
plt.close()

# vectorise the input
t_vec = t.reshape(n, 1)
s_vec = s.reshape(n, 1)
y_vec = y.reshape(n, 1)

# BayesOpt
iter_no = 0     # iteration counter
N_sample = 20   # sample size for each batch
MAX_iter = 20   # maximum iteration
EI_max = np.zeros([MAX_iter, 1])
EI_mean = np.zeros([MAX_iter, 1])

while(iter_no < MAX_iter):

    time_start = time.time()
    # design new evaluation points
    if iter_no == 0:
        # sample random locations initially

        ind_sampled = random.sample(range(n), N_sample)
        t_sampled = t_vec[ind_sampled]
        s_sampled = s_vec[ind_sampled]
        y_sampled = y_vec[ind_sampled]

    else:
        val_EI = np.sort(-ei.squeeze())
        ind_EI = np.argsort(-ei.squeeze())

        ind_sampled = ind_EI[:N_sample]
        t_sampled = np.append(t_sampled, t_vec[ind_sampled], axis = 0)
        s_sampled = np.append(s_sampled, s_vec[ind_sampled], axis = 0)
        y_sampled = np.append(y_sampled, y_vec[ind_sampled], axis = 0)

    # apply GP
    mu_updated, Cov_updated = gp(np.concatenate((s_vec, t_vec), axis = 1),
                                np.concatenate((s_sampled, t_sampled), axis = 1),
                                y_sampled)
    sigma = np.sqrt(np.diagonal(Cov_updated)).reshape(-1, 1)

    fig, ax = plt.subplots()
    plt.imshow(np.array(mu_updated).reshape(nt, ns))
    ax.set(title = 'Predication of mean for the {} time run'.format(iter_no + 1))
    fig_path = os.getcwd() + "/fig/mean_" + str(iter_no + 1) + ".pdf"
    plt.savefig(fig_path)
    plt.show()

    fig, ax = plt.subplots()
    plt.imshow(np.array(sigma).reshape(nt, ns))
    ax.set(title='Predication of std dev for the {} time run'.format(iter_no + 1))
    fig_path = os.getcwd() + "/fig/std_" + str(iter_no + 1) + ".pdf"
    plt.savefig(fig_path)
    plt.show()

    # update the current maximum value
    f_star = max(y_sampled.squeeze())
    f_star_vec = f_star * np.ones([n, 1])

    # apply EI to find where to sample next and display the expected improvement
    ei = EI(mu_updated, sigma, f_star_vec)
    fig, ax = plt.subplots()
    plt.imshow(ei.reshape(nt, ns))
    ax.set(title = 'Expected improvement for the {} time run'.format(iter_no + 1))
    fig_path = os.getcwd() + "/fig/EI_" + str(iter_no + 1) + ".pdf"
    plt.savefig(fig_path)
    plt.show()

    EI_max[iter_no] = max(ei)
    EI_mean[iter_no] = np.mean(ei)
    print("Iteration {} time, max_EI is {}, mean_EI is {}".format(iter_no + 1, max(ei), np.mean(ei)))

    iter_no = iter_no + 1
    # print(iter_no)
    # sleep()
    time_end = time.time()
    print(f"runtime is {time_end - time_start} seconds")

fig, ax = plt.subplots()
plt.plot(np.arange(0, iter_no), EI_max)
plt.plot(np.arange(0, iter_no), EI_mean)
plt.legend(['EImax', 'EImean'])
# ax.set(loc = 'lower')
plt.show()



