import numpy as np
from numba import jit

import timeit

# @jit(nopython=True)
def f(a1, a2):
    return a1 * a2
@jit
def f2(a1, a2):
    return a1 * a2

N = 1e5
a1 = np.arange(N)
a2 = np.arange(N)

timeit.timeit("f2(a1, a2)")

#%%


