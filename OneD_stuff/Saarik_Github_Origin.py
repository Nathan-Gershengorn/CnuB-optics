import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import special
import sys
import time

t0 = time.time()

jn = lambda n, z: special.spherical_jn(n, z)
jd = lambda n, z: special.spherical_jn(n, z, True)
hn = lambda n, z: special.spherical_jn(n, z) - 1j * special.spherical_yn(n, z)
hd = lambda n, z: special.spherical_jn(n, z, True) - 1j * special.spherical_yn(n, z, True)

def asym(params):
    k, delta = params
    k_minus = k * np.sqrt(1 - 2 * delta)
    k_plus = k * np.sqrt(1 + 2 * delta)

    l = 0
    total = 0
    while True:
        alt = 1j ** (l % 4)
        Cl_minus = -1j * alt * np.sqrt(4 * np.pi * (2 * l + 1)) / (k ** 2 * (hd(l, k) * jn(l, k_minus) - np.sqrt(1 - 2 * delta) * hn(l, k) * jd(l, k_minus)))
        Cl_plus = -1j * alt * np.sqrt(4 * np.pi * (2 * l + 1)) / (k ** 2 * (hd(l, k) * jn(l, k_plus) - np.sqrt(1 + 2 * delta) * hn(l, k) * jd(l, k_plus)))
        term = np.abs(Cl_minus * jn(l, k_minus)) ** 2 - np.abs(Cl_plus * jn(l, k_plus)) ** 2
        total += term
        if l > k and np.abs(term / total) <= 1e-8:
            break
        l += 1
    print(params, l, time.time() - t0)
    sys.stdout.flush()
    return total / (4 * np.pi)

deltas = np.logspace(-3., -2., 50)
ks = np.logspace(3.5, 5., 50)
inputs = [[k, delta] for k in ks for delta in deltas]
if __name__ == '__main__':
    pool = mp.Pool(20)
    asyms = pool.map_async(asym, inputs).get()
    pool.close()
    pool.join()

np.savetxt('origin.txt', np.block([np.array(inputs), np.array(asyms)[:,None]]))