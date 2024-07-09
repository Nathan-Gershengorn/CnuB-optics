import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import special
import sys
import time

t0 = time.time()

k = 30000
delta = 1e-2
num = 10000
width = 0.02
num_pts = 10

k_minus = k * np.sqrt(1 - 2 * delta)
k_plus = k * np.sqrt(1 + 2 * delta)

jn = lambda n, z: special.spherical_jn(n, z)
jd = lambda n, z: special.spherical_jn(n, z, True)
hn = lambda n, z: special.spherical_jn(n, z) + 1j * special.spherical_yn(n, z)
hd = lambda n, z: special.spherical_jn(n, z, True) + 1j * special.spherical_yn(n, z, True)

Bl_minus = []
Cl_minus = []
Bl_plus = []
Cl_plus = []
l_list = []
for l in range(num):
    l_list.append(l)
    matrix_minus = [[-hn(l, k), jn(l, k_minus)],
                    [-k * hd(l, k), k_minus * jd(l, k_minus)]]
    vector_minus = [1j ** l * np.sqrt(4 * np.pi * (2 * l + 1)) * jn(l, k),
                    1j ** l * np.sqrt(4 * np.pi * (2 * l + 1)) * k * jd(l, k)]
    coeff_minus = np.linalg.solve(matrix_minus, vector_minus)
    Bl_minus.append(coeff_minus[0])
    Cl_minus.append(coeff_minus[1])
    
    matrix_plus = [[-hn(l, k), jn(l, k_plus)],
                    [-k * hd(l, k), k_plus * jd(l, k_plus)]]
    vector_plus = [1j ** l * np.sqrt(4 * np.pi * (2 * l + 1)) * jn(l, k),
                    1j ** l * np.sqrt(4 * np.pi * (2 * l + 1)) * k * jd(l, k)]
    coeff_plus = np.linalg.solve(matrix_plus, vector_plus)
    Bl_plus.append(coeff_plus[0])
    Cl_plus.append(coeff_plus[1])
print(time.time() - t0)
print("Part 1 done")

plt.plot(l_list, np.abs(Cl_minus)**2, label='Cl_minus')
plt.plot(l_list, np.abs(Cl_plus)**2, label='Cl_plus')
plt.show()

in_minus_terms = lambda r: [np.abs(Cl_minus[l] * jn(l, k_minus * r)) ** 2 for l in range(num)]
in_plus_terms = lambda r: [np.abs(Cl_plus[l] * jn(l, k_plus * r)) ** 2 for l in range(num)]
out_minus_terms = lambda r: [np.abs(1j ** l * np.sqrt(4 * np.pi * (2 * l + 1)) * jn(l, k * r) + Bl_minus[l] * hn(l, k * r)) ** 2 for l in range(num)]
out_plus_terms = lambda r: [np.abs(1j ** l * np.sqrt(4 * np.pi * (2 * l + 1)) * jn(l, k * r) + Bl_plus[l] * hn(l, k * r)) ** 2 for l in range(num)]

in_minus = lambda r: np.sum(in_minus_terms(r)) / (4 * np.pi)
in_plus = lambda r: np.sum(in_plus_terms(r)) / (4 * np.pi)
out_minus = lambda r: np.sum(out_minus_terms(r)) / (4 * np.pi)
out_plus = lambda r: np.sum(out_plus_terms(r)) / (4 * np.pi)

plt.loglog(l_list, in_plus_terms(100), label='psi^2')
plt.legend()
plt.show()



# def asym(r):
#     if r <= 1:
#         asym = in_minus(r) - in_plus(r)
#     else:
#         asym = out_minus(r) - out_plus(r)
#     return asym

# xcoor = np.linspace(1 - width, 1 + width, num_pts)

# # ycoor = []
# # for x in xcoor:
# #     ycoor.append(asym(x))
# #     print("kdsjbfal")


# np.savetxt(f'k{k}_delta{delta}.txt', np.vstack((xcoor, ycoor)).T)