import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
from scipy import constants
from scipy.special import sph_harm, genlaguerre
from scipy.sparse import diags
from math import sqrt, factorial, pi
from scipy.optimize import curve_fit

hbar = 1   # unitless
m_H = 5e5 #ev
R_H = 2.68E-4 #1/ev

m = .1
R = 3E4
U = .01
distance = (R)*5
earth_cutoff = R / distance
e = .303 #unitless
epsilon_0 = 1 #unitless

# define derivative matrices

def first_deriv(N, r):
    diags0 = np.zeros(N)

    ddr = np.diag(diags0)

    for i in range(0, ddr.shape[0]):
        if i != 0:
            ddr[i,i-1] = -1
        if i != ddr.shape[0]-1:
            ddr[i,i+1] = 1
    
    delta_x = r[1] - r[0]

    return ddr / (2 * delta_x)


# Large numbers at the first few spots because this is how we remove the condition that
# the wavefunction must be zero at the origin
def second_deriv(N, r):
    diags2 = np.full(N, -2.0)

    d2dr = np.diag(diags2)

    for i in range(0, d2dr.shape[0]):
        if i == 0:
            d2dr[i,i] = -1E200
            d2dr[i,i+1] = 1E200
        if i > 0:
            d2dr[i,i-1] = 1.0
        if i != d2dr.shape[0]-1 and i != 0:
            d2dr[i,i+1] = 1.0

    delta_x =  r[1] - r[0]
    return d2dr / (delta_x**2)

# def second_deriv(N, r):
#     diags2 = np.full(N, -2)

#     d2dr = np.diag(diags2)

#     for i in range(0, d2dr.shape[0]):
#         if i != 0:
#             d2dr[i,i-1] = 1
#         if i != d2dr.shape[0]-1:
#             d2dr[i,i+1] = 1

#     delta_x =  r[1] - r[0]
#     return d2dr / (delta_x**2)

def normalize(psi):

    norm = np.sqrt(np.sum(np.abs(psi)**2))  # Compute the norm of the vector
    psi_norm = psi / norm  # Normalize the vector
    return psi_norm

# Potentials and Hamiltonian from equation 25


def potential_nu_25(N, r, l):

    H = np.zeros((N,N))
    for i in range(0, H.shape[0]):
        #try just a step function
        
        # inside the earth:
        
        V_eff_in = (l * (l + 1))/(r[i]**2) + 2 * m * U

        # outside the earth:
        V_eff_out = (l * (l + 1))/(r[i]**2)

        if r[i] < R:
            H[i,i] = V_eff_in
        else:
            H[i,i] = V_eff_out

    return H

def potential_antinu_25(N, r, l):

    H = np.zeros((N,N))
    for i in range(0, H.shape[0]):
        #try just a step function
        
        # inside the earth:
        V_eff_in_neg = (l * (l + 1))/(r[i]**2) - 2 * m * U

        # outside the earth:
        V_eff_out_neg = (l * (l + 1))/(r[i]**2)
        #print("In, Out neg: ", V_eff_in_neg, V_eff_out_neg)

        if r[i] < R:
            H[i,i] = V_eff_in_neg
        else:
            H[i,i] = V_eff_out_neg
    return H

def potential_H_atom(N, r, l):
    H = np.zeros((N,N))
    for i in range(0, H.shape[0]):
        H[i,i] = -e**2 / (4 * constants.pi * r[i] * epsilon_0) + (hbar**2 * l * (l + 1)) / (2 * m * r[i]**2)
    return H


# create the equation to be solved: Hψ = (k^2)ψ
def Hamiltonian_nu_25(N, r, l):
    H = - second_deriv(N, r) + potential_nu_25(N,r, l)
    return H

def Hamiltonian_antinu_25(N, r, l):
    H = - second_deriv(N, r) + potential_antinu_25(N,r, l)
    return H

def Hamiltonian_H_atom(N, r, l):
    H = - (hbar**2 / (2 * m)) * second_deriv(N, r) + potential_H_atom(N,r, l)
    return H

