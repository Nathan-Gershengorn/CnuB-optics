#import cnub.tools as tools
import numpy as np


protons = int(input("Number of protons: "))
mass_num = int(input("Mass number: "))
num_density = float(input("Number density of atoms (1/m^3): "))
#should be the right conversion to ev^3 -- check
num_density_ev = num_density * 8E-21

Gf = 1.166E-23

def ior_electron(Z, A, density, mass, k):
    U = Gf / (2 * np.sqrt(2)) * density * (-(3 * Z - A))
    return -(mass * U) / k**2

def ior_mu_tau(Z, A, density, mass, k):
    U = Gf / (2 * np.sqrt(2)) * density * (-(Z - A))
    return -(mass * U) / k**2

print("Electron Neutrinos: IOR is ", ior_electron(protons, mass_num, num_density_ev, .1, 2.68E-4))
print("Muon, Tau Neutrinos: IOR is ", ior_mu_tau(protons, mass_num, num_density_ev, .1, 2.68E-4))