import numpy as np

Gf = 1.166E-23

protons = int(input("Number of protons: "))
mass_num = int(input("Mass number: "))
density = float(input("Density: (g/cm^3) "))

#get density as g/cm^3 because that's easily accessible and converts it to number density
num_density = density / mass_num * (100)**3 * 6.022E23

#convert to ev
num_density_ev = num_density * 8E-21

def ior_electron(Z, A, density, mass, k):
    U = Gf / (2 * np.sqrt(2)) * density * (-(3 * Z - A))
    return -(mass * U) / k**2

def ior_mu_tau(Z, A, density, mass, k):
    U = Gf / (2 * np.sqrt(2)) * density * (-(Z - A))
    return -(mass * U) / k**2

print("Electron Neutrinos: IOR is ", ior_electron(protons, mass_num, num_density_ev, .1, 2.68E-4))
print("Muon, Tau Neutrinos: IOR is ", ior_mu_tau(protons, mass_num, num_density_ev, .1, 2.68E-4))