import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm, genlaguerre
from scipy.constants import physical_constants

# Define the quantum number
n = 6

# Define the theta, phi, and r arrays
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2*np.pi, 50)
a0 = physical_constants['Bohr radius'][0]  # Bohr radius
r = np.linspace(0, 5*a0, 50)  # r values in the range of 0 to 5 times the Bohr radius

# Create a 3D grid for theta, phi, and r
theta, phi, r = np.meshgrid(theta, phi, r)

# Define the radial function
def radial_func(r, n, l):
    # Constants
    Z = 1  # Atomic number for hydrogen

    # Generalized Laguerre polynomial
    L = genlaguerre(n-l-1, 2*l+1)

    # Radial function
    R = np.sqrt((2*Z/n/a0)**3 * np.math.factorial(n-l-1) / (2*n*np.math.factorial(n+l))) * np.exp(-Z*r/n/a0) * (2*Z*r/n/a0)**l * L(2*Z*r/n/a0)

    return R

# Calculate the total wavefunction for all possible l and m values
wavefunction = np.zeros_like(r, dtype=complex)
for l in range(n):
    for m in range(-l, l+1):
        # Calculate the spherical harmonics
        Y_lm = sph_harm(m, l, phi, theta)

        # Apply the radial function
        R = radial_func(r, n, l)

        # Add to the total wavefunction
        wavefunction += R * Y_lm

# Normalize the wavefunction
wavefunction /= np.sqrt(np.sum(np.abs(wavefunction)**2))

# Calculate the probability density
density = np.abs(wavefunction)**2

# Convert to Cartesian coordinates for plotting
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# Create a mask for the slice
mask = (phi < 5*np.pi/4) | (phi > 7*np.pi/4)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x[mask], y[mask], z[mask], c=density[mask], alpha=0.6, s=1)

# Add a colorbar
fig.colorbar(sc, ax=ax, label='Probability Density')

# Show the plot
plt.show()