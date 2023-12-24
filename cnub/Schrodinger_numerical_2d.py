import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0   # Planck's constant / 2π
N = 1000     # Number of grid points

# create hamiltonian matrix
def Hamiltonian(m, N):
    H = np.zeros((N,N))

    for i in range(0,N):
        H[i,i] = 2  # Changed from -2 to 2
        if i != 0:
            H[i,i-1] = -1
        if i != N-1:
            H[i,i+1] = -1

    return H * -hbar**2 / (2*m)

def normalize(psi):
    norm = np.sqrt(np.sum(np.abs(psi)**2, axis=1))  # Compute the norm of each row
    psi_norm = psi / norm[:, np.newaxis]  # Normalize each row
    return psi_norm

# Solve the Schrödinger equation for each dimension
E_x, psi_x = np.linalg.eigh(Hamiltonian(1,N))
E_y, psi_y = np.linalg.eigh(Hamiltonian(1,N))

# Normalize the wave functions
psi_x_norm = normalize(psi_x)
psi_y_norm = normalize(psi_y)

# Create a 2D grid for the x, y coordinates
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# Calculate the 2D wave function for the nth energy level
n = 2  # Change this to plot a different energy level
psi_2D = psi_x_norm[n-1, :, np.newaxis] * psi_y_norm[n-1, np.newaxis, :]

# Plot the 2D wave function
plt.imshow(psi_2D, extent=[0, 1, 0, 1], origin='lower')
plt.colorbar(label='Wave function')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'2D wave function for n = {n}')
plt.show()