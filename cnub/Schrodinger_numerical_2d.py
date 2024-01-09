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

# Analytical solution
def normalize_analytical(n, x, N):
    psi = np.sqrt(2) * np.sin(n*np.pi*x*N) # times N to make it work on the same domain
    dx = x[1] - x[0]  # calculate the step size
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)  # multiply the sum by the step size
    psi_norm_ana = psi / norm
    return psi_norm_ana

# Solve the Schrödinger equation for each dimension
E_x, psi_x = np.linalg.eigh(Hamiltonian(1,N))
E_y, psi_y = np.linalg.eigh(Hamiltonian(1,N))

# Normalize the wave functions
psi_x_norm = normalize(psi_x)
psi_y_norm = normalize(psi_y)

# Create a 2D grid for the x, y coordinates
x = np.linspace(0, N, 100)
y = np.linspace(0, N, 100)
X, Y = np.meshgrid(x, y)

# Calculate the 2D wave function for the nth energy level
n = 2  # Change this to plot a different energy level
psi_2D_num = psi_x_norm[n-1, :, np.newaxis] * psi_y_norm[n-1, np.newaxis, :]


# Calculate the 2D analytical wave function
psi_x_ana = normalize_analytical(n, x, N)
psi_y_ana = normalize_analytical(n, y, N)
psi_2D_ana = psi_x_ana[:, np.newaxis] * psi_y_ana[np.newaxis, :]

# Plot the 2D wave functions
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

im1 = axs[0].imshow(psi_2D_num**2, extent=[0, 1, 0, 1], origin='lower')
axs[0].set_title(f'Numerical 2D wave function for n = {n}')

im2 = axs[1].imshow(psi_2D_ana**2, extent=[0, 1, 0, 1], origin='lower')
axs[1].set_title(f'Analytical 2D wave function for n = {n}')

for ax in axs:
    ax.set_xlabel('x')
    ax.set_ylabel('y')

fig.colorbar(im1, ax=axs[0], label='Wave function')
fig.colorbar(im2, ax=axs[1], label='Wave function')

plt.tight_layout()
plt.savefig('plots/Schrodinger_2D_Simulation.png')
plt.show()
