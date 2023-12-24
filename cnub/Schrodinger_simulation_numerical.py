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

E, psi = np.linalg.eigh(Hamiltonian(1,N))


def normalize(psi):
    norm = np.sqrt(np.sum(np.abs(psi)**2, axis=1))  # Compute the norm of each row
    psi_norm = psi / norm[:, np.newaxis]  # Normalize each row
    return psi_norm

psi_norm = normalize(psi)

# function to graph the nth harmonic
def harmonic(n):
    for i, points_array in enumerate(psi_norm[n-1:n]):
        plt.plot(points_array, label=f'Harmonic {n}')
    
    # Set labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Harmonic {n}')

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()

harmonic(3)