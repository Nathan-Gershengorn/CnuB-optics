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
def harmonic(n, N):
    for i, points_array in enumerate(psi_norm[n-1:n]):
        plt.plot(points_array, label=f'Numerical: n = {n}')
    
    x = np.linspace(0, N, 100) # x ranges from 0 to 1000

    def normalize_analytical(n, x):
        psi = np.sqrt(2) * np.sin(n*np.pi*x*N) # times N to make it work on the same domain
        dx = x[1] - x[0]  # calculate the step size
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)  # multiply the sum by the step size
        psi_norm_ana = psi / norm
        return psi_norm_ana
    
    for a in range(n, n+1):
        psi_norm_ana = normalize_analytical(a, x)
        plt.plot(x, psi_norm_ana, label=f'Analytical: n = {a}')

    # Set labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Harmonic {n}')

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


harmonic(3, N)