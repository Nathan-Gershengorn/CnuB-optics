import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0   # Planck's constant / 2π
N = 1000     # Number of grid points


# create hamiltonian matrix
def Hamiltonian(m, N):
    H = np.zeros((N,N))

    for i in range(0,N):
        H[i,i] = 2
        if i != 0:
            H[i,i-1] = -1
        if i != N-1:
            H[i,i+1] = -1

    return H * hbar**2 / (2*m)

E, psi = np.linalg.eig(Hamiltonian(1,N))

def normalize(data):
    data_np = np.array(data)
    min_val = np.min(data_np)
    max_val = np.max(data_np)
    normalized_data = 2 * (data_np - min_val) / (max_val - min_val) - 1
    return normalized_data

psi_norm = normalize(psi)


# graph numerical
for i, points_array in enumerate(psi_norm[1:2]):
    plt.plot(points_array, label=f'Harmonic {i+2}')


# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Second harmonic')

# Display legend
plt.legend()

# Show the plot
plt.show()