import numpy as np
import matplotlib.pyplot as plt

# Define constants
m = 1.0  # mass of the particle
hbar = 1.0  # Planck's constant
E = 2.0  # energy of the plane wave
V0 = 1.0  # height of the potential barrier
L = 1.0  # width of the potential barrier
N = 1000  # number of grid points
x = np.linspace(-10, 10, N)  # x coordinates
dx = x[1] - x[0]  # grid spacing

# Define the potential
V = np.zeros(N)
V[x > 0] = V0

# Construct the Hamiltonian matrix
H = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            H[i, j] = -2
        elif np.abs(i - j) == 1:
            H[i, j] = 1
H = -hbar**2 / (2 * m * dx**2) * H + np.diag(V)

# Solve the Schrödinger equation
E, psi = np.linalg.eigh(H)

# Normalize the wave function
psi = psi / np.sqrt(dx)

# Plot the first few eigenstates
plt.figure(figsize=(10, 7))
for i in range(5):
    if psi[0, i] < 0:  # Flip the wave function if it starts with a negative value
        plt.plot(x, -psi[:, i], label=f"E = {E[i]:.2f}")
    else:
        plt.plot(x, psi[:, i], label=f"E = {E[i]:.2f}")
plt.plot(x, V, 'k')  # plot the potential
plt.title("Solutions to the Schrödinger equation for a potential barrier")
plt.xlabel("x")
plt.ylabel("Energy")
plt.legend()
plt.show()