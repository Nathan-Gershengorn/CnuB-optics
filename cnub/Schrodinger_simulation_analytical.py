import numpy as np
import matplotlib.pyplot as plt

N=1000

# Set up grid
x = np.linspace(0, N, 100) 

def normalize_analytical(n, x, N):
    psi = np.sqrt(2) * np.sin(n*np.pi*x*N) # times N to make it work on the same domain
    dx = x[1] - x[0]  # calculate the step size
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)  # multiply the sum by the step size
    psi_norm_ana = psi / norm
    return psi_norm_ana

# Plot harmonics
for n in range(3, 4):
    psi_norm = normalize_analytical(n, x, N)
    plt.plot(x, psi_norm, label='n='+str(n))

plt.title("1D Particle in a Box Wavefunctions")
plt.xlabel("Position")
plt.ylabel("Wavefunction")
plt.legend()
plt.show()