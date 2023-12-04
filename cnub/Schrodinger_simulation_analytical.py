import numpy as np
import matplotlib.pyplot as plt

# Set up grid
x = np.linspace(0, 1, 100) 

# Plot first 5 harmonics
for n in range(2, 3):
    psi = np.sqrt(2) * np.sin(n*np.pi*x)
    plt.plot(x, psi, label='n='+str(n))

plt.title("1D Particle in a Box Wavefunctions")
plt.xlabel("Position")
plt.ylabel("Wavefunction")
plt.legend()
plt.show()
