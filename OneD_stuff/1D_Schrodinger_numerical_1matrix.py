import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0   # Planck's constant / 2π
N = 1000     # Number of grid points



def Hamiltonians(m, N): # make an all diag(2) off-diag(-1) matrix
    def kinetic(N):
        diags = np.full(N, 2)

        H = np.diag(diags)

        for i in range(0, H.shape[0]):
            if i != 0:
                H[i,i-1] = -1
            if i != H.shape[0]-1:
                H[i,i+1] = -1
        print(H)
        return H

    def potential(N):
        H = np.zeros((N,N))
        for i in range(0, H.shape[0]):
            if i < int(N/4) or i > int(3*N/4):
                H[i,i] = 1
            else:
                H[i,i] = 0

        return H
    H = (kinetic(N)* (-hbar**2 / (2*m))  + potential(N))
    return H


def normalize(psi):
    norm = np.sqrt(np.sum(np.abs(psi)**2, axis=1))  # Compute the norm of each row
    psi_norm = psi / norm[:, np.newaxis]  # Normalize each row
    return psi_norm



# function to graph the nth harmonic
def harmonic(n, N):
    
    E, psi = np.linalg.eigh(Hamiltonians(1, N))
    psi = psi.T
    x_values = np.linspace(0, N, N)
    
    plt.plot(x_values, psi[n-1]**2, label=f'Numerical: n = {n}') # graphing numerically


    x1 = np.linspace(0, N/4, int(N/4)) 
    x2 = np.linspace(N/4, 3*N/4, int(N/2))
    x3 = np.linspace(3*N/4, N, int(N/4))
    


    def normalize_analytical_in(n, x):
        psi = np.sqrt(2)*np.sin(n*np.pi*x*N/2) # times N to make it work on the same domain
        dx = x[1] - x[0]  # calculate the step size
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)  # multiply the sum by the step size
        psi_norm_ana = psi / norm
        return psi_norm_ana
    
    for a in range(n, n+1):
        plt.plot(x1, 0*x1, ls=':', color='black')

    for a in range(n, n+1):
        psi_norm_ana = normalize_analytical_in(a, x2)
        plt.plot(x2, psi_norm_ana**2, label=f'Analytical: n = {a}', ls=':', color='black')

    for a in range(n, n+1):
        plt.plot(x3, 0*x3, ls=':', color='black')


    # Set labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Harmonic {n} Probability Distribution')

    # Display legend
    plt.legend()
    # Show the plot
    
    plt.show()


harmonic(1, N)

# find the decay constant for outside the box
# mess around more, understand it better
# expand to 2D or 3D or soemthing? idk just fuck around
# try 1/r case