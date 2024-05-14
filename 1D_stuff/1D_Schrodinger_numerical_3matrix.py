import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0   # Planck's constant / 2π
N = 200     # Number of grid points


# create hamiltonian matrix
def Hamiltonian(m, N, x):
    diag=np.where((N/4 <= x) & (x <= 3*N/4), 2, np.infty)
    H = np.diag(diag)

    for i in range(0, H.shape[0]):
        if i != 0:
            H[i,i-1] = -1
        if i != H.shape[0]-1:
            H[i,i+1] = -1
    return H * -hbar**2 / (2*m)

# def Hamiltonian(m, N, x): # THIS IS THE CODE THAT MAKES THE ALL INFINITIES MATRIX, EXCEPT FOR THE BOX IN THE MIDDLE
#     H = np.full((N, N), np.inf)  # fill the matrix with infinity

#     middle = N // 2  # find the middle index
#     box_size = (N + 1) // 4  # size of the box

#     # set all elements in the box to 0
#     H[middle - box_size:middle + box_size, middle - box_size:middle + box_size] = 0

#     # set the diagonal elements in the box to 2
#     for i in range(middle - box_size, middle + box_size):
#         H[i, i] = 2

#     # set the off-diagonal elements in the box to -1
#     for i in range(middle - box_size, middle + box_size - 1):
#         H[i, i + 1] = -1    
#         H[i + 1, i] = -1
#     return H



def normalize(psi):
    norm = np.sqrt(np.sum(np.abs(psi)**2, axis=1))  # Compute the norm of each row
    psi_norm = psi / norm[:, np.newaxis]  # Normalize each row
    return psi_norm



# function to graph the nth harmonic
def harmonic(n, N):
    
    x1 = np.linspace(0, N/4, int(N/4)) 
    x2 = np.linspace(N/4, 3*N/4, int(N/2))
    x3 = np.linspace(3*N/4, N, int(N/4))

    E, psi_1 = np.linalg.eigh(Hamiltonian(1, N, x1))
    E, psi_2 = np.linalg.eigh(Hamiltonian(1, N, x2))
    E, psi_3 = np.linalg.eigh(Hamiltonian(1, N, x3))
    psi_norm_1 = normalize(psi_1)
    psi_norm_2 = normalize(psi_2)
    psi_norm_3 = normalize(psi_3)
    
    

    plt.plot(x1, psi_norm_1[n-1]**2, label=f'Numerical: n = {n}') # graphing numerically
    plt.plot(x2, psi_norm_2[n-1]**2)
    plt.plot(x3, psi_norm_3[n-1]**2)
    


    def normalize_analytical_in(n, x):
        psi = np.sqrt(2) * np.sin(n*np.pi*x*N/2) # times N to make it work on the same domain
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
    plt.ylim(top=.05)
    # Show the plot
    
    plt.show()


harmonic(4, N)