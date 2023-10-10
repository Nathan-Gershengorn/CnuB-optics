import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad, quadrature


# define the variables
knu = 2.7E-4
A = 1 / np.sqrt(2)
com = complex(0, 1)
lambda_cr = 1.87E-8

# quantum section
def k_par(theta):
    return knu * np.sqrt((1 - theta**2))

def k_perp(theta):
    return knu * theta

def k_prime(theta, ior):
    return knu * theta * np.emath.sqrt((1 + (2 * ior) / theta**2))


def psi_incident(x, z, theta):
    return A * np.exp(com * (k_par(theta) * x + k_perp(theta) * z))

def psi_reflected(x, z, theta, ior):
    B = A * (k_perp(theta) - k_prime(theta, ior)) / (k_perp(theta) + k_prime(theta, ior))
    return B * np.exp(com * (k_par(theta) * x - k_perp(theta) * z))

def psi_transmitted(x, z, theta, ior):
    C = A * (2 * k_perp(theta)) / (k_perp(theta) + k_prime(theta, ior))
    return C * np.exp(com * (k_par(theta) * x + k_perp(theta) * z))
def integral_fxn_1(z):
    def integrand(theta):
       term1 = (np.absolute(psi_incident(0, z, theta) + psi_reflected(0, z, theta, -2.5E-8)))**2
       term2 = (np.absolute(psi_incident(0, z, theta) + psi_reflected(0, z, theta, 2.5E-8)))**2
       return term1 - term2
    integral, *_ = quad(integrand, 0, 1, epsrel = 10E-10)
    return integral

def integral_fxn_1_hardcode(z):
    def integrand(theta):
       term1 = (np.absolute(np.e**(com * knu * theta * z)/np.sqrt(2) + 1 / np.sqrt(2) * np.e**(-com * knu * theta * z) * (1 - np.sqrt(1 - 5E-8/theta**2))/(1 + np.sqrt(1 - 5E-8/theta**2))))**2
       term2 = (np.absolute(np.e**(com * knu * theta * z)/np.sqrt(2) + 1 / np.sqrt(2) * np.e**(-com * knu * theta * z) * (1 - np.sqrt(1 + 5E-8/theta**2))/(1 + np.sqrt(1 + 5E-8/theta**2))))**2
       return (term1 - term2)
    integral, *_ = quad(integrand, 0, 1, epsrel = 10E-10)
    return integral

print("FIRST",integral_fxn_1_hardcode(0))
print(integral_fxn_1(0))

def integral_fxn_2(z):
    def integrand(theta):
       term1 = np.absolute(psi_transmitted(0, z, theta, -2.5E-8))**2
       term2 = np.absolute(psi_transmitted(0, z, theta, 2.5E-8))**2
       return term1 - term2
    integral, *_ = quad(integrand, 1, 0)
    return integral




def quantum_graph():
    # Generate z values from -3.33E-8 to 0
    z_vals = np.linspace(-3.33E-8, 0, 100)

    # Calculate y values using your function
    integral_values = [integral_fxn_1_hardcode(z) for z in z_vals]
    axis_vals = 6 * z_vals / 3.3E-8

    # Create the plot
    plt.plot(axis_vals, integral_values, label='Asymmetry with distance from medium')
    plt.xlabel('distance from medium (in units of 3.3E-8)')
    plt.ylabel('asymmetry')
    plt.title('Asymmetry with distance from medium')
    plt.grid(True)
    plt.legend()

    # # Set y-axis limits
    # plt.ylim(1E-4, 2.6E-4)

    # Show the plot (or save it to a file)
    plt.show()

# # Now call the function to see the graph
quantum_graph()



# z approx section

ior = 2.5E-8

def f(z):
    return (2 / 15) * np.sqrt(2 * abs(ior)) * (3 + 5 * math.e ** (-abs(z) / lambda_cr))


def approx_graph():
    # Generate z values from -6 to 6
    z = np.linspace(-3.33E-8, 3.33E-8, 1000)  # 100 points for a smooth curve

    # Calculate y values using your function
    y = f(z)
    axis_vals = 6 * z / 3.3E-8

    # Create the plot
    plt.plot(axis_vals, y, label='Asymmetry with distance from medium')  # Change the label as needed
    plt.xlabel('distance from medium')
    plt.ylabel('asymmetry')
    plt.title('Asymmetry with distance from medium')
    plt.grid(True)
    plt.legend()

    # Show the plot (or save it to a file)
    plt.show()
# approx_graph()
