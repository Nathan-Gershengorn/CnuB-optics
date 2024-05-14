import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import math


a = 4
hbar = 1
m = 1
V0 = 10

x = np.linspace(0.1, 8, 1000)  # More points for higher accuracy

def RHS(x):
    return np.tan(x)

def LHS(x):
    return np.sqrt((a**2 * m * V0) / (2 * hbar**2 * x**2) - 1)

def diff(x):
    return RHS(x) - LHS(x)

# Initial guesses for energies
initial_guesses = [1.5, 4, 7]


# gives bound state energies
E = fsolve(diff, initial_guesses)

print("bound state energies:", E)


# Plot RHS
plt.plot(x, RHS(x), label='RHS: tan(x)')

# Plot LHS
plt.plot(x, LHS(x), label='LHS: sqrt((a^2 * m * V0) / (2 * hbar^2 * x^2) - 1)')

plt.ylim(bottom=0, top=10)

plt.legend()
plt.grid(True)
plt.title('Graphs of RHS and LHS functions')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# just try graphing first energy

# define terms
E1 = E[0]
k0 = np.sqrt(2 * m / hbar**2 * (V0 - E1))
k1 = np.sqrt(2 * m * E1 / hbar**2)
# below is almost certainly irrelevant, but i don't want to have to rewrite it if it's right
# F = np.sqrt((1 + 1/(4*k1**3) * ((k1**2 - k0**2) * np.sin(2 * k1 * a) - 4 * k0 * k1 * (np.cos(k1 * a))**2 + 2 * a * k1**3 + k1 * (2 * a * k0**2 + 4 * k0)) - 1/(2*k0))* 2 * k0 * np.e**(2 * k0 * a))

def psi_0(x):
    return np.exp(k0 * x)

def psi_1(x):
    return k0/k1 * np.sin(k1 * x) + np.cos(k1 * x)

def psi_2(x):
    return np.exp(-k0 * x)

# set up linspaces to graph all 3
x0 = np.linspace(-a/2, 0, 100)
x1 = np.linspace(0, a, 100)
x2 = np.linspace(a, 1.5*a, 100)

plt.plot(x0, psi_0(x0), label='Graph 1')
plt.plot(x1, psi_1(x1), label='Graph 2')
plt.plot(x2, psi_2(x2), label='Graph 3')

# Add a legend
plt.legend()

# Show the plot
plt.show()