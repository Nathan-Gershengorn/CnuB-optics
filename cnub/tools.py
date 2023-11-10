import numpy as np
from scipy.integrate import quad
import vegas

# Global variables (usually a bad thing)
knu = 2.7e-4  # eV
A = 1 / np.sqrt(2)
com = complex(0, 1)


def cm_to_eVinv(d):
    return d / 197.326e-7  # 197.326 MeV.fm = hbar c = 1


def m_to_eVinv(d):
    return cm_to_eVinv(d * 1e2)


def log_jacobian(ctheta):
    J = [[-np.sqrt(1-ctheta**2), 0], 
         [1/ctheta, 1/ctheta]]
    return np.absolute(J[0][0]*J[1][1] - J[0][1]*J[1][0])


# quantum section
def gamma(ctheta, delta):
    return (-1 - (ctheta**2 - np.emath.sqrt(ctheta**4 + 2 * delta * ctheta**2)/delta))

def k_par(ctheta):
    return knu * np.sqrt(1 - ctheta**2)


def k_perp(ctheta):
    return knu * ctheta


def k_prime(ctheta, delta):
    return knu * np.emath.sqrt((ctheta**2 + 2 * delta))


def psi_incident(x, z, ctheta):
    """psi_incident Incident wave function"""
    return A * np.exp(com * (k_par(ctheta) * x + k_perp(ctheta) * z))


def psi_reflected(x, z, ctheta, delta):
    """psi_incident Reflected wave function in vacuum"""
    B = (
        A
        * (k_perp(ctheta) - k_prime(ctheta, delta))
        / (k_perp(ctheta) + k_prime(ctheta, delta))
    )
    return B * np.exp(com * (k_par(ctheta) * x - k_perp(ctheta) * z))


def psi_transmitted(x, z, ctheta, delta):
    """psi_incident Transmitted wave function in matter"""
    C = A * (2 * k_perp(ctheta)) / (k_perp(ctheta) + k_prime(ctheta, delta))
    return C * np.exp(com * (k_par(ctheta) * x + k_prime(ctheta, delta) * z))


def integral_fxn_1(z, delta_abs=2.5e-8):
    z = m_to_eVinv(z)

    def integrand(ctheta):
        term1 = (
            np.absolute(
                psi_incident(0, z, ctheta) + psi_reflected(0, z, ctheta, -delta_abs)
            )
        ) ** 2
        term2 = (
            np.absolute(
                psi_incident(0, z, ctheta) + psi_reflected(0, z, ctheta, delta_abs)
            )
        ) ** 2
        return term1 - term2

    integ = vegas.Integrator([[0,1]])
    integral = integ(integrand, nitn=10, neval=1000)
    return integral[0].mean


def integral_fxn_1_log(z, delta_abs = 2.5E-8): #try doing this with gamma after checking it
    z = m_to_eVinv(z)

    def integrand(ctheta):
        term1 = (
            np.absolute(
                psi_incident(0, z, ctheta) + psi_reflected(0, z, ctheta, -delta_abs)
            )
        ) ** 2
        term2 = (
            np.absolute(
                psi_incident(0, z, ctheta) + psi_reflected(0, z, ctheta, delta_abs)
            )
        ) ** 2
        return (term1 - term2)*log_jacobian(ctheta)
    
    integ = vegas.Integrator([[-20,1]])
    integral = integ(integrand, nitn=10, neval=1000)
    return integral[0].mean



def integral_fxn_1_separate(z, delta_abs=2.5e-8):
    z = m_to_eVinv(z)

    def term1(ctheta):
        return (
            np.absolute(
                psi_incident(0, z, ctheta) + psi_reflected(0, z, ctheta, -delta_abs)
            )
        ) ** 2

    def term2(ctheta):
        return (
            np.absolute(
                psi_incident(0, z, ctheta) + psi_reflected(0, z, ctheta, delta_abs)
            )
        ) ** 2

    integ1 = vegas.Integrator([[0,1]])
    integ2 = vegas.Integrator([[0,1]])
    integral1 = integ1(term1, nitn=10, neval=1000)
    integral2 = integ2(term2, nitn=10, neval=1000)
    return integral1[0].mean, integral2[0].mean

def integral_vac_gamma(z, delta_abs = 2.5E-8):
    z = m_to_eVinv(z)

    def integrand(ctheta):
        return 0.5 * (
            2 * np.real(np.conjugate(gamma(ctheta, -delta_abs)) * np.exp(2j * knu * ctheta * z)
            ) + 
            
            np.absolute(
                gamma(ctheta, -delta_abs) * np.exp(-2 * 1j * knu * ctheta * z)
            )**2 - 

            2 * gamma(ctheta, delta_abs) * np.cos(2 * knu * ctheta * z)
            
            - gamma(ctheta, delta_abs)**2
        )
    integ = vegas.Integrator([[0,2.3E-4]])
    integral = integ(integrand, nitn=10, neval=1000)
    return integral[0].mean




def integral_fxn_2(z, delta_abs=2.5e-8):
    z = m_to_eVinv(z)

    def integrand(ctheta):
        term1 = np.absolute(psi_transmitted(0, z, ctheta, -delta_abs)) ** 2
        term2 = np.absolute(psi_transmitted(0, z, ctheta, delta_abs)) ** 2
        return term1 - term2

    integral, *_ = quad(integrand, 0, 1)
    return integral


def analytical_approximation(z):
    delta = 2.5e-8
    lambda_cr = 3.3  # 1.87e-8)
    return (2 / 15) * np.sqrt(2 * abs(delta)) * (3 + 5 * np.exp(-np.abs(z) / lambda_cr))


