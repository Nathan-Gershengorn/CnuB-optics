import numpy as np
from scipy.integrate import quad

# Global variables (usually a bad thing)
knu = 2.7e-4  # eV
A = 1 / np.sqrt(2)
com = complex(0, 1)


def cm_to_eVinv(d):
    return d / 197.326e-7  # 197.326 MeV.fm = hbar c = 1


def m_to_eVinv(d):
    return cm_to_eVinv(d * 1e2)


# quantum section
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

    integral, *_ = quad(integrand, 0, 1)
    return integral


def integral_fxn_1_hardcode(z, delta_abs=2.5e-8):
    """integral_fxn_1_hardcode asymmetry

    Parameters
    ----------
    z : distance from the boundary in meters
    delta_abs : |delta| absolute value of delta, by default 2.5e-8 (dimensionless)

    Returns
    -------
    float
        the integral value
    """
    z = m_to_eVinv(z)

    def integrand(ctheta):
        term1 = (
            np.absolute(
                np.exp(com * k_perp(ctheta) * z) / np.sqrt(2)
                + 1
                / np.sqrt(2)
                * np.exp(-com * k_perp(ctheta) * z)
                * (1 - np.sqrt(1 - 2 * delta_abs / ctheta**2))
                / (1 + np.sqrt(1 - 2 * delta_abs / ctheta**2))
            )
        ) ** 2
        term2 = (
            np.absolute(
                np.exp(com * k_perp(ctheta) * z) / np.sqrt(2)
                + 1
                / np.sqrt(2)
                * np.exp(-com * k_perp(ctheta) * z)
                * (1 - np.sqrt(1 + 2 * delta_abs / ctheta**2))
                / (1 + np.sqrt(1 + 2 * delta_abs / ctheta**2))
            )
        ) ** 2

        return term1 - term2

    integral, *_ = quad(integrand, 0, 1, epsrel=10e-10)
    return integral


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
