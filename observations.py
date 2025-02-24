""" Functions to generate noise and other observational effects on the field. """

import numpy as np
from postEoR.tools import nu_21, h, k_B, T_CMB, c
from postEoR.analysis import get_distance
T_408 = 30


def get_T_sys1(z, ang, T_rcv, T_spl, T_atm, tau_0):
    """
    Calculate the total system temperature. Formula from "Anticipated performance of the SKA" (Dec 2019).

    Parameters
    ----------
    z : float
        The redshift at which to evaluate the system temperature.
    ang : float
        The zenith angle of the observation, in radians.
    T_rcv : float
        The receiver temperature, in Kelvins.
    T_spl : float
        The spillover temperature, in Kelvins.
    T_atm: float
        The atmospheric temperature contribution to the observed signal, in Kelvins.
    tau_0 : float
        The atmospheric zenith opacity. Dimensionless.

    Returns
    -------
    T_sys : float
        The system temperature, in Kelvins.
    """
    nu = nu_21 / (1+z)
    T = T_rcv+T_spl+get_T_sky(tau_0, ang, z, T_atm)
    x = (h * nu) / (k_B * T)
    T_sys = (T * x / (np.exp(x) - 1)) / (np.exp(-tau_0 * (np.cos(ang))**(-1)))

    return T_sys


def get_T_sky(tau_0, ang, z, T_atm):
    """
    Calculate the total temperature contribution from sky effects. Formula from "Anticipated performance of the SKA" (Dec 2019).

    Parameters
    ----------
    tau_0 : float
        The atmospheric zenith opacity. Dimensionless.
    ang : float
        The zenith angle of the observation, in radians.
    z : float
        The redshift at which to evaluate the sky temperature.
    T_atm : float
        The atmospheric temperature contribution to the observed signal, in Kelvins.

    Returns
    -------
    T_sky : float
        The sky temperature, in Kelvins.
    """
    T_sky = (T_CMB + get_T_gal(z))*np.exp(-tau_0*(np.cos(ang))**(-1)) + T_atm
    
    return T_sky


def get_T_gal(z):
    """
    Calculate the temperature contribution of our own galaxy. Formula from SKA red book 2018.

    Parameters
    ----------
    z : float
        The redshift at which to evaluate our galaxy temperature.

    Returns
    -------
    T_gal : float
        The temperature contribution of our galaxy to the observed signal.
    """
    nu = (nu_21 / 10**6) / (1+z)
    T_gal = 25 *(408/nu)**2.75

    return T_gal


def get_T_rx(z):
    """
    Calculate the anticipated receiver temperature at a given redshift. Formula from SKA red book 2018.

    Parameters
    ----------
    z : float
        The redshift at which to evaluate the receiver temperature.

    Returns
    -------
    T_rx : float
        The receiver temperature, in Kelvins.
    """
    T_rx = 0.1 * get_T_gal(z) + 40

    return T_rx


def get_sigma_noise(T_sys, A_eff, bw, time_res):
    """
    Calculate the standard deviation in the observed signal due to noise. 
    Formula from "Towards optimal foreground mitigation strategies for interferometric HI intensity mapping in the low redshift universe" (Chen et al. 2023)

    Parameters
    ----------
    T_sys : float
        The system temperature, in Kelvins.
    A_eff : float
        The effective observing area, in metres squared. For MeerKAT, A_eff / T_sys = 6.22 m^2 / K.
    bw : float
        The bandwidth of the instrument making the observation, in Hz. For MeerKAT, this value is 208984 Hz.
    time_res : float
        The temporal resolution of the observation, in s. For MeerKAT, this value is 40s.

    Returns
    -------
    sigma_N : float
        The contribution to the standard deviation in the observed signal due to noise.
    """
    sigma_N = 2 * k_B * T_sys / (A_eff * (bw*time_res)**0.5) 

    return sigma_N


def get_beam_size(z, D_max):
    """
    Calculate the beam size, in radians.
    Formula from "Multipole expansion for H I intensity mapping experiments: simulations and modelling" (Cunnington et al. 2020)

    Parameters
    ----------
    z : float
        The redshift of observation.
    D_max : float
        The maximum baseline of the observing system, in metres.

    Returns
    -------
    size : float
        The beam size in radians.
    """
    wavelength = c / nu_21
    size = 1.22 * wavelength * (1+z) / D_max

    return size


def get_T_sys2(z): 
    """
    Calculate the total system temperature.
    Formula from "Impact of inhomogeneous reionization of post-reionization 21cm intensity mapping measurement of cosmological parameters" (Long et al. 2023)

    Parameters
    ----------
    z : float
        The redshift at which to evaluate the total system temperature.
    
    Returns
    -------
    T_sys : float
        The total system temperature, in Kelvins.
    """
    nu = (nu_21 / 10**6) / (1+z)
    T_sys = 60 * (300/ nu)**2.55 * 1.1 + 40

    return T_sys


def get_A_eff(z, A_crit):
    """
    Calculate the effective observation area of a survey.
    Formula from "Impact of inhomogeneous reionization on post-reionization 21cm intensity mapping measurement of cosmological parameters" (Long et al. 2023)

    Parameters
    ----------
    z : float
        The redshift at which to determine the effective observation area.
    A_crit : float
        The default observation area (for high redshift), in metres squared.

    Returns
    -------
    A_eff : float
        The effective observation area, in metres squared.
    """
    nu = (nu_21 / 10**6) / (1+z)
    if nu > 110: # nu_crit in Mhz
        A_eff = A_crit * (110/nu)**2
    else:
        A_eff = A_crit
    
    return A_eff


def get_survey_volume(z_max, z_min, f_sky=0.0024): 
    """
    Calculate the survey volume of an observation.
    Formula from "Impact of inhomogeneous reionization on post-reionization 21cm intensity mapping measurement of cosmological parameters" (Long et al. 2023)

    Parameters
    ----------
    z_max : float
        The maximum redshift of the observation.
    z_min : float
        The minimum redshift of the observation.
    f_sky : float
        The fraction of sky observed. The default value is 0.0024 (for SKA-LOW in z = 3-6 band).
    """
    D_c_max = get_distance(z_max)
    D_c_min = get_distance(z_min)
    V_sur = 4 * np.pi / 3 * f_sky * (D_c_max**3-D_c_min**3)

    return V_sur

# total observation area of ska-low in z=3-6 band is approx 0.24% of sky - for ref, moon covers 0.00048%