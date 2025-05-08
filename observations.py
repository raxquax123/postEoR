""" Functions to generate noise and other observational effects on the field. """

import numpy as np
from postEoR.tools import nu_21, h, k_B, T_CMB, c, hlittle, OMm, OMl, Mpc_to_m
from postEoR.analysis import get_distance
from abc import ABC, ABCMeta
from scipy.stats import binned_statistic
T_408 = 20
lambda_21 = 0.21
A_crit = 419000 / 512 # effective area of single station
from astropy.cosmology import Planck18
from postEoR.objects import Box, Ltcone
from ska_ost_array_config.simulation_utils import simulate_observation
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.time import Time
import ska_ost_array_config.UVW as UVW
from ska_ost_array_config.array_config import LowSubArray
from ska_ost_array_config.UVW import plot_uv_coverage
from scipy.interpolate import make_smoothing_spline
import matplotlib.pyplot as plt



def get_T_sys1(z, T_spl, T_atm):
    """
    Calculate the total system temperature. Formula from "Anticipated performance of the SKA" (Dec 2019).

    Parameters
    ----------
    z : float
        The redshift at which to evaluate the system temperature.
    T_spl : float
        The spillover temperature, in Kelvins.
    T_atm: float
        The atmospheric temperature contribution to the observed signal, in Kelvins.

    Returns
    -------
    T_sys : float
        The system temperature, in Kelvins.
    """
    nu = nu_21 / (1+z)
    T = get_T_rcv(z)+T_spl+get_T_sky(z, T_atm)
    x = (h * nu) / (k_B * T)
    T_sys = T * x / (np.exp(x) - 1)

    return T_sys


def get_T_sky(z, T_atm):
    """
    Calculate the total temperature contribution from sky effects. Formula from "Anticipated performance of the SKA" (Dec 2019).

    Parameters
    ----------
    z : float
        The redshift at which to evaluate the sky temperature.
    T_atm : float
        The atmospheric temperature contribution to the observed signal, in Kelvins.

    Returns
    -------
    T_sky : float
        The sky temperature, in Kelvins.
    """
    T_sky = T_CMB + get_T_gal(z) + T_atm
    
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
    T_gal = 25 * (408 / nu)**2.75

    return T_gal


def get_T_rcv(z):
    """
    Calculate the anticipated receiver temperature at a given redshift. Formula from SKA red book 2018.

    Parameters
    ----------
    z : float
        The redshift at which to evaluate the receiver temperature.

    Returns
    -------
    T_rcv : float
        The receiver temperature, in Kelvins.
    """
    T_rcv = 0.1 * get_T_gal(z) + 40

    return T_rcv


def get_sigma_noise(T_sys, A_eff, bw, time_res):
    """
    Calculate the standard deviation in the observed signal due to noise. 
    Formula from "Towards optimal foreground mitigation strategies for interferometric HI intensity mapping in the low redshift universe" (Chen et al. 2023)
    Uses the 'radiometer equation'

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
    sigma_N = 2 * k_B * T_sys / (A_eff * (bw * time_res)**0.5) 

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
    T_sys = 60 * (300 / nu)**2.55 * 1.1+40

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
        A_eff = A_crit * (110 / nu)**2
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

    Returns
    -------
    V_sur : float
        The survey volume of an observation, in Mpc^3.
    """
    D_c_max = get_distance(z_max)
    D_c_min = get_distance(z_min)
    V_sur = 4 * np.pi / 3 * f_sky * (D_c_max**3-D_c_min**3)

    return V_sur

# total observation area of ska-low in z=3-6 band is approx 0.24% of sky - for ref, moon covers 0.00048%


"""
Below are modified versions of Laura's code, edited to be specific to SKA-LOW.
"""

class Telescope(ABC):
    """
    The base telescope class, upon which we may build the various stages of SKA-LOW. Cannot be directly instantiated.
    """
    def __init__(self):
        self.fwhm21cm = lambda_21 / A_crit**0.5 # fwhm at 21cm in radians
        self.fov21cm = self.fwhm21cm**2 * self.nbeam # Field-of-view in steradians

    def fwhm_at_z(self, z): 
        """
        Calculate the FWHM of a single beam of the telescope at a given observation redshift, in radians.

        Parameters
        ----------
        z : float
            The redshift of the observation.
        
        1.220 * (1.+z) * (c / nu_21) / (0.8 * self.ddish)
        """

        return (lambda_21 * (1+z) / A_crit**0.5) # from phil's paper
    

    def fov_at_z(self, z):
        """
        Calculate the field of view of the telescope at a given redshift z for a single beam, in steradians.

        Parameters
        ----------
        z : float
            The redshift of the observation.
        """
        return self.fwhm_at_z(z)**2 * self.nbeam
    

    def z_to_f(self, z):
        """
        Determines the frequency of the 21 cm line at a given redshift z.

        Parameters
        ----------
        z : float
            The redshift of the observation.
        """
        return nu_21 / (1. + z)



class SKA1LOW_AAstar(Telescope):
    """
    Instantiating the SKA-LOW telescope for the AA* stage.

    Parameters
    ----------
    T_spl : float
        The spillover temperature of the instrument, in K.
    """
    def __init__(self, T_spl):
        self.maxB = 73.4 # Maximum Baseline in km
        self.ddish = 35. # diameter of a station, in m
        self.nbeam = 1 # number of beams
        self.npol = 2 # number of polarisations
        self.ndish = 307. # number of stations
        self.area = 419000 * self.ndish/512 # Total effective collecting area [m^2] CHECK
        self.dnu = 781250.0 # this is coarse channel width - fine channel width is 226 Hz. In Hz
        self.T_spl = T_spl
        self.stage = "AA*"



class SKA1LOW_AA4(Telescope):
    """
    Instantiating the SKA-LOW telescope for the A4 stage.

    Parameters
    ----------
    T_spl : float
        The spillover temperature of the instrument, in K.
    """
    def __init__(self, T_spl):
        self.maxB = 73.4 # Maximum Baseline in km
        self.ddish = 35. # diameter of a station, in m
        self.nbeam = 1 # number of beams
        self.npol = 2 # number of polarisations
        self.ndish = 512. # number of stations
        self.area = 419000 * self.ndish/512 # Total effective collecting area [m^2] CHECK
        self.dnu = 781250.0 # this is coarse channel width - fine channel width is 226 Hz. In Hz
        self.T_spl = T_spl
        self.stage = "AA4"



class SKA1LOW_AA05(Telescope):
    """
    Instantiating the SKA-LOW telescope for the AA0.5 stage.

    Parameters
    ----------
    T_spl : float
        The spillover temperature of the instrument, in K.
    """
    def __init__(self, T_spl):
        self.maxB = 73.4 # Maximum Baseline in km
        self.ddish = 35. # diameter of a station, in m
        self.nbeam = 1 # number of beams
        self.npol = 2 # number of polarisations
        self.ndish = 4. # number of stations
        self.area = 419000 * self.ndish/512 # Total effective collecting area [m^2] CHECK
        self.dnu = 781250.0 # this is coarse channel width - fine channel width is 226 Hz. In Hz
        self.T_spl = T_spl
        self.stage = "AA0.5"



class SKA1LOW_AA2(Telescope):
    """
    Instantiating the SKA-LOW telescope for the AA2 stage.

    Parameters
    ----------
    T_spl : float
        The spillover temperature of the instrument, in K.
    """
    def __init__(self, T_spl):
        self.maxB = 39.0 # Maximum Baseline in km
        self.ddish = 35. # diameter of a station, in m
        self.nbeam = 1 # number of beams
        self.npol = 2 # number of polarisations
        self.ndish = 64. # number of stations
        self.area = 419000 * self.ndish/512 # Total effective collecting area [m^2] CHECK
        self.dnu = 781250.0 # this is coarse channel width - fine channel width is 226 Hz. In Hz
        self.T_spl = T_spl
        self.stage = "AA2"



class SKA1LOW_AA1(Telescope):
    """
    Instantiating the SKA-LOW telescope for the AA1 stage.
    """
    def __init__(self, T_spl):
        self.maxB = 73.4 # Maximum Baseline in km
        self.ddish = 35. # diameter of a station, in m
        self.nbeam = 1 # number of beams
        self.npol = 2 # number of polarisations
        self.ndish = 16. # number of stations
        self.area = 419000 * self.ndish/512 # Total effective collecting area [m^2] CHECK
        self.dnu = 781250.0 # this is coarse channel width - fine channel width is 226 Hz. In Hz
        self.T_spl = T_spl
        self.stage = "AA1"



class Survey(Telescope):
    """
    The base survey class, upon which we may build the various surveys to be carried out by SKA-LOW. Cannot be directly instantiated.

    Parameters
    ----------
    Telescope : class Telescope
        The telescope used in the survey.
    """
    __metaclass__ = ABCMeta

    def __init__(self, Telescope): 
        self.Telescope = Telescope
        self.z_med = (self.z_min + self.z_max) / 2. # Median redshift
        
        self.asurv = self.asurv * (np.pi/180.)**2 # Survey area in sr

        self.npt = self.asurv / Telescope.fov_at_z(self.z_med) # Number of pointings
        self.t_single = (self.tsurv*3600.) / self.npt # Integration time per pointing
        
        self.T_sys = get_T_sys1(self.z_med, Telescope.T_spl, self.T_atm) + T_CMB + get_T_rcv(self.z_med) # in K
        
        self.nk = 20 # number of k bins

        self.aeffdish = get_A_eff(self.z_med, A_crit) # Effective collecting area per station in m^2 | np.pi * (Telescope.ddish / 2.)**2

        self.aeff = get_A_eff(self.z_med, Telescope.area) # Total effective collecting area [m^2] CHECK


    def volelement(self, z=None): # tested
        """
        Calculates the comoving volume of one element per pointing, i.e. volume per FoV and frequency channel.

        Parameters
        ----------
        z : float
            The redshift of the observation. Defaults to None. 
        """
        if z is None: # defaults to median redshift if none specified
            z = self.z_med
        r = get_distance(z) * hlittle # in Mpc/h
        H_z = (hlittle * 100 * (OMm * (1+z)**3 + OMl)**0.5) # in km s^-1 Mpc^-1
        y = c * 1e-3 * (1.+z)**2 / (H_z * nu_21) * hlittle # in Mpc s/h
        
        return r**2 * y  * self.Telescope.fov_at_z(z) * self.Telescope.dnu


    def volsurvey(self): # tested
        """
        Computes the comoving volume of the survey in (Mpc/h)^3.

        Returns
        -------
        vol : float
            The comoving volume of the survey, in (Mpc/h)^3.
        """
        vol = get_survey_volume(self.z_max, self.z_min) * hlittle**3 
        
        return vol


    def pk_gal(self, Ngal, bgal, field: Box | Ltcone): # tested
        """
        Generates the theoretical galaxy power spectrum, alongside the galaxy shot noise and the total noise contribution from galaxies.

        Parameters
        ----------
        Ngal : int
            The number of galaxies in the sample.
        bgal : float
            The linear galaxy bias.
        field : class Box | class Ltcone
            The field on which to evaluate the galaxy power spectrum.

        Returns
        -------
        karr : NDarray
            An array of wavenumbers, in h Mpc^-1.
        popt : NDarray
            The theoretical galaxy power spectrum.
        pshotgal : float
            The galaxy shot noise.
        sigma_cross : NDarray
            The total power contribution from galaxies to the observed power spectrum.
        """
        if isinstance(self, Interferometer):
            karr = self.karr()
        else:
            karr = field.gen_k_bins()

        _, pcosmo, _ = field.get_PS(field="dens", kbins=karr, save_fig=False, remove_nan=False)

        popt = bgal**2 * pcosmo
        
        pshotgal = 1. / (Ngal / self.volsurvey())
        
        sigma_cross = np.sqrt((popt**2+pshotgal**2)) 

        if isinstance(self, Interferometer):
            rhok = self.volsurvey() / ((2. * np.pi)**3)
            volk = 0.5 * (4. / 3.) * np.pi * ((karr[1:])**3 - (karr[:-1])**3) # Number of Fourier modes in this bin
            nummod = rhok * volk
            sigma_cross /= 2 * nummod

        k = (karr[1:] + karr[:-1]) / 2

        k = k[:np.size(sigma_cross)]

        k = k[~np.isnan(sigma_cross)]
        popt = popt[~np.isnan(sigma_cross)]
        sigma_cross = sigma_cross[~np.isnan(sigma_cross)]
        
        return k, popt, pshotgal, sigma_cross


    def fisher_pkshotnoise(self, pkm, pkcrosserr):
        """
        Calculates the standard deviation in the shot noise, using the Fisher information matrix.

        Parameters
        ----------
        pkm : NDarray

        pkcrosserr : NDarray

        Returns
        -------
        pkshoterr : float
            The standard deviation in the shot noise.
        """
        fish = np.empty(shape=(2,2))
        fish[0,0] = np.sum((pkm**2) / (pkcrosserr**2))
        fish[1,0] = np.sum(pkm / (pkcrosserr**2))
        fish[0,1] = fish[1,0]
        fish[1,1] = np.sum(1. / (pkcrosserr**2))
        fishinv = np.linalg.inv(fish)
        pkshoterr = np.sqrt(fishinv[1,1])
        
        return pkshoterr



class Interferometer(Survey):
    """
    The interferometer class, which can be instantiated to reflect the properties of a given observation.

    Parameters
    ----------
    Telescope : class Telescope
        The telescope object with which the survey is carried out.
    z_max : float
        The maximum redshift of the survey.
    z_min : float
        The minimum redshift of the survey.
    asurv : float
        The survey area, in square degrees.
    tsurv : float
        The total collecting time of the survey, in hours.
    T_atm : float
        The atmospheric temperature contribution to the system temperature, in Kelvins.
    """
    def __init__(self, Telescope, z_max, z_min, asurv, tsurv, T_atm):
        self.asurv = asurv
        self.tsurv = tsurv # tsurv is in hours
        self.T_atm = T_atm
        self.z_max = z_max
        self.z_min = z_min
        Survey.__init__(self, Telescope)
        self.gen_nvis(Telescope)
        self.kperparr = self.kperp_at_z(self.z_med)
        self.Telescope = Telescope
        
        # Set min and max k modes via the visibility coverage 
        kmin1 = self.kperparr[0]
        # min k in line-of-sight direction
        kmin2 = (2. * np.pi) / (get_distance(self.z_max) - get_distance(self.z_min))
        
        # use the smallest kmin possible
        self.kmin = min(kmin1, kmin2)
        self.kmax = self.kperparr[-1]


    def gen_nvis(self, Telescope):
        """
        Generates the baseline number density for a given interferometric array, and assigns it and the baselines to the object.

        Parameters
        ----------
        Telescope : Telescope object
            The telescope array whose n(u) is to be calculated.
        """
        # Target coordinate - Polaris Australis
        phase_centre = SkyCoord("21:08:46.8 -88:57:23.4", unit=(units.hourangle, units.deg))

        # Start time of the observation
        start_time = Time("2024-09-25T03:23:21.6", format='isot', scale='utc')
        nu = nu_21 / (1+self.z_med)

        observation = simulate_observation(
            array_config=LowSubArray(subarray_type=Telescope.stage).array_config,
            phase_centre=phase_centre,
            start_time=start_time,
            duration=5000,
            integration_time=500,
            ref_freq=nu,
            chan_width=Telescope.dnu,
            n_chan=1,
            horizon=0,
        )

        uvw = UVW.UVW(observation, ignore_autocorr=False)

        u_vals = (uvw.u_wave**2 + uvw.v_wave**2)**0.5

        bins = np.geomspace(20, 15000, 51)

        counts, _ = np.histogram(u_vals, bins)

        self.u = (bins[1:] + bins[:-1]) / 2

        self.nvis = counts.astype(float)

        self.du = bins[1:] - bins[:-1] 

        self.nvis /= self.du * self.u * 2 * np.pi

        sumn = 2. * np.pi * sum(self.nvis * self.u * self.du) # numerical approximation to the half plane integral

        self.nvis *= (0.5 * Telescope.ndish * (Telescope.ndish-1.)) / sumn # this normalises the half plane integral to the total number of baseline pairs


    def nvis_to_spline(self, lam=1e6):
        """
        Fits the baseline density distribution n(u) to a spline.

        Parameters
        ----------
        lam : float (optional)
            The degree of smoothing applied to the spline creation. Defaults to 1e6.

        Returns
        -------
        spl : BSpline object
            The smoothed spline for the baseline density distribution.
        """
        spl = make_smoothing_spline(self.u, self.nvis, lam=lam)

        return spl


    def karr(self):
        """
        Generates an array of wavenumbers based on the maximum and minimum wavenumbers present in the array.

        Returns
        -------
        karr : NDarray
            The wavenumber array, in h Mpc^-1.
        dk : float
            The logarithmic spacing of the wavenumbers in the array, in h Mpc^-1.
        """
        dk = (np.log10(self.kmax)-np.log10(self.kmin)) / self.nk
        karr = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.nk)
        
        return karr, dk


    def renorm_vis(self, z=None): 
        """
        Renormalises the baseline variables to a given redshift z.

        Parameters
        ----------
        z : float (optional)
            The redshift of the observation. Defaults to None. 

        Returns
        -------
        u_at_z : NDarray
            The renormalised u coordinate distribution.
        nvis_at_z : NDarray
            The renormalised baseline number density.
        du_at_z : float
            The renormalised u coordinate separation.
        """
        if z is None: # defaults to median redshift if none specified
            z = self.z_med
        u_at_z = self.u / (1.+z)
        nvis_at_z = self.nvis * (1.+z)**2
        du_at_z = self.du / (1.+z)
        
        return u_at_z, nvis_at_z, du_at_z


    def kperp_at_z(self, z=None): 
        """
        Returns array of k_perp values for the given redshift, in units [Mpc/h].

        Parameters
        ----------
        z : float (optional)
            The redshift of the observation. Defaults to None. 

        Returns
        -------
        kperpatz : NDarray
            An array of the perpendicular k values for the given array layout, in h Mpc^-1.
        """
        if z is None: # defaults to median redshift if none specified
            z = self.z_med
        r = get_distance(z) * hlittle # [Mpc/h]
        kperpatz = (2. * np.pi * self.u) / r
        
        return kperpatz


    def sigmaT_per_pointing(self, z=None): 
        """
        Implementation of Equ. 18 from Wolz et al.
        Calculate the deviation in the temperature signal due to the system temperature, for a given telescope pointing.
        In Kelvins. 

        Parameters
        ----------
        z : float (optional)
            The redshift of the observation. Defaults to None. 

        Returns
        -------
        sig_T : float
            The contribution to the observed temperature signal from the observing system, in Kelvins.
        """
        if z is None: # defaults to median redshift if none specified
            z = self.z_med

        u_at_z, nvis_at_z, du_at_z = self.renorm_vis(z) # renorm visibilities to redshift of observation
        
        dusq = 2. * np.pi * u_at_z * du_at_z # 2d differential
        
        lambda_z = lambda_21 * (1.+z)

        sig_T = ((lambda_z**2) * self.T_sys) / (self.aeffdish * np.sqrt(self.Telescope.dnu * self.t_single * nvis_at_z * dusq))

        sig_T /= np.sqrt(self.Telescope.nbeam * self.Telescope.npol)

        sig_T[np.isinf(sig_T)] = np.nan
        
        return sig_T


    def noise_power_perp(self, z=None):
        """
        Returns the noise power spectrum for the given visibility coverage, as a function of k_perp.

        Parameters
        ----------
        z : float (optional)
            The redshift of the observation. Defaults to None. 

        Returns
        -------
        pnoise : NDarray
            The noise power spectrum, in K^2 (Mpc h^-1)^3.
        k_perp : NDarray
            An array containing the perpendicular k values of the array layout, in h Mpc^-1.
        """
        if z is None: # defaults to median redshift if none specified
            z = self.z_med

        sig_T = self.sigmaT_per_pointing(z)
        vpix = self.volelement(z)
        u_at_z, _, du_at_z = self.renorm_vis(z)
        
        dusq = 2. * np.pi * u_at_z * du_at_z
        fov = self.Telescope.fov_at_z(z)

        pnoise = (sig_T**2) * vpix * dusq / fov
        
        k_perp = self.kperp_at_z(z)
        
        return pnoise, k_perp
    

    def plot_uv(self, pointing_time=None, duration=None):
        """
        Plots the uv distribution of a given survey, using the functionality provided by SKA-ost-array-config.

        Parameters
        ----------
        integration_time : float (optional)
            The time per pointing, in seconds. If no value is specified, this defaults to the integration time of the survey.
        duration : float (optional)
            The total duration of the observation, in hours. If no value is specified, this defaults to the survey time.
        """

        if pointing_time is None:
            pointing_time = self.t_single
        if duration is None:
            duration = self.tsurv

        duration *= 3600

        phase_centre = SkyCoord("21:08:46.8 -88:57:23.4", unit=(units.hourangle, units.deg))

        start_time = Time("2024-09-25T03:23:21.6", format='isot', scale='utc') # Start time of the observation

        nu = nu_21 / (1+self.z_med)

        observation = simulate_observation(
            array_config=LowSubArray(subarray_type=self.Telescope.stage).array_config,
            phase_centre=phase_centre,
            start_time=start_time,
            duration=duration,
            integration_time=pointing_time,
            ref_freq=nu,
            chan_width=self.Telescope.dnu,
            n_chan=1,
            horizon=0,
        )

        uvw = UVW.UVW(observation, ignore_autocorr=False)
        plot_uv_coverage(uvw)

        plt.title("Duration: " + str(duration / 3600) + " hours, pointing time: " + str(pointing_time) + " seconds, for " + str(self.Telescope.stage))




class SingleDish(Survey): # tested
    """
    The single-dish class, which can be instantiated to reflect the properties of a given observation.

    Parameters
    ----------
    Telescope : class Telescope
        The telescope object with which the survey is carried out.
    z_max : float
        The maximum redshift of the survey.
    z_min : float
        The minimum redshift of the survey.
    asurv : float
        The survey area, in square degrees.
    tsurv : float
        The total collecting time of the survey, in hours.
    T_atm : float
        The atmospheric temperature contribution to the system temperature, in Kelvins.
    """
    def __init__(self, Telescope, z_max, z_min, asurv, tsurv, T_atm):  
        self.z_max = z_max
        self.z_min = z_min
        self.asurv = asurv
        self.tsurv = tsurv
        self.Telescope = Telescope
        self.T_atm = T_atm
        Survey.__init__(self, Telescope)
        # Determine min and max wavenumber according to beam fwhm amd survey volume
        # min k in perpendicular direction
        comovarcmin = Planck18.kpc_comoving_per_arcmin(self.z_med).value * 1e-3 * hlittle # in Mpc/h 
        asurv_in_arcmin = np.sqrt(self.asurv) * 180. / np.pi * 60. # convert in deg and then in arcmin
        kmin1 = (2. * np.pi) / (comovarcmin * asurv_in_arcmin)

        # min k in line-of-sight direction
        kmin2 = (2. * np.pi) / ((get_distance(self.z_max) - get_distance(self.z_min)) * hlittle) 

        # use the smallest kmin possible
        self.kmin = min(kmin1, kmin2)
        
        # k max is set arbitrary as we convolve with Gaussian beam in pknoise
        self.kmax = 1e3


    def sigmaT_per_pointing(self, z=None): # tested
        """
        Calculate the deviation in the temperature signal due to the system temperature, for a given telescope pointing.
        In Kelvins. 
        Implementation of Equ.18 from Wolz et al, for single dish, so n(u) -> 1 
        
        Parameters
        ----------
        z : float (optional)
            The redshift of the observation. Defaults to None. 

        Returns
        -------
        sig_T : float
            The contribution to the observed temperature signal from the observing system, in Kelvins.
        """
        if z is None: # defaults to median redshift if none specified
            z = self.z_med
        lambda_z = lambda_21 * (1. + z)
        sig_T = lambda_z**2 * self.T_sys / (self.Telescope.fov_at_z(z) * self.aeffdish * np.sqrt(self.Telescope.dnu * self.t_single))
        
        # Scale to multiple dishes/beams/polarizations
        sig_T /= np.sqrt(self.Telescope.ndish * self.Telescope.nbeam * self.Telescope.npol)
        
        return sig_T


    def noise_power_perp(self, z=None): # tested
        """
        Parameters
        ----------
        z : float (optional)
            The redshift of the observation. Defaults to None. 

        Returns
        -------
        pnoise : float
            The noise power contribution.
        """
        if z is None: # defaults to median redshift if none specified
            z = self.z_med
        
        sig_T = self.sigmaT_per_pointing(z)
        vpix =  self.volelement(z)
        
        pnoise = (sig_T**2) * vpix # normalise by sampling volume
        
        return pnoise




def FT_gals(mock, HII_dim, box_len):
    """
    Calculates the FT and the corresponding wavenumbers for the input mock data.

    Parameters
    ----------
    mock : NDarray
        Mock data field to be FT-ed.
    HII_dim : int
        The number of cells along the spatial dimensions of the field.
    box_len : float
        The comoving distance along the spatial dimensions of the field, in Mpc / h.

    Returns
    -------
    k : NDarray
        The wavenumbers of the Fourier-transformed field.
    FT : NDarray
        The Fourier transformed field.
    """
    # Generate a histogram of the data, with appropriate number of bins.
    edges = [np.linspace(0, box_len, HII_dim+1)] * 3
    deltax = np.histogramdd(mock['pos_tracer']%box_len, bins=edges)[0].astype("float")

    # Convert sampled data to mean-zero data
    deltax = deltax/np.mean(deltax) - 1

    # Calculate the n-D power spectrum and align it with the k from powerbox.
    FT = np.fft.fftn(deltax)
    dims = np.shape(deltax)

    # obtaining k values and k bins to use in ps
    ksx = np.fft.fftfreq(dims[0], (box_len / HII_dim)) * 2 * np.pi # max accessible wavenumber corresponds to 2 * pi
    ksy = np.fft.fftfreq(dims[1], (box_len / HII_dim)) * 2 * np.pi
    ksz = np.fft.fftfreq(dims[2], (box_len / HII_dim)) * 2 * np.pi
    kx, ky, kz = np.meshgrid(ksx, ksy, ksz) # converting to a 3d array
    k = (kx**2+ky**2+kz**2)**0.5 # spherical k-values
    k = k.reshape(np.size(k)) # converting to 1d array for use in binned_statistic
    return k, FT


def power_from_ft(ft, k, n, HII_dim, box_len, ft2=None):
    """
    Calculate the power spectrum from the input Fourier-transformed field. If only one field is input, its autopower is calculated.

    Parameters
    ----------
    ft : NDarray
        A Fourier-transformed field.
    k : NDarray
        The wavenumbers corresponding to the Fourier-transformed field.
    n : int
        The total number of cells comprising the Fourier-transformed field.
    HII_dim : int
        The number of cells along the spatial dimensions of the field.
    ft2 : NDarray (optional)
        An additional Fourier-transformed field to calculate the cross-power spectrum with. Defaults to None.

    Returns
    -------
    p_k_HI : NDarray
        The power spectrum.
    kbins : NDarray
        The corresponding wavenumbers for the power spectrum.
    """
    if ft2 is None: # compute auto power if only one field is input
        ft2 = ft
        
    bins = int(len(ft) / 2.2)

    # HI power spectrum
    PHI = np.real(ft * np.conj(ft2))
    p_k_HI, _, _ = binned_statistic(k, PHI, statistic = "mean", bins = bins) # binning power
    p_k_HI /= (n * (HII_dim / box_len)**3)

    return p_k_HI, bins


def noisetokbin(kperparr, pnoise, karr):

    """
    Given the perpendicular noise power spectrum pnoise as a function of perpendicular wavenumber kperp,
    interpolates the noise power spectrum for different wavenumber bins karr.

    Parameters
    ----------
    kperparr : NDarray
        The original wavenumber bins corresponding to the power spectrum pnoise.
    pnoise : NDarray
        The input power spectrum to be interpolated.
    karr : NDarray
        The wavenumber bins to interpolate the power spectrum to.

    Returns
    -------
    pknoise : NDarray
        The interpolated noise power spectrum.
    """
    nk = len(karr)
    pknoise = np.zeros(nk)
    kperpmin = kperparr[0]
    kperpmax = kperparr[-1]
    nint = 10000
    kperpint = np.linspace(kperpmin, kperpmax, nint)
    dkperpint = (kperpmax-kperpmin) / nint
    pnoiseint = np.interp(kperpint, kperparr, pnoise)
    
    # Average noise power spectrum into k-bins
    for ik in range(nk):
        k = karr[ik]
        # If k is less than the minimum k_perp, no measurement is possible
        if (k < kperpmin):
            pknoise[ik] = 1.e+99
        else:
        # This approximates : int P_N(k_perp) dOmega / int dOmega
        # where dOmega = sin(theta) dtheta dphi and k_perp = k sin(theta)
            sum1 = 0.
            sum2 = 0.
            for ikperp in range(nint):
                kperp = kperpint[ikperp]
                if (kperp < k):
                    wei = (kperp / k) * (dkperpint / np.sqrt((k**2)-(kperp**2)))
                    sum1 += pnoiseint[ikperp] * wei
                    sum2 += wei

            pknoise[ik] = sum1 / sum2

    return pknoise

