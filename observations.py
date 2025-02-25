""" Functions to generate noise and other observational effects on the field. """

import numpy as np
from postEoR.tools import nu_21, h, k_B, T_CMB, c, hlittle, OMm, OMl, Mpc_to_m
from postEoR.analysis import get_distance, get_PS
from abc import ABC, ABCMeta
from scipy.stats import norm, binned_statistic
T_408 = 20
lambda_21 = 0.21
from astropy.cosmology import Planck18
from postEoR.objects import Box, Ltcone


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
        self.aeffdish = get_A_eff(self.z_med, np.pi * (self.ddish / 2.)**2) # Total effective collecting area m^2 
        self.fwhm21cm = 1.220 * lambda_21 / self.ddish # fwhm at 21cm [sr]
        self.fov21cm = self.fwhm21cm**2 * self.nbeam # Field-of-view [sr] 
    
    def fwhm_at_z(self, z): 
        """
        Calculate the FWHM of the telescope at a given observation redshift, in radians.

        Parameters
        ----------
        z : float
            The redshift of the observation.
        """
        return 1.220 * (1.+z) * (c / nu_21) / (0.8 * self.ddish)
    
    def fov_at_z(self, z):
        """
        Calculate the field of view of the telescope at a given redshift z, in steradians.

        Parameters
        ----------
        z : float
            The redshift of the observation.
        """
        return self.fwhm_at_z(z)**2 
    
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
    """
    def __init__(self):
        self.maxB = 73.4 # Maximum Baseline in km
        self.z_med = (self.z_min + self.z_max) / 2
        self.aeff = get_A_eff(self.z_med, 419000) # Total effective collecting area [m^2]
        self.ddish = 35. # diameter of a station, in m
        self.nbeam = 1. # number of beams
        self.npol = 2. # number of polarisations
        self.ndish = 307. # number of stations
        self.dnu = 781250.0 # this is coarse channel width - fine channel width is 226 Hz. In Hz


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
        
        self.asurv *= (np.pi/180.)**2 # Survey area in sr
        self.fov_at_zmed = self.Telescope.fov_at_z(self.z_med)

        self.npt = self.asurv / self.fov_at_zmed # Number of pointings
        self.tint = (self.tsurv*3600.) / self.npt # Integration time per pointing
        
        self.T_sys = get_T_sys1(self.z_med, self.T_spl, self.T_atm) + T_CMB + get_T_rcv(self.z_med) # in K
        
        self.nk = 50 # number of k bins


    def volelement(self, z=None):
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
        H_z = (hlittle * 100 * 1000 * (OMm * (1+z)**3 + OMl)**0.5) / Mpc_to_m # in s^-1
        y = c * 1e-3 * (1.+z)**2 / (H_z * nu_21) # in m s
        
        return r**2 * y  * self.Telescope.fov_at_z(z) * self.Telescope.dnu


    def volsurvey(self):
        """
        Computes the comoving volume of the survey in (Mpc/h)^3.

        Returns
        -------
        vol : float
            The comoving volume of the survey, in (Mpc/h)^3.
        """
        vol = get_survey_volume(self.zmax, self.zmin) * (self.asurv / (4. * np.pi)) * hlittle**3 
        
        return vol


    def karr(self):
        """
        Outputs an array of wavenumbers, alongside their logarithmic separation.

        Returns
        -------
        """
        dk = (np.log10(self.kmax) - np.log10(self.kmin)) / self.nk
        karr = np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.nk)

        return karr, dk


    def pk_gal(self, Ngal, bgal, field: Box | Ltcone):
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

        """
        karr, dk = self.karr()
        
        _, pcosmo, _ = get_PS(field.density_field, field.box_len, field.HII_dim)
        
        popt = bgal**2 * pcosmo
        
        pshotgal = 1. / (Ngal / self.volsurvey())
        
        rhok = self.volsurvey() / ((2. * np.pi)**3)
        
        # Number of Fourier modes in this bin
        volk = 0.5 * (4. / 3.) * np.pi * ((karr[:]+dk)**3 - (karr[:]-dk)**3)
        
        nummod = rhok * volk
        
        sigma_cross = np.sqrt((popt**2+pshotgal**2) / (2. * nummod))
        
        return karr, popt, pshotgal, sigma_cross


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
    visfile : str
        The path to the visibility file.
    """
    def __init__(self, Telescope, visfile):
        Survey.__init__(self, Telescope)
        self.read_vis(visfile)
        self.kperparr = self.kperp_at_z(self.z_med)
        
        # Set min and max k modes via the visibility coverage 
        kmin1 = self.kperparr[0]
        # min k in line-of-sight direction
        kmin2 = (2. * np.pi) / (get_distance(self.z_max) - get_distance(self.z_min))
        
        # use the smallest kmin possible
        self.kmin = min(kmin1, kmin2)
        self.kmax = self.kperparr[-1]


    def read_vis(self, visfile): 
        """
        Reads in a visibility file (stored as a text file), and returns the normalised visibility values.
        Sets u, n(u) at z=0, lambda=0.21.

        Parameters
        ----------
        visfile : str
            The path to the visibility file.
        """
        vis = np.loadtxt(visfile)
        self.u = vis[:,0] / lambda_21 # extracts the u coordinate
        self.nvis = vis[:,1] # extracts the visibility value
        self.du = self.u[1] - self.u[0]
        
        sumn = 2. * np.pi * self.du * sum(self.nvis * self.u)
        self.nvis *= (0.5 * self.Telescope.ndish * (self.Telescope.ndish-1.)) / sumn


    def renorm_vis(self, z=None): 
        """
        Renormalises the visibility files (read in by read_vis(visfile)) to a given redshift z.

        Parameters
        ----------
        z : float (optional)
            The redshift of the observation. Defaults to None. 
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
        """
        if z is None: # defaults to median redshift if none specified
            z = self.z_med

        u_at_z, nvis_at_z, du_at_z = self.renorm_vis(z) # renorm visibilities to redshift
        
        dusq = 2. * np.pi * u_at_z * du_at_z # 2d differential
        
        lambda_z = lambda_21 * (1.+z)

        sig_T = ((lambda_z**2) * self.T_sys) / (self.Telescope.aeffdish * np.sqrt(self.Telescope.dnu * self.tint * nvis_at_z * dusq))

        sig_T /= np.sqrt(self.Telescope.nbeam * self.Telescope.npol)
        
        return sig_T


    def noise_power_perp(self, z=None):
        """
        Returns the noise power spectrum for the given visibility coverage, as a function of k_perp.

        Parameters
        ----------
        z : float (optional)
            The redshift of the observation. Defaults to None. 
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




class SingleDish(Survey):
    """
    The single-dish class, which can be instantiated to reflect the properties of a given observation.

    Parameters
    ----------
    Telescope : class Telescope
    """
    def __init__(self, Telescope):  
        Survey.__init__(self, Telescope)
        # Determine min and max wavenumber according to beam fwhm amd survey volume
        # min k in perpendicular direction
        comovarcmin = Planck18.kpc_comoving_per_arcmin(self.zmed).value * 1e-3 * hlittle # in Mpc/h 
        asurv_in_arcmin = np.sqrt(self.asurv) * 180. / np.pi * 60. # convert in deg and then in arcmin
        kmin1 = (2. * np.pi) / (comovarcmin * asurv_in_arcmin)

        # min k in line-of-sight direction
        kmin2 = (2. * np.pi) / ((get_distance(self.z_max) - get_distance(self.z_min)) * hlittle) 

        # use the smallest kmin possible
        self.kmin = min(kmin1, kmin2)
        
        # k max is set arbitrary as we convolve with Gaussian beam in pknoise
        self.kmax = 1e3


    def sigmaT_per_pointing(self, z=None): 
        """
        Calculate the deviation in the temperature signal due to the system temperature, for a given telescope pointing.
        In Kelvins. 
        Implementation of Equ.18 from Wolz et al, for single dish, so n(u) -> 1 
        
        Parameters
        ----------
        z : float (optional)
            The redshift of the observation. Defaults to None. 
        """
        if z is None: # defaults to median redshift if none specified
            z = self.z_med
        lambda_z = lambda_21 * (1. + z)
        sig_T = lambda_z**2 * self.T_sys / (self.Telescope.fov_at_z(z) * self.Telescope.aeffdish * np.sqrt(self.Telescope.dnu * self.tint))
        
        # Scale to multiple dishes/beams/polarizations
        sig_T /= np.sqrt(self.Telescope.ndish * self.Telescope.nbeam * self.Telescope.npol)
        
        return sig_T


    def noise_power_perp(self, z=None): 
        """
        Parameters
        ----------
        z : float (optional)
            The redshift of the observation. Defaults to None. 
        """
        if z is None: # defaults to median redshift if none specified
            z = self.z_med
        
        sig_T = self.sigmaT_per_pointing(z)
        vpix =  self.volelement(z)
        
        pnoise = (sig_T**2) * vpix # normalise by sampling volume
        
        return pnoise




def noisetokbin(kperparr, pnoise, karr):
    """
    Determines the noise in each element of an array of k bins.
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


def FT_gals(mock, HII_dim, box_len):
    """
    Calculates the FT and the corresponding wavenumber for the input mock data.
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
        The number of cells along the spatial direction
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