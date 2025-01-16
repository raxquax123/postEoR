""" Analysis functions for the fields generated. """

import scipy.stats as stats
import numpy as np
from scipy.integrate import quad
from .constants import *
import py21cmfast as p21c
import generation

def get_PS(x, box_len, HII_dim, kbins=np.asarray([np.nan, np.nan]), remove_nan=True): 
    """
    Calculates the power spectrum for the input field x.

    Parameters
    ----------
    x : NumPy array
        The field whose power spectrum is to be calculated.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc. 
    HII_dim : int
        The number of cells in each of the spatial dimensions of the box / cone.
    kbins : NumPy array (optional)
        The wavenumber bins to use in binning the power. If none / invalid bins are provided, bins will be generated automatically based on the minimum and maximum wavenumber and the number of cells in each dimension.
    remove_nan : bool (optional)
        Whether to remove NaN values from the binned power spectrum (need to keep if calculating the ratio between two power spectra). Defaults to True.

    Returns
    -------
    new_k : NumPy array
        The wavenumbers corresponding to the power spectrum.
    plot1 : NumPy array
        The power spectrum of the input field x.
    error1 : NumPy array
        The error in each bin of the power spectrum.
    """
    n = np.size(x)
    dims = np.shape(x)

    # obtaining k values and k bins to use in ps
    ksx = np.fft.fftfreq(dims[0], (box_len / HII_dim)) * 2 * np.pi # max accessible wavenumber corresponds to 2 * pi
    ksy = np.fft.fftfreq(dims[1], (box_len / HII_dim)) * 2 * np.pi
    ksz = np.fft.fftfreq(dims[2], (box_len / HII_dim)) * 2 * np.pi
    kx, ky, kz = np.meshgrid(ksx, ksy, ksz) # converting to a 3d array
    k = (kx**2+ky**2+kz**2)**0.5 # spherical k-values
    k = k.reshape(np.size(k)) # converting to 1d array for use in binned_statistic

    try: # check for input bins, and generate if none / invalid provided
        if np.isnan(kbins.any()):
            kbins = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), HII_dim//2+1) # sampling in log space - defining bin edges
        else:
            print("Using input bins")
    except AttributeError:
        print("Incorrect bin type. Generating bins")
        kbins = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), HII_dim//2+1)

    kvals = ((kbins[1:] + kbins[:-1])) / 2
    power1 = np.abs(np.fft.fftn(x))**2 # computing fft of field and taking absolute squared values
    ps1 = power1.reshape(np.size(power1)) / (n * (2 * np.pi * HII_dim / box_len)) # converting to 1d array | normalise by sampling volume

    bin_count1, _, _ = stats.binned_statistic(k, ps1, statistic="count", bins=kbins) # obtaining number of data points in each bin
    Abins1, _, _ = stats.binned_statistic(k, ps1, statistic = "mean", bins = kbins) # binning power
    error1, _, _ = stats.binned_statistic(k, ps1, statistic = "std", bins = kbins) # obtaining standard deviation in each bin

    new_k = np.array([x for x in kvals if x <= (2*np.pi / (2*box_len / HII_dim))]) # removing values above the nyquist frequency (corresponds to sampling inside the cells)
    plot1 = Abins1[0:(np.size(new_k))]
    error1 = error1[0:(np.size(new_k))] / (bin_count1[0:(np.size(new_k))])**0.5

    if remove_nan:
        new_k = new_k[~np.isnan(plot1)]
        error1 = error1[~np.isnan(plot1)]
        plot1 = plot1[~np.isnan(plot1)]

    return new_k, plot1, error1


def get_dimless_PS(x, HII_dim, box_len, kbins=np.asarray([np.nan, np.nan]), remove_nan=True): 
    """
    Calculates the dimensionless power spectrum for the input field x.

    Parameters
    ----------
    x : NumPy array
        The field whose dimensionless power spectrum is to be calculated.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc. 
    HII_dim : int
        The number of cells in each of the spatial dimensions of the box / cone.
    kbins : NumPy array (optional)
        The wavenumber bins to use in binning the power. If none / invalid bins are provided, bins will be generated automatically based on the minimum and maximum wavenumber and the number of cells in each dimension.
    remove_nan : bool (optional)
        Whether to remove NaN values from the binned power spectrum (need to keep if calculating the ratio between two power spectra). Defaults to True.

    Returns
    -------
    new_k : NumPy array
        The wavenumbers corresponding to the dimensionless power spectrum.
    plot1 : NumPy array
        The dimensionless power spectrum of the input field x.
    error1 : NumPy array
        The error in each bin of the dimensionless power spectrum.
    """
    n = np.size(x)
    dims = np.shape(x)

    # obtaining k values and k bins to use in ps
    ksx = np.fft.fftfreq(dims[0], (box_len / HII_dim)) * 2 * np.pi # max accessible wavenumber corresponds to 2 * pi
    ksy = np.fft.fftfreq(dims[1], (box_len / HII_dim)) * 2 * np.pi
    ksz = np.fft.fftfreq(dims[2], (box_len / HII_dim)) * 2 * np.pi
    kx, ky, kz = np.meshgrid(ksx, ksy, ksz) # converting to a 3d array
    k = (kx**2+ky**2+kz**2)**0.5 # spherical k-values
    k = k.reshape(np.size(k)) # converting to 1d array for use in binned_statistic

    try: # check for input bins, and generate if none / invalid provided
        if np.isnan(kbins.any()):
            kbins = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), HII_dim//2+1) # sampling in log space - defining bin edges
        else:
            print("Using input bins")
    except AttributeError:
        kbins = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), HII_dim//2+1)
    
    kvals = ((kbins[1:] + kbins[:-1])) / 2
    power1 = np.abs(np.fft.fftn(x))**2 # computing fft of field and taking absolute squared values
    ps1 = power1.reshape(np.size(power1)) / (n * (2 * np.pi * HII_dim / box_len)) # converting to 1d array | normalise by sampling volume

    bin_count1, _, _ = (stats.binned_statistic(k, ps1, statistic="count", bins=kbins)) # obtaining number of data points in each bin
    Abins1, _, _ = stats.binned_statistic(k, ps1, statistic = "mean", bins = kbins) # binning power
    error1, _, _ = stats.binned_statistic(k, ps1, statistic = "std", bins = kbins) # obtaining standard deviation in each bin

    new_k = np.array([x for x in kvals if x <= (2*np.pi / (2*box_len / HII_dim))]) # removing values above the nyquist frequency (corresponds to sampling inside the cells)
    plot1 = Abins1[0:(np.size(new_k))] 
    error1 = error1[0:(np.size(new_k))] / (bin_count1[0:(np.size(new_k))])**0.5 

    if remove_nan:
        new_k = new_k[~np.isnan(plot1)]
        error1 = error1[~np.isnan(plot1)] 
        plot1 = plot1[~np.isnan(plot1)]
    
    plot1 *= new_k**3 / (2 * np.pi**2) 
    error1 *= new_k**3 / (2 * np.pi**2) 

    return new_k, plot1, error1

def get_distance(z_start, z_end=0):
    """
    Calculates distance corresponding to a given redshift interval, in units of Mpc. Assumes a flat ΛCDM cosmology.

    Parameters
    ----------
    z_start : float
        The maximum redshift of the distance interval calculated.
    z_end : float (optional)
        The minimum redshift of the distance interval calculated. Defaults to 0.

    Returns
    -------
    dist : float
        The distance corresponding to the input redshift interval, in Mpc.
    """
    c_km = c / 1000
    dx = lambda z: 1 / (H_0*(omega_m*(1+z)**3+omega_lambda)**0.5)
    dist = quad(dx, z_end, z_start)
    dist = dist[0] * c_km

    return dist


def gen_hmf(z, HII_dim, box_len, set_bins=True):
    """
    Generates a halo mass function at a given redshift. Used for testing of the halo finder.

    Parameters
    ----------
    z : float
        The redshift at which to calculate the HMF.
    HII_dim : int
        The number of cells in each of the spatial dimensions of the box / cone.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc. 
    set_bins : bool (optional)
        Whether to use the set bins (instead of the bins generated by np.histogram). Default is True.

    Returns
    -------
    bins : NumPy array
        The mass bins used to create the HMF.
    counts : NumPy array
        The number of halos falling in each halo mass bin.
    los_dist : float
        The physical distance corresponding to the redshift interval the lightcone is generated over, used for some plots.
    """
    z_min = z - 0.2
    z_max = z + 0.2
    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=z_min,
        max_redshift=z_max,
        quantities=('brightness_temp', 'density', 'velocity_z'),
        resolution=user_params.cell_size,
        get_los_velocity = True,
        # index_offset=0,
    )

    # run lightcone using 21cmFAST functionality
    user_params = p21c.UserParams(
    HII_DIM=HII_dim, BOX_LEN=box_len, KEEP_3D_VELOCITIES=True, USE_2LPT=True, HMF=3, 
    )
    lightcone = p21c.run_lightcone(
        lightconer=lcn,
        global_quantities=("brightness_temp", "density", 'xH_box'),
        direc='_cache',
        user_params=user_params,
        random_seed=1122
    )
    dens_ltcone = getattr(lightcone, "density") # getting density for post-processing
    
    halos_ltcone = generation.find_halos(dens_ltcone, overdens_cap=1.686) # find the halos on the lightcone. 
    bins1 = np.geomspace(1e+10, 1e+13, 10)
    if set_bins:
        counts, bins = np.histogram(halos_ltcone, bins1)
    else:
        counts, bins = np.histogram(halos_ltcone)
    los_dist = get_distance(z_max, z_min) # used for plotting

    return bins, counts, los_dist


def gen_himf(z, HII_dim, box_len, bins):
    """
    Generates a HI mass function at a given redshift. Used for testing of the halo finder and HI-halo mass relation.

    z : float
        The redshift at which to calculate the HIMF.
    HII_dim : int
        The number of cells in each of the spatial dimensions of the box / cone.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc. 
    set_bins : bool (optional)
        Whether to use the set bins (instead of the bins generated by np.histogram). Default is True.

    Returns
    -------
    bins : NumPy array
        The mass bins used to create the HIMF.
    counts : NumPy array
        The number of halos falling in each HI mass bin.
    los_dist : float
        The physical distance corresponding to the redshift interval the lightcone is generated over, used for some plots.
    """
    z_min = z - 0.2
    z_max = z + 0.2
    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=z_min,
        max_redshift=z_max,
        quantities=('brightness_temp', 'density', 'velocity_z'),
        resolution=user_params.cell_size,
        get_los_velocity = True,
        # index_offset=0,
    )

    # run lightcone using 21cmFAST functionality
    user_params = p21c.UserParams(
    HII_DIM=HII_dim, BOX_LEN=box_len, KEEP_3D_VELOCITIES=True, USE_2LPT=True, HMF=3, 
    )
    lightcone = p21c.run_lightcone(
        lightconer=lcn,
        global_quantities=("brightness_temp", "density", 'xH_box'),
        direc='_cache',
        user_params=user_params,
        random_seed=1122
    )
    dens_ltcone = getattr(lightcone, "density") # getting density for post-processing

    halos_ltcone = generation.find_halos(dens_ltcone, overdens_cap=1.686) # find the halos on the lightcone. 
    mean_z = (z_max+z_min) / 2
    hi_ltcone = generation.hi_from_halos_2(halos_ltcone, mean_z)
    bins = np.geomspace(1e+10, 1e+12, 10)
    counts, bins = np.histogram(hi_ltcone)
    los_dist = get_distance(z_max, z_min)

    return bins, counts, los_dist