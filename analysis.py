""" Analysis functions. Not needed if analysing quantities produced by postEoR, as the classes import the required analysis functions, but here for analysis of other arrays if needed. """

import scipy.stats as stats
import numpy as np
from scipy.integrate import quad
import py21cmfast as p21c
import postEoR.tools as tools
from postEoR.tools import hlittle, OMm, OMl
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})


def get_PS(x, box_len, HII_dim, kbins=None, remove_nan=True): 
    """
    Calculates the power spectrum for the input field x.

    Parameters
    ----------
    x : NumPy array
        The field whose power spectrum is to be calculated.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc/h. 
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
        if kbins is None:
            kbins = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), HII_dim//2+1) # sampling in log space - defining bin edges
            print("Generated bins.")
        else:
            print("Using input bins")
    except AttributeError:
        print("Incorrect bin type. Generating bins")
        kbins = np.geomspace(np.min(k[np.nonzero(k)]), np.max(k), HII_dim//2+1)

    kvals = ((kbins[1:] + kbins[:-1])) / 2
    power1 = np.abs(np.fft.fftn(x))**2 # computing fft of field and taking absolute squared values
    ps1 = power1.reshape(np.size(power1)) / (n * (HII_dim / box_len)**3) # converting to 1d array | normalise by sampling volume

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


def get_dimless_PS(x, box_len, HII_dim:int, kbins=None, remove_nan=True): 
    """
    Calculates the dimensionless power spectrum for the input field x.

    Parameters
    ----------
    x : NumPy array
        The field whose dimensionless power spectrum is to be calculated.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc/h. 
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
    new_k, plot1, error1 = get_PS(x, box_len, HII_dim, kbins, remove_nan)
    
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
        The distance corresponding to the input redshift interval, in Mpc/h.
    """
    c_km = 299792458 / 1000
    H_0 = hlittle * 100
    dx = lambda z: 1 / (H_0*(OMm*(1+z)**3+OMl)**0.5)
    dist = quad(dx, z_end, z_start)
    dist = dist[0] * c_km * hlittle

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
        The physical length of each of the spatial dimensions of the box / cone, in Mpc/h. 
    set_bins : bool (optional)
        Whether to use the set bins (instead of the bins generated by np.histogram). Default is True.

    Returns
    -------
    bins : NumPy array
        The mass bins used to create the HMF.
    counts : NumPy array
        The number of halos falling in each halo mass bin.
    los_dist : float
        The physical distance corresponding to the redshift interval the lightcone is generated over, used for some plots, in Mpc/h.
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
    
    halos_ltcone = tools.find_halos(dens_ltcone, overdens_cap=1.686) # find the halos on the lightcone. 
    if set_bins:
        bins1 = np.geomspace(1e+10, 1e+13, 10)
        counts, bins = np.histogram(halos_ltcone, bins1)
    else:
        counts, bins = np.histogram(halos_ltcone)
    los_dist = get_distance(z_max, z_min) # used for plotting

    return bins, counts, los_dist


def gen_himf(z, HII_dim, box_len, set_bins=True):
    """
    Generates a HI mass function at a given redshift. Used for testing of the halo finder and HI-halo mass relation.

    Parameters
    ----------
    z : float
        The redshift at which to calculate the HIMF.
    HII_dim : int
        The number of cells in each of the spatial dimensions of the box / cone.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc/h. 
    set_bins : bool (optional)
        Whether to use the set bins (instead of the bins generated by np.histogram). Default is True.

    Returns
    -------
    bins : NumPy array
        The mass bins used to create the HIMF.
    counts : NumPy array
        The number of halos falling in each HI mass bin.
    los_dist : float
        The physical distance corresponding to the redshift interval the lightcone is generated over, used for some plots, in Mpc/h.
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

    halos_ltcone = tools.find_halos(dens_ltcone, overdens_cap=1.686) # find the halos on the lightcone. 
    mean_z = (z_max+z_min) / 2
    hi_ltcone = tools.hi_from_halos_2(halos_ltcone, mean_z)
    if set_bins:
        bins1 = np.geomspace(1e+10, 1e+13, 10)
        counts, bins = np.histogram(hi_ltcone, bins1)
    else:
        counts, bins = np.histogram(hi_ltcone)
    los_dist = get_distance(z_max, z_min)

    return bins, counts, los_dist


def flatten(x, axis): # takes all the values along a given axis and sums them (essentially flattening along one dimension)
    """
    
    """
    dims = np.shape(x)
    if axis == 0:
        flat = np.zeros([dims[1], dims[2]])
        for i in range(dims[1]):
            for j in range(dims[2]):
                for k in range(dims[0]):
                    flat[i, j] += x[k, i, j]
    elif axis == 1:
        flat = np.zeros([dims[0], dims[2]])
        for i in range(dims[0]):
            for j in range(dims[2]):
                for k in range(dims[1]):
                    flat[i, j] += x[i, k, j]
    else: # assumes flattening along redshift direction, if axis not specified or invalid
        flat = np.zeros([dims[0], dims[1]])
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    flat[i, j] += x[i, j ,k]
    return flat


def reduce_res(reduction_factor, object):
    """
    Reduces the resolution of a field by the specified factor.

    Parameters
    ----------
    reduction_factor : float
        The factor by which the resolution of the field is to be reduced.
    object : NDarray
        The field whose resolution is to be reduced.

    Returns
    -------
    """
    if reduction_factor >= object.HII_dim:
        print("Reduction factor too large.")
        return 

    dx, dy = object.cell_size, object.cell_size 
    y1_new, x1_new = np.mgrid[slice(dy * reduction_factor / 2, object.box_len, dy*reduction_factor), slice(dx*reduction_factor/ 2, object.box_len, dx*reduction_factor)]
    y1p, x1p = np.mgrid[slice(dy / 2, object.box_len, dy), slice(dx / 2, object.box_len, dx)]

    xbins = np.linspace(0, object.HII_dim, int(object.HII_dim / reduction_factor))
    ybins = np.linspace(0, object.HII_dim, int(object.HII_dim / reduction_factor))
    x1 = np.linspace(0, object.HII_dim, object.HII_dim+1)
    y1 = np.linspace(0, object.HII_dim, object.HII_dim+1)
    x1 = (x1[1:]+x1[:-1])/2
    y1 = (y1[1:]+y1[:-1])/2

    new_HII_dim = np.size(x1_new[0]) - 1

    binned_1 = np.zeros([new_HII_dim, object.HII_dim])
    binned_fin = np.zeros([new_HII_dim, new_HII_dim])

    two_d_BT = flatten(object.BT_field[:, :, :5], 2)

    for i in range(object.HII_dim):
        ith_row = np.array(two_d_BT[:, i])
        Abins1, _, _ = stats.binned_statistic(x1, ith_row, statistic = "mean", bins = xbins) 
        binned_1[:, i] = Abins1
    for i in range(new_HII_dim):
        ith_row = np.array(binned_1[i, :])
        Abins1, _, _ = stats.binned_statistic(y1, ith_row, statistic = "mean", bins = ybins) 
        binned_fin[i, :] = Abins1

    plt.rcParams['figure.figsize'] = [16, 5.5]
    fig, (ax3, ax4) = plt.subplots(1, 2)

    cb3 = ax3.pcolormesh(x1_new, y1_new, binned_fin, cmap = "viridis")
    ax3.set_xlabel('$x$ (Mpc)')
    cbar3 = fig.colorbar(cb3)
    cbar3.set_label('Brightness temperature, mK', rotation=270, labelpad = 12)
    cbar3.formatter.set_powerlimits((0, 0))
    cb4 = ax4.pcolormesh(x1p, y1p, two_d_BT, cmap = "viridis")
    ax4.set_xlabel('$x$ (Mpc)')
    cbar4 = fig.colorbar(cb4)
    cbar4.set_label('Brightness temperature, mK', rotation=270, labelpad = 12)
    cbar4.formatter.set_powerlimits((0, 0))

    return binned_fin


def get_clustering_ps(x, box_len, HII_dim, kbins=None, remove_nan=True):
    """
    Calculates the clustering power spectrum of the input field x.

    Parameters
    ----------

    """
    y = x.copy()
    y[y>0] = 1
    y[y<0] = 0
    y /= np.mean(y)
    k, ps, err = get_PS(y, box_len, HII_dim, kbins, remove_nan)

    return k, ps, err