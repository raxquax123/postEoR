""" Analysis functions. Not needed if analysing quantities produced by postEoR, as the classes import the required analysis functions, but here for analysis of other arrays if needed. """

import scipy.stats as stats
import numpy as np
from scipy.integrate import quad
import py21cmfast as p21c
import postEoR.tools as tools
from postEoR.tools import hlittle, OMm, OMl, nu_21
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
import gc
plt.style.use('seaborn-v0_8-ticks')


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


def flatten(x, axis):
    """
    Takes all the cells along a given axis and sums them, essentially flattening the field along one dimension.

    Parameters
    ----------
    x : NDarray
        The field to be flattened.
    axis : int
        The axis along which the field is flattened.

    Returns
    -------
    flat : NDarray
        The flattened field.

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
    binned_fin : NDarray
        The reduced-resolution field.
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
    x : NDarray
        The field whose clustering power spectrum is to be calculated.
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
    k : NumPy array
        The wavenumbers corresponding to the power spectrum.
    ps : NumPy array
        The power spectrum of the input field x.
    err : NumPy array
        The error in each bin of the power spectrum.
    """
    y = x.copy()
    y[y > 0] = 1
    y[y < 0] = 0
    y /= np.mean(y)
    k, ps, err = get_PS(y, box_len, HII_dim, kbins, remove_nan)

    return k, ps, err


def get_2d_ps(x, par_bins=10, perp_bins=10):
        """
        Calculates and plots the cylindrical power spectrum for the input box/cone x.

        Parameters
        ----------
        x : Box object / Cone object
            The object whose cylindrical BT power spectrum is to be calculated.
        par_bins : int (optional)
            The number of wavenumber bins to bin the line-of-sight power into for plotting. Defaults to 10.
        perp_bins : int (optional)
            The number of wavenumber bins to bin the perpendicular power into for plotting. Defaults to 10.
        """
        par_bins+=2
        perp_bins+=2

        BT = x.BT_field
        power0 = np.zeros(np.shape(BT)[:2])
        power2 = np.zeros(np.shape(BT)[2])

        # taking the mean of all the individual perp and parallel ps
        for i in range(np.shape(BT)[0]):
            for j in range(np.shape(BT)[1]):
                power2 += np.abs(np.fft.fft(BT[i, j, :]))
        for i in range(np.shape(BT)[2]):
            power0 += np.abs(np.fft.fftn(BT[:, :, i]))
        power2 /= np.shape(BT)[0] * np.shape(BT)[1]
        power0 /= np.shape(BT)[2]

        n1 = np.size(power0)
        n2 = np.size(power2)
        dims1 = np.shape(BT)[0]
        dims2 = np.shape(BT)[2]

        perp = power0.reshape(n1)
        para = power2.reshape(n2)

        # obtaining the corresponding wavenumbers
        ks1 = np.fft.fftfreq(dims1, x.cell_size) * 2 * np.pi
        kx, ky = np.meshgrid(ks1, ks1) # converting to a 2d array
        k1 = (kx**2+ky**2)**0.5 # perp k-values
        k1 = k1.reshape(np.size(k1)) # converting to 1d array for use in binned_statistic
        kbins1 = np.geomspace(np.min(k1[np.nonzero(k1)]), np.max(k1), perp_bins) # sampling in log space - defining bin edges
        kvals1 = ((kbins1[1:] + kbins1[:-1])) / 2
        k2 = np.abs(np.fft.fftfreq(dims2, x.cell_size) * 2 * np.pi)
        kbins2 = np.geomspace(np.min(k2[np.nonzero(k2)]), np.max(k2), par_bins) # sampling in log space - defining bin edges
        kvals2 = ((kbins2[1:] + kbins2[:-1])) / 2

        Abins1, _, _ = stats.binned_statistic(k1, perp, statistic = "mean", bins = kbins1)
        Abins2, _, _ = stats.binned_statistic(k2, para, statistic = "mean", bins = kbins2)

        # removing values past the nyquist frequency 
        new_k_1 = np.array([y for y in kvals1 if y <= (2*np.pi / (2*x.cell_size))])
        new_k_2 = np.array([y for y in kvals2 if y <= (2*np.pi / (2*x.cell_size))])
        plot1 = Abins1[0:(np.size(new_k_1))] 
        plot2 = Abins2[0:(np.size(new_k_2))] 

        # removing NaN values and adding a k=0 value for plotting
        new_k_1 = new_k_1[~np.isnan(plot1)]
        plot1 = plot1[~np.isnan(plot1)]
        plot1 /= (np.max(new_k_1)**2 * np.pi * 4)
        new_k_2 = new_k_2[~np.isnan(plot2)]
        plot2 = plot2[~np.isnan(plot2)]
        new_k_2 = new_k_2[:-1]
        plot2 = plot2[:-1] # this truncation is just bc some weird sampling things were happening at the final wavenumber - think it was going inside the cell
        plot2 /= (np.max(new_k_2) * 2 * np.pi)

        cross = np.zeros((np.size(plot2-1), np.size(plot1-1))) # calculating the cross power spectrum
        for i in range(np.size(plot1)):
            for j in range(np.size(plot2)):
                cross[j,i] = plot1[i] * plot2[j]

        cb = plt.pcolormesh(np.log10(new_k_1), np.log10(new_k_2), np.log10(cross), cmap = "viridis")
        plt.xlabel("log(k$_{\perp}$, h/Mpc)")
        plt.ylabel("log(k$_\|$, h/Mpc)")
        cbar = plt.colorbar(cb)
        cbar.set_label("log(P$_{cross}$, (Mpc/h)$^3$)")


def len_to_ang(len, z):
    """
    Converts from spatial size to angular size, assuming small angle approximation.

    Parameters
    ----------
    box_len : float
        The spatial size to be converted, in Mpc/h.
    z : float
        The redshift of observation.

    Returns
    -------
    ang : float
        The angular size corresponding to the input spatial size and redshift, in degrees.
    """
    ang = len / get_distance(z) * 180 / np.pi

    return ang


def ang_to_len(ang, z):
    """
    Converts from angular size to spatial size, assuming small angle approximation.

    Parameters
    ----------
    ang : float
        The angular size to be converted, in degrees.
    z : float
        The redshift of observation.

    Returns
    -------
    len : float
        The spatial size corresponding to the input angular size and redshift, in Mpc/h.
    """
    len = np.pi / 180 * get_distance(z) * ang

    return len