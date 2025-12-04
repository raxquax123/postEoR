""" Functions to be used within generating the coeval boxes and lightcones. """

import numpy as np
from scipy.integrate import quad
import scipy.ndimage as ndimage
from skimage.segmentation import watershed
import py21cmfast as p21c

""" Defining / importing parameters used. """
""" Defining the cosmology used. Here, using Planck18. """
OMm = 0.30964144154550644
OMb = 0.04897468161869667
hlittle = 67.66 / 100
cosmo_params = p21c.CosmoParams.new(hlittle=hlittle, OMm=OMm, OMb=OMb) # adding cosmology used to 21cmFAST.
OMl = 1 - OMm # assuming a flat cosmology - as is done by 21cmFAST.

""" Defining physical constants. """
c = 299792458 # speed of light, in m/s
k_B = 1.380649e-23 # Boltzmann constant, in J/K
T_CMB = 2.7255 # CMB temperature, in K
A_10 = 2.86888e-15 # einstein coefficient for hi spin-flip transition
m_H = 1.6735e-27 # mass of hydrogen atom
nu_21 = 1420.405751768 * 10**6 # frequency of hi spin-flip transition, in Hz
h = 6.63e-34 # Planck's constant
solar_mass = 1.989 * 10**30 # mass of the sun in kg
Mpc_to_m = 3.0857e+22 # conversion from Mpc to m
G = 6.67430e-11 # gravitational constant, in N m^2 / kg^2
hydrogen_baryon_frac = 0.75


def get_delta_vir(z): 
    """
    Calculates the mean halo overdensity within the virial radius at a given redshift.

    Parameters
    ----------
    z : float
        The redshift at which the mean overdensity will be calculated.

    Returns
    -------
    delta_vir : float
        The mean halo overdensity within the virial radius at the input redshift. Dimensionless.
    """
    x = OMm * (1+z)**3 / (OMm*(1+z)**3 + OMl) - 1
    delta_vir = 18 * np.pi**2 + 82 * x - 39 * x**2 # mean overdensity within virial radius formula

    return delta_vir


def get_vc(M, z):
    """
    Calculates the virial velocity of an input object of mass M at redshift z.

    Parameters
    ----------
    M : float
        The mass of the object, in solar masses.
    z : float
        The redshift at which to evaluate the virial velocity.

    Returns
    -------
    vc : float
        The virial velocity of the object, in km/s.
    """
    vc = np.array(163 * (M / 10**12)**(1/3) * (get_delta_vir(z) / 200)**(1/6) * OMm**(1/6) * (1+z)**0.5)  # virial velocity formula
    vc[vc < 0.01] = 0.01 # avoid dividing by zero errors later on.

    return vc


def get_r_vir(M, z): 
    """
    Calculates the virial radius of an input object of mass M at redshift z.

    Parameters
    ----------
    M : float
        The mass of the object, in solar masses/h.
    z : float
        The redshift at which to evaluate the virial radius.

    Returns
    -------
    r : float
        The virial radius of the object, in Mpc/h.
    """
    v_c = get_vc(M, z) * 1000 # convert to m/s from km/s
    M *= solar_mass # convert to kg from solar masses
    r = G * M / v_c**2
    r /= Mpc_to_m # convert from m to Mpc

    return r 


def get_conc(M, z, c_HI=28.65, gamma=1.45): # returns halo concentration (dimensionless)
    """
    Calculates the halo concentration (dimensionless) of a halo of mass M, at redshift z.

    Parameters
    ----------
    M : float
        The mass of the object, in solar masses/h.
    z : float
        The redshift at which to evaluate the halo concentration.
    c_HI : float (optional)
        The HI concentration parameter. Defaults to 28.65.
    gamma : float (optional)


    Returns
    -------
    conc : float
        The halo concentration of the halo (dimensionless).
    
    """

    conc = c_HI * (M /(hlittle * 10**11))**(-0.109) * 4 / (1+z)**gamma 

    return conc


def get_rho_0(M, z): # calculates the rho_0 constant for a given modified NFW profile
    """
    Calculates the rho_0 constant for a given modified NFW profile.

    Parameters
    ----------
    M : float
        The mass of the halo, in solar masses.
    z : float
        The redshift at which to evaluate rho_0 at.

    Returns
    -------
    rho : float
        The rho_0 constant for the input halo, in solar masses * hlittle^2 / Mpc^3.
    """
    r_vir = get_r_vir(M, z) # obtaining virial radius in Mpc/h
    r_s = r_vir / get_conc(M, z) # obtaining scale radius in Mpc/h

    dx = lambda r: r_s**3 / ((r+0.75 * r_s)*(r+r_s)**2) * 4 * np.pi * r**2 # defining integrand
    integ, _ = quad(dx, r_vir, 0)
    rho = M / integ 

    return rho


def obtain_HI_at_dist(M, z, r_end, r_start, HI_model=3): 
    """
    Calculates the total HI mass over a given radius range from a halo of mass M, in solar masses/h.

    Parameters
    ----------
    M : float
        The mass of the central halo, in solar masses/h.
    z : float
        The redshift at which to evaluate the HI mass at.
    r_end : float
        The maximum radius of the shell in consideration, in Mpc/h.
    r_start : float
        The minimum radius of the shell in consideration, in Mpc/h.
    HI_model : int (optional)
        Which analytic HI-halo relation to use. 1 corresponds to the simulation-based model from Spinelli et al. 2020, 2 is an observational-based model from Padmanabhan and Refegier (2017), and 3 is an observational-based model from Padamanabhan, Refregier, and Amara (2017). Defaults to 3.

    Returns
    -------
    shell_mass : float
        The total HI mass contained within the shell, in solar masses/h.
    """
    if HI_model==1:
        M_hi = hi_from_halos_1(M, z)
        c_HI = 28.65
        gamma = 1.45
    elif HI_model==2:
        M_hi = hi_from_halos_2(M, z)
        c_HI = 113.80
        gamma = 0.22
    else:
        M_hi = hi_from_halos_3(M, z)
        c_HI = 28.65
        gamma = 1.45
    r_s = get_r_vir(M, z) / get_conc(M, z, c_HI, gamma) # obtaining scale radius

    integrand = lambda r: get_rho_0(M_hi, z) * r_s**3 / ((r + 0.75*r_s)*(r+r_s)**2) * 4 * np.pi * r**2 # defining integrand
    shell_mass, _ = quad(integrand, r_end, r_start) # obtaining hi mass contained within a shell corresponding to specified radius range
    shell_mass = abs(shell_mass)

    return shell_mass


def hi_from_halos_2(halo_field, z): 
    """
    HI mass-halo mass relation from Padmanabhan and Refregier (2017).

    Parameters
    ----------
    halo_field : NDarray
        The distribution of halo masses across the field, in solar masses/h .
    z : float
        The redshift at which to evaluate the quantity.

    Returns
    -------
    xHI_mass : NDarray
        The total neutral hydrogen mass associated with each of the halos in the field, in solar masses/h.
    """
    f_H = OMb * hydrogen_baryon_frac / OMm
    alpha = 0.17
    v_c0 = 10**1.57 # potentially edit limits to match observations 
    v_c1 = 10**4.39
    beta = -0.55
    v_c = get_vc(halo_field, z)

    xHI_mass = alpha * f_H * halo_field**(1+beta) * (10**-11)**beta * np.exp(-(v_c0 / v_c)**3) * np.exp(-(v_c / v_c1)**3)

    return xHI_mass


def bin_halos(halo_field, no_bins=25): # collecting halos into mass bins
    """
    Collects the halos within an input halo field into equally spaced mass bins.

    Parameters
    ----------
    halo_field : NDarray
        The distribution of halo masses across the field, in solar masses/h.
    no_bins : int (optional)
        The number of mass bins into which the halos are collected. Defaults to 25.

    Returns
    -------
    binned_halo_field : NDarray
        The halo field, arranged into mass bins, in solar masses/h.
    bins : NDarray
        The bin edges that define the mass bins, in solar masses/h.
    """
    binned_halo_field = np.zeros([no_bins, np.shape(halo_field)[0], np.shape(halo_field)[1], np.shape(halo_field)[2]])

    bins = np.linspace((np.min(halo_field)-1), (np.max(halo_field)+1), (no_bins+1))

    for i in range(no_bins):
        in_bin = np.multiply((halo_field > bins[i]).astype(int), (halo_field < bins[i+1]).astype(int))
        binned_halo_field[i] = np.multiply(halo_field, in_bin)

    return binned_halo_field, bins


def get_HI_field(halo_field, z, cell_size, no_bins=25, max_rad=1, HI_model=3):
    """
    Convolves the HI profiles with the halo locations to produce the complete final HI mass field.

    Parameters
    ----------
    halo_field : NDarray
        The distribution of halo masses across the field, in solar masses/h.
    z : float
        The redshift at which the field is evaluated.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc/h.
    HII_dim: int
        The number of cells in each of the spatial dimensions of the box / cone.
    no_bins: int (optional)
        The number of mass bins into which the halos are collected. Defaults to 25.
    max_rad: float (optional)
        The maximum radius out to which the spherical HI profile is evaluated, in Mpc/h. Defaults to 1 Mpc/h.
    HI_model : int (optional)
        Which analytic HI-halo relation to use. 1 corresponds to the simulation-based model from Spinelli et al. 2020, 2 is an observational-based model from Padmanabhan and Refegier (2017), and 3 is an observational-based model from Padamanabhan, Refregier, and Amara (2017). Defaults to 3.

    Returns
    -------
    final_HI_field : NDarray
        The distribution of HI mass across the field, in solar masses/h.
    """
    binned_halos, bins = bin_halos(halo_field, no_bins)
    M_centr = (bins[1:]+bins[:-1]) / 2
    final_HI_field = np.zeros(np.shape(halo_field))

    for i in range(no_bins):
        spherical_profile = create_spherical_profile(M_centr[i], z, cell_size, max_rad, HI_model)
        halo_pos = (binned_halos[i] > 0.01).astype(int)
        print("abt to convolve")
        final_HI_field += ndimage.convolve(halo_pos, spherical_profile)

    return final_HI_field


def create_spherical_profile(halo, z, cell_size, max_rad=1, HI_model=3): 
    """
    Calculates the radial distribution of HI surrounding a halo of given mass.

    Parameters
    ----------
    halo : float
        The mass of the halo whose HI profile is being calculated.
    z : float
        The redshift at which the profile is being calculated.
    box_len : float
        The size in Mpc/h of the full box/spatial dimensions of cone.
    HII_dim: int
        The number of cells in each dimension of the full box/spatial dimensions of cone.
    max_rad: float (optional)
        The maximum radius out to which the profile will be evaluated. Defaults to 1 Mpc/h.
    HI_model : int (optional)
        Which analytic HI-halo relation to use. 1 corresponds to the simulation-based model from Spinelli et al. 2020, 2 is an observational-based model from Padmanabhan and Refegier (2017), and 3 is an observational-based model from Padamanabhan, Refregier, and Amara (2017). Defaults to 3.

    Returns
    -------
    sph_prof_fin : NDarray
        The HI distribution around the input halo.
    """
    centre_ind = int(max_rad / cell_size) # array index that will correspond to the centre of the profile
    prof_size = centre_ind * 2 + 1
    sph_prof = np.zeros([prof_size, prof_size, prof_size])
    max_rad_fin = cell_size * (np.ceil(max_rad / cell_size-0.5)+0.5)

    r_set = np.asarray(np.linspace((cell_size / 2), max_rad_fin, centre_ind+1)) # radius bins to integrate over, in Mpc/h
    r_set = np.insert(r_set, 0, 0)

    for i in range(prof_size):
        for j in range(prof_size):
            for k in range(prof_size):
                dist_from_cent_i = abs(centre_ind - i)
                dist_from_cent_j = abs(centre_ind - j)
                dist_from_cent_k = abs(centre_ind - k)
                count_at_dist = ((dist_from_cent_i+1)**3+(dist_from_cent_j+1)**3+(dist_from_cent_k+1)**3) / 3 # calculating how many 'similar' cells there are, to distribute shell mass amongst them
                r_max = (r_set[dist_from_cent_i + 1]**2+r_set[dist_from_cent_j + 1]**2+r_set[dist_from_cent_k + 1]**2)**0.5
                r_min = (r_set[dist_from_cent_i]**2+r_set[dist_from_cent_j]**2+r_set[dist_from_cent_k]**2)**0.5
                sph_prof[i, j, k] = obtain_HI_at_dist(halo, z, r_max, r_min, HI_model) / count_at_dist
    if HI_model==1:
        sph_prof_fin = sph_prof * hi_from_halos_1(halo, z) / np.sum(sph_prof)
    elif HI_model==2:
        sph_prof_fin = sph_prof * hi_from_halos_2(halo, z) / np.sum(sph_prof)
    else:
        sph_prof_fin = sph_prof * hi_from_halos_3(halo, z) / np.sum(sph_prof)

    return sph_prof_fin


def find_halos_watershed(dens, cell_size, overdens_cap=None, connectivity=1, compactness=0, normalise=False, smooth=False, sigma=1):
    """
    Returns mass and centre position of halos in solar masses. 
    Stops at either no change after algorithm applied, or when maximum number of iterations has been reached.

    Parameters
    ----------
    overdensity_field : NDarray
        The overdensity field on which the halos are to be found.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc/h. 
    HII_dim : int
        The number of cells in each of the spatial dimensions of the box / cone.
    overdens_cap : float (optional)
        The minimum overdensity for which a cell is considered to be associated with a halo. Defaults to None (applies simple linear approximation to optimal overdens_cap based on matching to theoretical HMF).
    connectivity : float (optional)
        The neighbour connectivity parameter used in the watershed algorithm. Defaults to 1 (3 is maximum - going beyond this doesn't change the halos).
    compactness : float (optional)
        The object compactness parameter used in the watershed algorithm. Defaults to 1.
    normalise : bool (optional)
        Whether or not to approximate halo merging by normalising the total halo mass based on the overdensity cap (to account for otherwise 'lost'/unresolved halos). Defaults to False.
    smooth : bool (optional)
        Whether or not to apply Gaussian smoothing to the overdensity field before halo finding on it (here for testing purposes). Defaults to False.
    sigma : float (optional)
        The degree of Gaussian smoothing applied to the overdensity field before halo finding on it. Only functional if smooth=True. Defaults to 1.
        
    Returns 
    -------
    halo_field : NDarray
        The distribution of halo masses across the field, in solar masses/h.
    """
    print("poo")
    if overdens_cap is None:
        #overdens_cap = -54.7 * box_len / HII_dim + 16.75 # simple linear approximation to the optimal overdens cap to match hmf to theoretical
        overdens_cap = 8 * (0.16 / cell_size)
        print("Optimal overdensity cap used is " + str(overdens_cap))

    image = dens.copy()
    image[dens < overdens_cap] = 0

    if smooth:
        image = ndimage.gaussian_filter(image, sigma=(sigma, sigma, sigma), order=0)

    labels = watershed(-image, connectivity=connectivity, compactness=compactness) 
    print(np.max(labels)) # prints total number of halos, to keep track of mass allocation progress
    dims = np.shape(dens)

    halo_field = np.zeros([dims[0], dims[1], dims[2]], dtype=np.float64)

    H_0_std_units = (hlittle * 100 * 1000) / (Mpc_to_m)
    z_comov = 0
    H = H_0_std_units * (OMm*(1+z_comov)**3 + OMl) ** 0.5
    crit_M_dens = (3 * H ** 2) / (8 * np.pi * G) * (OMm * (1+z_comov)**3) / (OMm * (1+z_comov)**3 + OMl) / hlittle**2 # using the critical density at a set redshift as the simulation is comoving. h-agnostic
    print("crit dens " + str(crit_M_dens))

    new_overdensity_field = dens.copy()
    new_overdensity_field = new_overdensity_field.astype(np.float64)
    mass_field = (1 + new_overdensity_field) * crit_M_dens * (cell_size * Mpc_to_m)**3 / (solar_mass) * (1 / (1 + np.mean(dens))) * (OMm-OMb)/OMm

    mass_field[dens < overdens_cap] = 0 # removing underdense regions
    try:
        print("min mass of cell before halos " + str(np.min(mass_field[np.nonzero(mass_field)])))

        for i in range(np.max(labels)):
            current_halo = np.multiply((labels == (i+1)).astype(int), mass_field)
            halo_centre = np.unravel_index(np.argmax(current_halo, axis=None), current_halo.shape)
            halo_mass = np.sum(current_halo)
            halo_field[halo_centre] += halo_mass
            print(int(i+1), end="\r")

        if normalise:
            halo_field *= np.max([1., overdens_cap])
            bl_halos = (crit_M_dens * (cell_size * Mpc_to_m)**3 * np.size(dens) / hlittle**2) / solar_mass * (OMm-OMb) / OMm # baseline total halo mass - to add mass that has been removed from small halos to larger halos. division by h^2 is to account for the H^2 in calculating the critical mass density.
            if np.sum(halo_field) >= bl_halos:
                halo_field *= bl_halos / np.sum(halo_field) # normalise total mass to the baseline mass

        halo_field[halo_field < 10**7] = 0 # removing small mass regions that are unlikely to be halos
        
    except ValueError:
        print("No halos found.")
        halo_field = np.zeros(dims)
    
    return halo_field


def get_vel_grad(vel, HII_dim, box_len, axis=2): 
    """
    Calculate the velocity gradient along a given axis. Used in calculating the effects of RSDs.

    Parameters
    ----------
    vel : NDarray
        The velocity field output by 21cmFAST.
    HII_dim : int
        The number of cells along the spatial directions of the field.
    box_len : float
        The physical length along the spatial dimensions of the field, in Mpc/h.
    axis : int (optional)
        The axis along which the velocity gradient is to be calculated. Valid options are 0, 1, 2. Defaults to 2 (generally the LoS axis).

    Returns
    -------
    grad : NDarray
        The velocity gradient field.
    """
    dv = np.empty((HII_dim, HII_dim, HII_dim))
    dr = box_len / HII_dim
    dv = np.roll(vel, 1, int(axis)) - vel
    grad = dv / dr

    return grad


def bin_freq_to_z(z_start, z_end, bin_width):
        """
        Converts a given frequency channel width into redshift bins.

        Parameters
        ----------
        z_start : float
            The highest redshift of the lightcone.
        z_end : float
            The lowest redshift of the lightcone.
        bin_width : float
            The frequency channel width in Hz.

        Returns
        -------
        z_bins : NDarray
            The redshift bins corresponding to the given channel width, starting from the highest redshift of the lightcone.
        """
        freq_start = nu_21 / (1 + z_start)
        freq_end = nu_21 / (1 + z_end)
        bin_edge=freq_start
        freq_bins = list([])
        while bin_edge <= freq_end:
            freq_bins.append(bin_edge)
            bin_edge += bin_width
        z_bins = nu_21 / np.asarray(freq_bins) - 1

        return z_bins


def z_to_index(cone, bin_width):
    """
    Calculates the indices corresponding to the cone being divided into redshift bins of a specified width.

    Parameters
    ----------
    cone : Ltcone object
        The cone being divided.
    bin_width : float
        The redshift bin width to divide the cone into.
    
    Returns
    -------
    indices : NDarray
        The indices corresponding to the edges of the redshift bins that the lightcone has been divided into.
    """
    bin_edge=cone.z_end
    z_bins=list([])

    while bin_edge <= cone.z_start:
        z_bins.append(bin_edge)
        bin_edge += bin_width

    dist_bins = list([])
    for i in range(len(z_bins)):
        dist_bins.append(get_distance(z_bins[i]))
    start_dist = get_distance(cone.z_end)
    indices = np.abs(np.round((np.asarray(dist_bins) - start_dist) / cone.cell_size, 0))

    return indices


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


# defining the functions used to get the model parameters for the HI-halo mass relation from spinelli et al. 2020. valid for 5 >= z >= 3
def get_a1(z):
    x = 3.7863578022788724e-02 - 1.1398128309436184e-01*z + 1.1924070362185538e-01*z**2 - 5.9990895868714915e-02*z**3 + 1.5986366990635947e-02*z**4 - 2.1832848451906338e-03*z**5 + 1.2072787216388951e-04*z**6
    return x
def get_a2(z):
    x = 1.3499999999999285e-02 - 1.1043333333332635e-02*z + 3.1049999999997885e-03*z**2 - 2.9166666666664652e-04*z**3
    return x
def get_alpha(z):
    x = -3.9099999997916162e+00 + 6.5069999997397030e+00*z - 3.4613333332161793e+00*z**2 + 7.8699999997741299e-01*z**3 - 6.4666666665086803e-02*z**4
    return x
def get_beta(z):
    x = 4.3899999999994961e+00 - 3.3366666666661766e+00*z + 8.8499999999985191e-01*z**2 - 7.8333333333319252e-02*z**3
    return x
def get_Mbreak(z): # in h^-1 solar masses
    y = -1.8939995579430693e+01 + 4.1194659930500073e+01*z - 2.1521107131317578e+01*z**2 + 5.5722210776839693e+00*z**3 - 7.1555539481099406e-01*z**4 + 3.6444435601936033e-02*z**5
    x = 10**y
    return x
def get_Mmin(z): # in h^-1 solar masses
    y = -1.2909999999998245e+01 + 8.5399999999982867e+00*z - 2.1399999999994801e+00*z**2 + 1.5999999999995043e-01*z**3 
    x = 10**y
    return x


def hi_from_halos_1(halo_field, z): # HI-halo mass relation from spinelli et al. 2020
    """
    HI mass-halo mass relation from Spinelli et al. (2020).

    Parameters
    ----------
    halo_field : NDarray
        The distribution of halo masses across the field, in solar masses/h .
    z : float
        The redshift at which to evaluate the quantity.

    Returns
    -------
    xHI_mass : NDarray
        The total neutral hydrogen mass associated with each of the halos in the field, in solar masses/h.
    """
    if z >= 3 and z <= 5:
        pass
    elif z > 5:
        z=5
        print("HI-halo mass relation evaluated at upper limit of z=5.", end="\r")
    else:
        z=3
        print("HI-halo mass relation evaluated at lower limit of z=3.", end="\r")

    xHI_mass = halo_field * (get_a1(z) * (halo_field / 10**10)**(get_beta(z)) * np.exp(-(halo_field / get_Mbreak(z))**(get_alpha(z))) + get_a2(z)) * np.exp(-(get_Mmin(z) / halo_field)**0.5) # parameterisation from spinelli et al 2020

    return xHI_mass


def hi_from_halos_3(halo_field, z): 
    """
    HI mass-halo mass relation from Padmanabhan, Refregier, and Amara (2017).

    Parameters
    ----------
    halo_field : NDarray
        The distribution of halo masses across the field, in solar masses/h .
    z : float
        The redshift at which to evaluate the quantity.

    Returns
    -------
    xHI_mass : NDarray
        The total neutral hydrogen mass associated with each of the halos in the field, in solar masses/h.
    """
    f_H = OMb * hydrogen_baryon_frac / OMm
    alpha = 0.09
    v_c0 = 10**1.56
    beta = -0.58
    v_c = get_vc(halo_field, z)

    xHI_mass = alpha * f_H * halo_field**(1+beta) * (10**-11)**beta * np.exp(-(v_c0 / v_c)**3)

    return xHI_mass