""" Functions to be used within generating the coeval boxes and lightcones. """

import numpy as np
from scipy.integrate import quad
import scipy.ndimage as ndimage
from skimage.segmentation import watershed
from scipy import ndimage as ndi

""" Defining / importing parameters used. """
from postEoR.generation import hlittle, OMm, OMl, OMb, Mpc_to_m, solar_mass
G = 6.67430e-11 # gravitational constant, in N m^2 / kg^2
hydrogen_baryon_frac = 0.75

def push_mass_to_halo(x, y):
    """
    Iterates over the input overdensity field x, pushing overdensities to their local maxima, moving them to the output array y. Iterates once forwards and once backwards.

    Parameters
    ----------
    x : NDarray
        The overdensity field produced by 21cmFAST (dimensionless).
    y : NDarray
        An empty array that will contain the final halo mass distribution (in solar masses).
    """
    dims_i = np.shape(x)[0]
    dims_j = np.shape(x)[1]
    dims_k = np.shape(x)[2]

    # forward iteration:
    for i in range(int(dims_i-1)):
        for j in range(int(dims_j-1)):
            for k in range(int(dims_k-1)):
                if x[i, j, k] >= 0.01:
                    if x[i+1, j, k] >= x[i, j, k]:
                        if x[i, j+1, k] >= x[i+1, j, k]:
                            if x[i, j, k+1] >= x[i, j+1, k]:
                                y[i, j, k+1] += x[i, j, k]
                            else: 
                                y[i, j+1, k] += x[i, j, k]
                        elif x[i, j, k+1] >= x[i+1, j, k]:
                            y[i, j, k+1] += x[i, j, k]
                        else:
                            y[i+1, j, k] += x[i, j, k]
                    elif x[i, j+1, k] >= x[i, j, k]:
                        if x[i, j, k+1] >= x[i, j+1, k]:
                            y[i, j, k+1] += x[i, j, k]
                        else:
                            y[i, j+1, k] += x[i, j, k]
                    elif x[i, j, k+1] >= x[i, j, k]:
                        y[i, j, k+1] += x[i, j, k]
                    else:
                        y[i, j, k] += x[i, j, k]

    end_ind_i = int(dims_i - 1)
    end_ind_j = int(dims_j - 1)
    end_ind_k = int(dims_k - 1)
    for j in range(int(dims_j)):
        for k in range(int(dims_k)):
            y[end_ind_i, j, k] += x[end_ind_i, j, k]
    for i in range(int(dims_i-1)):
        for k in range(int(dims_k)):
            y[i, end_ind_j, k] += x[i, end_ind_j, k]
    for i in range(int(dims_i-1)):
        for j in range(int(dims_j-1)):
            y[i, j, end_ind_k] += x[i, j, end_ind_k]


    # reverse iteration:
    x = y.copy()
    y = np.zeros(np.shape(x))
    for i in reversed(range(1, int(dims_i))):
        for j in reversed(range(1, int(dims_j))):
            for k in reversed(range(1, int(dims_k))):
                if x[i, j, k] >= 0.01:
                    if x[i-1, j, k] >= x[i, j, k]:
                        if x[i, j-1, k] >= x[i-1, j, k]:
                            if x[i, j, k-1] >= x[i, j-1, k]:
                                y[i, j, k-1] += x[i, j, k]
                            else: 
                                y[i, j-1, k] += x[i, j, k]
                        elif x[i, j, k-1] >= x[i-1, j, k]:
                            y[i, j, k-1] += x[i, j, k]
                        else:
                            y[i-1, j, k] += x[i, j, k]
                    elif x[i, j-1, k] >= x[i, j, k]:
                        if x[i, j, k-1] >= x[i, j-1, k]:
                            y[i, j, k-1] += x[i, j, k]
                        else:
                            y[i, j-1, k] += x[i, j, k]
                    elif x[i, j, k-1] >= x[i, j, k]:
                        y[i, j, k-1] += x[i, j, k]
                    else:
                        y[i, j, k] += x[i, j, k]

    for j in reversed(range(0, int(dims_j))):
        for k in reversed(range(0, int(dims_k))):
            y[0, j, k] += x[0, j, k]
    for i in reversed(range(1, int(dims_i))):
        for k in reversed(range(0, int(dims_k))):
            y[i, 0, k] += x[i, 0, k]
    for i in reversed(range(1, int(dims_i))):
        for j in reversed(range(1, int(dims_j))):
            y[i, j, 0] += x[i, j, 0]


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
    delta_vir = 18*np.pi**2 + 82 * x - 39 * x**2 # mean overdensity within virial radius formula

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
    vc = np.array(163 *(M * hlittle / 10**12)**(1/3)*(get_delta_vir(z)/200)**(1/6)*OMm**(1/6)*(1+z)**0.5)  # virial velocity formula
    vc[vc < 0.01] = 0.01 # avoid dividing by zero errors later on.

    return vc


def get_r_vir(M, z): # returns virial radius in Mpc
    """
    Calculates the virial radius of an input object of mass M at redshift z.

    Parameters
    ----------
    M : float
        The mass of the object, in solar masses.
    z : float
        The redshift at which to evaluate the virial radius.

    Returns
    -------
    r : float
        The virial radius of the object, in Mpc.
    """
    v_c = get_vc(M, z) * 1000 # convert to m/s from km/s
    M *= solar_mass # convert to kg from solar masses
    r = G * M / v_c**2
    r /= Mpc_to_m # convert from m to Mpc

    return r 


def get_conc(M, z): # returns halo concentration (dimensionless)
    """
    Calculates the halo concentration (dimensionless) of a halo of mass M, at redshift z.

    Parameters
    ----------
    M : float
        The mass of the object, in solar masses.
    z : float
        The redshift at which to evaluate the halo concentration.

    Returns
    -------
    conc : float
        The halo concentration of the halo (dimensionless).
    
    """
    c_HI = 113.80
    gamma = 0.22

    conc = c_HI * (M/10**11)**(-0.109) * 4 / (1+z)**gamma 

    return conc


def find_halos(overdensity_field, box_len, HII_dim, max_count=50, overdens_cap=1.686, sanity_check=False): 
    """
    Returns mass and centre position of halos in solar masses. 
    Stops at either no change after algorithm applied, or when maximum number of iterations has been reached.

    Parameters
    ----------
    overdensity_field : NDarray
        The overdensity field on which the halos are to be found.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc. 
    HII_dim : int
        The number of cells in each of the spatial dimensions of the box / cone.
    max_count : int (optional)
        The maximum number of times the halo finder will iterate over the input field. Defaults to 50.
    overdens_cap : float (optional)
        The minimum overdensity for which a cell is considered to be associated with a halo. Defaults to 1.686 (from P-S).
    sanity_check : bool (optional)
        Whether to print the total mass moved after each iteration, for bug testing. Defaults to False.

    Returns 
    -------
    halo_field : NDarray
        The distribution of halo masses across the field, in solar masses.
    """
    H_0_std_units = (hlittle * 100 * 1000) / (Mpc_to_m)
    z_comov = 0
    H = H_0_std_units * (OMm*(1+z_comov)**3 + OMl) ** 0.5
    crit_M_dens = (3 * H ** 2) / (8 * np.pi * G) * (OMm * (1+z_comov)**3) / (OMm*(1+z_comov)**3 + OMl) # using the critical density at a set redshift as the simulation is comoving.

    new_overdensity_field = overdensity_field.copy()
    mass_field = (1 + new_overdensity_field) * crit_M_dens * (box_len / HII_dim * Mpc_to_m)**3 / (solar_mass) * (1 / (1 + np.mean(overdensity_field))) # converting to solar masses (critical density at z = 3 since comoving box) CHECK

    mass_field[overdensity_field < overdens_cap] = 0 # removing underdense regions
    halo_field = np.zeros(np.shape(overdensity_field)) # empty array to put masses into
    push_mass_to_halo(mass_field, halo_field)
    count = 0
    match = 10 # initialising check as arbitrary non-zero number

    while count <= max_count and match >= 1:
        old_field = halo_field.copy()
        halo_field = np.zeros(np.shape(overdensity_field)) 
        push_mass_to_halo(old_field, halo_field) # algorithm to push masses to central maxima
        count += 1
        match = np.sum(abs(halo_field - old_field))
        if sanity_check:
            print("change after iteration: ", match,". iteration number", count) # sanity check - expect to decrease with each iteration.

    return halo_field


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
        The rho_0 constant for the input halo, in solar masses / Mpc^3.
    """
    r_vir = get_r_vir(M, z) # obtaining virial radius in Mpc
    r_s = r_vir / get_conc(M, z) # obtaining scale radius in Mpc

    dx = lambda r: r_s**3 / ((r + 0.75*r_s)*(r+r_s)**2) * 4 * np.pi * r**2 # defining integrand
    integ, _ = quad(dx, r_vir, 0)
    rho = M / integ 

    return rho


def obtain_HI_at_dist(M, z, r_end, r_start): 
    """
    Calculates the total HI mass over a given radius range from a halo of mass M, in solar masses.

    Parameters
    ----------
    M : float
        The mass of the central halo, in solar masses.
    z : float
        The redshift at which to evaluate the HI mass at.
    r_end : float
        The maximum radius of the shell in consideration, in Mpc.
    r_start : float
        The minimum radius of the shell in consideration, in Mpc.

    Returns
    -------
    shell_mass : float
        The total HI mass contained within the shell, in solar masses.
    """
    M_hi = hi_from_halos_2(M, z)
    r_s = get_r_vir(M, z) / get_conc(M, z) # obtaining scale radius

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
        The distribution of halo masses across the field, in solar masses.
    z : float
        The redshift at which to evaluate the quantity.

    Returns
    -------
    xHI_mass : NDarray
        The total neutral hydrogen mass associated with each of the halos in the field, in solar masses.
    """
    f_H = OMb * hydrogen_baryon_frac / OMm
    alpha = 0.17
    v_c0 = 10**1.57 # potentially edit limits to match observations 
    v_c1 = 10**4.39
    beta = -0.55
    v_c = get_vc(halo_field, z)

    xHI_mass = alpha*f_H*halo_field**(1+beta)*(hlittle / 10**11)**beta*np.exp(-(v_c0/v_c)**3)*np.exp(-(v_c/v_c1)**3)

    return xHI_mass


def bin_halos(halo_field, no_bins=25): # collecting halos into mass bins
    """
    Collects the halos within an input halo field into equally spaced mass bins.

    Parameters
    ----------
    halo_field : NDarray
        The distribution of halo masses across the field, in solar masses.
    no_bins : int (optional)
        The number of mass bins into which the halos are collected. Defaults to 25.

    Returns
    -------
    binned_halo_field : NDarray
        The halo field, arranged into mass bins, in solar masses.
    bins : NDarray
        The bin edges that define the mass bins, in solar masses.
    """
    binned_halo_field = np.zeros([no_bins, np.shape(halo_field)[0], np.shape(halo_field)[1], np.shape(halo_field)[2]])

    bins = np.linspace((np.min(halo_field)-1), (np.max(halo_field)+1), (no_bins+1))

    for i in range(no_bins):
        in_bin = np.multiply((halo_field > bins[i]).astype(int), (halo_field < bins[i+1]).astype(int))
        binned_halo_field[i] = np.multiply(halo_field, in_bin)

    return binned_halo_field, bins


def get_HI_field(halo_field, z, box_len, HII_dim, no_bins=25, max_rad=1):
    """
    Convolves the HI profiles with the halo locations to produce the complete final HI mass field.

    Parameters
    ----------
    halo_field : NDarray
        The distribution of halo masses across the field, in solar masses.
    z : float
        The redshift at which the field is evaluated.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc.
    HII_dim: int
        The number of cells in each of the spatial dimensions of the box / cone.
    no_bins: int (optional)
        The number of mass bins into which the halos are collected. Defaults to 25.
    max_rad: float (optional)
        The maximum radius out to which the spherical HI profile is evaluated, in Mpc. Defaults to 1 Mpc.

    Returns
    -------
    final_HI_field : NDarray
        The distribution of HI mass across the field, in solar masses.
    """
    binned_halos, bins = bin_halos(halo_field, no_bins)
    M_centr = (bins[1:]+bins[:-1])/2
    final_HI_field = np.zeros(np.shape(halo_field))

    for i in range(no_bins):
        spherical_profile = create_spherical_profile(M_centr[i], z, box_len, HII_dim, max_rad)
        halo_pos = (binned_halos[i] > 0.01).astype(int)
        print("abt to convolve")
        final_HI_field += ndimage.convolve(halo_pos, spherical_profile)

    return final_HI_field


def create_spherical_profile(halo, z, box_len, HII_dim, max_rad=1): 
    """
    Calculates the radial distribution of HI surrounding a halo of given mass.

    Parameters
    ----------
    halo : float
        The mass of the halo whose HI profile is being calculated.
    z : float
        The redshift at which the profile is being calculated.
    box_len : float
        The size in Mpc of the full box/spatial dimensions of cone.
    HII_dim: int
        The number of cells in each dimension of the full box/spatial dimensions of cone.
    max_rad: float (optional)
        The maximum radius out to which the profile will be evaluated. Defaults to 1 Mpc.

    Returns
    -------
    sph_prof_fin : NDarray
        The HI distribution around the input halo.
    """
    centre_ind = int(max_rad / (box_len/HII_dim)) # array index that will correspond to the centre of the profile
    prof_size = centre_ind * 2 + 1
    sph_prof = np.zeros([prof_size, prof_size, prof_size])
    max_rad_fin = (box_len/HII_dim) * (np.ceil(max_rad / (box_len/HII_dim)-0.5)+0.5)

    r_set = np.asarray(np.linspace(((box_len/HII_dim) / 2), max_rad_fin, centre_ind+1)) # radius bins to integrate over, in Mpc
    r_set = np.insert(r_set, 0, 0)

    for i in range(prof_size):
        for j in range(prof_size):
            for k in range(prof_size):
                dist_from_cent_i = abs(centre_ind - i)
                dist_from_cent_j = abs(centre_ind - j)
                dist_from_cent_k = abs(centre_ind - k)
                count_at_dist = ((dist_from_cent_i+1)**3+(dist_from_cent_j+1)**3+(dist_from_cent_k+1)**3)/3 # calculating how many 'similar' cells there are, to distribute shell mass amongst them
                sph_prof[i, j, k] = (obtain_HI_at_dist(halo, z, r_set[dist_from_cent_i + 1], r_set[dist_from_cent_i]) + obtain_HI_at_dist(halo, z, r_set[dist_from_cent_j+1], r_set[dist_from_cent_j])+ obtain_HI_at_dist(halo, z, r_set[dist_from_cent_k+1], r_set[dist_from_cent_k])) / (3 * count_at_dist)
    sph_prof_fin = sph_prof * hi_from_halos_2(halo, z) / np.sum(sph_prof)

    return sph_prof_fin


def find_halos_watershed(dens, box_len, HII_dim, overdens_cap=0):
    """
    Returns mass and centre position of halos in solar masses. 
    Stops at either no change after algorithm applied, or when maximum number of iterations has been reached.

    Parameters
    ----------
    overdensity_field : NDarray
        The overdensity field on which the halos are to be found.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc. 
    HII_dim : int
        The number of cells in each of the spatial dimensions of the box / cone.
    overdens_cap : float (optional)
        The minimum overdensity for which a cell is considered to be associated with a halo. Defaults to 0.

    Returns 
    -------
    halo_field : NDarray
        The distribution of halo masses across the field, in solar masses.
    """
    image = dens
    image[dens < 1.686] = 0

    distance = ndi.distance_transform_edt(image)
    labels = watershed(-distance)
    print(np.max(labels))

    halo_field = np.zeros([HII_dim, HII_dim, HII_dim])

    H_0_std_units = (hlittle * 100 * 1000) / (Mpc_to_m)
    z_comov = 0
    H = H_0_std_units * (OMm*(1+z_comov)**3 + OMl) ** 0.5
    crit_M_dens = (3 * H ** 2) / (8 * np.pi * G) * (OMm * (1+z_comov)**3) / (OMm*(1+z_comov)**3 + OMl) # using the critical density at a set redshift as the simulation is comoving.

    new_overdensity_field = dens.copy()
    mass_field = (1 + new_overdensity_field) * crit_M_dens * (box_len / HII_dim * Mpc_to_m)**3 / (solar_mass) * (1 / (1 + np.mean(dens))) 

    mass_field[dens < overdens_cap] = 0 # removing underdense regions

    for i in range(np.max(labels)):
        current_halo = np.multiply((labels == i).astype(int), mass_field)
        halo_centre = np.unravel_index(np.argmax(current_halo, axis=None), current_halo.shape)
        halo_mass = np.sum(current_halo)
        halo_field[halo_centre] += halo_mass
        print(i, end="\r")
    
    return halo_field
