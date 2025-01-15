import py21cmfast as p21c
import numpy as np
from astropy.cosmology import WMAP7
from scipy.integrate import quad
import scipy.ndimage as ndimage
import os

from postEoR.constants import *

p21c.global_params.RecombPhotonCons = 1
p21c.global_params.PhotonConsEndCalibz = 2.5
p21c.FlagOptions.USE_TS_FLUCT = False
p21c.FlagOptions.INHOMO_RECO = True
p21c.FlagOptions.PHOTON_CONS = True

if not os.path.exists('_cache'):
    os.mkdir('_cache')
p21c.config['direc'] = '_cache'


def push_mass_to_halo(x, y): # iterates over the overdensity field, pushing overdensities to their local maxima. iterates once forward and once backwards
    # forward iteration:
    dims_i = np.shape(x)[0]
    dims_j = np.shape(x)[1]
    dims_k = np.shape(x)[2]
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


def get_delta_vir(z): # returns mean halo overdensity within virial radius (dimensionless)
    x = omega_m * (1+z)**3 / (omega_m*(1+z)**3 + omega_lambda) - 1
    delta_vir = 18*np.pi**2 + 82 * x - 39 * x**2 # mean overdensity within virial radius
    return delta_vir


def get_vc(M, z): # returns virial velocity in km/s
    vc = np.array(163 *(M * little_h / 10**12)**(1/3)*(get_delta_vir(z)/200)**(1/6)*omega_m**(1/6)*(1+z)**0.5)  # virial velocity
    vc[vc < 0.01] = 0.01 # avoid dividing by zero errors later on.
    return vc


def get_r_vir(M, z): # returns virial radius in Mpc
    v_c = get_vc(M, z) * 1000 # convert to m/s from km/s
    M *= 1.989 * 10**30 # convert to kg from solar masses
    r = G * M / v_c**2
    r /= 3.086e+22
    return r # in Mpc


def get_conc(M, z): # returns halo concentration (dimensionless)
    c_HI = 113.80
    gamma = 0.22
    conc = c_HI * (M/10**11)**(-0.109) * 4 / (1+z)**gamma # takes in mass in solar masses
    return conc


def find_halos(density_field, box_len, HII_dim, max_count=50, overdens_cap=1.686): # returns mass and centre position of halos in solar masses. stops at either no change after algorithm applied, or when maximum number of iterations has been reached
    overdensity_field = density_field.copy()
    #print(np.mean(overdensity_field))
    #print(WMAP7.critical_density(3).value * 1000 * (box_len / HII_dim * 3.086*10**22)**3 / (1.989 * 10**30) * (1 / (1 + np.mean(density_field))))
    overdensity_field = (1 + overdensity_field) * WMAP7.critical_density(3).value * 1000 * (box_len / HII_dim * 3.086*10**22)**3 / (1.989 * 10**30) * (1 / (1 + np.mean(density_field))) # converting to solar masses (critical density at z = 3 since comoving box)
    overdensity_field[density_field < overdens_cap] = 0 # removing underdense regions
    #print(np.count_nonzero(overdensity_field))
    #vmin = vmin_km / (3.086e+19) # convert from km/s to Mpc/s. vmin_km defaults to 25km/s from villaescusa-navarro, bull, and viel (2015b)
    #overdensity_field[abs(velocity_field) < vmin] = 0 # removing unbound regions
    halo_field = np.zeros(np.shape(density_field)) # empty array to put masses into
    push_mass_to_halo(overdensity_field, halo_field)
    count = 0
    match = 10 # initialising check as arbitrary non-zero number
    while count <= max_count and match >= 1:
        old_field = halo_field.copy()
        halo_field = np.zeros(np.shape(density_field)) 
        push_mass_to_halo(old_field, halo_field) # algorithm to push masses to central maxima
        count += 1
        match = np.sum(abs(halo_field - old_field))
        print("change after iteration: ", match,". iteration number", count) # sanity check - expect to decrease with each iteration.
    halo_field[halo_field < (10**7)] = 0
    return halo_field


def get_rho_0(M, z): # calculates the rho_0 constant for a given modified NFW profile
    r_vir = get_r_vir(M, z) # obtaining virial radius in Mpc
    r_s = r_vir / get_conc(M, z) # obtaining scale radius in Mpc
    dx = lambda r: r_s**3 / ((r + 0.75*r_s)*(r+r_s)**2) * 4 * np.pi * r**2 # defining integrand
    integ, _ = quad(dx, r_vir, 0)
    rho = M / integ 
    return rho # in solar masses / Mpc^3


def obtain_HI_at_dist(M, z, r_end, r_start): # calculates the total HI mass over a given radius range from a given halo centre. in solar masses
    M_hi = hi_from_halos_2(M, z)
    #print("M_HI = ", M_hi)
    r_s = get_r_vir(M, z) / get_conc(M, z) # obtaining scale radius
    #print("r_s = ", r_s)
    integrand = lambda r: get_rho_0(M_hi, z) * r_s**3 / ((r + 0.75*r_s)*(r+r_s)**2) * 4 * np.pi * r**2 # defining integrand
    shell_mass, _ = quad(integrand, r_end, r_start) # obtaining hi mass contained within a shell corresponding to specified radius range
    #print("shell mass = ", shell_mass)
    return abs(shell_mass)


def hi_from_halos_2(halo_field, z): # HI-halo mass relation from padmanabhan and refregier 2017
    f_H = omega_b * hydrogen_baryon_frac / omega_m
    alpha = 0.17
    v_c0 = 10**1.57 # potentially edit limits to match observations 
    v_c1 = 10**4.39
    beta = -0.55
    v_c = get_vc(halo_field, z)
    xHI_mass = alpha*f_H*halo_field**(1+beta)*(little_h / 10**11)**beta*np.exp(-(v_c0/v_c)**3)*np.exp(-(v_c/v_c1)**3)
    return xHI_mass


def bin_halos(halo_field, no_bins=20): # collecting halos into mass bins
    binned_halo_field = np.zeros([no_bins, np.shape(halo_field)[0], np.shape(halo_field)[1], np.shape(halo_field)[2]])
    bins = np.linspace((np.min(halo_field)-1), (np.max(halo_field)+1), (no_bins+1))
    for i in range(no_bins):
        in_bin = np.multiply((halo_field > bins[i]).astype(int), (halo_field < bins[i+1]).astype(int))
        binned_halo_field[i] = np.multiply(halo_field, in_bin)
    return binned_halo_field


def get_HI_field(halo_field, z, box_len, HII_dim, no_bins=25, max_rad=1): # convolves the hi density profiles with the halo centre locations to produce the final hi mass field.
    binned_halos = bin_halos(halo_field, no_bins)
    bins = np.linspace((np.min(halo_field)-1), (np.max(halo_field)+1), (no_bins+1))
    M_centr = (bins[1:]+bins[:-1])/2
    final_HI_field = np.zeros(np.shape(halo_field))
    for i in range(no_bins):
        spherical_profile = create_spherical_profile(M_centr[i], z, max_rad, box_len, HII_dim)
        halo_pos = (binned_halos[i] > 0.01).astype(int)
        print("abt to convolve")
        final_HI_field += ndimage.convolve(halo_pos, spherical_profile)
    return final_HI_field


def create_spherical_profile(halo, z, max_rad, box_len, HII_dim): # creates spherical profile of default maximum radius 1 Mpc
    centre_ind = int(max_rad / (box_len/HII_dim)) # array index that will correspond to the centre of the profile
    prof_size = centre_ind * 2 + 1
    sph_prof = np.zeros([prof_size, prof_size, prof_size])
    max_rad_fin = (box_len/HII_dim) * (np.ceil(max_rad / (box_len/HII_dim)-0.5)+0.5)
    r_set = np.asarray(np.linspace(((box_len/HII_dim) / 2), max_rad_fin, centre_ind+1)) # radius bins to integrate over, in Mpc
    r_set = np.insert(r_set, 0, 0)
    print(r_set)
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


def generate_box(z, HII_dim=256, box_len=64): # generates a coeval box at specified redshift using base functionality of 21cmFAST and post-processing functions above
    # 21cmFAST - generates and evolves the density field using 2LPT, and produces the bt expected from eor
    initial_conditions = p21c.initial_conditions(
        user_params = {"HII_DIM": HII_dim, "BOX_LEN": box_len, "USE_2LPT": True, "HMF": 3},
        random_seed=1122
    )
    perturbed_field = p21c.perturb_field(
        redshift = z,
        init_boxes = initial_conditions
    )
    ionized_field = p21c.ionize_box(perturbed_field = perturbed_field) # export neutral fraction from 21cmFAST (EoR bubbles)
    dens = getattr(perturbed_field, "density") # export overdensity field for use in post-processing
    
    halos = find_halos(dens, box_len, HII_dim) # obtain halo distribution and masses from the overdensity field by pushing overdensities to their local maxima
    HI_distr = get_HI_field(halos, z, box_len, HII_dim) # obtain the neutral hydrogen distribution, given a halo field and the redshift of evaluation
    BT_21c = p21c.brightness_temperature(ionized_box=ionized_field, perturbed_field=perturbed_field) # calculate 21cm bt from 21cmFAST - eor / neutral igm contribution
    HI_dens = HI_distr * (1.989 * 10**30) / (box_len / HII_dim * 3.086*10**22)**3 # calculate \rho_{HI} in kg/m^3
    BT = (3 * h * c**3 * A_10)/(32 * np.pi * m_H * k_B * (nu_21)**2) * ((1+z)**2 / ((H_0*1000/3.0857e+22)*(omega_m*(1+z)**3+omega_lambda)**0.5)) * HI_dens # bt formula from wolz et al. 2017
    BT_EoR = getattr(BT_21c, "brightness_temp") # getting bt from neutral igm (pre-reionization)
    BT_fin = np.maximum(BT, BT_EoR) # avoiding 'double-counting' of bt from post-processing and 21cmFAST
    return dens, BT_fin, halos



def generate_cone(z_centr, delta_z=0.5, HII_dim=800, box_len=400): # generates a cone at specified central redshift using base functionality of 21cmFAST and post-processing functions above
    # 21cmFAST - generates and evolves the density field using 2LPT, and produces the bt expected from eor
    user_params = p21c.UserParams(
    HII_DIM=HII_dim, BOX_LEN=box_len, KEEP_3D_VELOCITIES=True, USE_2LPT=True, HMF=3
)
    # defining the redshift bounds
    min_redshift=z_centr - delta_z / 2
    max_redshift=z_centr + delta_z / 2

    # set up lightconer class
    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=min_redshift,
        max_redshift=max_redshift,
        quantities=('brightness_temp', 'density', 'velocity_z'),
        resolution=user_params.cell_size,
        get_los_velocity = True,
        # index_offset=0,
    )
    # run lightcone using 21cmFAST functionality
    lightcone = p21c.run_lightcone(
        lightconer=lcn,
        global_quantities=("brightness_temp", "density", 'xH_box'),
        direc='_cache',
        user_params=user_params,
        random_seed=1122
    )

    BT_EoR_ltcone = getattr(lightcone, "brightness_temp") # getting the bt from the pre-reionization neutral igm
    dens_ltcone = getattr(lightcone, "density") # getting density for post-processing
    halos_ltcone = find_halos(dens_ltcone, box_len, HII_dim) # find the halos on the lightcone.
    redshift = (min_redshift+max_redshift) / 2 # taking the mean redshift of the lightcone to use in evaluating the BT
    HI_ltcone = get_HI_field(halos_ltcone, redshift, box_len, HII_dim, no_bins=25, max_rad=5) # find the hi field on the lightcone
    HI_dens = HI_ltcone * (1.989 * 10**30) / (box_len / HII_dim * 3.086*10**22)**3 # calculate \rho_{HI} in kg/m^3
    BT_HI_ltcone = (3 * h * c**3 * A_10)/(32 * np.pi * m_H * k_B * (nu_21)**2) * ((1+redshift)**2 / ((H_0*1000/3.0857e+22)*(omega_m*(1+redshift)**3+omega_lambda)**0.5)) * HI_dens # bt formula from wolz et al. 2017
    BT_ltcone = np.maximum(BT_HI_ltcone, BT_EoR_ltcone)
    return dens_ltcone, BT_ltcone, halos_ltcone