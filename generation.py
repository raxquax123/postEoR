""" Functions to generate coeval boxes and lightcones containing the neutral hydrogen distribution in the redshift range 6 > z > 3, using the outputs from 21cmFAST as a base. """

import py21cmfast as p21c
import numpy as np
import os
import postEoR.tools as tools
from postEoR.objects import Ltcone, Box
from postEoR.tools import cosmo_params, hlittle, solar_mass, Mpc_to_m, h, c, A_10, m_H, k_B, nu_21, OMm, OMl

p21c.global_params.RecombPhotonCons = 1
p21c.global_params.PhotonConsEndCalibz = 2.5
p21c.FlagOptions.USE_TS_FLUCT = False
p21c.FlagOptions.INHOMO_RECO = True
p21c.FlagOptions.PHOTON_CONS = True

if not os.path.exists('_cache'):
    os.mkdir('_cache')
p21c.config['direc'] = '_cache'


def generate_box(
    z : float, 
    HII_dim=256, 
    box_len=64,
    overdens_cap=1.686,
    use_watershed=False,
) -> Box:
    """
    Generates a coeval box at specified redshift using the base functionality of 21cmFAST and post-processing functions in tools.py.

    Parameters
    ----------
    z : float
        Redshift at which to produce the coeval box.
    HII_dim : int (optional)
        The number of cells in each dimension of the box.
    box_len : float (optional)
        The size of the box in Mpc.
    overdens_cap : float (optional)
        The minimum overdensity required for a cell to be considered part of a halo. Defaults to 1.686 (Press-Schechter critical overdensity for collapse).
    use_watershed : bool (optional)
        Whether to use a watershed-based method for finding the halos. Defaults to False.

    Returns
    -------
    box : Box object
        Object containing the BT, overdensity, and halo fields of the coeval box, in addition to defining information such as number of cells in each dimension and physical box length.
    
    Example usage
    -------------
    >>> from postEoR import generation as gen
    >>> box = gen.generate_box(z=4, HII_dim=250, box_len=50, overdens_cap=0, use_watershed=True)
    >>> print(box)
    <postEoR.objects.Box object at 0x199b7d690>
    >>> print(box.cell_size())
    0.2
    """
    # 21cmFAST - generates and evolves the density field using 2LPT, and produces the bt expected from eor
    initial_conditions = p21c.initial_conditions(
        user_params = {"HII_DIM": HII_dim, "BOX_LEN": box_len, "USE_2LPT": True, "HMF": 3},
        cosmo_params=cosmo_params,
        random_seed=1122
    )
    perturbed_field = p21c.perturb_field(
        redshift = z,
        init_boxes = initial_conditions
    )
    ionized_field = p21c.ionize_box(perturbed_field = perturbed_field) # export neutral fraction from 21cmFAST (EoR bubbles)
    dens = getattr(perturbed_field, "density") # export overdensity field for use in post-processing
    if use_watershed:
        halos = tools.find_halos_watershed(dens, box_len, HII_dim, overdens_cap=overdens_cap)
    else:
        halos = tools.find_halos(dens, box_len, HII_dim, overdens_cap=overdens_cap) # obtain halo distribution and masses from the overdensity field by pushing overdensities to their local maxima
    HI_distr = tools.get_HI_field(halos, z, box_len, HII_dim) # obtain the neutral hydrogen distribution, given a halo field and the redshift of evaluation

    H_0 = hlittle * 100
    BT_21c = p21c.brightness_temperature(ionized_box=ionized_field, perturbed_field=perturbed_field) # calculate 21cm bt from 21cmFAST - eor / neutral igm contribution
    HI_dens = HI_distr * solar_mass / (box_len / HII_dim * Mpc_to_m)**3 # calculate \rho_{HI} in kg/m^3
    BT = (3 * h * c**3 * A_10)/(32 * np.pi * m_H * k_B * (nu_21)**2) * ((1+z)**2 / ((H_0*1000/Mpc_to_m)*(OMm*(1+z)**3+OMl)**0.5)) * HI_dens # bt formula from wolz et al. 2017
    BT_EoR = getattr(BT_21c, "brightness_temp") # getting bt from neutral igm (pre-reionization)
    BT_fin = np.maximum(BT, BT_EoR) # avoiding 'double-counting' of bt from post-processing and 21cmFAST
    box = Box(z, box_len, HII_dim, dens, halos, BT_fin)

    return box



def generate_cone(
    z_centr : float, 
    delta_z=0.5, 
    HII_dim=200, 
    box_len=400,
    overdens_cap=1.686,
    use_watershed=False,
) -> Ltcone: 
    """
    Generates a lightcone using the base functionality of 21cmFAST and post-processing functions in tools.py.

    Parameters
    ----------
    z_centr : float
        The central redshift at which to evaluate the lightcone.
    delta_z : float
        The size of the redshift range over which to evaluate the lightcone. Defaults to 0.5.
    HII_dim : int
        The number of cells in each spatial dimension of the lightcone. Defaults to 200.
    box_len : float
        The length in Mpc of each spatial dimension of the lightcone. Defaults to 400.
    overdens_cap : float
        The minimum overdensity required for a cell to be considered part of a halo. Defaults to 1.686 (Press-Schechter critical overdensity for collapse).
    use_watershed : bool (optional)
        Whether to use a watershed-based method for finding the halos. Defaults to False.

    Returns
    -------
    ltcone : Ltcone object
        Object containing the BT, overdensity, and halo fields of the lightcone, in addition to defining information.

    Example usage
    -------------
    >>> from postEoR import generation as gen
    >>> cone = gen.generate_cone(z_centr=4, delta_z=0.4, HII_dim=250, box_len=400, overdens_cap=0, use_watershed=True)
    >>> print(cone)
    <postEoR.objects.Ltcone object at 0x199b7d690>
    >>> print(cone.cell_size())
    1.6
    """
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
    )
    # run lightcone using 21cmFAST functionality
    lightcone = p21c.run_lightcone(
        lightconer=lcn,
        global_quantities=("brightness_temp", "density", 'xH_box'),
        direc='_cache',
        user_params=user_params,
        cosmo_params=cosmo_params,
        random_seed=1122
    )
    BT_EoR_ltcone = getattr(lightcone, "brightness_temp") # getting the bt from the pre-reionization neutral igm
    dens_ltcone = getattr(lightcone, "density") # getting density for post-processing

    if use_watershed:
        halos_ltcone = tools.find_halos_watershed(dens_ltcone, box_len, HII_dim, overdens_cap=overdens_cap)
    else:
        halos_ltcone = tools.find_halos(dens_ltcone, box_len, HII_dim, overdens_cap=overdens_cap) # find the halos on the lightcone.
    redshift = (min_redshift+max_redshift) / 2 # taking the mean redshift of the lightcone to use in evaluating the BT
    HI_ltcone = tools.get_HI_field(halos_ltcone, redshift, box_len, HII_dim, no_bins=25, max_rad=5) # find the hi field on the lightcone

    H_0 = hlittle * 100
    HI_dens = HI_ltcone * solar_mass / (box_len / HII_dim * Mpc_to_m)**3 # calculate \rho_{HI} in kg/m^3
    BT_HI_ltcone = (3 * h * c**3 * A_10)/(32 * np.pi * m_H * k_B * (nu_21)**2) * ((1+redshift)**2 / ((H_0*1000/Mpc_to_m)*(OMm*(1+redshift)**3+OMl)**0.5)) * HI_dens # bt formula from wolz et al. 2017
    BT_ltcone = np.maximum(BT_HI_ltcone, BT_EoR_ltcone)

    # set up Ltcone object, containing the post-EoR data
    ltcone = Ltcone(max_redshift, min_redshift, box_len, HII_dim, dens_ltcone, halos_ltcone, BT_ltcone, lightcone)

    return ltcone