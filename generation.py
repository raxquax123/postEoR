""" Functions to generate coeval boxes and lightcones containing the neutral hydrogen distribution in the redshift range 6 > z > 3, using the outputs from 21cmFAST as a base. """

import py21cmfast as p21c
import numpy as np
import os
import postEoR.tools as tools
from postEoR.objects import Ltcone, Box
from postEoR.tools import cosmo_params, hlittle, solar_mass, Mpc_to_m, h, c, A_10, m_H, k_B, nu_21, OMm, OMl
from postEoR.analysis import ang_to_len
import gc

matter_options = p21c.MatterOptions(USE_HALO_FIELD=False, HALO_STOCHASTICITY=False, USE_RELATIVE_VELOCITIES=True, PERTURB_ALGORITHM="2LPT", HMF="ST") # choose sheth-tormen hmf, and 2LPT to evolve density field to the required late times
astro_params = p21c.AstroParams(R_BUBBLE_MAX=50, PHOTONCONS_CALIBRATION_END=2.99)
p21c.config['ignore_R_BUBBLE_MAX_error'] = True

if not os.path.exists('_cache'):
    os.mkdir('_cache')
cache = p21c.OutputCache('_cache')


def generate_box(
    z : float, 
    HII_dim=256, 
    box_size=64,
    size_type="len",
    overdens_cap=None,
    connectivity=1,
    normalise_halos=False,
    max_rad=1,
    smooth=False,
    sigma=0.5,
    include_RSD=False,
    random_seed=1122,
    incl_EoR_contr=True,
    HI_model=3,
) -> Box:
    """
    Generates a coeval box at specified redshift using the base functionality of 21cmFAST and post-processing functions in tools.py.

    Parameters
    ----------
    z : float
        Redshift at which to produce the coeval box.
    HII_dim : int (optional)
        The number of cells in each dimension of the box.
    box_size : float (optional)
        The size of each spatial dimension of the lightcone. Defaults to 400.
    size_type : str (optional)
        The unit type of the given size. Defaults to length in Mpc/h ("len"). Valid options are "len" and "ang" (angular size in deg).
    overdens_cap : float (optional)
        The minimum overdensity required for a cell to be considered part of a halo. Defaults to None (applies simple linear scaling to optimal overdens_cap based on matching to theoretical HMF).
    connectivity : int (optional)
        The connectivity parameter used in the watershed algorithm. Defaults to 3 (maximum).
    normalise_halos : bool (optional)
        Whether to normalise the halo mass based on the overdensity cutoff for halo inclusion. Defaults to False.
    max_rad : float (optional)
        The maximum radius, in Mpc/h, to calculate the HI density profile out to. Defaults to 1.
    smooth : bool (optional)
        Whether to apply Gaussian smoothing to the density field before applying the halo finder to it. Defaults to False.
    sigma : float (optional)
        The degree of Gaussian smoothing to be applied. Only has effect if smooth=True. Defaults to 0.5.
    include_RSD : bool (optional)
        Whether to calculate the corrections due to redshift space distortions. Defaults to False.
    random_seed : int (optional)
        The seed used for generating the 21cmFAST density field. Used for testing. Defaults to 1122.
    incl_EoR_contr : bool (optional)
        Whether to include the neutral IGM contribution to the brightness temperature. Defaults to True.
    HI_model : int (optional)
        Which analytic HI-halo relation to use. 1 corresponds to the simulation-based model from Spinelli et al. 2020, 2 is an observational-based model from Padmanabhan and Refegier (2017), and 3 is an observational-based model from Padamanabhan, Refregier, and Amara (2017). Defaults to 3.

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
    if size_type=="ang":
        box_len = ang_to_len(box_size, z)
    elif size_type=="len":
        box_len=box_size
    else:
        print("Unrecognised size type. Defaulting to length in Mpc.")
        box_len=box_size

    # 21cmFAST - generates and evolves the density field using 2LPT, and produces the bt expected from eor
    simulation_options = p21c.SimulationOptions(HII_DIM=HII_dim, BOX_LEN=box_len / hlittle)
    astro_options = p21c.AstroOptions(USE_UPPER_STELLAR_TURNOVER=False, USE_EXP_FILTER=False, INHOMO_RECO=True, USE_TS_FLUCT=False, HII_FILTER = 'sharp-k', USE_MINI_HALOS=False, CELL_RECOMB=True)
    inputs = p21c.InputParameters(cosmo_params=cosmo_params, matter_options=matter_options, simulation_options=simulation_options, astro_options=astro_options, astro_params=astro_params, random_seed=random_seed)
    print("inputs in")
    initial_conditions = p21c.compute_initial_conditions(
        inputs=inputs,
        cache=cache,
        write=True,
    )   
    print("init conds in")

    coevals = p21c.run_coeval(
        inputs=inputs,
        out_redshifts=[z],
        cache=cache,
        initial_conditions=initial_conditions,
        write=True,
    )
    print("box ran")

    dens = getattr(coevals[0], "density").astype("float64") # export overdensity field for use in post-processing)
    
    if incl_EoR_contr:
        BT_EoR = getattr(coevals[0], "brightness_temp") # calculate 21cm bt from 21cmFAST - eor / neutral igm contribution
        print("Max BT value=" + str(np.max(BT_EoR)))
    else:
        BT_EoR = np.zeros(np.shape(dens))
        print("EoR BT not used.")

    if include_RSD: # export velocity field and calculate dv/dr if including effects of rsd [NEEDS FIXING FOR USE IN LIGHTCONE]
        vel = getattr(coevals[0], "velocity_z")
        vel_grad = tools.get_vel_grad(vel)
    else:
        vel = 0
        vel_grad = 0

    # clear memory of things no longer in use
    del simulation_options
    del inputs
    del coevals
    del vel
    gc.collect()

    if overdens_cap is None:
        #overdens_cap = -54.7 * box_len / HII_dim + 16.75 # simple linear approximation to the optimal overdens cap to match hmf to theoretical
        overdens_cap = 8 * (0.16 / (box_len / HII_dim))
        print("Optimal overdensity cap used is " + str(overdens_cap))
    
    halos = tools.find_halos_watershed(dens, box_len / HII_dim, overdens_cap=overdens_cap, connectivity=connectivity, normalise=normalise_halos, smooth=smooth, sigma=sigma)
    if np.max(halos) == 0:
        HI_distr = np.zeros(np.shape(dens))
    else:
        HI_distr = tools.get_HI_field(halos, z, box_len / HII_dim, max_rad=max_rad, HI_model=HI_model) # obtain the neutral hydrogen distribution, given a halo field and the redshift of evaluation

    H_0 = hlittle * 100
    H = H_0 * 1000 / Mpc_to_m * (OMm * (1+z)**3+OMl)**0.5
    HI_dens = HI_distr * solar_mass / (box_len / HII_dim * Mpc_to_m)**3 # calculate \rho_{HI} in kg/m^3 h^2
    BT = (3 * h * c**3 * A_10)/(32 * np.pi * m_H * k_B * (nu_21)**2) * ((1+z)**2 / H) * HI_dens * hlittle**2 * H / (H * np.ones(np.shape(vel_grad)) + vel_grad) # bt formula from wolz et al. 2017, removing h-agnosticity as now looking at observable, adding rsd term
    BT_fin = np.maximum(BT, BT_EoR) # avoiding 'double-counting' of bt from post-processing and 21cmFAST
    box = Box(z, box_len, HII_dim, dens, halos, BT_fin)

    return box



def generate_cone(
    min_redshift : float, 
    max_redshift : float, 
    HII_dim=200, 
    box_size=400,
    size_type="len",
    overdens_cap=None,
    connectivity=1,
    normalise_halos=False,
    max_rad=1,
    smooth=False,
    sigma=False,
    include_RSD=False,
    random_seed=1122,
    nchunks=20,
    HI_model=3,
) -> Ltcone: 
    """
    Generates a lightcone using the base functionality of 21cmFAST and post-processing functions in tools.py.

    Parameters
    ----------
    z_centr : float
        The central redshift at which to evaluate the lightcone.
    delta_z : float (optional)
        The size of the redshift range over which to evaluate the lightcone. Defaults to 0.5.
    HII_dim : int (optional)
        The number of cells in each spatial dimension of the lightcone. Defaults to 200.
    box_size : float (optional)
        The size of each spatial dimension of the lightcone. Defaults to 400.
    size_type : str (optional)
        The unit type of the given size. Defaults to length in Mpc ("len"). Valid options are "len" and "ang" (angular size in deg).
    overdens_cap : float (optional)
        The minimum overdensity required for a cell to be considered part of a halo. Defaults to None (applies simple linear scaling to optimal overdens_cap based on matching to theoretical HMF).
    connectivity : int (optional)
        The connectivity parameter used in the watershed algorithm. Defaults to 3 (maximum).
    normalise_halos : bool (optional)
        Whether to normalise the halo mass based on the overdensity cutoff for halo inclusion. Defaults to False.
    max_rad : float (optional)
        The maximum radius, in Mpc/h, to calculate the HI density profile out to. Defaults to 1.
    smooth : bool (optional)
        Whether to apply Gaussian smoothing to the density field before applying the halo finder to it. Defaults to False.
    sigma : float (optional)
        The degree of Gaussian smoothing to be applied. Only has effect if smooth=True. Defaults to 0.5.
    include_RSD : bool (optional)
        Whether to calculate the corrections due to redshift space distortions. Defaults to False.
    random_seed : int (optional)
        The seed used for generating the 21cmFAST density field. Used for testing. Defaults to 1122.
    nchunks : int (optional)
        The number of redshift bins to divide the lightcone into for calculating redshift-dependent values. Defaults to 20.
    HI_model : int (optional)
        Which analytic HI-halo relation to use. 1 corresponds to the simulation-based model from Spinelli et al. 2020, 2 is an observational-based model from Padmanabhan and Refegier (2017), and 3 is an observational-based model from Padamanabhan, Refregier, and Amara (2017). Defaults to 3.

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

    if size_type=="ang":
        box_len = ang_to_len(box_size, (min_redshift+max_redshift)*0.5)
    elif size_type=="len":
        box_len=box_size
    else:
        print("Unrecognised size type. Defaulting to length in Mpc.")
        box_len=box_size

    # defining the redshift bounds
    simulation_options = p21c.SimulationOptions(HII_DIM=HII_dim, BOX_LEN=box_len / hlittle)
    if max_redshift < 35:
        astro_options = p21c.AstroOptions(USE_UPPER_STELLAR_TURNOVER=False, USE_EXP_FILTER=False, INHOMO_RECO=False, USE_TS_FLUCT=False, HII_FILTER = 'sharp-k', USE_MINI_HALOS=False, CELL_RECOMB=True)
        print("INHOMO_RECO changed to False, since max_redshift < 35.")
    else:
        astro_options = p21c.AstroOptions(USE_UPPER_STELLAR_TURNOVER=False, USE_EXP_FILTER=False, INHOMO_RECO=True, USE_TS_FLUCT=False, HII_FILTER = 'sharp-k', USE_MINI_HALOS=False, CELL_RECOMB=True)
    inputs = p21c.InputParameters(cosmo_params=cosmo_params, matter_options=matter_options, simulation_options=simulation_options, 
                                  astro_options=astro_options, astro_params=astro_params, random_seed=random_seed, node_redshifts=np.geomspace(min_redshift, max_redshift, nchunks)
    )
    print("inputs in")
    initial_conditions = p21c.compute_initial_conditions(
        inputs=inputs,
        cache=cache,
        write=True,
    )
    print("init conds in")

    # 21cmFAST - generates and evolves the density field using 2LPT, and produces the bt expected from eor
    lightcone_quantities = (
        "brightness_temp",
        "neutral_fraction",
        "density",
        "velocity_z"
    )

    # set up lightconer class
    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=min(inputs.node_redshifts),
        max_redshift=max(inputs.node_redshifts),
        quantities=lightcone_quantities,
        resolution=inputs.simulation_options.cell_size,
    )
    print("lcn made")

    _, _, _, lightcone = p21c.run_lightcone(
        lightconer=lcn, 
        initial_conditions=initial_conditions,
        global_quantities=lightcone_quantities,
        cache=cache,
        write=True,
        inputs=inputs,
    )

    print("lightcone ran")

    BT_EoR_ltcone = lightcone.lightcones["brightness_temp"] # getting the bt from the pre-reionization neutral igm
    dens_ltcone = lightcone.lightcones["density"] # getting density for post-processing

    if include_RSD: # export velocity field and calculate dv/dr if including effects of rsd [NEEDS FIXING FOR USE IN LIGHTCONE]
        vel = lightcone.lightcones["velocity_z"] #* Mpc_to_m
        vel_grad = tools.get_vel_grad(vel, HII_dim, box_len)
    else:
        vel = 0
        vel_grad = np.zeros(np.shape(dens_ltcone))


    # clear memory of things no longer in use
    del lcn
    del vel
    del initial_conditions
    del inputs
    gc.collect()

    if overdens_cap is None:
        #overdens_cap = -54.7 * box_len / HII_dim + 16.75 # simple linear approximation to the optimal overdens cap to match hmf to theoretical
        overdens_cap = 8 * (0.16 / (box_len / HII_dim))
        print("Optimal overdensity cap used is " + str(overdens_cap))
    
    halos_ltcone = tools.find_halos_watershed(dens_ltcone, box_len / HII_dim, overdens_cap=overdens_cap, connectivity=connectivity, normalise=normalise_halos, smooth=smooth, sigma=sigma)
    BT = np.zeros(np.shape(dens_ltcone))
    if np.max(halos_ltcone) != 0:
        redshifts = np.flip(np.geomspace(min_redshift, max_redshift, nchunks+1))
        chunk_indices = tools.z_to_index(box_len / HII_dim, redshifts)

        for ii in range(nchunks):
            start = round(chunk_indices[ii])
            end = round(chunk_indices[ii + 1])
            redshift = 0.5 * (redshifts[ii]+redshifts[ii+1])
            try:
                if np.max(halos_ltcone[:,:,start:end]) == 0:
                    BT[:, :, start:end] = np.zeros(np.shape(dens_ltcone[:, :, start:end]))
                else:
                    HI_distr = tools.get_HI_field(halos_ltcone[:, :, start:end], redshift, box_len / HII_dim, max_rad=max_rad, HI_model=HI_model) # obtain the neutral hydrogen distribution, given a halo field and the redshift of evaluation
                    H_0 = hlittle * 100
                    H = H_0 * 1000 / Mpc_to_m * (OMm * (1+redshift)**3+OMl)**0.5
                    HI_dens = HI_distr * solar_mass / (box_len / HII_dim * Mpc_to_m)**3 # calculate \rho_{HI} in kg/m^3 h^2
                    BT[:, :, start:end] = (3 * h * c**3 * A_10)/(32 * np.pi * m_H * k_B * (nu_21)**2) * ((1+redshift)**2 / H) * HI_dens * hlittle**2 * 1 / (np.ones([HII_dim,HII_dim,(end-start)]) + vel_grad[:, :, start:end] / H) # bt formula from wolz et al. 2017, removing h-agnosticity as now looking at observable
            except ValueError:
                pass

    BT_ltcone = np.maximum(BT, BT_EoR_ltcone) # avoiding 'double-counting' of bt from post-processing and 21cmFAST

    # set up Ltcone object, containing the post-EoR data
    ltcone = Ltcone(max_redshift, min_redshift, box_len, HII_dim, dens_ltcone, halos_ltcone, BT_ltcone, lightcone)

    return ltcone