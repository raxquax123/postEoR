import postEoR.generation as gen
import postEoR.observations as obs
import matplotlib.pyplot as plt
import numpy as np
import camb
import hickle as hkl
from postEoR.tools import hlittle, OMb, OMm
from scipy.interpolate import make_interp_spline
plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = [9, 6]


def gen_ska_low_forecast(z, lin_bias=2.5, asurv=100, tsurv=5000, freq_bin=10e6, plot_dens=False, color="tab:orange", use_AA4=False, fig=None, ax1=None, HI_model=3):
    """
    Calculates and plots a theoretical HI power spectrum at the specified redshift for SKA-Low, using CAMB for the non-linear matter power spectrum (halofit).

    Parameters
    ----------
    z : float
        The redshift at which to calculate the theoretical forecasts.
    lin_bias : float (optional)
        The constant value of the HI bias at the largest scales. Defaults to 2.5.
    asurv : float (optional)
        The survey area, in deg^2. Defaults to 100 (corresponding to the planned SKA-Low deep survey).
    tsurv : float (optional)
        The total observing time, in hours. Defaults to 5000 (corresponding to the planned SKA-Low deep survey).
    freq_bin : float (optional)
        The frequency channel width of the instrument, in Hz. Defaults to 10 MHz.
    plot_dens : bool (optional)
        Whether to also plot the forecasted density power spectrum (in color "tab:blue"). Defaults to False.
    color : str (optional)
        The color to plot the HI power spectrum in. Defaults to "tab:orange".
    use_AA4 : bool (optional)
        Whether to use AA4 when generating the forecast. Defaults to False.
    fig : Matplotlib figure object (optional)
        A pre-existing figure object to plot the forecast onto. Defaults to None.
    fig : Matplotlib axis object (optional)
        A pre-existing axis object to plot the forecast onto. Defaults to None.
    HI_model : int (optional)
        Which analytic HI-halo relation to use. 1 corresponds to the simulation-based model from Spinelli et al. 2020, 2 is an observational-based model from Padmanabhan and Refegier (2017), and 3 is an observational-based model from Padamanabhan, Refregier, and Amara (2017). Defaults to 3.

    """
    pars = camb.set_params(H0=hlittle*100, ombh2=OMb*hlittle**2, omch2=(OMm - OMb)*hlittle**2, ns=1)
    pars.set_matter_power(redshifts=[z], kmax=81)

    if fig is None or ax1 is None:
        fig, ax1 = plt.subplots()

    results = camb.get_results(pars)
    k_matter, _, p_matter = results.get_matter_power_spectrum(minkh=0.026, maxkh=81, npoints = 100)

    pars.NonLinear = camb.model.NonLinear_both
    results = camb.get_results(pars)

    results.calc_power_spectra(pars)
    kh_nonlin, _, pk_nonlin = results.get_matter_power_spectrum(minkh=0.026, maxkh=81, npoints = 100)

    try: 
        mean_BT=hkl.load("mean_BT_cache.hkl")[str(z)]
    except FileNotFoundError:
        BT_file={}
        print("Generating small...")
        small = gen.generate_box(z, 250, 40, max_rad=1, incl_EoR_contr=False, HI_model=HI_model) # obtain bt contribution from hi in halos
        print("Generating large...")
        large = gen.generate_box(z, 100, 200, max_rad=0, incl_EoR_contr=True) # obtain bt contribution from eor
        mean_BT = np.mean(small.BT_field) + np.mean(large.BT_field)
        BT_file[str(z)]=mean_BT
        hkl.dump(BT_file, "mean_BT_cache.hkl")
    except KeyError:
        BT_file=hkl.load("mean_BT_cache.hkl")
        print("Generating small...")
        small = gen.generate_box(z, 250, 40, max_rad=1, incl_EoR_contr=False, HI_model=HI_model) # obtain bt contribution from hi in halos
        print("Generating large...")
        large = gen.generate_box(z, 100, 200, max_rad=0, incl_EoR_contr=True) # obtain bt contribution
        mean_BT = np.mean(small.BT_field) + np.mean(large.BT_field)
        BT_file[str(z)]=mean_BT
        hkl.dump(BT_file, "mean_BT_cache.hkl")

    print("mean_BT " + str(mean_BT))
    camb_BT = pk_nonlin[0,:] * lin_bias**2 * mean_BT**2

    AA4 = obs.SKA1LOW_AA4(0)
    AAstar = obs.SKA1LOW_AAstar(0)

    some_survey = obs.Interferometer(AA4, 6, 3, asurv, tsurv, 0, freq_bin)
    comp_survey = obs.Interferometer(AAstar, 6, 3, asurv, tsurv, 0, freq_bin)

    ax1.plot(kh_nonlin, camb_BT, color=color, label="HI PS, z=" + str(z))

    if use_AA4:
        ps, k_perp = some_survey.noise_power_perp(z)
        noise = make_interp_spline(k_perp, ps, k=1)
        print("AA4 used")
    else:
        ps, k_perp = comp_survey.noise_power_perp(z)
        noise = make_interp_spline(k_perp, ps, k=1)
        print("AA* used")

    ax1.fill_between(kh_nonlin, camb_BT - noise(kh_nonlin), camb_BT + noise(kh_nonlin), alpha=0.2, color=color)

    if plot_dens:
        ax2 = ax1.twinx() 
        ax2.plot(k_matter, p_matter[0, :], color="tab:green", linestyle="--", label="Matter PS, z=" + str(z))
        ax2.set_ylabel("P$_m$(k), (Mpc/h)$^3$")
        ax2.set_yscale("log")
        ax2.legend(loc="upper right")
        ax1.yaxis.label.set_color("tab:orange")
        ax2.yaxis.label.set_color("tab:green")

    ax1.set_ylabel("P$_{HI}$(k), K$^2$(Mpc/h)$^3$")
    ax1.set_xlabel("k, h/Mpc")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.legend(loc="lower left")

