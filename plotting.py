""" Plotting functions for outputs and analysis objects. """

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from hmf import MassFunction
from postEoR.generation import hlittle, OMm
plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = [9, 6]
plt.style.use('seaborn-v0_8-ticks')


def plot_lightcone(
    lightcone, 
    lightcone_item, 
    save_loc : str,
    title : str,
    label="Brightness temperature, K",
):
    """
    Plot and save the redshift evolution of a given quantity of the lightcone (e.g. brightness temperature, density).

    Parameters
    ----------
    lightcone : Lightconer class
        The lightcone class object output by 21cmFAST.
    lightcone_item : NDarray
        Contains the 3D distribution of the quantity to be plot (output by the functions in generation.py).
    save_loc : str
        The path and filename for the plot to be saved to.
    label : str (optional)
        The label to be put on the colorbar on the plot (defaults to brightness temperature in K if no input given).
    title : str (optional)
        The title of the plot.
    """
    plt.clf() # clearing any previous plots
    fig, ax = plt.subplots(1,1, constrained_layout=True)

    z_axis = "x"
    slice_axis = 0
    axis_dct = {
            "x": 2 if z_axis == "x" else [1, 0, 0][slice_axis],
            "y": 2 if z_axis == "y" else [1, 0, 1][slice_axis],
        }
    extent = (
            0,
            lightcone.lightcone_dimensions[axis_dct["x"]],
            0,
            lightcone.lightcone_dimensions[axis_dct["y"]],
        )

    plot = ax.imshow(lightcone_item[:, 10, :], extent=extent)
    cbar = fig.colorbar(plot, ax=ax)
    cbar.set_label(str(label), rotation=270, labelpad = 12)

    fig.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    labels = [item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels([str(round(float(x), 1)) for x in labels if label != ''])
    ax.set_xlabel("Redshift")
    ax.set_ylabel("y, Mpc/h")
    fig.suptitle(title)

    plt.savefig(save_loc)


def plot_colormaps(
    BT, 
    dens, 
    z : float, 
    HII_dim : float, 
    box_len : float,
    save_loc : str,
):
    """
    Plot and save colormaps of a slice of the field, for the brightness temperature and overdensity.

    Parameters
    ----------
    BT : NDarray
        The brightness temperature field.
    dens : NDarray
        The overdensity field.
    z : float
        The redshift at which the fields were evaluated.
    HII_dim : int
        The number of cells in each of the spatial dimensions of the box / cone.
    box_len : float
        The physical length of each of the spatial dimensions of the box / cone, in Mpc. 
    save_loc : str
        The path and filename for the plot to be saved to.
    """
    plt.clf() # clearing any previous plots
    plt.rcParams['figure.figsize'] = [13, 6]
    fig, (ax1, ax3) = plt.subplots(1,2)

    # box coordinates
    fin_size = np.shape(BT)[0] / HII_dim * box_len
    dx, dy = box_len / HII_dim, box_len / HII_dim 
    y1, x1 = np.mgrid[slice(dy / 2, fin_size, dy), slice(dx / 2, fin_size, dx)]

    # plotting colormaps of overdensity, neutral fraction, brightness temperature
    cb1 = ax1.pcolormesh(x1, y1, dens[:, :, 10], cmap = "magma")
    ax1.set_xlabel('$x$, Mpc/h')
    ax1.set_ylabel('$y$, Mpc/h')
    cbar1 = fig.colorbar(cb1)
    cbar1.set_label('Overdensity', rotation=270, labelpad = 12)

    cb3 = ax3.pcolormesh(x1, y1, BT[:, :, 10], cmap = "magma")
    ax3.set_xlabel('$x$, Mpc/h')
    cbar3 = fig.colorbar(cb3)
    cbar3.set_label('Brightness temperature, K', rotation=270, labelpad = 12)
    cbar3.formatter.set_powerlimits((0, 0))

    fig.suptitle("HII_dim " + str(HII_dim) + ", box_len " + str(box_len) + ",  z = " + str(z))
    fig.tight_layout()

    plt.savefig(str(save_loc))


def plot_mfs(counts, 
    bins, 
    los_dist : float, 
    box_len : float, 
    z : float, 
    save_loc : str,
    title : str,
    color="tab:blue",
):
    """
    Plot and save the mass functions output by the get_hmf, get_himf functions.

    Parameters
    ----------
    counts : NDarray
        The number of objects falling in each halo mass bin.
    bins : NDarray
        The mass bins used to create the HMF.
    los_dist : float
        The physical distance corresponding to the redshift interval the lightcone is generated over.
    box_len : float
        The physical length of the spatial dimensions of the box / lightcone.
    z : float
        The redshift at which the mass function was evaluated.
    save_loc : str
        The path and filename for the plot to be saved to.
    title : str 
        The title of the plot.
    color : str (optional)
        The desired color of the plot.
    """
    plt.clf() # clearing any previous plots
    bins_plot = (bins[1:] + bins[:-1]) / 2
    plt.hist(bins_plot, bins, weights=(2 * counts / (np.log(bins[1:]+bins[:-1])*box_len**2*los_dist)), histtype='step', label='z = ' + str(z), color=str(color))
    mf1 = MassFunction(z = z,
                  cosmo_params={"Om0":OMm}, 
                  hmf_model="Watson") 
    
    plt.plot(mf1.m * hlittle,mf1.dndm * mf1.m, label="z = " + str(z), linestyle="--", color=str(color))
    plt.title(title)
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel('$\dfrac{dn}{d\log M}$, (h/Mpc)$^3$')
    plt.xlabel('M, h$^{-1}$M$_{\odot}$')

    plt.savefig(str(save_loc))


def plot_ps(ps, 
    k, 
    z : float, 
    marker : str, 
    ax, 
    label : str, 
    save_loc : str, 
    title : str,
    color="tab:blue", 
):
    """
    Plot and save power spectra.

    Parameters
    ----------
    ps : NDarray
        The power spectrum.
    k : NDarray
        The k values corresponding to the power spectrum
    z : float
        The redshift at which the power spectrum was evaluated.
    linestyle : str
        The desired linestyle of the plot.
    ax : Axes object
        The axis on which to plot the power spectra.
    label : str
        The label of the item being plotted
    save_loc : str
        The path and filename for the plot to be saved to.
    title : str 
        The title of the plot.
    color : str (optional)
        The desired color of the plot.
    """
    plt.clf() # clearing any previous plots
    ax.scatter(k, ps, color=str(color), label="z = " + str(z) + ", " + str(label), marker=str(marker))

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("P(k), (Mpc/h)$^3$")
    ax.set_xlabel("k, (h/Mpc)$^{-1}$")
    ax.legend()
    ax.set_title(title)

    plt.savefig(str(save_loc))