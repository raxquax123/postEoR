import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from hmf import MassFunction
from .constants import *


def plot_lightcone(lightcone, lightcone_item, label="Brightness temperature, mK"): # processed lightcone plotting
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
    cbar.set_label(label, rotation=270, labelpad = 12)

    fig.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    labels = [item.get_text() for item in ax.get_xticklabels()]
    ax.set_xticklabels([str(round(float(label), 1)) for label in labels if label != ''])

    ax.set_xlabel("Redshift")
    ax.set_ylabel("y (Mpc)")



def plot_colormaps(BT, dens, z, HII_dim, box_len): # producing colormaps
    plt.rcParams['figure.figsize'] = [13, 6]
    fig, (ax1, ax3) = plt.subplots(1,2)

    # box coordinates
    fin_size = np.shape(BT)[0] / HII_dim * box_len
    dx, dy = box_len / HII_dim, box_len / HII_dim 
    y1, x1 = np.mgrid[slice(dy / 2, fin_size, dy), slice(dx / 2, fin_size, dx)]

    # plotting colormaps of overdensity, neutral fraction, brightness temperature
    cb1 = ax1.pcolormesh(x1, y1, dens[:, :, 10], cmap = "viridis")
    ax1.set_xlabel('$x$ (Mpc)')
    ax1.set_ylabel('$y$ (Mpc)')
    cbar1 = fig.colorbar(cb1)
    cbar1.set_label('Overdensity', rotation=270, labelpad = 12)

    cb3 = ax3.pcolormesh(x1, y1, BT[:, :, 10], cmap = "viridis")
    ax3.set_xlabel('$x$ (Mpc)')
    cbar3 = fig.colorbar(cb3)
    cbar3.set_label('Brightness temperature, mK', rotation=270, labelpad = 12)
    cbar3.formatter.set_powerlimits((0, 0))

    fig.suptitle("HII_dim " + str(HII_dim) + ", box_len " + str(box_len) + ",  z = " + str(z))

    fig.tight_layout()

    plt.show()


def plot_mfs(counts, bins, los_dist, box_len, z, color):
    plt.hist(bins[:-1], bins, weights=(2 * counts / (np.log(bins[1:]+bins[:-1])*box_len**2*los_dist)), histtype='step', label='z = ' + str(z), color=str(color))
    mf1 = MassFunction(z = z,
                  cosmo_params={"Om0":omega_m}, 
                  hmf_model="Watson") 
    plt.plot(mf1.m * little_h,mf1.dndm * mf1.m, label="z = " + str(z), linestyle="--", color=str(color))


def plot_ps(ps, k, z, color, linestyle):
    