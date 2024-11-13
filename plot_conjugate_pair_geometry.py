"""
Plot the geometric interpretation of the conjugate pair F_LL and E at zero
coupling. The line v = - partial_sigma F_LL is drawn in the (sigma, v) plane
along with its mirroring about the line sigma + v = 0.

We use t = 1.
"""
import numpy as np
import matplotlib.pyplot as plt

from plot_config import PlotConfig as PC


PC.use_tex()


def main():
    fig, ax = plt.subplots(
        figsize=(PC.fig_width, PC.fig_height - 0.6)
        # figsize=(PC.fig_width, PC.fig_width)
    )
    lim = 2 / np.sqrt(5)
    sigma = np.linspace(-lim, lim, 100)
    v = - sigma / np.sqrt(1 - sigma**2)
    ax.plot(
        sigma,
        v,
        ls='-',
        lw=PC.linewidth,
        # 'o',
        c='k',
        label=r'$-\frac{\partial}{\partial \sigma} F_\mathrm{LL}^0$'
    )

    ax.plot(
        2*np.array([-1, 1]),
        2*np.array([1, -1]),
        ls=':',
        lw=PC.linewidth,
        c='k',
        label=r'$\sigma + v = 0$'
    )

    mirrored_x_coords = - v
    mirrored_y_coords = - sigma
    ax.plot(
        mirrored_x_coords,
        mirrored_y_coords,
        ls='--',
        lw=PC.linewidth,
        # 'x',
        c='k',
        label=r'$\frac{\partial}{\partial v} E^0$'
    )


    PC.set_ax_info(
        ax=ax,
        ylabel=r'$v$',  # temporary to get the save width as the other plots
        legend=True,
    )

    PC.tight_layout(fig, ax_aspect=1.75)
    ax.set_ylabel('')
    fig.savefig(PC.save_dir + '/conjugate-pair-geometry.pdf')
    # plt.show()


if __name__ == '__main__':
    main()
