import numpy as np
import matplotlib.pyplot as plt

from quantum_rabi import QuantumRabi
from plot_config import PlotConfig as PC


PC.use_tex()


def main():
    omega = 1.0
    t = 1.0 * omega
    g = 3.0 * omega**(3/2)
    xi = 0.0 / np.sqrt(omega)

    lmbda_values = np.linspace(0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(PC.fig_width, PC.fig_height))
    for sigma, ls in zip([0.0, 0.4, 0.6], PC.line_styles):
        F_values = np.zeros_like(lmbda_values)
        for i, lmbda in enumerate(lmbda_values):
            qr = QuantumRabi(omega, t, g, lmbda=lmbda)
            F_values[i] = qr.F(sigma, xi)

        ax.plot(
            lmbda_values,
            F_values / omega,
            label=r'$\sigma = ' f'{sigma:.2f}$',
            ls=ls,
            lw=PC.linewidth,
            color='k',
        )
    ax.set_ylim(-4.2, 0.1)
    PC.set_ax_info(
        ax,
        xlabel=r'$\lambda$',
        ylabel=r'$F_\mathrm{LL}^\lambda (\sigma, \xi)$',
        legend=True,
    )

    PC.parameter_text_box(
        ax,
        s=r'$ \omega = 1, \; t = 1, \; g = 3, \; \xi = 0 $',
        loc='upper right',
    )

    PC.tight_layout(fig, ax_aspect=3/2)

    p = {
        'omega': omega,
        't': t,
        'g': g,
        'xi': xi,
    }
    fig.savefig(PC.save_fname('functional-in-lambda', '.pdf', p))
    # plt.show()


if __name__ == '__main__':
    main()
