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
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(PC.fig_width, 1.5*PC.fig_height),
        sharex=True
    )

    # first plot, functional in lambda for different values of sigma
    for sigma, ls in zip([0.0, 0.4, 0.6], PC.line_styles):
        F_values = np.zeros_like(lmbda_values)
        for i, lmbda in enumerate(lmbda_values):
            qr = QuantumRabi(omega, t, g, lmbda=lmbda)
            F_values[i] = qr.F_from_minimization(sigma, xi)

        ax1.plot(
            lmbda_values,
            F_values / omega,
            label=r'$\sigma = ' f'{sigma:.1f}$',
            ls=ls,
            lw=PC.linewidth,
            color='k',
        )
    ax1.set_ylim(-4.2, 0.1)
    PC.set_ax_info(
        ax1,
        ylabel=r'$F_\mathrm{LL}^\lambda (\sigma, \xi)$',
        legend=True,
    )

    PC.parameter_text_box(
        ax1,
        s=r'$ \omega = 1, \; t = 1, \; g = 3, \; \xi = 0 $',
        loc='upper right',
    )


    # second plot, functional and all terms except T
    sigma = 0.6

    F_values = np.zeros_like(lmbda_values)
    analytic_terms = np.zeros_like(lmbda_values)
    for i, lmbda in enumerate(lmbda_values):
        qr = QuantumRabi(omega, t, g, lmbda=lmbda)
        F_values[i] = qr.F_from_minimization(sigma, xi)
        analytic_terms[i] = qr.analytic_terms_of_F(sigma, xi)

    ax2.plot(
        lmbda_values,
        F_values / omega,
        label=r'$F_\mathrm{LL}^\lambda$',
        ls='-',
        lw=PC.linewidth,
        color='k',
    )
    ax2.plot(
        lmbda_values,
        analytic_terms / omega,
        label=r'$F_\mathrm{LL}^\lambda - T_c^\lambda$',
        ls='--',
        lw=PC.linewidth,
        color='k',
    )

    ax2.set_ylim(-3.5, 0.1)
    PC.set_ax_info(
        ax2,
        xlabel=r'$\lambda$',
        legend=True,
    )

    PC.parameter_text_box(
        ax2,
        s=r'$ \omega = 1, \; t = 1, \; g = 3, \; \sigma = 0.6, \; \xi = 0 $',
        loc='upper right',
    )

    fig.tight_layout(pad=0.1)

    p = {
        'omega': omega,
        't': t,
        'g': g,
    }
    fig.savefig(PC.save_fname('functional-in-lambda', '.pdf', p))
    # plt.show()


if __name__ == '__main__':
    main()
