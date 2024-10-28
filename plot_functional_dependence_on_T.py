import numpy as np
import matplotlib.pyplot as plt

from quantum_rabi import QuantumRabi
from plot_config import PlotConfig as PC


PC.use_tex()


def main():
    omega = 1.0
    t = 1.0 * omega
    g = 3.0 * omega**(3/2)
    sigma = 0.6
    xi = 0.0 / np.sqrt(omega)

    lmbda_values = np.linspace(0, 1.0, 41)
    # height adjusted to avoid problems in rescaling later
    fig, ax = plt.subplots(figsize=(PC.fig_width, PC.fig_height - 0.0591))

    F_values = np.zeros_like(lmbda_values)
    analytic_terms = np.zeros_like(lmbda_values)
    for i, lmbda in enumerate(lmbda_values):
        qr = QuantumRabi(omega, t, g, lmbda=lmbda)
        F_values[i] = qr.F(sigma, xi)
        analytic_terms[i] = qr.analytic_terms_of_the_coupling(sigma, xi)

    ax.plot(
        lmbda_values,
        F_values / omega,
        label=r'$F_\mathrm{LL}^\lambda$',
        ls='-',
        color='k',
    )
    ax.plot(
        lmbda_values,
        analytic_terms / omega,
        label=r'$F_\mathrm{LL}^\lambda - T_c^\lambda$',
        ls='--',
        color='k',
    )

    ax.set_ylim(-3.5, 0.1)
    PC.set_ax_info(
        ax,
        xlabel=r'$\lambda$',
        ylabel=r'Functional $/ \, \omega$',
        title=r'The Adiabatic Connection of $F_\mathrm{LL}$',
        legend=True,
    )

    PC.parameter_text_box(
        ax,
        s=r'$ t = \omega, \; g = 3 \omega^{3/2}, \; \sigma = 0.6, \; \xi = 0 $',
        loc='upper right',
    )

    PC.tight_layout(fig, ax_aspect=3/2)

    p = {
        'omega': omega,
        't': t,
        'g': g,
        'sigma': sigma,
        'xi': xi,
    }
    fig.savefig(PC.save_fname('functional-with-and-without-T', '.pdf', p))
    # plt.show()


if __name__ == '__main__':
    main()
