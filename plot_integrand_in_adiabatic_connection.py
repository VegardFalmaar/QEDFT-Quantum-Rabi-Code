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
    for sigma, ls in zip([-0.4, 0.0, 0.6], ['--', '-', ':']):
        integrand_values = np.zeros_like(lmbda_values)
        for i, lmbda in enumerate(lmbda_values):
            qr = QuantumRabi(omega, t, g, lmbda=lmbda)
            integrand_values[i] = qr.T_integrand(lmbda, sigma)

        ax.plot(
            lmbda_values,
            integrand_values,
            label=r'$\sigma = ' f'{sigma:.2f}$',
            ls=ls,
            color='k',
        )
    ax.set_ylim(-0.2, 0.02)
    PC.set_ax_info(
        ax,
        xlabel=r'$\tau$',
        ylabel='Integrand',
        title=r'The integrand of $T_c^\lambda$, ' \
            r'$ \frac{1}{2} \langle \psi^\tau |  \hat \sigma_y \hat p | \psi^\tau \rangle$',
        legend=True,
    )

    # textbox for parameter values
    ax.text(
        x=0.99,
        y=0.98,
        s=r'$ t = \omega, \; g = 3 \omega^{3/2}, \; \xi = 0 $',
        alpha=0.7,
        fontsize=PC.fontsize_parameters,
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes
    )

    PC.tight_layout(fig, ax_aspect=3/2)

    p = {
        'omega': omega,
        't': t,
        'g': g,
        'xi': xi,
    }
    # fig.savefig(PC.save_fname('functional-title-params', '.pdf', p))
    fig.savefig(PC.save_fname('integrand-scaling', '.pdf', p))
    # plt.show()


if __name__ == '__main__':
    main()
