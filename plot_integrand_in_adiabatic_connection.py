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
        integrand_values = np.zeros_like(lmbda_values)
        for i, lmbda in enumerate(lmbda_values):
            qr = QuantumRabi(omega, t, g, lmbda=lmbda)
            integrand_values[i] = qr.T_integrand(lmbda, sigma)

        ax.plot(
            lmbda_values,
            integrand_values * omega**(-0.5),
            label=r'$\sigma = ' f'{sigma:.2f}$',
            ls=ls,
            color='k',
        )
    ax.set_ylim(-0.2, 0.02)
    PC.set_ax_info(
        ax,
        xlabel=r'$\tau$',
        ylabel=r'Integrand $/ \, \sqrt \omega$',
        title=r'The integrand of $T_c^\lambda$, ' \
            r'$ \frac{1}{2} \langle \psi^\tau |  \hat \sigma_y \hat p | \psi^\tau \rangle$',
        legend=True,
    )

    PC.parameter_text_box(
        ax,
        s=r'$ t = \omega, \; g = 3 \omega^{3/2}, \; \xi = 0 $',
        loc='upper right',
    )

    PC.tight_layout(fig, ax_aspect=3/2)

    p = {
        'omega': omega,
        't': t,
        'g': g,
        'xi': xi,
    }
    fig.savefig(PC.save_fname('integrand-of-T-scaling', '.pdf', p))
    # plt.show()


if __name__ == '__main__':
    main()
