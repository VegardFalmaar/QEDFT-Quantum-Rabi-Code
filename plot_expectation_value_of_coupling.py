import numpy as np
import matplotlib.pyplot as plt

from quantum_rabi import QuantumRabi
from plot_config import PlotConfig as PC


PC.use_tex()


def main():
    omega = 1.0
    t = 1.0 * omega
    g = 3.0 * omega**(3/2)

    lmbda_values = np.linspace(0, 1.0, 41)
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(PC.fig_width, 1.5*PC.fig_height),
        sharex=True
    )

    qr = QuantumRabi(omega, t, g)
    for sigma, ls in zip([0.0, 0.4, 0.6], PC.line_styles):
        expectation_values = np.zeros_like(lmbda_values)
        for i, nu in enumerate(lmbda_values):
            expectation_values[i] = qr.G_integrand(nu=nu, sigma=sigma)

        ax1.plot(
            lmbda_values,
            expectation_values / omega,
            label=r'$\sigma = ' f'{sigma:.1f}$',
            ls=ls,
            lw=PC.linewidth,
            color='k',
        )
    PC.set_ax_info(
        ax1,
        ylabel=r'$ g \langle \varphi^\lambda ' \
            r'| \hat \sigma_z \hat x | \varphi^\lambda \rangle $',
        legend=True,
    )

    PC.parameter_text_box(
        ax1,
        s=r'$ \omega = 1, \; t = 1, \; g = 3 $',
        loc='upper right',
    )

    for sigma, ls in zip([0.0, 0.4, 0.6], PC.line_styles):
        G_values = np.zeros_like(lmbda_values)
        for i, lmbda in enumerate(lmbda_values):
            if i == 0:
                continue
            qr = QuantumRabi(omega, t, g, lmbda=lmbda, oscillator_size=40)
            G_values[i] = qr.G_from_T(sigma)

        ax2.plot(
            lmbda_values,
            G_values / omega,
            label=r'$\sigma = ' f'{sigma:.1f}$',
            ls=ls,
            lw=PC.linewidth,
            color='k',
        )
    PC.set_ax_info(
        ax2,
        xlabel=r'$\lambda$',
        ylabel=r'$ G^\lambda (\sigma) $',
        legend=True,
    )

    PC.parameter_text_box(
        ax2,
        s=r'$ \omega = 1, \; t = 1, \; g = 3 $',
        loc='upper right',
    )

    fig.tight_layout(pad=0.1)

    p = {
        'omega': omega,
        't': t,
        'g': g,
    }
    fig.savefig(PC.save_fname('expectation-value-of-the-coupling-new', '.pdf', p))


if __name__ == '__main__':
    main()
