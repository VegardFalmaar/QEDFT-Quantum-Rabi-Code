import numpy as np
import matplotlib.pyplot as plt

from quantum_rabi import QuantumRabi
from plot_config import PlotConfig as PC


PC.use_tex()


def plot_expectation_value():
    omega = 1.0
    t = 1.0 * omega
    g = 3.0 * omega**(3/2)

    nu_values = np.linspace(0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(PC.fig_width, PC.fig_height))

    qr = QuantumRabi(omega, t, g)
    for sigma, ls in zip([0.0, 0.4, 0.6], PC.line_styles):
        expectation_values = np.zeros_like(nu_values)
        for i, nu in enumerate(nu_values):
            expectation_values[i] = qr.G_integrand(nu=nu, sigma=sigma)

        ax.plot(
            nu_values,
            expectation_values / omega,
            label=r'$\sigma = ' f'{sigma:.1f}$',
            ls=ls,
            lw=PC.linewidth,
            color='k',
        )
    PC.set_ax_info(
        ax,
        xlabel=r'$\nu$',
        ylabel=r'$ g \langle \varphi^\nu ' \
            r'| \hat \sigma_z \hat x | \varphi^\nu \rangle $',
        legend=True,
    )

    PC.parameter_text_box(
        ax,
        s=r'$ \omega = 1, \; t = 1, \; g = 3, \; \xi = 0 $',
        loc='upper right',
    )

    PC.tight_layout(fig, ax_aspect=1.75)

    p = {
        'omega': omega,
        't': t,
        'g': g,
    }
    fig.savefig(PC.save_fname('expectation-value-of-the-coupling', '.pdf', p))


def plot_G():
    omega = 1.0
    t = 1.0 * omega
    g = 3.0 * omega**(3/2)

    lmbda_values = np.linspace(0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(PC.fig_width, PC.fig_height - 0.5))
    for sigma, ls in zip([0.0, 0.4, 0.6], PC.line_styles):
        G_values = np.zeros_like(lmbda_values)
        for i, lmbda in enumerate(lmbda_values):
            if i == 0:
                continue
            qr = QuantumRabi(omega, t, g, lmbda=lmbda, oscillator_size=40)
            # G_values[i] = qr.G_from_T(sigma)
            G_values[i] = qr.G_from_integration(sigma)

        ax.plot(
            lmbda_values,
            G_values / omega,
            label=r'$\sigma = ' f'{sigma:.1f}$',
            ls=ls,
            lw=PC.linewidth,
            color='k',
        )
    PC.set_ax_info(
        ax,
        xlabel=r'$\lambda$',
        ylabel=r'$ G^\lambda (\sigma) $',
        legend=True,
    )

    PC.parameter_text_box(
        ax,
        s=r'$ \omega = 1, \; t = 1, \; g = 3, \; \xi = 0 $',
        loc='upper right',
    )

    PC.tight_layout(fig, ax_aspect=1.75)

    p = {
        'omega': omega,
        't': t,
        'g': g,
    }
    fig.savefig(PC.save_fname('expectation-value-of-the-coupling-G-in-lambda', '.pdf', p))


if __name__ == '__main__':
    plot_expectation_value()
    plot_G()
