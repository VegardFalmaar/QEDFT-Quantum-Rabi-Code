import numpy as np
import matplotlib.pyplot as plt

from quantum_rabi import QuantumRabi
from plot_config import PlotConfig as PC


PC.use_tex()


def plot_in_lambda():
    omega = 1.0
    t = 1.0 * omega
    g = 3.0 * omega**(3/2)
    xi = 0.0 / np.sqrt(omega)

    lmbda_values = np.linspace(0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(PC.fig_width, PC.fig_height))
    for sigma, ls in zip([0.0, 0.4, 0.6], PC.line_styles):
        T_values = np.zeros_like(lmbda_values)
        for i, lmbda in enumerate(lmbda_values):
            qr = QuantumRabi(omega, t, g, lmbda=lmbda)
            T_values[i] = qr.F(sigma, xi) - qr.analytic_terms_of_the_coupling(sigma, xi)

        ax.plot(
            lmbda_values,
            T_values / omega,
            label=r'$\sigma = ' f'{sigma:.1f}$',
            ls=ls,
            lw=PC.linewidth,
            color='k',
        )
    PC.set_ax_info(
        ax,
        xlabel=r'$\lambda$',
        ylabel=r'$T_c^\lambda (\sigma)$',
        legend=True,
    )

    PC.parameter_text_box(
        ax,
        s=r'$ \omega = 1, \; t = 1, \; g = 3 $',
        loc='lower right',
    )

    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    PC.tight_layout(fig, ax_aspect=1.75)

    p = {
        'omega': omega,
        't': t,
        'g': g,
        'xi': xi,
    }
    fig.savefig(PC.save_fname('kinetic-correlation-functional-in-lambda', '.pdf', p))


def plot_in_sigma():
    omega = 1.0
    t = 1.0 * omega
    g = 3.0 * omega**(3/2)
    xi = 0.0 / np.sqrt(omega)

    # The y-values change rapidly at the end points, so to make the plot nice
    # we should have denser plotting points there. The following section
    # creates an array of N points between -1 and 1 which is more densely
    # populated around the end points using numpys logspace.
    def logspace_0_to_1(n: int):
        eps = 1e-8
        return (np.logspace(0, np.log10(11) - eps, n) - 1) / 10
    N = 100
    sigma_values = np.hstack([
        logspace_0_to_1(N//2) - 1,
        1 - logspace_0_to_1(N//2)[::-1]
    ])
    # sigma_values = np.linspace(-1.0, 1.0, 11)

    fig, ax = plt.subplots(figsize=(PC.fig_width, PC.fig_height))
    for lmbda, ls in zip([1.0, 0.6, 0.4], PC.line_styles):
        T_values = np.zeros_like(sigma_values)
        for i, sigma in enumerate(sigma_values):
            qr = QuantumRabi(omega, t, g, lmbda=lmbda)
            T_values[i] = qr.F(sigma, xi) - qr.analytic_terms_of_the_coupling(sigma, xi)

        ax.plot(
            sigma_values,
            T_values / omega,
            label=r'$\lambda = ' f'{lmbda:.1f}$',
            ls=ls,
            lw=PC.linewidth,
            color='k',
        )
    PC.set_ax_info(
        ax,
        xlabel=r'$\sigma$',
        ylabel=r'$T_c^\lambda (\sigma)$',
        legend=True,
    )

    PC.parameter_text_box(
        ax,
        s=r'$ \omega = 1, \; t = 1, \; g = 3 $',
        loc='upper right',
    )

    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    PC.tight_layout(fig, ax_aspect=1.75)

    p = {
        'omega': omega,
        't': t,
        'g': g,
        'xi': xi,
    }
    fig.savefig(PC.save_fname('kinetic-correlation-functional-in-sigma', '.pdf', p))


if __name__ == '__main__':
    plot_in_lambda()
    plot_in_sigma()
