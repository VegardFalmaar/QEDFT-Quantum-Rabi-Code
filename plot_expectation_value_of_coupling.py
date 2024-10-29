import numpy as np
import matplotlib.pyplot as plt

from quantum_rabi import QuantumRabi
from plot_config import PlotConfig as PC


PC.use_tex()


def plot_expectation_value():
    omega = 1.0
    t = 1.0 * omega
    g = 3.0 * omega**(3/2)

    tau_values = np.linspace(0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(PC.fig_width, PC.fig_height))
    for sigma, ls in zip([0.0, 0.4, 0.6], PC.line_styles):
        expectation_values = np.zeros_like(tau_values)
        for i, tau in enumerate(tau_values):
            qr = QuantumRabi(omega, t, g, lmbda=tau)
            v, j = qr.minimizer_potential(sigma, 0)
            op_H = qr.op_H_0 + v*qr.op_sigma_z + j*qr.op_x
            gs = op_H.eig(hermitian=True)['eigenvectors'][0]
            expectation_values[i] = g * (qr.op_sigma_z*qr.op_x).expval(gs, transform_real=True)

        ax.plot(
            tau_values,
            expectation_values / omega,
            label=r'$\sigma = ' f'{sigma:.2f}$',
            ls=ls,
            lw=PC.linewidth,
            color='k',
        )
    PC.set_ax_info(
        ax,
        xlabel=r'$\tau$',
        ylabel=r'$ g \langle \varphi^\tau | \hat \sigma_z \hat x | \varphi^\tau \rangle \,/ \, \omega$',
        title=r'The expectation value of the coupling',
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
    }
    fig.savefig(PC.save_fname('expectation-value-of-the-coupling', '.pdf', p))


def plot_G():
    omega = 1.0
    t = 1.0 * omega
    g = 3.0 * omega**(3/2)

    lmbda_values = np.linspace(0, 1.0, 41)
    fig, ax = plt.subplots(figsize=(PC.fig_width, PC.fig_height))
    for sigma, ls in zip([0.0, 0.4, 0.6], PC.line_styles):
        G_values = np.zeros_like(lmbda_values)
        for i, lmbda in enumerate(lmbda_values):
            if i == 0:
                continue
            qr = QuantumRabi(omega, t, g, lmbda=lmbda, oscillator_size=40)
            G_values[i] = qr.G(sigma)

        ax.plot(
            lmbda_values,
            G_values / omega,
            label=r'$\sigma = ' f'{sigma:.2f}$',
            ls=ls,
            lw=PC.linewidth,
            color='k',
        )
    PC.set_ax_info(
        ax,
        xlabel=r'$\lambda$',
        ylabel=r'$ G^\lambda (\sigma) $',
        title=r'The Scaling of the Correlation Coupling',
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
    }
    fig.savefig(PC.save_fname('G-scaling-with-lambda', '.pdf', p))


if __name__ == '__main__':
    # plot_expectation_value()
    plot_G()
