import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit    # type: ignore

from quantum_rabi import QuantumRabi
from plot_config import PlotConfig as PC


# PC.use_tex()


def sigma_values_from_logspace():
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
    return sigma_values


def ellipse_fit(sigma, a, b):
    return b/a * (np.sqrt(a**2 - sigma**2) - np.sqrt(a**2 - 1))
    # return b * np.sqrt(1 - sigma**2/a**2) - d


def plot_in_sigma():
    omega = 1.0
    t = 1.0 * omega
    g = 3.0 * omega**(3/2)
    xi = 0.0 / np.sqrt(omega)

    # sigma_values = np.linspace(-1.0, 1.0, 11)
    sigma_values = sigma_values_from_logspace()

    fig, ax = plt.subplots()
    for lmbda, c in zip([1.0, 0.6, 0.4], PC.colors):
        T_values = np.zeros_like(sigma_values)
        for i, sigma in enumerate(sigma_values):
            qr = QuantumRabi(omega, t, g, lmbda=lmbda)
            T_values[i] = qr.F(sigma, xi) - qr.analytic_terms_of_the_coupling(sigma, xi)

        ax.plot(
            sigma_values,
            T_values / omega,
            label=r'$\lambda = ' f'{lmbda:.2f}$',
            ls='-',
            color=c,
        )
        ellipsis_params, _ = curve_fit(ellipse_fit, sigma_values, T_values/omega)
        a, b, d = ellipsis_params
        ax.plot(
            sigma_values,
            ellipse_fit(sigma_values, a, b, d),
            ls='--',
            color='r'
        )
    PC.set_ax_info(
        ax,
        xlabel=r'$\sigma$',
        ylabel=r'$T_c^\lambda (\sigma) \, / \, \omega$',
        title='The Kinetic Correlation Functional',
        legend=True,
    )

    PC.parameter_text_box(
        ax,
        s=r'$ t = \omega, \; g = 3 \omega^{3/2} $',
        loc='upper right',
    )

    PC.tight_layout(fig, ax_aspect=3/2)

    p = {
        'omega': omega,
        't': t,
        'g': g,
        'xi': xi,
    }
    # fig.savefig(PC.save_fname('T-ellipse-fit', '.pdf', p))
    plt.show()


class Writer:
    def header(self):
        header = '| t / omega | lg /omega |       a       |       b       |       d       |'
        print('-'*len(header))
        print(header)
        print('-'*len(header))

    def line(self, t_factor, g_factor, a, b, d):
        for value in (t_factor, g_factor):
            print(f'| {value:9.5f} ', end='')
        for value in (a, b, d):
            print(f'| {value:13.10f} ', end='')
        print('|')


def tabulate_parameters():
    writer = Writer()
    writer.header()

    omega = 1.0
    sigma_values = sigma_values_from_logspace()

    fig, ax = plt.subplots()

    def line(t_factor, lg_factor):
        t = t_factor * omega
        g = lg_factor * omega**(3/2)
        y = np.zeros_like(sigma_values)
        for i, sigma in enumerate(sigma_values):
            qr = QuantumRabi(omega, t, g, lmbda=1)
            y[i] = qr.F(sigma, 0) - qr.analytic_terms_of_the_coupling(sigma, 0) / omega

        ellipsis_params, _ = curve_fit(ellipse_fit, sigma_values, y)
        a, b, d = ellipsis_params
        writer.line(t_factor, lg_factor, a, b, d)
        ax.plot(sigma_values, y, label=f'{t_factor=:.3f}, {lg_factor=:.3f}', color='k')
        ax.plot(sigma_values, ellipse_fit(sigma_values, a, b, d), color='r', ls=':')

    line(0.5, 0.5)
    line(0.5, 1.0)
    line(1.0, 0.5)
    line(1.0, 1.0)
    line(1.0, 2.0)
    line(1.0, 3.0)
    ax.legend()
    plt.show()


def plot_parameters():
    omega = 1.0
    sigma_values = sigma_values_from_logspace()[1:-1]

    fig, ax = plt.subplots()
    def fit():
        y = np.zeros_like(sigma_values)
        for i, sigma in enumerate(sigma_values):
            qr = QuantumRabi(omega, t, g, lmbda=1)
            T = qr.F(sigma, 0) - qr.analytic_terms_of_the_coupling(sigma, 0)
            y[i] = T

        ellipsis_params, pcov = curve_fit(
            ellipse_fit,
            sigma_values,
            y,
            # bounds=([1, 0], [np.inf, 10]),
            # bounds=([1, 0, -10], [np.inf, 10, 10]),
            bounds=([1, -np.inf], [np.inf, np.inf]),
        )
        perr = np.sqrt(np.diag(pcov))
        return ellipsis_params, perr

    t = 1.0 * omega
    # g = 1.0 * omega**(3/2)

    x_values = np.linspace(0.1, 3.0, 40)

    a_values = np.zeros_like(x_values)
    b_values = np.zeros_like(x_values)
    a_std_values = np.zeros_like(x_values)
    b_std_values = np.zeros_like(x_values)
    # for i, t in enumerate(x_values):
    for i, g in enumerate(x_values):
        (a, b), (a_std, b_std) = fit()
        # (a, b, d), (a_std, b_std, d_std) = fit()
        a_values[i] = a
        b_values[i] = b
        a_std_values[i] = a_std
        b_std_values[i] = b_std

    # ax.plot(x_values, a_values, label='a', color='b')
    # ax.fill_between(x_values, a_values - a_std, a_values + a_std, color='b', alpha=0.5)
    # ax.plot(x_values, b_values, label='b', color='g')
    # ax.fill_between(x_values, b_values - b_std, b_values + b_std, color='g', alpha=0.5)
    # ax.plot(x_values, d_values, label='d', color='r')
    # ax.fill_between(x_values, d_values - d_std, d_values + d_std, color='r', alpha=0.5)

    # t_values = x_values
    # ax.plot(x_values, a_values, label=r'$a$', color='b')
    # ax.plot(x_values, b_values / t_values, label=r'$b / t$', color='g')

    ax.plot(x_values, a_values, label=r'$a$')
    ax.plot(x_values, b_values, label=r'$b$')

    ax.set_title(r'$T_c^\lambda = \frac{b}{a} \left( \sqrt{a^2 - \sigma^2} - \sqrt{a^2 - 1} \right)$')
    ax.set_xlabel(r'$\lambda g / \omega^{3/2}$')
    # ax.set_xlabel(r'$t / \omega$')

    ax.legend()
    plt.show()

if __name__ == '__main__':
    # plot_in_sigma()
    # tabulate_parameters()
    plot_parameters()
