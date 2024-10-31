from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy.optimize import curve_fit    # type: ignore

from quantum_rabi import QuantumRabi
from plot_config import PlotConfig as PC
from utils import ProgressBar


# PC.use_tex()


def sigma_values_from_logspace():
    # The y-values change rapidly at the end points, so to make the plot nice
    # we should have denser plotting points there. The following section
    # creates an array of N points between -1 and 1 which is more densely
    # populated around the end points using numpys logspace.
    # The endpoints sigma=+1 and sigma=-1 are removed.
    def logspace_0_to_1(n: int):
        eps = 1e-8
        return (np.logspace(0, np.log10(11) - eps, n) - 1) / 10
    N = 100
    sigma_values = np.hstack([
        logspace_0_to_1(N//2) - 1,
        1 - logspace_0_to_1(N//2)[::-1]
    ])
    return sigma_values[1:-1]


def ellipse_fit(sigma, a, b):
    return b/a * (np.sqrt(a**2 - sigma**2) - np.sqrt(a**2 - 1))


def plot_T_and_ellipse_in_sigma_from_compute():
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
        ellipse_params, _ = curve_fit(ellipse_fit, sigma_values, T_values/omega)
        a, b = ellipse_params
        ax.plot(
            sigma_values,
            ellipse_fit(sigma_values, a, b),
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


def tabulate_ellipse_parameters():
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

        ellipse_params, _ = curve_fit(ellipse_fit, sigma_values, y)
        a, b, d = ellipse_params
        writer.line(t_factor, lg_factor, a, b, d)
        ax.plot(sigma_values, y, label=f'{t_factor=:.3f}, {lg_factor=:.3f}', color='k')
        ax.plot(sigma_values, ellipse_fit(sigma_values, a, b), color='r', ls=':')

    line(0.5, 0.5)
    line(0.5, 1.0)
    line(1.0, 0.5)
    line(1.0, 1.0)
    line(1.0, 2.0)
    line(1.0, 3.0)
    ax.legend()
    plt.show()


def plot_ellipse_parameters_from_compute():
    omega = 1.0
    sigma_values = sigma_values_from_logspace()

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
            bounds=([1, -np.inf], [np.inf, np.inf]),
        )
        perr = np.sqrt(np.diag(pcov))
        return ellipsis_params, perr

    # t = 1.0 * omega
    g = 1.0 * omega**(3/2)

    x_values = np.linspace(0.05, 0.15, 3)

    a_values = np.zeros_like(x_values)
    b_values = np.zeros_like(x_values)
    a_std_values = np.zeros_like(x_values)
    b_std_values = np.zeros_like(x_values)
    # for i, g in enumerate(x_values):
    for i, t in enumerate(x_values):
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

    t_values = x_values
    ax.plot(x_values, a_values, label=r'$a$', color='b')
    ax.plot(x_values, b_values / t_values, label=r'$b / t$', color='g')

    # ax.plot(x_values, a_values, label=r'$a$')
    # ax.plot(x_values, b_values, label=r'$b$')

    ax.set_title(
        r'$T_c^\lambda = \frac{b}{a} \left( \sqrt{a^2 - \sigma^2} ' \
        r'- \sqrt{a^2 - 1} \right)$'
    )
    # ax.set_xlabel(r'$\lambda g / \omega^{3/2}$')
    ax.set_xlabel(r'$t / \omega$')

    ax.legend()
    plt.show()


def precompute_data():
    folder = Path('parameters_to_circle_fit')
    run = 2
    omega = 1.0
    sigma_values = sigma_values_from_logspace()

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
            bounds=([1, -np.inf], [np.inf, np.inf]),
        )
        perr = np.sqrt(np.diag(pcov))
        return ellipsis_params, perr, y

    t_values = np.linspace(0.05, 3.0, 60)
    g_values = np.linspace(0.00, 2.5, 51)
    # t_values = np.linspace(0.05, 3.0, 2)
    # g_values = np.linspace(0.0, 2.5, 2)

    a_values = np.zeros((len(t_values), len(g_values)))
    b_values = np.zeros_like(a_values)
    a_std_values = np.zeros_like(a_values)
    b_std_values = np.zeros_like(a_values)
    T_values = np.zeros((len(t_values), len(g_values), len(sigma_values)))

    total_iterations = len(t_values) * len(g_values)
    bar = ProgressBar(total_iterations)
    error_messages = []
    for i, t in enumerate(t_values):
        for j, g in enumerate(g_values):
            msg = f'Starting {t = :.3f}, {g = :.3f}'
            bar.write(msg)
            try:
                (a, b), (a_std, b_std), T = fit()
            except ValueError as e:
                error_messages.append(msg + ':' + e)
                print(f'Error: {e}')
            a_values[i, j] = a
            b_values[i, j] = b
            a_std_values[i, j] = a_std
            b_std_values[i, j] = b_std
            T_values[i, j] = T
    bar.write('Done!')

    num_errors = len(error_messages)
    print(f'{num_errors} errors')
    if num_errors > 0:
        for error_message in error_messages:
            print(error_message)

    np.save(folder / f'run_{run}_t', t_values)
    np.save(folder / f'run_{run}_lmbda', g_values)
    np.save(folder / f'run_{run}_a', a_values)
    np.save(folder / f'run_{run}_b', b_values)
    np.save(folder / f'run_{run}_a_std', a_std_values)
    np.save(folder / f'run_{run}_b_std', b_std_values)
    np.save(folder / f'run_{run}_T', T_values)


def interactive_plot_of_ellipse_params():
    fig, axes = plt.subplots(figsize=(18, 8), nrows=1, ncols=2)
    fig.subplots_adjust(bottom=0.3)
    fig.suptitle(r'$T_c^\lambda = \frac{b}{a} ' \
        r'\left( \sqrt{a^2 - \sigma^2} - \sqrt{a^2 - 1} \right)$'
    )

    def load_for_specific_t(idx):
        return (
            np.load('parameters_to_circle_fit/run_1_a.npy')[idx, :],
            np.load('parameters_to_circle_fit/run_1_b.npy')[idx, :],
            np.load('parameters_to_circle_fit/run_1_a_std.npy')[idx, :],
            np.load('parameters_to_circle_fit/run_1_b_std.npy')[idx, :]
        )

    def load_for_specific_lmbda(idx):
        return (
            np.load('parameters_to_circle_fit/run_1_a.npy')[:, idx],
            np.load('parameters_to_circle_fit/run_1_b.npy')[:, idx],
            np.load('parameters_to_circle_fit/run_1_a_std.npy')[:, idx],
            np.load('parameters_to_circle_fit/run_1_b_std.npy')[:, idx]
        )

    t_values = np.load('parameters_to_circle_fit/run_1_t.npy')
    lmbda_values = np.load('parameters_to_circle_fit/run_1_lmbda.npy')

    r"""
    ### left plot in lambda
    ax = axes[0]
    # smallest value of t
    t = t_values[0]
    a, b, a_std, b_std = load_for_specific_t(0)
    ax.plot(lmbda_values, a, label=f'$a, {t=:.3f}$', color='b', ls=':')
    # ax.fill_between(lmbda_values, a - a_std, a + a_std, color='b', alpha=0.5)
    ax.plot(lmbda_values, b, label=f'$b, {t=:.3f}$', color='r', ls=':')
    # ax.fill_between(lmbda_values, b - b_std, b + b_std, color='r', alpha=0.5)
    # largest value of t
    t = t_values[-1]
    a, b, a_std, b_std = load_for_specific_t(-1)
    ax.plot(lmbda_values, a, label=f'$a, {t=:.3f}$', color='b', ls='--')
    # ax.fill_between(lmbda_values, a - a_std, a + a_std, color='b', alpha=0.5)
    ax.plot(lmbda_values, b, label=f'$b, {t=:.3f}$', color='r', ls='--')
    # ax.fill_between(lmbda_values, b - b_std, b + b_std, color='r', alpha=0.5)

    ### right plot in t
    ax = axes[1]
    # smallest value of lmbda
    lmbda = lmbda_values[0]
    a, b, a_std, b_std = load_for_specific_lmbda(0)
    # ax.plot(t_values, a, label=r'$a, \lambda=' f'{lmbda:.3f}$', color='b', ls=':')
    # ax.fill_between(t_values, a - a_std, a + a_std, color='b', alpha=0.5)
    ax.plot(t_values, b, label=r'$b, \lambda=' f'{lmbda:.3f}$', color='r', ls=':')
    # ax.fill_between(t_values, b - b_std, b + b_std, color='r', alpha=0.5)
    # largest value of lmbda
    lmbda = lmbda_values[-1]
    a, b, a_std, b_std = load_for_specific_lmbda(-1)
    ax.plot(t_values, a, label=r'$a, \lambda=' f'{lmbda:.3f}$', color='b', ls='--')
    # ax.fill_between(t_values, a - a_std, a + a_std, color='b', alpha=0.5)
    ax.plot(t_values, b, label=r'$b, \lambda=' f'{lmbda:.3f}$', color='r', ls='--')
    """
    # ax.fill_between(t_values, b - b_std, b + b_std, color='r', alpha=0.5)

    # add axes for the sliders
    # fig.add_axes([
        # horisontal start position from left,
        # vertical start position from bottom,
        # width of the ax panel,
        # height of the ax panel,
    # ])
    ax_t_slider = fig.add_axes((0.1, 0.15, 0.25, 0.03))
    ax_lmbda_slider = fig.add_axes((0.1, 0.1, 0.25, 0.03))

    # create the sliders
    slider_t = Slider(
        ax_t_slider,
        r'Index $t$',
        valmin=0,
        valmax=len(t_values) - 1,
        valinit=len(t_values) // 2,
        valstep=1,
    )
    slider_lmbda = Slider(
        ax_lmbda_slider,
        r'Index $\lambda$',
        valmin=0,
        valmax=len(lmbda_values) - 1,
        valinit=len(lmbda_values) // 2,
        valstep=1,
    )

    # plot lines for adjustable value of t
    t = t_values[slider_t.val]
    a, b, a_std, b_std = load_for_specific_t(slider_t.val)
    plots_left = [
        axes[0].plot(lmbda_values, a, label='$a$', color='b', ls='-'),
        axes[0].fill_between(lmbda_values, a - a_std, a + a_std, color='b', alpha=0.5),
        axes[0].plot(lmbda_values, b, label='$b$', color='r', ls='-'),
        axes[0].fill_between(lmbda_values, b - b_std, b + b_std, color='r', alpha=0.5)
    ]
    # plot lines for adjustable value of t
    lmbda = lmbda_values[slider_lmbda.val]
    a, b, a_std, b_std = load_for_specific_lmbda(slider_lmbda.val)
    plots_right = [
        axes[1].plot(t_values, a, label='$a$', color='b', ls='-'),
        axes[1].fill_between(t_values, a - a_std, a + a_std, color='b', alpha=0.5),
        axes[1].plot(t_values, b, label='$b$', color='r', ls='-'),
        axes[1].fill_between(t_values, b - b_std, b + b_std, color='r', alpha=0.5)
    ]

    axes[0].legend(loc='upper left')
    axes[1].legend(loc='upper left')
    axes[0].set_title(r'Scaling in $\lambda$ with ' f'${t = :.3f}$')
    axes[1].set_title(r'Scaling in $t$ with $\lambda = ' f'{lmbda:.3f}$')
    axes[0].set_xlabel(r'$\lambda$')
    axes[1].set_xlabel(r'$t / \omega$')
    axes[0].set_ylim([-0.1, 5.1])
    axes[1].set_ylim([-0.1, 4.1])
    axes[0].grid(True)
    axes[1].grid(True)

    def update_left(_):
        t = t_values[slider_t.val]
        a, b, a_std, b_std = load_for_specific_t(slider_t.val)

        plots_left[0][0].set_ydata(a)

        # create invisible dummy object to extract the vertices
        dummy = axes[0].fill_between(
            lmbda_values, a - a_std, a + a_std, alpha=0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        # update the vertices of the PolyCollection
        plots_left[1].set_paths([dp.vertices])

        plots_left[2][0].set_ydata(b)

        # create invisible dummy object to extract the vertices
        dummy = axes[0].fill_between(
            lmbda_values, b - b_std, b + b_std, alpha=0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        # update the vertices of the PolyCollection
        plots_left[3].set_paths([dp.vertices])

        axes[0].set_title(r'Scaling in $\lambda$ with ' f'${t = :.3f}$')
        fig.canvas.draw_idle()

    def update_right(_):
        lmbda = lmbda_values[slider_lmbda.val]
        axes[1].set_title(r'Scaling in $t$ with $\lambda = ' f'{lmbda:.3f}$')
        a, b, a_std, b_std = load_for_specific_lmbda(slider_lmbda.val)

        plots_right[0][0].set_ydata(a)

        # create invisible dummy object to extract the vertices
        dummy = axes[1].fill_between(
            t_values, a - a_std, a + a_std, alpha=0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        # update the vertices of the PolyCollection
        plots_right[1].set_paths([dp.vertices])

        plots_right[2][0].set_ydata(b)

        # create invisible dummy object to extract the vertices
        dummy = axes[1].fill_between(
            t_values, b - b_std, b + b_std, alpha=0)
        dp = dummy.get_paths()[0]
        dummy.remove()
        # update the vertices of the PolyCollection
        plots_right[3].set_paths([dp.vertices])

        fig.canvas.draw_idle()

    slider_t.on_changed(update_left)
    slider_lmbda.on_changed(update_right)

    ax_reset = fig.add_axes((0.25, 0.025, 0.1, 0.04))
    button = Button(ax_reset, 'Reset', hovercolor='0.975')

    def reset(_):
        slider_t.reset()
        slider_lmbda.reset()
    button.on_clicked(reset)

    plt.show()


def interactive_plot_in_sigma():
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.subplots_adjust(bottom=0.3)
    fig.suptitle(
        r'$T_c^\lambda$ and the ellipse segment fit ' \
        r'$T_c^\lambda = \frac{b}{a} \left( \sqrt{a^2 - \sigma^2} ' \
        r'- \sqrt{a^2 - 1} \right)$'
    )

    t_values = np.load('parameters_to_circle_fit/run_1_t.npy')[::3]
    lmbda_values = np.load('parameters_to_circle_fit/run_1_lmbda.npy')[::3]
    sigma_values = sigma_values_from_logspace()
    a_values = np.load('parameters_to_circle_fit/run_1_a.npy')[::3, ::3]
    b_values = np.load('parameters_to_circle_fit/run_1_b.npy')[::3, ::3]
    T_values = np.load('parameters_to_circle_fit/run_1_T-every-third.npy')

    # add axes for the sliders
    # fig.add_axes([
        # horisontal start position from left,
        # vertical start position from bottom,
        # width of the ax panel,
        # height of the ax panel,
    # ])
    ax_t_slider = fig.add_axes((0.1, 0.15, 0.25, 0.03))
    ax_lmbda_slider = fig.add_axes((0.1, 0.1, 0.25, 0.03))

    # create the sliders
    slider_t = Slider(
        ax_t_slider,
        r'Index $t$',
        valmin=0,
        valmax=len(t_values) - 1,
        valinit=len(t_values) // 2,
        valstep=1,
    )
    slider_lmbda = Slider(
        ax_lmbda_slider,
        r'Index $\lambda$',
        valmin=0,
        valmax=len(lmbda_values) - 1,
        valinit=len(lmbda_values) // 2,
        valstep=1,
    )
    def get_values():
        i = slider_t.val
        j = slider_lmbda.val
        a = a_values[i, j]
        b = b_values[i, j]
        ellipse = ellipse_fit(sigma_values, a, b)
        return T_values[i, j], ellipse, a, b


    T, e, a, b = get_values()
    plots = [
        ax.plot(sigma_values, T, label=r'$T_c^\lambda$', color='k', ls='-'),
        ax.plot(sigma_values, e, label='Ellipse', color='r', ls=':'),
    ]

    def set_title(a, b):
        ax.set_title(
            r'$\lambda=' f'{lmbda_values[slider_lmbda.val]:.3f},' r'\quad' \
            f't={t_values[slider_t.val]:.3f},' r'\quad' \
            f'{a=:8.5f},' r'\quad' \
            f'{b=:8.5f}$'
        )

    set_title(a, b)
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylim([-0.1, 2.1])
    ax.grid(True)

    def update(_):
        T, e, a, b = get_values()
        plots[0][0].set_ydata(T)
        plots[1][0].set_ydata(e)
        set_title(a, b)
        fig.canvas.draw_idle()

    slider_t.on_changed(update)
    slider_lmbda.on_changed(update)

    ax_reset = fig.add_axes((0.25, 0.025, 0.1, 0.04))
    button = Button(ax_reset, 'Reset', hovercolor='0.975')

    def reset(_):
        slider_t.reset()
        slider_lmbda.reset()
    button.on_clicked(reset)

    plt.show()


def compute_T_for_all_t_lmbda():
    t_values = np.load('parameters_to_circle_fit/run_1_t.npy')[::3]
    lmbda_values = np.load('parameters_to_circle_fit/run_1_lmbda.npy')[::3]
    sigma_values = sigma_values_from_logspace()

    T_values = np.zeros((len(t_values), len(lmbda_values), len(sigma_values)))
    def calculate():
        for k, sigma in enumerate(sigma_values):
            qr = QuantumRabi(omega=1.0, t=t, g=1.0, lmbda=lmbda)
            T = qr.F(sigma, 0) - qr.analytic_terms_of_the_coupling(sigma, 0)
            T_values[i, j, k] = T

    total_iterations = T_values.shape[0] * T_values.shape[1]
    bar = ProgressBar(total_iterations)
    for i, t in enumerate(t_values):
        for j, lmbda in enumerate(lmbda_values):
            bar.write(f'Starting {t = :.3f}, {lmbda = :.3f}')
            try:
                calculate()
            except ValueError as e:
                print(f'Error: {e}')
    bar.write('Done!')
    folder = Path('parameters_to_circle_fit')
    np.save(folder / 'run_1_T-every-third', T_values)


if __name__ == '__main__':
    interactive_plot_in_sigma()
