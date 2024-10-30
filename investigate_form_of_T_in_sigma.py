from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
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

    ax.set_title(r'$T_c^\lambda = \frac{b}{a} \left( \sqrt{a^2 - \sigma^2} - \sqrt{a^2 - 1} \right)$')
    # ax.set_xlabel(r'$\lambda g / \omega^{3/2}$')
    ax.set_xlabel(r'$t / \omega$')

    ax.legend()
    plt.show()


def prepare_data():
    folder = Path('parameters_to_circle_fit')
    omega = 1.0
    sigma_values = sigma_values_from_logspace()[1:-1]

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

    t_values = np.linspace(0.05, 3.0, 60)
    g_values = np.linspace(0.00, 2.5, 51)
    # t_values = np.linspace(0.05, 3.0, 2)
    # g_values = np.linspace(0.0, 2.5, 2)
    np.save(folder / 'run_1_t', t_values)
    np.save(folder / 'run_1_lmbda', g_values)

    a_values = np.zeros((len(t_values), len(g_values)))
    b_values = np.zeros_like(a_values)
    a_std_values = np.zeros_like(a_values)
    b_std_values = np.zeros_like(a_values)

    total_iterations = len(t_values) * len(g_values)
    bar = StatusBar(total_iterations)
    for i, t in enumerate(t_values):
        for j, g in enumerate(g_values):
            bar.write(f'Starting {t = :.3f}, {g = :.3f}')
            try:
                (a, b), (a_std, b_std) = fit()
            except ValueError as e:
                print(f'Error: {e}')
            # (a, b, d), (a_std, b_std, d_std) = fit()
            a_values[i, j] = a
            b_values[i, j] = b
            a_std_values[i, j] = a_std
            b_std_values[i, j] = b_std
    bar.write('Done!')

    np.save(folder / 'run_1_a', a_values)
    np.save(folder / 'run_1_b', b_values)
    np.save(folder / 'run_1_a_std', a_std_values)
    np.save(folder / 'run_1_b_std', b_std_values)


class StatusBar():
    def __init__(self, total: int):
        self.total = total
        self.counter = 0
        self.start_time = int(time.perf_counter())

    def write(self, msg: str):
        pbar_length = 30
        num_progress_bars = pbar_length * self.counter // self.total
        progress = self.counter / self.total
        t = int(time.perf_counter()) - self.start_time
        total_t = int(t / progress) if self.counter > 0 else 0
        line = '[' + '#'*num_progress_bars + ' '*(pbar_length - num_progress_bars) \
            + f'] {progress*100:5.1f} % ' \
            + f'({t//60:3d}:{t%60:02d} / {total_t//60:3d}:{total_t%60:02d})  ' \
            + f'({msg})'
        print(line)
        self.counter += 1


def interactive_plot_from_file():
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


if __name__ == '__main__':
    # plot_in_sigma()
    # tabulate_parameters()
    # plot_parameters()
    # prepare_data()
    interactive_plot_from_file()
