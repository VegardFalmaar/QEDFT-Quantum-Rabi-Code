from typing import Tuple
from pathlib import Path

import scipy as sp  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from orgmin import ComputeDB

from quantum_rabi import QuantumRabi
from investigate_form_of_T_in_sigma import sigma_values_from_logspace
from plot_config import PlotConfig as PC


class OptimalGaussian:
    def __init__(
        self,
        omega: float,
        t: float,
        g: float,
        lmbda: float,
        sigma: float,
    ) -> None:
        self.omega = omega
        self.t = t
        self.g = g
        self.lmbda = lmbda
        self.sigma = sigma
        self.a = 2 * t / omega**2 * (1 - sigma) / (1 + sigma) * np.sqrt(1 - sigma**2)
        self.b = omega / (1 - sigma)**2
        self.c = 2 * lmbda * g * (1 - sigma) / omega**2

    @property
    def s_star(self):
        return self.lmbda * self.g * (self.sigma - 1) / self.omega**2

    def print_simplified_parameters(self):
        print('Simplified parameters:')
        print(f'\t{self.a = :.2e}')
        print(f'\t{self.b = :.2e}')
        print(f'\t{self.c = :.2e}')

    def find_optimal_translation(self, verbose: bool = False) -> float:
        starting_points = [self.s_star, 0.0]
        local_minima = []
        for s in starting_points:
            res = self._minimize(s)
            x, y = float(res.x[0]), res.fun
            bound = self.kinetic_correlation_bound(x)
            if verbose:
                print(f'Optimum starting from {s:9.2e}: ({x:.2e}, {y:9.2e}), {bound = :.2e}')
            local_minima.append((x, bound))
        return sorted(local_minima, key=lambda e: e[1])[0][0]


    def _minimize(self, starting_point: float):
        return sp.optimize.minimize(
            self.simplified_target,
            x0=starting_point,
            method='BFGS',
            options={
                'gtol': 1e-8,
            },
        )

    def simplified_target(
        self,
        s: float,
        abc: None | Tuple[float, float, float] = None
    ) -> float:
        if abc is None:
            a, b, c = self.a, self.b, self.c
        else:
            a, b, c = abc
        return s**2 - a*np.exp(-b*s**2) + c*s

    def kinetic_correlation_bound(self, s: float) -> float:
        o, t, g, l, sm = self.omega, self.t, self.g, self.lmbda, self.sigma
        result = 0.5 * o**2 * s**2 * (1 + sm) / (1 - sm) \
            + l * g * (1 + sm) * s \
            + l**2 * g**2 * (1 - sm**2) / (2 * o**2) \
            + t * np.sqrt(1 - sm**2) * (1 - np.exp(-o*s**2/(1 - sm)**2))
        return float(result)



def _test():
    # om_factor = 20
    om_factor = 1
    og = OptimalGaussian(1/om_factor, 16/om_factor**2, 2, 2/om_factor**2, 0.5)
    og.print_simplified_parameters()
    s = og.find_optimal_translation(verbose=True)
    print(s)


def interactive_plot_of_kinetic_correlation():
    db = ComputeDB(Path('/home/vegard/Git/Database/quantum-rabi/F-v-s-2'))
    t_values = np.linspace(0.2, 3, 60)
    lmbda_values = np.linspace(0.0, 2.0, 60)
    sigma_values = sigma_values_from_logspace()[1:-1]

    I_values = np.zeros((len(t_values), len(lmbda_values), len(sigma_values)))
    gaussian_values = np.zeros_like(I_values)
    for i, t in enumerate(t_values):
        for j, lmbda in enumerate(lmbda_values):
            for k, sigma in enumerate(sigma_values):
                precomputed = db[{'lmbda': lmbda, 'sigma': sigma, 't': t}]
                qr = QuantumRabi(omega=1.0, t=t, g=1.0, lmbda=lmbda)
                og = OptimalGaussian(
                    omega=1.0, t=t, g=1.0, lmbda=lmbda, sigma=sigma
                )
                I = precomputed['F'] - qr.analytic_terms_of_F(sigma, xi=0.0)
                I_values[i, j, k] = I
                g = og.kinetic_correlation_bound(s=precomputed['s'])
                gaussian_values[i, j, k] = g

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.subplots_adjust(bottom=0.3)
    fig.suptitle(r'$I^\lambda$ and the bound from the best fit Gaussian')

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
        I = I_values[i, j]
        g = gaussian_values[i, j]
        precomputed = db[{
            'lmbda': lmbda_values[j],
            'sigma': sigma_values[len(sigma_values) // 2],
            't': t_values[i]
        }]
        s = precomputed['s']
        v = precomputed['v']
        return I, g, s, v

    I, g, s, v = get_values()
    plots = [
        ax.plot(sigma_values, I, label=r'$I^\lambda$', color='k', ls='-'),
        ax.plot(sigma_values, g, label='Gaussian', color='r', ls=':'),
    ]

    def set_title(s, v):
        ax.set_title(
            r'$\lambda=' f'{lmbda_values[slider_lmbda.val]:.3f},' r'\quad' \
            f't={t_values[slider_t.val]:.3f},' r'\quad' \
            f'{s=:8.5e},' r'\quad' \
            f'{v=:8.5e}$'
        )

    set_title(s, v)
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylim([-0.1, 2.1])
    ax.grid(True)

    def update(_):
        I, g, s, v = get_values()
        plots[0][0].set_ydata(I)
        plots[1][0].set_ydata(g)
        set_title(s, v)
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


def plot_of_kinetic_correlation():
    PC.use_tex()
    db = ComputeDB(Path('/home/vegard/Git/Database/quantum-rabi/F-v-s-2'))
    t_values = np.linspace(0.2, 3, 60)
    lmbda_values = np.linspace(0.0, 2.0, 60)
    sigma_values = sigma_values_from_logspace()[1:-1]

    fig, ax = plt.subplots(figsize=(PC.fig_width, PC.fig_height -0.13))

    I_values = np.zeros_like(sigma_values)
    gaussian_values = np.zeros_like(sigma_values)

    t_lmbda_indices = ((38, 59), (9, 59), (38, 21))
    colors = ["#9986A5", "#79402E", "#CCBA72", "#0F0D0E", "#D9D0D3", "#8D8680"]
    for (i, j), c in zip(t_lmbda_indices, colors):
        for k, sigma in enumerate(sigma_values):
            t = t_values[i]
            lmbda = lmbda_values[j]
            precomputed = db[{'lmbda': lmbda, 'sigma': sigma, 't': t}]
            qr = QuantumRabi(omega=1.0, t=t, g=1.0, lmbda=lmbda)
            og = OptimalGaussian(
                omega=1.0, t=t, g=1.0, lmbda=lmbda, sigma=sigma
            )
            I = precomputed['F'] - qr.analytic_terms_of_F(sigma, xi=0.0)
            I_values[k] = I
            g = og.kinetic_correlation_bound(s=precomputed['s'])
            gaussian_values[k] = g
        ax.plot(
            sigma_values,
            I_values,
            c=c,
            ls='-',
            lw=PC.linewidth,
            label=f'${t = :.2f}$,\n' r'$\lambda = ' f'{lmbda:.2f}$'
        )
        ax.plot(
            sigma_values,
            gaussian_values,
            c=c,
            ls='--',
            lw=PC.linewidth,
        )

    PC.set_ax_info(
        ax=ax,
        xlabel=r'$\sigma$',
        title=r'$I^\lambda$ (solid) and Gaussian bound (dashed)',
        legend=True,
    )

    PC.tight_layout(fig, ax_aspect=1.75)
    fig.savefig(PC.save_dir + '/gaussian-fit-to-kinetic-corr.pdf')
    # plt.show()


def main():
    # interactive_plot_of_kinetic_correlation()
    plot_of_kinetic_correlation()


if __name__ == '__main__':
    main()
