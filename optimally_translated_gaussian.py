from typing import Tuple

import scipy as sp  # type: ignore
import numpy as np


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



def main():
    # om_factor = 20
    om_factor = 1
    og = OptimalGaussian(1/om_factor, 16/om_factor**2, 2, 2/om_factor**2, 0.5)
    og.print_simplified_parameters()
    s = og.find_optimal_translation(verbose=True)
    print(s)


if __name__ == '__main__':
    main()
