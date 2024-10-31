from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from ft.formats import Table    # type: ignore


class OLS:
    """
    Ordinary Least Squares for functions of two variables.
    """
    def __init__(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y: np.ndarray,
        deg: Tuple[int, int],
    ) -> None:
        assert x1.ndim == x2.ndim == 1
        assert y.shape == (len(x1), len(x2))
        self.x1 = x1.reshape(-1, 1)
        self.x2 = x2.reshape(-1, 1)

        self.num_params = (deg[0] + 1) * (deg[1] + 1)
        self.deg = deg
        self.X = np.zeros((len(x1)*len(x2), self.num_params))
        self.fill_design_matrix()

        self.y = y.reshape(-1, 1)
        self.beta = np.linalg.inv(self.X.T@self.X) @ self.X.T @ self.y
        assert self.beta.shape == (self.num_params, 1)

    def fill_design_matrix(self) -> None:
        for i, x1 in enumerate(self.x1):
            for j, x2 in enumerate(self.x2):
                self.X[i*len(self.x2) + j] \
                    = self._x_combination_matrix(x1, x2).ravel()

    def _x_combination_matrix(self, x1: float, x2: float) -> np.ndarray:
        return x1**np.arange(self.deg[0] + 1).reshape(-1, 1) \
            @ x2**np.arange(self.deg[1] + 1).reshape(1, -1)

    def print_coeffs(self, x1_name: str, x2_name: str) -> None:
        M = self.deg[0] + 1
        N = self.deg[1] + 1
        beta = self.beta.reshape(M, N)
        tab = [[''] + [f'{x2_name}^{i}' for i in range(N)]]
        for j, beta_line in enumerate(beta):
            line = [f'{x1_name}^{j}'] + [f'{b:.5f}' for b in beta_line]
            tab.append(line)
        table = Table(tab)
        table.write()

    def predict(self, x1: float, x2: float) -> float:
        result = self.beta.ravel().dot(self._x_combination_matrix(x1, x2).ravel())
        return result


def test():
    def f(x, y):
        return 5 + x + 3*y - 2*x**2 + np.pi*x*y - 8*x**2*y
    x1 = np.arange(5)
    x2 = np.linspace(5, 6, 4)
    z = np.zeros((5, 4))
    for i in range(5):
        for j in range(4):
            z[i, j] = f(x1[i], x2[j])

    ols = OLS(x1, x2, z, (2, 1))
    ols.print_coeffs('x', 'y')


def main():
    t_values = np.load('run_1_t.npy')
    lmbda_values = np.load('run_1_lmbda.npy')
    a_values = np.load('run_1_a.npy')
    b_values = np.load('run_1_b.npy')

    ols = OLS(t_values, lmbda_values, a_values, (4, 5))
    ols.print_coeffs('t', 'l')

    x = t_values
    # x = lmbda_values
    lmbda = lmbda_values[23]
    # t = t_values[20]

    y = np.zeros_like(x)
    for i, t in enumerate(x):
        y[i] = ols.predict(t, lmbda)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set_title(r'OLS fit of $a$ for $\lambda=' f'{lmbda:.3f}$')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$a$')
    ax.set_ylim([-0.1, 4.1])
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    # test()
    main()
