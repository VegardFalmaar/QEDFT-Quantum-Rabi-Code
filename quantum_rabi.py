from typing import Tuple

import numpy as np
import scipy    # type: ignore

import qmodel as q  # type: ignore


class QuantumRabi:
    def __init__(
        self,
        omega: float,
        t: float,
        g: float,
        lmbda: float = 1,
        oscillator_size: int = 40
    ) -> None:
        assert t >= 0, f't should be >= 0, not {t}.'
        self.omega = omega
        self.t = t
        self.g = g
        self.lmbda = lmbda
        self.oscillator_size = oscillator_size

        b_spin = q.SpinBasis()
        b_osc = q.NumberBasis(oscillator_size)
        basis = b_osc.tensor(b_spin)

        self.op_sigma_x = b_spin.sigma_x().extend(basis)
        self.op_sigma_y = b_spin.sigma_y().extend(basis)
        self.op_sigma_z = b_spin.sigma_z().extend(basis)
        self.op_hop = basis.hop
        a_dag = b_osc.creator()
        a = b_osc.annihilator()
        self.op_p = 1j * np.sqrt(omega/2) * (a_dag - a).extend(basis)
        self.op_x = (2*omega)**(-0.5) * (a_dag + a).extend(basis)
        self.op_H_0 = 0.5 * self.op_p**2 \
            + 0.5 * omega**2 * self.op_x**2 \
            - t * self.op_sigma_x \
            + lmbda * g * self.op_sigma_z * self.op_x

    @staticmethod
    def check_sigma_xi(
        omega: float,
        lmbda_times_g: float,
        sigma: float,
        xi: float,
        j: float,
        tol: float = 1e-3,
    ):
        # sigma and xi must be in a precise relation after d/d xi applied on
        # displacement rule
        error = abs(lmbda_times_g*sigma + j + omega**2*xi)
        if error > tol:
            msg = f'sigma--xi check:\n\tFAIL at {lmbda_times_g=:.8f}, ' \
                + f'{sigma=:.8f}, {xi=:.8f}, {j=:.8f}: {error=:.2e}!\n' \
                + '\tConsider increasing the oscillator_size value.'
            raise ValueError(msg)

    def photon_filling(self, v: float, j: float) -> Tuple[np.ndarray, np.ndarray]:
        ground_state = q.EnergyFunctional(
            self.op_H_0,
            [self.op_sigma_z, self.op_x]
        ).solve([v, j])['gs_vector']

        sigma = self.op_sigma_z.expval(ground_state, transform_real = True)
        xi = self.op_x.expval(ground_state, transform_real = True)
        print('In QuantumRabi.photon_filling:')
        print(f'\tsigma_z expectation value = {sigma}')
        print(f'\tx expectation value = {xi}')
        self.check_sigma_xi(self.omega, self.lmbda*self.g, sigma, xi, j)

        rho0_up = np.array([
            self.op_hop({'n': n, 's': +1}).expval(ground_state, transform_real=True)
            for n in range(self.oscillator_size)
        ])
        rho0_down = np.array([
            self.op_hop({'n': n, 's': -1}).expval(ground_state, transform_real=True)
            for n in range(self.oscillator_size)
        ])
        return rho0_up, rho0_down

    def minimizer_potential(self, sigma: float, xi: float):
        energy = q.EnergyFunctional(self.op_H_0, [self.op_sigma_z, self.op_x])
        lt = energy.legendre_transform([sigma, xi], verbose=False)
        v, j = lt['pot']
        self.check_sigma_xi(self.omega, self.lmbda*self.g, sigma, xi, j)
        return v, j

    def F_from_minimization(self, sigma: float, xi: float) -> float:
        energy = q.EnergyFunctional(self.op_H_0, [self.op_sigma_z, self.op_x])
        lt = energy.legendre_transform([sigma, xi], verbose=False, gtol=1e-5)
        _, j = lt['pot']
        self.check_sigma_xi(self.omega, self.lmbda*self.g, sigma, xi, j)
        return lt['F']

    def optimal_j(self, sigma: float, xi: float) -> float:
        return - self.omega**2*xi - self.lmbda*self.g*sigma

    def F_from_constrained_minimization(self, sigma: float, xi: float) -> float:
        optimal_j = self.optimal_j(sigma, xi)
        energy = q.EnergyFunctional(
            self.op_H_0 + optimal_j*self.op_x,
            [self.op_sigma_z]
        )
        lt = energy.legendre_transform([sigma], verbose=False, gtol=1e-6)
        return lt['F'] - optimal_j*xi

    def T_integrand(self, tau: float, sigma: float) -> float:
        qr_tau = QuantumRabi(
            self.omega,
            self.t,
            self.g,
            lmbda=tau,
            oscillator_size=self.oscillator_size
        )
        v, j = qr_tau.minimizer_potential(sigma, 0)
        H = qr_tau.op_H_0 + v*qr_tau.op_sigma_z + j*qr_tau.op_x
        ground_state = H.eig(hermitian = True)['eigenvectors'][0]
        return 0.5 * (qr_tau.op_sigma_y*qr_tau.op_p).expval(
            ground_state, transform_real=True
        )

    def T_from_integration(self, sigma: float) -> float:
        integration, error = scipy.integrate.quad(
            self.T_integrand, 0, self.lmbda,
            args=(sigma),
            # epsabs=1e-12, epsrel=1e-12, limit=80
        )
        # print(f'Integration error estimate: {error:.3e}')
        return - 4 * self.t * self.g * integration / self.omega**2

    def analytic_terms_of_F(self, sigma: float, xi: float):
        return 0.5 * self.omega \
            - self.t * np.sqrt(1 - sigma**2) \
            + 0.5 * self.omega**2 * xi**2 \
            + self.lmbda * self.g * sigma * xi \
            - 0.5 * self.lmbda**2 * self.g**2 * (1 - sigma**2) / self.omega**2

    def T_from_minimization(self, sigma: float) -> float:
        return self.F_from_minimization(sigma, 0.0) \
            - self.analytic_terms_of_F(sigma, 0.0)

    def G_from_T(self, sigma: float) -> float:
        o, l, g = self.omega, self.lmbda, self.g
        T = self.T_from_minimization(sigma)
        return - 0.5 * l * g**2 * (1 - sigma**2) / o**2 + T / l

    def G_integrand(self, nu: float, sigma: float) -> float:
        qr_nu = QuantumRabi(
            self.omega,
            self.t,
            self.g,
            lmbda=nu,
            oscillator_size=self.oscillator_size
        )
        v, j = qr_nu.minimizer_potential(sigma, 0.0)
        op_H = qr_nu.op_H_0 + v*qr_nu.op_sigma_z + j*qr_nu.op_x
        gs = op_H.eig(hermitian=True)['eigenvectors'][0]
        ev = (qr_nu.op_sigma_z*qr_nu.op_x).expval(gs, transform_real=True)
        return self.g * ev

    def G_from_integration(self, sigma: float) -> float:
        integration, error = scipy.integrate.quad(
            self.G_integrand, 0, self.lmbda,
            args=(sigma),
            # epsabs=1e-12, epsrel=1e-12, limit=80
        )
        # print(f'Integration error estimate: {error:.3e}')
        return integration / self.lmbda
