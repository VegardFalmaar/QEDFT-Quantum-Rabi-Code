import pytest
import scipy    # type: ignore
import numpy as np

from quantum_rabi import QuantumRabi


# The tolerance used when verifying equations.
# An equation
#   LHS = RHS
# passes the test if
#   abs(LHS - RHS) < TOL.
# An inequality
#   LHS < RHS
# passes the test if
#   LHS - RHS < TOL.
TOL = 1e-5
OSCILLATOR_SIZE = 40

omega_values = [0.7, 1.3]
t_values = [0.2, 0.6]
g_values = [-1.0, 0.0, 0.5]
sigma_values = [-0.7, 0.3]
xi_values = [-0.15, 0.0, 0.2]
zeta_values = [-0.9, 0.6]
lmbda_values = [0.0, 0.5, 1.0]
v_values = np.linspace(-0.5, 0.5, 5)
j_values = np.linspace(-0.5, 0.5, 4)


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('xi', xi_values)
def test_thm_4_2_item_1(omega, t, g, sigma, xi):
    qr = QuantumRabi(omega, t, g, oscillator_size=OSCILLATOR_SIZE)
    LHS = qr.F_from_minimization(sigma, xi)
    RHS = qr.F_from_minimization(-sigma, -xi)
    assert abs(LHS - RHS) < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('xi', xi_values)
@pytest.mark.parametrize('zeta', zeta_values)
def test_thm_4_2_item_2(omega, t, g, sigma, xi, zeta):
    qr = QuantumRabi(omega, t, g, oscillator_size=OSCILLATOR_SIZE)
    LHS = qr.F_from_minimization(sigma, xi + zeta)
    RHS = qr.F_from_minimization(sigma, xi) \
        + omega**2 * zeta * (xi + 0.5*zeta) \
        + g * sigma * zeta
    assert abs(LHS - RHS) < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
def test_thm_4_2_item_4(omega, t, g, sigma):
    xi = 0
    qr = QuantumRabi(omega, t, g, oscillator_size=OSCILLATOR_SIZE)
    v, j = qr.minimizer_potential(sigma=sigma, xi=xi)
    op_H = qr.op_H_0 + v*qr.op_sigma_z + j*qr.op_x
    gs = op_H.eig(hermitian=True)['eigenvectors'][0]
    LHS = 0.5 * (qr.op_p**2 - omega**2 * qr.op_x**2).expval(gs)
    RHS = 0.5 * g * (qr.op_sigma_z*qr.op_x).expval(gs)
    assert abs(LHS - RHS) < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('xi', xi_values)
def test_thm_4_2_item_5(omega, t, g, sigma, xi):
    qr = QuantumRabi(omega, t, g, oscillator_size=OSCILLATOR_SIZE)
    v, j = qr.minimizer_potential(sigma=sigma, xi=xi)
    op_H = qr.op_H_0 + v*qr.op_sigma_z + j*qr.op_x
    gs = op_H.eig(hermitian=True)['eigenvectors'][0]
    LHS = 0.5 * xi + 0.5 * (qr.op_sigma_z*qr.op_x).expval(gs)
    RHS = - t / omega**2 * (qr.op_sigma_y*qr.op_p).expval(gs) \
        - g * (1 - sigma**2) / (2 * omega**2) \
        + 0.5 * xi * (1 + sigma)
    assert abs(LHS - RHS) < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('xi', xi_values)
def test_thm_4_2_item_6(omega, t, g, sigma, xi):
    qr = QuantumRabi(omega, t, g, oscillator_size=OSCILLATOR_SIZE)
    v, j = qr.minimizer_potential(sigma=sigma, xi=xi)
    op_H = qr.op_H_0 + v*qr.op_sigma_z + j*qr.op_x
    gs = op_H.eig(hermitian=True)['eigenvectors'][0]
    LHS = - 0.5 * (qr.op_sigma_x*qr.op_p**2).expval(gs)
    RHS = omega**2 / (8*t) * (1 - sigma**2)
    assert LHS.imag < TOL
    assert LHS.real - RHS < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('xi', xi_values)
@pytest.mark.parametrize('lmbda', lmbda_values)
def test_thm_5_2(omega, t, g, sigma, xi, lmbda):
    qr = QuantumRabi(omega, t, g, lmbda=lmbda, oscillator_size=OSCILLATOR_SIZE)
    T = qr.T_from_integration(sigma)
    LHS = qr.F_from_minimization(sigma, xi)
    RHS = qr.analytic_terms_of_F(sigma, xi) + T
    assert abs(LHS - RHS) < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('lmbda', [l for l in lmbda_values if l > 0.1])
def test_cor_5_4(omega, t, g, sigma, lmbda):
    LHS = QuantumRabi(omega, t, g, lmbda=lmbda).G_from_T(sigma)
    def integrand(nu):
        qr = QuantumRabi(omega, t, g, lmbda=nu)
        v, j = qr.minimizer_potential(sigma=sigma, xi=0.0)
        op_H = qr.op_H_0 + v*qr.op_sigma_z + j*qr.op_x
        gs = op_H.eig(hermitian=True)['eigenvectors'][0]
        return (qr.op_p**2 - omega**2 * qr.op_x**2).expval(gs) / nu
    RHS = scipy.integrate.quad(integrand, 0, lmbda)[0] / lmbda
    assert abs(LHS - RHS) < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('lmbda', [l for l in lmbda_values if l > 0.1])
def test_G_from_integration_and_T(omega, t, g, sigma, lmbda):
    LHS = QuantumRabi(omega, t, g, lmbda=lmbda).G_from_integration(sigma)
    RHS = QuantumRabi(omega, t, g, lmbda=lmbda).G_from_T(sigma)
    assert abs(LHS - RHS) < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('v', v_values)
@pytest.mark.parametrize('j', j_values)
def test_eq_20(omega, t, g, v, j):
    qr = QuantumRabi(omega, t, g, lmbda=1.0, oscillator_size=OSCILLATOR_SIZE)
    op_H = qr.op_H_0 + v*qr.op_sigma_z + j*qr.op_x
    eigenvectors = op_H.eig(hermitian=True)['eigenvectors']
    # v = eigenvectors[0]
    for eigenvector in eigenvectors[:15]:
        LHS = omega**2 * (qr.op_sigma_z*qr.op_x).expval(eigenvector) \
            + 2 * t * (qr.op_sigma_y*qr.op_p).expval(eigenvector) \
            + g + j*qr.op_sigma_z.expval(eigenvector)
        RHS = 0.0
        assert LHS.imag < TOL
        assert LHS.real - RHS < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('xi', xi_values)
@pytest.mark.parametrize('lmbda', lmbda_values)
def test_F_from_constrained_minimization(omega, t, g, sigma, xi, lmbda):
    qr = QuantumRabi(omega, t, g, lmbda=lmbda, oscillator_size=OSCILLATOR_SIZE)
    LHS = qr.F_from_minimization(sigma, xi)
    RHS, _ = qr.F_from_constrained_minimization(sigma, xi)
    assert abs(LHS - RHS) < TOL
