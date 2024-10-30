import pytest
import scipy    # type: ignore

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
TOL = 1e-6
OSCILLATOR_SIZE = 40

omega_values = [0.7, 1.3]
t_values = [0.2, 0.6]
g_values = [-1.0, 0.0, 0.5]
sigma_values = [-0.7, 0.3]
xi_values = [-0.15, 0.0, 0.2]
zeta_values = [-0.9, 0.6]
lmbda_values = [0.0, 0.5, 1.0]


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('xi', xi_values)
def test_thm_4_9_item_1(omega, t, g, sigma, xi):
    qr = QuantumRabi(omega, t, g, oscillator_size=OSCILLATOR_SIZE)
    LHS = qr.F(sigma, xi)
    RHS = qr.F(-sigma, -xi)
    assert abs(LHS - RHS) < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('xi', xi_values)
@pytest.mark.parametrize('zeta', zeta_values)
def test_thm_4_9_item_2(omega, t, g, sigma, xi, zeta):
    qr = QuantumRabi(omega, t, g, oscillator_size=OSCILLATOR_SIZE)
    LHS = qr.F(sigma, xi + zeta)
    RHS = qr.F(sigma, xi) \
        + omega**2 * zeta * (xi + 0.5*zeta) \
        + g * sigma * zeta
    assert abs(LHS - RHS) < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
def test_thm_4_9_item_4(omega, t, g, sigma):
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
def test_thm_4_9_item_5(omega, t, g, sigma, xi):
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
def test_thm_4_9_item_6(omega, t, g, sigma, xi):
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
def test_thm_5_1(omega, t, g, sigma, xi, lmbda):
    qr = QuantumRabi(omega, t, g, lmbda=lmbda, oscillator_size=OSCILLATOR_SIZE)
    T = qr.T(sigma)
    LHS = qr.F(sigma, xi)
    RHS = qr.analytic_terms_of_the_coupling(sigma, xi) + T
    assert abs(LHS - RHS) < TOL


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('lmbda', lmbda_values)
def test_cor_5_3(omega, t, g, sigma, lmbda):
    # TODO: finish this test
    LHS = QuantumRabi(omega, t, g, lmbda=lmbda).G(sigma)
    def integrand(nu):
        pass
    RHS = scipy.integrate.quad(integrand, 0, lmbda)[0] / lmbda


@pytest.mark.parametrize('omega', omega_values)
@pytest.mark.parametrize('t', t_values)
@pytest.mark.parametrize('g', g_values)
@pytest.mark.parametrize('sigma', sigma_values)
@pytest.mark.parametrize('xi', xi_values)
def test_eq_40(omega, t, g, sigma, xi):
    qr = QuantumRabi(omega, t, g, lmbda=1.0, oscillator_size=OSCILLATOR_SIZE)
    v, j = qr.minimizer_potential(sigma=sigma, xi=xi)
    op_H = qr.op_H_0 + v*qr.op_sigma_z + j*qr.op_x
    eigenvectors = op_H.eig(hermitian=True)['eigenvectors']
    # v = eigenvectors[0]
    for v in eigenvectors[:15]:
        LHS = omega**2 * (qr.op_sigma_z*qr.op_x).expval(v) \
            + 2 * t * (qr.op_sigma_y*qr.op_p).expval(v) \
            + g + j*qr.op_sigma_z.expval(v)
        RHS = 0.0
        assert LHS.imag < TOL
        assert LHS.real - RHS < TOL

if __name__ == '__main__':
    test_eq_40(1, 1, 1, 0.3, 0.1)
