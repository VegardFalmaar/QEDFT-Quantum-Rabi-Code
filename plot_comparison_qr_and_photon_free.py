'''
code for QRabi model paper plot
compare spectra of the QRabi model and the photon-free Hamiltonian
also compare expectation values for 'densities' and parts of the Hamiltonian
'''
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from qmodel import NumberBasis, SpinBasis   # type: ignore

from quantum_rabi import QuantumRabi
from plot_config import PlotConfig as PC


def ev(op, state):
    """Shorthand function for the exp val of an operator in a given state."""
    return op.expval(state, transform_real=True)


PC.use_tex()


om = 1
t = 1
oscillator_size = 50

b_oscillator = NumberBasis(oscillator_size)
b_spin = SpinBasis()
b = b_oscillator.tensor(b_spin)

num_op = b_oscillator.diag().extend(b) # number operator for oscillator
x_op = b_oscillator.x_operator().extend(b) / sqrt(om)
p_op = -1j*b_oscillator.dx_operator().extend(b) * sqrt(om)
sigma_x = b_spin.sigma_x().extend(b)
sigma_z = b_spin.sigma_z().extend(b)

H0_Rabi = om*(num_op + 1/2) - t*sigma_x # without coupling term
CouplingRabi = x_op*sigma_z

spec_QR = []
spec_pf = []
d_xi = []
d_sigma = []
d_kin = []
d_coupl = []
d_num = []
d_E = []

g_span = np.linspace(0, 3, 50)
v = 0.1
j = 0.1

# photon free operators
sigma_x_pf = b_spin.sigma_x()
sigma_y_pf = b_spin.sigma_y()
sigma_z_pf = b_spin.sigma_z()

for g in g_span:
    H_QR = H0_Rabi + g*CouplingRabi + v*sigma_z + j*x_op
    H_pf = -t*sigma_x_pf + (v - g*j/om**2)*sigma_z_pf - (j**2+g**2)/(2*om**2) + om/2
    # photon-free operators
    x_pf = -(g*sigma_z_pf+j)/om**2
    a_pf = sqrt(om/2)*x_pf
    num_pf = a_pf**2
    p_pf = 2*g*t/om**2*sigma_y_pf
    # alternative way of writing H_pf with photon-free operators
    # H_pf = 1/2*om**2*x_pf**2  + om/2 - t*sigma_x_pf + g*x_pf*sigma_z_pf + v*sigma_z_pf + j*x_pf

    res_QR = H_QR.eig(hermitian = True)
    res_pf = H_pf.eig(hermitian = True)

    spec_QR.append(res_QR['eigenvalues'][0:10]) # lowest 10 eigenvalues
    spec_pf.append(res_pf['eigenvalues']) # both eigenvalues

    # ground states
    Psi_QR = res_QR['eigenvectors'][0]
    Psi_pf = res_pf['eigenvectors'][0]
    # check
    QuantumRabi.check_sigma_xi(
        omega=om,
        lmbda_times_g=g,
        sigma=sigma_z.expval(Psi_QR, transform_real=True),
        xi=x_op.expval(Psi_QR, transform_real=True),
        j=j,
        tol=1e-6,
    )

    # compare expvals
    d_E.append(spec_QR[-1][0] - spec_pf[-1][0]) # groundstate energy
    d_xi.append(ev(x_op, Psi_QR) - ev(x_pf, Psi_pf))
    d_sigma.append(ev(sigma_z, Psi_QR) - ev(sigma_z_pf, Psi_pf))
    d_kin.append(-t*ev(sigma_x, Psi_QR) + t*ev(sigma_x_pf, Psi_pf))
    d_coupl.append(g*ev(sigma_z*x_op, Psi_QR) - g*ev(sigma_z_pf*x_pf, Psi_pf))
    d_num.append(ev(num_op, Psi_QR) - ev(num_pf, Psi_pf))

fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(PC.fig_width, 1.5*PC.fig_height), sharex=True
)

lw = PC.linewidth
ms = PC.markersize

ax1.plot(g_span, spec_QR, ls='-', c='k', lw=lw)
ax1.plot(g_span, spec_pf, ls='--', c='k', lw=lw)
ax1.plot([], ls='-', c='k', label='Quantum Rabi', lw=lw)
ax1.plot([], ls='--', c='k', label='Photon-free', lw=lw)

ax2.plot(g_span, d_E, ls='-', c='k', label=r'$\hat H$', lw=lw)
ax2.plot(g_span, d_xi, ls='--', c='k', label=r'$\hat x$', lw=lw)
ax2.plot(g_span, d_sigma, ls='-.', c='k', label=r'$\hat\sigma_z$', lw=lw)
ax2.plot(g_span, d_kin, ls=':', c='k', label=r'$-t\hat\sigma_x$', lw=lw)
lbl = r'$g\hat\sigma_z\hat x$'
ax2.plot(g_span, d_coupl, marker='o', ls='', c='k', label=lbl, ms=ms)
lbl = r'$\hat a^\dagger\hat a$'
ax2.plot(g_span, d_num, marker='^', ls='', c='k', label=lbl, ms=ms)

PC.set_ax_info(ax1, legend=True)   # also sets the fontsize of axis ticks
PC.set_ax_info(ax2, xlabel='$g$', legend=False)
ax2.legend(loc='lower left', ncols=2, fontsize=PC.fontsize_legends)

# do the normal tight_layout with the usual padding and then shift a little to
# the left
fig.tight_layout(pad=0.1, rect=(-0.03, 0, 1, 1))

p = {
    'omega': om,
    't': t,
    'v': v,
    'j': j,
}
fig.savefig(PC.save_fname('spectrum-photon-free', '.pdf', p))
