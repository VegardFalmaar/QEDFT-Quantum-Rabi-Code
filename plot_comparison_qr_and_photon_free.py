'''
code for QRabi model paper plot
compare spectra of the QRabi model and the photon-free Hamiltonian
also compare expectation values for 'densities' and parts of the Hamiltonian
'''
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from qmodel import NumberBasis, SpinBasis

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

g_span = np.linspace(0,3,50)
v = 0.1
j = 0.1

# photon free operators
sigma_x_pf = b_spin.sigma_x()
sigma_y_pf = b_spin.sigma_y()
sigma_z_pf = b_spin.sigma_z()

def check_sigma_x(g, sigma, x):
    # sigma and x must be in a precise relation
    if abs(g*sigma + j + om**2*x) > 1e-6:
        print(f'sigma--xi check: FAIL at g={g}, sigma={sigma}, xi={x}, j={j}! maybe increase oscillator_size value')

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
    check_sigma_x(g, sigma_z.expval(Psi_QR, transform_real=True), x_op.expval(Psi_QR, transform_real=True))
    
    # compare expvals
    d_E.append(spec_QR[-1][0] - spec_pf[-1][0]) # groundstate energy
    d_xi.append(x_op.expval(Psi_QR, transform_real=True) - x_pf.expval(Psi_pf, transform_real=True))
    d_sigma.append(sigma_z.expval(Psi_QR, transform_real=True) - sigma_z_pf.expval(Psi_pf, transform_real=True))
    d_kin.append(-t*sigma_x.expval(Psi_QR, transform_real=True) + t*sigma_x_pf.expval(Psi_pf, transform_real=True))
    d_coupl.append(g*(sigma_z*x_op).expval(Psi_QR, transform_real=True) - g*(sigma_z_pf*x_pf).expval(Psi_pf, transform_real=True))
    d_num.append(num_op.expval(Psi_QR, transform_real=True) - num_pf.expval(Psi_pf, transform_real=True))
    
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(g_span, spec_QR, '-k')
ax1.plot(g_span, spec_pf, '--k')

ax2.plot(g_span, d_E, '-k', label=r'$\hat H$')
ax2.plot(g_span, d_xi, '--k', label=r'$\hat x$')
ax2.plot(g_span, d_sigma, '.k', label=r'$\hat\sigma_z$')
ax2.plot(g_span, d_kin, 'xk', label=r'$-t\hat\sigma_x$')
ax2.plot(g_span, d_coupl, 'ok', label=r'$g\hat\sigma_z\hat x$')
ax2.plot(g_span, d_num, '^k', label=r'$\hat a^\dagger\hat a$')
ax2.set_xlabel('$g$')
ax2.legend()
plt.show()