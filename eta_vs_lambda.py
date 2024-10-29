from qmodel import *
from dft import *
import matplotlib.pyplot as plt
import numpy as np

# Define the check_sigma_x function with adjusted error tolerance
def check_sigma_x(lam, sigma, x, j):
    # Adjusted error threshold to 1e-3
    if abs(lam * sigma + j + x) > 1e-3:
        print(f'sigma--xi check: FAIL at lam={lam}, sigma={sigma}, xi={x}, j={j}! '
              'Maybe increase oscillator_size value')

def compute_v_hxc(sigma_space, lam, x, t, E_KS, E_full, sigma_x, x_op):
    vxc = []
    for sigma in sigma_space:
        # Legendre transforms
        LT_full = E_full.legendre_transform([sigma, x])
        LT_KS = E_KS.legendre_transform([sigma, x])

        # Check sigma and x consistency
        check_sigma_x(lam, sigma, x, LT_full['pot'][1])
        check_sigma_x(0, sigma, x, LT_KS['pot'][1])

        # Solve for ground states
        sol_full = E_full.solve(LT_full['pot'])
        sol_KS = E_KS.solve(LT_KS['pot'])

        # Compute v_xc from force-balance equation
        sigma_x_expval_full = sigma_x.expval(sol_full['gs_vector'])
        sigma_x_expval_KS = sigma_x.expval(sol_KS['gs_vector'])
        numerator_full = t * sigma + lam * (x_op * sigma_x).expval(sol_full['gs_vector'])
        vxc_value = (-t * sigma / sigma_x_expval_KS + numerator_full / sigma_x_expval_full)
        vxc.append(vxc_value)
    return np.array(vxc)

def compute_approximation(sigma_space, lam, x, eta):
    # Approximation from Eq. (65)
    vx_eta = lam * x + lam**2 * sigma_space * eta
    return np.array(vx_eta)

def plot_v_hxc_vs_approximations(lam, x, t, oscillator_size):
    # Setup basis and operators
    b_oscillator = NumberBasis(oscillator_size)
    b_spin = SpinBasis()
    b = b_oscillator.tensor(b_spin)

    # Define operators within the function scope
    num_op = b_oscillator.diag().extend(b)  # Number operator for oscillator
    x_op = b_oscillator.x_operator().extend(b)
    p_op = -1j * b_oscillator.dx_operator().extend(b)
    sigma_z = b_spin.sigma_z().extend(b)
    sigma_x = b_spin.sigma_x().extend(b)
    sigma_y = b_spin.sigma_y().extend(b)

    H0_Rabi_KS = num_op + 0.5 - t * sigma_x  # With 1/2 in harmonic oscillator
    CouplingRabi = x_op * sigma_z

    # Energy functionals
    E_KS = EnergyFunctional(H0_Rabi_KS, [sigma_z, x_op])
    E_full = EnergyFunctional(H0_Rabi_KS + lam * CouplingRabi, [sigma_z, x_op])

    # Define sigma_space for full range and for derivative calculation
    sigma_space = np.linspace(-0.95, 0.95, 201)
    sigma_space_deriv = np.linspace(-0.01, 0.01, 5)  # Focused around zero for derivative calculation

    # Compute exact v_Hxc over full sigma_space
    vxc = compute_v_hxc(sigma_space, lam, x, t, E_KS, E_full, sigma_x, x_op)
    vxc = np.real(vxc)

    # Compute v_Hxc at sigma values around sigma=0 for derivative calculation
    vxc_deriv = compute_v_hxc(sigma_space_deriv, lam, x, t, E_KS, E_full, sigma_x, x_op)
    vxc_deriv = np.real(vxc_deriv)

    # Compute derivative of v_Hxc at sigma = 0 using central difference
    delta_sigma = sigma_space_deriv[1] - sigma_space_deriv[0]
    dvxc_dsigma = (vxc_deriv[2] - vxc_deriv[0]) / (2 * delta_sigma)  # Using points around sigma=0

    # Compute eta for tangency
    eta_tangent = dvxc_dsigma / lam**2
    print(f"Computed η for tangency at σ=0 and λ={lam}: η = {eta_tangent}")

    # Compute approximation using computed eta
    vx_eta_tangent = compute_approximation(sigma_space, lam, x, eta_tangent)
    vx_eta_tangent = np.real(vx_eta_tangent)

    # Plotting
    plt.figure()
    plt.plot(sigma_space, vxc, 'r-', label=r'$v_{\mathrm{Hxc}}$ (Exact)')
    plt.plot(sigma_space, vx_eta_tangent, 'g--', label=rf'Approximation with $\eta_c={eta_tangent:.2f}$')
    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'$v$')
    plt.title(r'$v_{{\mathrm{{Hxc}}}}$ and Approximation for $\lambda={}$'.format(lam))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the plot before showing it
    plt.savefig('v_hxc_lambda_{}.pdf'.format(lam), format='pdf')
    plt.show()

def main():
    x = 0  # xi = 0
    t = 1

    # Plot 1: λ = 1
    lam1 = 1
    oscillator_size1 = 30  # Default size is sufficient
    print(f"Generating Plot 1 for λ = {lam1}")
    plot_v_hxc_vs_approximations(lam1, x, t, oscillator_size1)

    # Plot 2: λ = 3
    lam2 = 3
    oscillator_size2 = 50  # Increase oscillator_size for better accuracy
    print(f"Generating Plot 2 for λ = {lam2}")
    plot_v_hxc_vs_approximations(lam2, x, t, oscillator_size2)

    # Now compute η vs λ and plot
    oscillator_size = 50  # Use a reasonable oscillator size for accuracy

    # Define a range of lambda values
    lam_values = np.linspace(0.1, 3.0, 50)  # 50 points from 0.1 to 3.0

    eta_tangent_values = []

    # Setup basis and operators once outside the loop
    b_oscillator = NumberBasis(oscillator_size)
    b_spin = SpinBasis()
    b = b_oscillator.tensor(b_spin)
    num_op = b_oscillator.diag().extend(b)
    x_op = b_oscillator.x_operator().extend(b)
    p_op = -1j * b_oscillator.dx_operator().extend(b)
    sigma_z = b_spin.sigma_z().extend(b)
    sigma_x = b_spin.sigma_x().extend(b)
    sigma_y = b_spin.sigma_y().extend(b)

    sigma_space_deriv = np.linspace(-0.01, 0.01, 5)

    for lam in lam_values:
        print(f"Computing η for λ = {lam:.2f}")

        H0_Rabi_KS = num_op + 0.5 - t * sigma_x
        CouplingRabi = x_op * sigma_z

        E_KS = EnergyFunctional(H0_Rabi_KS, [sigma_z, x_op])
        E_full = EnergyFunctional(H0_Rabi_KS + lam * CouplingRabi, [sigma_z, x_op])

        # Compute v_Hxc at sigma values around sigma=0
        vxc_deriv = compute_v_hxc(sigma_space_deriv, lam, x, t, E_KS, E_full, sigma_x, x_op)
        vxc_deriv = np.real(vxc_deriv)

        # Compute derivative
        delta_sigma = sigma_space_deriv[1] - sigma_space_deriv[0]
        dvxc_dsigma = (vxc_deriv[2] - vxc_deriv[0]) / (2 * delta_sigma)

        eta_tangent = dvxc_dsigma / lam**2
        eta_tangent_values.append(eta_tangent)

    # Convert lists to arrays
    lam_values = np.array(lam_values)
    eta_tangent_values = np.array(eta_tangent_values)

    # Now plot η_tangent versus lam_values
    plt.figure()
    plt.plot(lam_values, eta_tangent_values, 'bo-', label=r'$\eta$ vs $\lambda$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\eta$ (tangent at $\sigma=0$)')
    plt.title(r'Relationship between $\lambda$ and $\eta$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('eta_vs_lambda.pdf')
    plt.show()

if __name__ == "__main__":
    main()


