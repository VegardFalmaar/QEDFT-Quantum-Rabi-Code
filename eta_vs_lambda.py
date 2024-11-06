import qmodel as q
import matplotlib.pyplot as plt
import numpy as np
from plot_config import PlotConfig  # Import the PlotConfig class

# Enable LaTeX in plots using PlotConfig
PlotConfig.use_tex()

# Define the check_sigma_x function with adjusted error tolerance
def check_sigma_x(lam, sigma, xi, j):
    if abs(lam * sigma + j + xi) > 1e-3:
        print(f'sigma--xi check: FAIL at λ={lam}, σ={sigma}, ξ={xi}, j={j}! '
              'Consider increasing oscillator_size')

def compute_v_hxc(sigma_space, lam, xi, t, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full):
    vxc = []
    for sigma in sigma_space:
        # Legendre transforms
        # For KS system, only sigma is used (photon-free)
        LT_KS = E_KS.legendre_transform(sigma)
        # For full system, both sigma and xi are used
        LT_full = E_full.legendre_transform([sigma, xi])

        # Check sigma and xi consistency for full system
        check_sigma_x(lam, sigma, xi, LT_full['pot'][1])

        # Solve for ground states
        sol_KS = E_KS.solve(LT_KS['pot'])
        sol_full = E_full.solve(LT_full['pot'])

        # Compute expectation values
        sigma_x_expval_KS = sigma_x_pf.expval(sol_KS['gs_vector'])
        sigma_x_expval_full = sigma_x_full.expval(sol_full['gs_vector'])
        x_sigma_x_expval_full = (x_op_full * sigma_x_full).expval(sol_full['gs_vector'])

        # Compute v_Hxc from force-balance equation
        numerator_full = t * sigma + lam * x_sigma_x_expval_full
        vxc_value = (-t * sigma / sigma_x_expval_KS) + (numerator_full / sigma_x_expval_full)
        vxc.append(vxc_value)
    return np.array(vxc)

def compute_approximation(sigma_space, lam, xi, eta):
    # Approximation with eta_c
    vx_eta = lam * xi + lam**2 * sigma_space * eta
    return np.array(vx_eta)

def plot_v_hxc_vs_approximations(lam, xi, t, oscillator_size):
    # Kohn-Sham system without photons (photon-free)
    b_spin = q.SpinBasis()
    sigma_z_pf = b_spin.sigma_z()
    sigma_x_pf = b_spin.sigma_x()

    H0_pf = -t * sigma_x_pf  # KS Hamiltonian without photons

    # Energy functional for KS system
    E_KS = q.EnergyFunctional(H0_pf, sigma_z_pf)

    # Full system with photons
    b_oscillator = q.NumberBasis(oscillator_size)
    b = b_oscillator.tensor(b_spin)

    # Define operators on the full basis
    num_op_full = b_oscillator.diag().extend(b)
    x_op_full = b_oscillator.x_operator().extend(b)
    sigma_z_full = sigma_z_pf.extend(b)
    sigma_x_full = sigma_x_pf.extend(b)

    H0_full = num_op_full + 0.5 - t * sigma_x_full
    CouplingRabi = x_op_full * sigma_z_full

    # Energy functional for full system
    E_full = q.EnergyFunctional(H0_full + lam * CouplingRabi, [sigma_z_full, x_op_full])

    # Define sigma_space for full range and for derivative calculation
    sigma_space = np.linspace(-0.95, 0.95, 201)
    sigma_space_deriv = np.linspace(-0.01, 0.01, 5)

    # Compute exact v_Hxc over full sigma_space
    vxc = compute_v_hxc(sigma_space, lam, xi, t, E_KS, E_full,
                        sigma_x_pf, sigma_x_full, x_op_full)
    vxc = np.real(vxc)

    # Compute v_Hxc at sigma values around sigma=0 for derivative calculation
    vxc_deriv = compute_v_hxc(sigma_space_deriv, lam, xi, t, E_KS, E_full,
                              sigma_x_pf, sigma_x_full, x_op_full)
    vxc_deriv = np.real(vxc_deriv)

    # Compute derivative of v_Hxc at sigma = 0 using central difference
    delta_sigma = sigma_space_deriv[1] - sigma_space_deriv[0]
    dvxc_dsigma = (vxc_deriv[2] - vxc_deriv[0]) / (2 * delta_sigma)

    # Compute eta for tangency
    eta_tangent = dvxc_dsigma / lam**2
    print(f"Computed η for tangency at σ=0 and λ={lam}: η = {eta_tangent}")

    # Compute approximation using computed eta
    vx_eta_tangent = compute_approximation(sigma_space, lam, xi, eta_tangent)
    vx_eta_tangent = np.real(vx_eta_tangent)

    # Prepare parameter dictionary for filenames and annotations
    params = {
        'lam': lam,
        'xi': xi,
        't': t,
        'osc_size': oscillator_size
    }

    # Convert lam to string without periods for filenames
    lam_str = str(lam).replace('.', '_')

    # Plotting using PlotConfig
    fig, ax = plt.subplots(figsize=(PlotConfig.fig_width, PlotConfig.fig_height))

    # Plot exact v_Hxc
    ax.plot(
        sigma_space,
        vxc,
        linestyle=PlotConfig.line_styles[0],
        color=PlotConfig.colors[0],
        linewidth=PlotConfig.linewidth,
        label=r'Exact $v_{\mathrm{Hxc}}$'
    )

    # Plot approximation with eta_tangent
    ax.plot(
        sigma_space,
        vx_eta_tangent,
        linestyle=PlotConfig.line_styles[1],
        color=PlotConfig.colors[0],  # Use a different color
        linewidth=PlotConfig.linewidth,
        label=rf'Approximation with $\eta_c={eta_tangent:.2f}$'
    )

    # Set axis information using PlotConfig
    PlotConfig.set_ax_info(
        ax,
        xlabel=r'$\sigma$',
        ylabel=r'$v$',
        title=rf'$v_{{\mathrm{{Hxc}}}}$ and Approximation for $\lambda={lam}$',
        legend=True
    )

    # Add parameter text box
    PlotConfig.parameter_text_box(
        ax,
        s=rf'$t = {t}, \; \xi = {xi}, \; \lambda = {lam}$',
        loc='lower right'
    )

    # Add grid lines
    #ax.grid(True)
    PlotConfig.tight_layout(fig, ax_aspect=3/2)

    # Save the plot using PlotConfig.save_fname
    fig.savefig(PlotConfig.save_fname(f'v_hxc_lambda_{lam_str}', '.pdf', params), format='pdf')
    # Uncomment the following line if you want to save as EPS
    # fig.savefig(PlotConfig.save_fname(f'v_hxc_lambda_{lam_str}', '.eps', params), format='eps')
    # plt.show()  # Uncomment to display the plot


def main():
    xi = 0  # xi = 0
    t = 1

    # Plot 1: λ = 1
    lam1 = 1
    oscillator_size1 = 30
    print(f"Generating Plot 1 for λ = {lam1}")
    plot_v_hxc_vs_approximations(lam1, xi, t, oscillator_size1)

    # Plot 2: λ = 2.5 (changed from 3 to 2.5 to avoid numerical difficulties)
    lam2 = 2.5
    oscillator_size2 = 50
    print(f"Generating Plot 2 for λ = {lam2}")
    plot_v_hxc_vs_approximations(lam2, xi, t, oscillator_size2)

    # Now compute η vs λ and plot
    oscillator_size = 50

    # Define a range of lambda values
    lam_values = np.linspace(0.1, 3.0, 50)
    eta_tangent_values = []

    # Setup basis and operators once outside the loop
    b_spin = q.SpinBasis()
    sigma_z_pf = b_spin.sigma_z()
    sigma_x_pf = b_spin.sigma_x()

    b_oscillator = q.NumberBasis(oscillator_size)
    b = b_oscillator.tensor(b_spin)
    num_op_full = b_oscillator.diag().extend(b)
    x_op_full = b_oscillator.x_operator().extend(b)
    sigma_z_full = sigma_z_pf.extend(b)
    sigma_x_full = sigma_x_pf.extend(b)

    sigma_space_deriv = np.linspace(-0.01, 0.01, 5)

    for lam in lam_values:
        print(f"Computing η for λ = {lam:.2f}")

        # Kohn-Sham Hamiltonian (photon-free)
        H0_pf = -t * sigma_x_pf
        E_KS = q.EnergyFunctional(H0_pf, sigma_z_pf)

        # Full Hamiltonian
        H0_full = num_op_full + 0.5 - t * sigma_x_full
        CouplingRabi = x_op_full * sigma_z_full
        E_full = q.EnergyFunctional(H0_full + lam * CouplingRabi, [sigma_z_full, x_op_full])

        # Compute v_Hxc at sigma values around sigma=0
        vxc_deriv = compute_v_hxc(
            sigma_space_deriv, lam, xi, t, E_KS, E_full,
            sigma_x_pf, sigma_x_full, x_op_full)
        vxc_deriv = np.real(vxc_deriv)

        # Compute derivative
        delta_sigma = sigma_space_deriv[1] - sigma_space_deriv[0]
        dvxc_dsigma = (vxc_deriv[2] - vxc_deriv[0]) / (2 * delta_sigma)

        eta_tangent = dvxc_dsigma / lam**2
        eta_tangent_values.append(eta_tangent)

    # Convert lists to arrays
    lam_values = np.array(lam_values)
    eta_tangent_values = np.array(eta_tangent_values)

    # Prepare parameter dictionary
    params = {
        'xi': xi,
        't': t,
        'osc_size': oscillator_size
    }

    # Plot η_tangent versus lam_values using PlotConfig
    fig, ax = plt.subplots(figsize=(PlotConfig.fig_width, PlotConfig.fig_height))

    ax.plot(
        lam_values,
        eta_tangent_values,
        linestyle='-',
        color=PlotConfig.colors[0],  # Use PlotConfig color
        linewidth=PlotConfig.linewidth,
        label=r'$\eta$ vs $\lambda$'
    )

    # Add horizontal dotted line at η = 1
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    #ax.text(lam_values[-1], 1.02, r'$\eta=1$', ha='right', va='bottom', color='gray')

    # Set y-axis limits to include η = 1
    ax.set_ylim(bottom=eta_tangent_values.min()-0.1, top=1.05)

    PlotConfig.set_ax_info(
        ax,
        xlabel=r'$\lambda$',
        ylabel=r'$\eta$ (tangent at $\sigma=0$)',
        title=r'Relationship between $\lambda$ and $\eta$',
        legend=False  # Legend is not necessary here
    )

    # Add parameter text box
    PlotConfig.parameter_text_box(
        ax,
        s=rf'$t = {t}, \; \xi = {xi}$',
        loc='lower right'
    )

    # Add grid lines
    #ax.grid(True)
    PlotConfig.tight_layout(fig, ax_aspect=3/2)

    # Save the plot using PlotConfig.save_fname
    fig.savefig(PlotConfig.save_fname('eta_vs_lambda', '.pdf', params), format='pdf')
    # Uncomment the following line if you want to save as EPS
    # fig.savefig(PlotConfig.save_fname('eta_vs_lambda', '.eps', params), format='eps')
    # plt.show()  # Uncomment to display the plot

if __name__ == "__main__":
    main()
