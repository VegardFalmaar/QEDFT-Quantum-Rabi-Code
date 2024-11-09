import qmodel as q
import matplotlib.pyplot as plt
import numpy as np
from plot_config import PlotConfig 

PlotConfig.use_tex()

# Function to check consistency between sigma and xi
def check_sigma_x(lam, sigma, xi, j):
    if abs(lam * sigma + j + xi) > 1e-3:
        print(f'sigma--xi check: FAIL at λ={lam}, σ={sigma}, ξ={xi}, j={j}! '
              'Consider increasing oscillator_size')

# Function to compute v_Hxc over a range of sigma values
def compute_v_hxc(sigma_space, lam, xi, t, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full):
    vxc = []
    for sigma in sigma_space:
        # --- KS System Calculations (Spin Basis) ---
        # Legendre transform for KS system with density sigma
        LT_KS = E_KS.legendre_transform(sigma)
        # Solve for ground state in KS system
        sol_KS = E_KS.solve(LT_KS['pot'])
        # Compute expectation value of sigma_x in KS system
        sigma_x_expval_KS = sigma_x_pf.expval(sol_KS['gs_vector'])

        # --- Full System Calculations (Spin-Photon Basis) ---
        # Legendre transform for full system with densities [sigma, xi]
        LT_full = E_full.legendre_transform([sigma, xi])
        # Check sigma and xi consistency for full system
        check_sigma_x(lam, sigma, xi, LT_full['pot'][1])
        # Solve for ground state in full system
        sol_full = E_full.solve(LT_full['pot'])
        # Compute expectation values in full system
        sigma_x_expval_full = sigma_x_full.expval(sol_full['gs_vector'])
        x_sigma_x_expval_full = (x_op_full * sigma_x_full).expval(sol_full['gs_vector'])

        # --- Compute v_Hxc using force-balance equation ---
        numerator_full = t * sigma + lam * x_sigma_x_expval_full
        vxc_value = (-t * sigma / sigma_x_expval_KS) + (numerator_full / sigma_x_expval_full)
        vxc.append(vxc_value)
    return np.array(vxc)

# Function to compute the approximation using eta
def compute_approximation(sigma_space, lam, xi, eta):
    # Approximation with eta_c
    vx_eta = lam * xi + lam**2 * sigma_space * eta
    return np.array(vx_eta)

# Function to plot v_Hxc and its approximations
def plot_v_hxc_vs_approximations(lam, xi, t, oscillator_size, ax=None):
    # --- KS System (Spin Basis) ---
    # Define spin basis
    b_spin = q.SpinBasis()
    # Define spin operators in spin basis
    sigma_z_pf = b_spin.sigma_z()
    sigma_x_pf = b_spin.sigma_x()
    # Define KS Hamiltonian without photons
    H0_pf = -t * sigma_x_pf
    # Energy functional for KS system
    E_KS = q.EnergyFunctional(H0_pf, sigma_z_pf)

    # --- Full System (Spin-Photon Basis) ---
    # Define photon basis
    b_oscillator = q.NumberBasis(oscillator_size)
    # Define combined spin-photon basis
    b = b_oscillator.tensor(b_spin)
    # Define operators on the full basis
    num_op_full = b_oscillator.diag().extend(b)
    x_op_full = b_oscillator.x_operator().extend(b)
    sigma_z_full = sigma_z_pf.extend(b)
    sigma_x_full = sigma_x_pf.extend(b)
    # Define full Hamiltonian with coupling
    H0_full = num_op_full + 0.5 - t * sigma_x_full
    CouplingRabi = x_op_full * sigma_z_full
    # Energy functional for full system
    E_full = q.EnergyFunctional(H0_full + lam * CouplingRabi, [sigma_z_full, x_op_full])

    # --- Compute v_Hxc over sigma_space ---
    sigma_space = np.linspace(-0.95, 0.95, 201)
    sigma_space_deriv = np.linspace(-0.01, 0.01, 5)

    # Compute exact v_Hxc
    vxc = compute_v_hxc(sigma_space, lam, xi, t, E_KS, E_full,
                        sigma_x_pf, sigma_x_full, x_op_full)
    vxc = np.real(vxc)

    # Compute derivative for eta calculation
    vxc_deriv = compute_v_hxc(sigma_space_deriv, lam, xi, t, E_KS, E_full,
                              sigma_x_pf, sigma_x_full, x_op_full)
    vxc_deriv = np.real(vxc_deriv)
    delta_sigma = sigma_space_deriv[1] - sigma_space_deriv[0]
    dvxc_dsigma = (vxc_deriv[2] - vxc_deriv[0]) / (2 * delta_sigma)
    eta_tangent = dvxc_dsigma / lam**2
    print(f"Computed η for tangency at σ=0 and λ={lam}: η = {eta_tangent}")

    # Compute approximation using computed eta
    vx_eta_tangent = compute_approximation(sigma_space, lam, xi, eta_tangent)
    vx_eta_tangent = np.real(vx_eta_tangent)

    # --- Plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(PlotConfig.fig_width, PlotConfig.fig_height))

    # Plot exact v_Hxc
    ax.plot(
        sigma_space,
        vxc,
        linestyle=PlotConfig.line_styles[0],
        linewidth=PlotConfig.linewidth,
        label=rf'$v_{{\mathrm{{Hxc}}}}$',
        color=PlotConfig.colors[0]
    )

    # Plot approximation with eta_tangent
    ax.plot(
        sigma_space,
        vx_eta_tangent,
        linestyle=PlotConfig.line_styles[1],
        linewidth=PlotConfig.linewidth,
        label=rf'$v_{{\mathrm{{Hxc}}}}^{{\mathrm{{pf}}, \eta_c}}$, $\eta_c={eta_tangent:.2f}$',
        color=PlotConfig.colors[0]
    )

    PlotConfig.set_ax_info(
        ax,
        ylabel=r'$v$',
        legend=True
    )

    PlotConfig.parameter_text_box(
        ax,
        s=rf'$\lambda = {lam}, \; t = {t}, \; \xi = {xi}$',
        loc='lower right'
    )

    return ax  


def main():
    xi = 0  # xi = 0
    t = 1

    # Define lambda values and oscillator sizes
    lam1 = 1
    oscillator_size1 = 30
    lam2 = 2.5
    oscillator_size2 = 50

    # Create figure with two subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(PlotConfig.fig_width, 2 * PlotConfig.fig_height),
        sharex=True
    )

    # Plot for λ = 1 on ax1
    print(f"Generating Plot for λ = {lam1}")
    plot_v_hxc_vs_approximations(lam1, xi, t, oscillator_size1, ax=ax1)

    # Plot for λ = 2.5 on ax2
    print(f"Generating Plot for λ = {lam2}")
    plot_v_hxc_vs_approximations(lam2, xi, t, oscillator_size2, ax=ax2)

    # Set x-label only on the bottom subplot
    PlotConfig.set_ax_info(
        ax2,
        xlabel=r'$\sigma$',
    )

    fig.tight_layout(pad=0.1)

    params = {
        'lam1': lam1,
        'lam2': lam2,
        'xi': xi,
        't': t,
        'osc_size1': oscillator_size1,
        'osc_size2': oscillator_size2
    }

    fig.savefig(PlotConfig.save_fname('v_hxc_lambda_shared_x', '.pdf', params), format='pdf')
    # plt.show()  # Uncomment to display the plot

    oscillator_size = 200  

    lam_values = np.linspace(0.1, 5.0, 50)
    eta_tangent_values = []

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


        H0_pf = -t * sigma_x_pf
        E_KS = q.EnergyFunctional(H0_pf, sigma_z_pf)

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

    lam_values = np.array(lam_values)
    eta_tangent_values = np.array(eta_tangent_values)


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
        linewidth=PlotConfig.linewidth,
        label=r'$\eta_c$ vs $\lambda$',
        color=PlotConfig.colors[0]
    )

    # Add horizontal dotted line at η = 1
    ax.axhline(y=1, color=PlotConfig.colors[0], linestyle='--', linewidth=1)

    # Set y-axis limits to include η = 1
    ax.set_ylim(bottom=eta_tangent_values.min() - 0.1, top=1.05)

    PlotConfig.set_ax_info(
        ax,
        xlabel=r'$\lambda$',
        ylabel=r'$\eta$',
        legend=True
    )

    PlotConfig.parameter_text_box(
        ax,
        s=rf'$t = {t}, \; \xi = {xi}$',
        loc='lower right'
    )


    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.9))


    fig.tight_layout(pad=0.1)

    fig.savefig(PlotConfig.save_fname('eta_vs_lambda', '.pdf', params), format='pdf')
    # plt.show()  # Uncomment to display the plot

if __name__ == "__main__":
    main()
