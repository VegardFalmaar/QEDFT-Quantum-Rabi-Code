import qmodel as q
import matplotlib.pyplot as plt
import numpy as np
from plot_config import PlotConfig
import csv
from pathlib import Path

PlotConfig.use_tex()

# Function to check consistency between sigma and xi
def check_sigma_x(lam, sigma, xi, j):
    if abs(lam * sigma + j + xi) > 1e-3:
        print(f'sigma--xi check: FAIL at λ={lam}, σ={sigma}, ξ={xi}, j={j}! '
              'Consider increasing oscillator_size')

# Function to compute v_dc over a range of sigma values
def compute_v_dc(sigma_space, lam, xi, t, omega, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full):
    vdc = []
    for sigma in sigma_space:
        # KS System Calculations
        LT_KS = E_KS.legendre_transform(sigma)
        sol_KS = E_KS.solve(LT_KS['pot'])
        sigma_x_expval_KS = sigma_x_pf.expval(sol_KS['gs_vector'])

        # Full System Calculations
        LT_full = E_full.legendre_transform([sigma, xi])
        check_sigma_x(lam, sigma, xi, LT_full['pot'][1])
        sol_full = E_full.solve(LT_full['pot'])
        sigma_x_expval_full = sigma_x_full.expval(sol_full['gs_vector'])
        x_sigma_x_expval_full = (x_op_full * sigma_x_full).expval(sol_full['gs_vector'])

        # Compute v_dc
        numerator_full = t * sigma + lam * x_sigma_x_expval_full
        vdc_value = (-t * sigma / sigma_x_expval_KS) + (numerator_full / sigma_x_expval_full)
        vdc.append(vdc_value)
    return np.array(vdc)

# Function to compute the approximation using eta_c
def compute_approximation(sigma_space, lam, xi, eta_c):
    vx_eta = lam * xi + lam**2 * sigma_space * eta_c
    return np.array(vx_eta)

# Main function for computing and plotting v_dc vs. approximations
def plot_v_dc_vs_approximations(lam, xi, t, omega, oscillator_size, ax=None):
    # Define basis and operators
    b_spin = q.SpinBasis()
    sigma_z_pf = b_spin.sigma_z()
    sigma_x_pf = b_spin.sigma_x()
    H0_pf = -t * sigma_x_pf
    E_KS = q.EnergyFunctional(H0_pf, sigma_z_pf)

    b_oscillator = q.NumberBasis(oscillator_size)
    b = b_oscillator.tensor(b_spin)
    num_op_full = b_oscillator.diag().extend(b)
    x_op_full = b_oscillator.x_operator().extend(b)
    sigma_z_full = sigma_z_pf.extend(b)
    sigma_x_full = sigma_x_pf.extend(b)
    H0_full = omega * (num_op_full + 0.5) - t * sigma_x_full
    CouplingRabi = x_op_full * sigma_z_full
    E_full = q.EnergyFunctional(H0_full + lam * CouplingRabi, [sigma_z_full, x_op_full])

    # Compute v_dc over sigma_space
    sigma_space = np.linspace(-0.95, 0.95, 201)
    sigma_space_deriv = np.linspace(-0.01, 0.01, 5)

    vdc = compute_v_dc(sigma_space, lam, xi, t, omega, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full)
    vdc = np.real(vdc)

    # Compute eta_c
    vdc_deriv = compute_v_dc(sigma_space_deriv, lam, xi, t, omega, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full)
    vdc_deriv = np.real(vdc_deriv)
    delta_sigma = sigma_space_deriv[1] - sigma_space_deriv[0]
    dvdc_dsigma = (vdc_deriv[2] - vdc_deriv[0]) / (2 * delta_sigma)
    eta_c = dvdc_dsigma / lam**2
    print(f"Computed η for tangency at σ=0 and λ={lam}: η = {eta_c}")

    vx_eta_tangent = compute_approximation(sigma_space, lam, xi, eta_c)
    vx_eta_tangent = np.real(vx_eta_tangent)

    if ax is None:
        fig, ax = plt.subplots(figsize=(PlotConfig.fig_width, PlotConfig.fig_height))

    ax.plot(sigma_space, vdc, linestyle=PlotConfig.line_styles[0], linewidth=PlotConfig.linewidth, label=rf'$v_{{\mathrm{{dc}}}}$', color=PlotConfig.colors[0])
    ax.plot(sigma_space, vx_eta_tangent, linestyle=PlotConfig.line_styles[1], linewidth=PlotConfig.linewidth, label=rf'$v_{{\mathrm{{dc}}}}^{{\mathrm{{pf}}, \eta_c}}$, $\eta_c={eta_c:.2f}$', color=PlotConfig.colors[0])

    PlotConfig.set_ax_info(ax, ylabel=r'$v$', legend=True)
    PlotConfig.parameter_text_box(ax, s=rf'$\lambda = {lam}, \; t = {t}, \; \xi = {xi}, \; \omega = {omega}$', loc='lower right')
    return ax  

# Function to write computed eta_c values to CSV
def write_to_csv(lam, eta_c, oscillator_size, csv_filename='eta_vs_lambda.csv'):
    csv_file_path = Path(csv_filename)
    file_exists = csv_file_path.exists()

    with open(csv_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['lambda', 'eta', 'oscillator_size'])
        writer.writerow([lam, eta_c, oscillator_size])

def main():
    xi = 0
    t = 1
    omega = 1

    # Shared x-axis plot for λ = 1 and λ = 2.5
    lam1, lam2 = 1, 2.5
    oscillator_size1, oscillator_size2 = 300, 500
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(PlotConfig.fig_width, 1.5 * PlotConfig.fig_height), sharex=True)
    
    # Plot for λ = 1
    print(f"Generating shared plot for λ = {lam1}")
    plot_v_dc_vs_approximations(lam1, xi, t, omega, oscillator_size1, ax=ax1)

    # Plot for λ = 2.5
    print(f"Generating shared plot for λ = {lam2}")
    plot_v_dc_vs_approximations(lam2, xi, t, omega, oscillator_size2, ax=ax2)

    # Set x-label on the bottom subplot only
    PlotConfig.set_ax_info(ax2, xlabel=r'$\sigma$')
    fig.tight_layout(pad=0.1)

    # Save the combined plot
    params = {'lam1': lam1, 'lam2': lam2, 'xi': xi, 't': t, 'omega': omega, 'osc_size1': oscillator_size1, 'osc_size2': oscillator_size2}
    fig.savefig(PlotConfig.save_fname('v_dc_lambda_shared_x', '.pdf', params), format='pdf')
    plt.close(fig)  # Close figure to free memory

    # Incremental plotting and CSV writing for additional λ values
    lam_values = np.linspace(0.1, 4, 40)
    eta_c_values = []
    b_spin = q.SpinBasis()
    sigma_z_pf = b_spin.sigma_z()
    sigma_x_pf = b_spin.sigma_x()
    sigma_space_deriv = np.linspace(-0.02, 0.02, 9)

    for lam in lam_values:
        if lam < 1:
            current_oscillator_size = 300
        elif lam < 2:
            current_oscillator_size = 500
        elif lam < 3:
            current_oscillator_size=1000
        else:
            current_oscillator_size = 1500

        b_oscillator = q.NumberBasis(current_oscillator_size)
        b = b_oscillator.tensor(b_spin)
        num_op_full = b_oscillator.diag().extend(b)
        x_op_full = b_oscillator.x_operator().extend(b)
        sigma_z_full = sigma_z_pf.extend(b)
        sigma_x_full = sigma_x_pf.extend(b)

        H0_pf = -t * sigma_x_pf
        E_KS = q.EnergyFunctional(H0_pf, sigma_z_pf)
        H0_full = omega * (num_op_full + 0.5) - t * sigma_x_full
        CouplingRabi = x_op_full * sigma_z_full
        E_full = q.EnergyFunctional(H0_full + lam * CouplingRabi, [sigma_z_full, x_op_full])

        vdc_deriv = compute_v_dc(sigma_space_deriv, lam, xi, t, omega, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full)
        vdc_deriv = np.real(vdc_deriv)
        delta_sigma = sigma_space_deriv[1] - sigma_space_deriv[0]
        dvdc_dsigma = np.gradient(vdc_deriv, delta_sigma)
        dvdc_dsigma_at_zero = dvdc_dsigma[len(dvdc_dsigma) // 2]
        eta_c = dvdc_dsigma_at_zero / lam**2
        eta_c_values.append(eta_c)

        # Write to CSV at each point
        write_to_csv(lam, eta_c, current_oscillator_size)

        # Plot η_c vs λ up to current lam
        fig, ax = plt.subplots(figsize=(PlotConfig.fig_width, PlotConfig.fig_height))
        ax.plot(lam_values[:len(eta_c_values)], eta_c_values, linestyle='-', linewidth=PlotConfig.linewidth, label=r'$\eta_c$ vs $\lambda$', color=PlotConfig.colors[0])
        ax.axhline(y=1, color=PlotConfig.colors[0], linestyle='--', linewidth=1)
        ax.set_ylim(bottom=min(0.9, min(eta_c_values) - 0.1), top=1.05)

        PlotConfig.set_ax_info(ax, xlabel=r'$\lambda$', ylabel=r'$\eta_c$', legend=True)
        PlotConfig.parameter_text_box(ax, s=rf'$t = {t}, \; \xi = {xi}, \; \omega = {omega}$', loc='lower right')
        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.9))

        fig.tight_layout(pad=0.1)
        params = {'xi': xi, 't': t, 'omega': omega, 'lam': lam}
        fname = PlotConfig.save_fname(f'eta_vs_lambda_{lam:.2f}'.replace('.', '_'), '.pdf', params)
        fig.savefig(fname, format='pdf')
        plt.close(fig)  # Close figure to free memory

if __name__ == "__main__":
    main()
