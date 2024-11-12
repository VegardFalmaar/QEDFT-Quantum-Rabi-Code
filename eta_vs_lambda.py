import qmodel as q
import matplotlib.pyplot as plt
import numpy as np
from plot_config import PlotConfig

from numba import njit
from concurrent.futures import ProcessPoolExecutor, as_completed

import functools
import csv
from typing import Dict
import re
from pathlib import Path

PlotConfig.use_tex() 

# Function to check consistency between sigma and xi
def check_sigma_x(lam, sigma, xi, j):
    if abs(lam * sigma + j + xi) > 1e-3:
        print(f'sigma--xi check: FAIL at λ={lam}, σ={sigma}, ξ={xi}, j={j}! '
              'Consider increasing oscillator_size')

# **Moved to top level to enable pickling**
def compute_single_vxc(sigma, lam, xi, t, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full):
    """
    Compute a single v_Hxc value for a given sigma.
    This function is designed to be used with parallel execution.
    """
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
    # Extract j from LT_full
    j = LT_full['pot'][1]
    # Check sigma and xi consistency for full system
    check_sigma_x(lam, sigma, xi, j)
    # Solve for ground state in full system
    sol_full = E_full.solve(LT_full['pot'])
    # Compute expectation values in full system
    sigma_x_expval_full = sigma_x_full.expval(sol_full['gs_vector'])
    x_sigma_x_expval_full = (x_op_full * sigma_x_full).expval(sol_full['gs_vector'])

    # --- Compute v_Hxc using force-balance equation ---
    numerator_full = t * sigma + lam * x_sigma_x_expval_full
    # To prevent division by zero, add a small epsilon
    epsilon = 1e-12
    vxc_value = (-t * sigma / (sigma_x_expval_KS + epsilon)) + (numerator_full / (sigma_x_expval_full + epsilon))
    return vxc_value

# Function to compute v_Hxc over a range of sigma values
def compute_v_hxc(sigma_space, lam, xi, t, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full):
    """
    Compute v_Hxc over a range of sigma values.
    This function is parallelized to utilize multiple CPU cores.
    """
    # Create a partial function with fixed parameters except sigma
    partial_func = functools.partial(
        compute_single_vxc,
        lam=lam,
        xi=xi,
        t=t,
        E_KS=E_KS,
        E_full=E_full,
        sigma_x_pf=sigma_x_pf,
        sigma_x_full=sigma_x_full,
        x_op_full=x_op_full
    )

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(partial_func, sigma_space))

    return np.array(results)

# Function to compute the approximation using eta
@njit
def compute_approximation(sigma_space, lam, xi, eta):
    vx_eta = lam * xi + lam**2 * sigma_space * eta
    return vx_eta

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
    # Use central difference for derivative
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

# Helper function to compute eta_tangent for a given lambda
def compute_eta_for_lambda(lam, xi, t, oscillator_size_base):
    """
    Compute eta_tangent for a specific lambda value.
    This function is designed to be used with parallel execution.
    """
    # Set oscillator_size based on lambda ranges
    if lam < 1:
        current_oscillator_size = 300
    elif lam < 2:
        current_oscillator_size = 500
    elif lam < 3:
        current_oscillator_size = 1000
    else:
        current_oscillator_size = 1500  # For λ ≥ 3

    print(f"Computing η for λ = {lam:.2f} with oscillator_size = {current_oscillator_size}")

    # --- KS System (Spin Basis) ---
    b_spin = q.SpinBasis()
    sigma_z_pf = b_spin.sigma_z()
    sigma_x_pf = b_spin.sigma_x()
    H0_pf = -t * sigma_x_pf
    E_KS = q.EnergyFunctional(H0_pf, sigma_z_pf)

    # --- Full System (Spin-Photon Basis) ---
    b_oscillator = q.NumberBasis(current_oscillator_size)
    b = b_oscillator.tensor(b_spin)
    num_op_full = b_oscillator.diag().extend(b)
    x_op_full = b_oscillator.x_operator().extend(b)
    sigma_z_full = sigma_z_pf.extend(b)
    sigma_x_full = sigma_x_pf.extend(b)
    H0_full = num_op_full + 0.5 - t * sigma_x_full
    CouplingRabi = x_op_full * sigma_z_full
    E_full = q.EnergyFunctional(H0_full + lam * CouplingRabi, [sigma_z_full, x_op_full])

    # Compute v_Hxc at sigma values around sigma=0
    sigma_space_deriv = np.linspace(-0.02, 0.02, 9)
    vxc_deriv = compute_v_hxc(
        sigma_space_deriv, lam, xi, t, E_KS, E_full,
        sigma_x_pf, sigma_x_full, x_op_full)
    vxc_deriv = np.real(vxc_deriv)

    # Compute derivative using central difference
    delta_sigma = sigma_space_deriv[1] - sigma_space_deriv[0]
    dvxc_dsigma = np.gradient(vxc_deriv, delta_sigma)
    dvxc_dsigma_at_zero = dvxc_dsigma[len(dvxc_dsigma) // 2]
    eta_tangent = dvxc_dsigma_at_zero / lam**2

    # Return the lambda, eta_tangent, and oscillator_size used
    return lam, eta_tangent, current_oscillator_size

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
        figsize=(PlotConfig.fig_width, 1.5 * PlotConfig.fig_height),
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

    # Replace 'alex' with 'upd' in the filename
    fig.savefig(PlotConfig.save_fname('v_hxc_lambda_shared_x_upd', '.pdf', params), format='pdf')
    # plt.show()  # Uncomment to display the plot

    # Increase oscillator_size for higher λ
    max_lambda = 5
    oscillator_size_base = 200  # Base oscillator size

    lam_values = np.linspace(0.1, max_lambda, 50)
    # We will use data_dict to store the results
    data_dict = {}

    # Read existing data from CSV file into data_dict
    csv_filename = 'eta_vs_lambda_upd.csv'
    csv_file_path = Path(csv_filename)
    if csv_file_path.exists():
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                lam_csv = float(row[0])
                eta_csv = float(row[1])
                oscillator_size_csv = int(row[2])
                data_dict[lam_csv] = (eta_csv, oscillator_size_csv)
    else:
        # Initialize CSV file with header
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['lambda', 'eta', 'oscillator_size'])

    # Precompute common spin operators to avoid redundant computations
    b_spin = q.SpinBasis()
    sigma_z_pf = b_spin.sigma_z()
    sigma_x_pf = b_spin.sigma_x()

    # Define a partial function with fixed parameters except lam
    partial_compute_eta = functools.partial(
        compute_eta_for_lambda,
        xi=xi,
        t=t,
        oscillator_size_base=oscillator_size_base
    )

    with ProcessPoolExecutor() as executor:
        # Submit tasks and collect futures
        futures = {executor.submit(partial_compute_eta, lam): lam for lam in lam_values}

        for future in as_completed(futures):
            lam = futures[future]
            try:
                lam_result, eta_tangent, oscillator_size_used = future.result()
                print(f"Finished computation for λ={lam_result}")


                # Update data_dict
                data_dict[lam_result] = (eta_tangent, oscillator_size_used)

                # Write the updated data_dict to CSV file
                with open(csv_file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['lambda', 'eta', 'oscillator_size'])
                    for lam_key in sorted(data_dict.keys()):
                        eta_value, osc_size_value = data_dict[lam_key]
                        writer.writerow([lam_key, eta_value, osc_size_value])

                # Update plot
                fig, ax = plt.subplots(figsize=(PlotConfig.fig_width, PlotConfig.fig_height))

                # Prepare data for plotting
                lam_plot = np.array(sorted(data_dict.keys()))
                eta_plot = np.array([data_dict[lam_key][0] for lam_key in lam_plot])

                ax.plot(
                    lam_plot,
                    eta_plot,
                    linestyle='-',
                    linewidth=PlotConfig.linewidth,
                    label=r'$\eta_c$ vs $\lambda$',
                    color=PlotConfig.colors[0]
                )

                # Add horizontal dotted line at η = 1
                ax.axhline(y=1, color=PlotConfig.colors[0], linestyle='--', linewidth=1)

                # Set y-axis limits to include η = 1
                ax.set_ylim(bottom=min(0.9, eta_plot.min() - 0.1), top=1.05)

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

                # Save the plot with filename including current lambda
                params = {
                    'xi': xi,
                    't': t,
                    'lam': lam_result
                }
                # Format lam_result to avoid periods in the filename
                lam_str = f"{lam_result:.2f}".replace('.', '_')
                fname = PlotConfig.save_fname(f'eta_vs_lambda_upd_{lam_str}', '.pdf', params)
                fig.savefig(fname, format='pdf')
                plt.close(fig)  # Close the figure to free memory

            except Exception as e:
                print(f"Exception occurred for λ={lam}: {e}")

    # Optionally, save the final plot with all data
    fig, ax = plt.subplots(figsize=(PlotConfig.fig_width, PlotConfig.fig_height))

    # Prepare data for plotting
    lam_plot = np.array(sorted(data_dict.keys()))
    eta_plot = np.array([data_dict[lam_key][0] for lam_key in lam_plot])

    ax.plot(
        lam_plot,
        eta_plot,
        linestyle='-',
        linewidth=PlotConfig.linewidth,
        label=r'$\eta_c$ vs $\lambda$',
        color=PlotConfig.colors[0]
    )

    # Add horizontal dotted line at η = 1
    ax.axhline(y=1, color=PlotConfig.colors[0], linestyle='--', linewidth=1)

    # Set y-axis limits to include η = 1
    ax.set_ylim(bottom=min(0.9, eta_plot.min() - 0.1), top=1.05)

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

    # Save the final plot
    params = {
        'xi': xi,
        't': t,
    }
    fname = PlotConfig.save_fname('eta_vs_lambda_upd_final', '.pdf', params)
    fig.savefig(fname, format='pdf')
    plt.close(fig)

if __name__ == "__main__":
    main()
