import qmodel as q
import matplotlib.pyplot as plt
import numpy as np
from plot_config import PlotConfig
from numba import njit
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import csv
from typing import Dict
from pathlib import Path

PlotConfig.use_tex()

# Check consistency function
def check_sigma_x(lam, sigma, xi, j):
    if abs(lam * sigma + j + xi) > 1e-3:
        print(f'sigma--xi check: FAIL at λ={lam}, σ={sigma}, ξ={xi}, j={j}! '
              'Consider increasing oscillator_size')

# Optimized function for computing v_dc values with cached operators
def compute_single_vdc(params):
    sigma, lam, xi, t, omega, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full = params
    LT_KS = E_KS.legendre_transform(sigma)
    sol_KS = E_KS.solve(LT_KS['pot'])
    sigma_x_expval_KS = sigma_x_pf.expval(sol_KS['gs_vector'])
    
    LT_full = E_full.legendre_transform([sigma, xi])
    j = LT_full['pot'][1]
    check_sigma_x(lam, sigma, xi, j)
    sol_full = E_full.solve(LT_full['pot'])
    sigma_x_expval_full = sigma_x_full.expval(sol_full['gs_vector'])
    x_sigma_x_expval_full = (x_op_full * sigma_x_full).expval(sol_full['gs_vector'])
    
    epsilon = 1e-12
    numerator_full = t * sigma + lam * x_sigma_x_expval_full
    vdc_value = (-t * sigma / (sigma_x_expval_KS + epsilon)) + (numerator_full / (sigma_x_expval_full + epsilon))
    return vdc_value

def compute_v_dc(sigma_space, lam, xi, t, omega, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full):
    params = [(sigma, lam, xi, t, omega, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full) for sigma in sigma_space]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(compute_single_vdc, params))
    return np.array(results)

@njit
def compute_approximation(sigma_space, lam, xi, eta_c):
    return lam * xi + lam**2 * sigma_space * eta_c

def plot_v_dc_vs_approximations(lam, xi, t, omega, oscillator_size, ax=None):
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

    sigma_space = np.linspace(-0.95, 0.95, 201)
    sigma_space_deriv = np.linspace(-0.01, 0.01, 5)

    vdc = compute_v_dc(sigma_space, lam, xi, t, omega, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full)
    vdc = np.real(vdc)

    vdc_deriv = compute_v_dc(sigma_space_deriv, lam, xi, t, omega, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full)
    vdc_deriv = np.real(vdc_deriv)
    delta_sigma = sigma_space_deriv[1] - sigma_space_deriv[0]
    dvdc_dsigma = (vdc_deriv[2] - vdc_deriv[0]) / (2 * delta_sigma)
    eta_tangent = dvdc_dsigma / lam**2
    print(f"Computed η for tangency at σ=0 and λ={lam}: η = {eta_tangent}")

    vx_eta_tangent = compute_approximation(sigma_space, lam, xi, eta_tangent)
    vx_eta_tangent = np.real(vx_eta_tangent)

    if ax is None:
        fig, ax = plt.subplots(figsize=(PlotConfig.fig_width, PlotConfig.fig_height))

    ax.plot(sigma_space, vdc, linestyle=PlotConfig.line_styles[0], linewidth=PlotConfig.linewidth, label=rf'$v_{{\mathrm{{dc}}}}$', color=PlotConfig.colors[0])
    ax.plot(sigma_space, vx_eta_tangent, linestyle=PlotConfig.line_styles[1], linewidth=PlotConfig.linewidth, label=rf'$v_{{\mathrm{{dc}}}}^{{\mathrm{{pf}}, \eta_\mathrm{{c}}}}$, $\eta_\mathrm{{c}}={eta_tangent:.2f}$', color=PlotConfig.colors[0])

    PlotConfig.set_ax_info(ax, ylabel=r'$v$', legend=True)
    PlotConfig.parameter_text_box(ax, s=rf'$\lambda = {lam}, \; t = {t}, \; \xi = {xi}, \; \omega = {omega}$', loc='lower right')
    return ax  

def compute_eta_for_lambda(lam, xi, t, omega, oscillator_size_base):
    if lam < 1:
        current_oscillator_size = 300
    elif lam < 2:
        current_oscillator_size = 500
    else:
        current_oscillator_size = 1000

    b_spin = q.SpinBasis()
    sigma_z_pf = b_spin.sigma_z()
    sigma_x_pf = b_spin.sigma_x()
    H0_pf = -t * sigma_x_pf
    E_KS = q.EnergyFunctional(H0_pf, sigma_z_pf)

    b_oscillator = q.NumberBasis(current_oscillator_size)
    b = b_oscillator.tensor(b_spin)
    num_op_full = b_oscillator.diag().extend(b)
    x_op_full = b_oscillator.x_operator().extend(b)
    sigma_z_full = sigma_z_pf.extend(b)
    sigma_x_full = sigma_x_pf.extend(b)
    H0_full = omega * (num_op_full + 0.5) - t * sigma_x_full
    CouplingRabi = x_op_full * sigma_z_full
    E_full = q.EnergyFunctional(H0_full + lam * CouplingRabi, [sigma_z_full, x_op_full])

    sigma_space_deriv = np.linspace(-0.02, 0.02, 9)
    vdc_deriv = compute_v_dc(sigma_space_deriv, lam, xi, t, omega, E_KS, E_full, sigma_x_pf, sigma_x_full, x_op_full)
    vdc_deriv = np.real(vdc_deriv)

    delta_sigma = sigma_space_deriv[1] - sigma_space_deriv[0]
    dvdc_dsigma = np.gradient(vdc_deriv, delta_sigma)
    dvdc_dsigma_at_zero = dvdc_dsigma[len(dvdc_dsigma) // 2]
    eta_tangent = dvdc_dsigma_at_zero / lam**2

    return lam, eta_tangent, current_oscillator_size

def main():
    xi, t, omega = 0, 1, 1
    lam1, oscillator_size1 = 1, 300
    lam2, oscillator_size2 = 2.5, 500

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(PlotConfig.fig_width, 1.5 * PlotConfig.fig_height), sharex=True)
    plot_v_dc_vs_approximations(lam1, xi, t, omega, oscillator_size1, ax=ax1)
    plot_v_dc_vs_approximations(lam2, xi, t, omega, oscillator_size2, ax=ax2)

    PlotConfig.set_ax_info(ax2, xlabel=r'$\sigma$')
    fig.tight_layout(pad=0.1)
    fig.savefig(PlotConfig.save_fname('v_dc_lambda_shared_x_upd', '.pdf', {'lam1': lam1, 'lam2': lam2, 'xi': xi, 't': t, 'omega': omega, 'osc_size1': oscillator_size1, 'osc_size2': oscillator_size2}), format='pdf')

    max_lambda = 5
    lam_values = np.linspace(0.1, max_lambda, 50)
    data_dict = {}
    csv_filename = 'eta_vs_lambda_upd.csv'
    csv_file_path = Path(csv_filename)

    if csv_file_path.exists():
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                data_dict[float(row[0])] = (float(row[1]), int(row[2]))

    partial_compute_eta = functools.partial(compute_eta_for_lambda, xi=xi, t=t, omega=omega, oscillator_size_base=200)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(partial_compute_eta, lam): lam for lam in lam_values}
        for future in as_completed(futures):
            lam = futures[future]
            try:
                lam_result, eta_tangent, oscillator_size_used = future.result()
                data_dict[lam_result] = (eta_tangent, oscillator_size_used)
            except Exception as e:
                print(f"Exception occurred for λ={lam}: {e}")

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda', 'eta', 'oscillator_size'])
        for lam_key in sorted(data_dict.keys()):
            eta_value, osc_size_value = data_dict[lam_key]
            writer.writerow([lam_key, eta_value, osc_size_value])

if __name__ == "__main__":
    main()
