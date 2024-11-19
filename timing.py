import time

from quantum_rabi import QuantumRabi
from test_results_of_paper import (
    omega_values,
    t_values,
    g_values,
    sigma_values,
    xi_values,
    lmbda_values,
)


OSCILLATOR_SIZE = 120


def time_one_run(omega, t, g, sigma, xi, lmbda):
    qr = QuantumRabi(omega, t, g, lmbda=lmbda, oscillator_size=OSCILLATOR_SIZE)
    t1 = time.perf_counter()
    qr.F_from_minimization(sigma, xi)
    t2 = time.perf_counter()
    qr.F_from_constrained_minimization(sigma, xi)
    t3 = time.perf_counter()
    t_full = t2 - t1
    t_constr = t3 - t2
    return t_full, t_constr


def main():
    total_time_full = 0.0
    total_time_constrained = 0.0
    total_runs = 0
    for o in omega_values:
        for t in t_values:
            for g in g_values:
                for s in sigma_values:
                    for xi in xi_values:
                        for l in lmbda_values:
                            full, constr = time_one_run(o, t, g, s, xi, l)
                            total_time_full += full
                            total_time_constrained += constr
                            total_runs += 1
    print(f'Running with oscillator size {OSCILLATOR_SIZE}')
    print('Two-dimensional optim.:')
    print(f'  Total time: {total_time_full:.3f} s')
    print(f'  Per run:    {total_time_full / total_runs:.5f} s')
    print('Constrained optim.:')
    print(f'  Total time: {total_time_constrained:.3f} s')
    print(f'  Per run:    {total_time_constrained / total_runs:.5f} s')


if __name__ == '__main__':
    main()
