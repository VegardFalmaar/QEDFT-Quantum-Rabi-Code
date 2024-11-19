from pathlib import Path

import numpy as np
from orgmin import ComputeDB

from investigate_form_of_T_in_sigma import sigma_values_from_logspace
from quantum_rabi import QuantumRabi
from optimally_translated_gaussian import OptimalGaussian


def initialize_db():
    ComputeDB.initialize(
        Path('numerical-results'),
        'F-v-s-1',
        {'lmbda': float, 'sigma': float, 't': float},
        {'F': float, 'v': float, 's': float},
        description='''A first run to store results regarding the QR model.
        Results are the (Levy--)Lieb-type functional F, the value of v which
        correspond to F, and the optimal translation s of the Gaussian trial
        states.

        Runs are performed with the constrained optimization (i.e. fixing j to
        the optimal value from the hypervirial theorem) using oscillator size
        40.

        Larger values for lambda create trouble for the small values of t,
        therefore the parameters here are in the ranges
        lambda: [0, 2]
        t: [0.2, 3]
        sigma: (-1, 1)
        '''
    )

def main():
    sigma_values = sigma_values_from_logspace()[1:-1]
    lmbda_values = np.linspace(0.0, 2.0, 60)
    t_values = np.linspace(0.2, 3, 60)
    db = ComputeDB(Path('numerical-results/F-v-s-1'))
    for l in lmbda_values:
        for t in t_values:
            print(f'{l = :f}, {t = :f}')
            for s in sigma_values:
                qr = QuantumRabi(
                    omega=1.0, t=t, g=1.0, lmbda=l, oscillator_size=40)
                F, v = qr.F_from_constrained_minimization(s, 0)
                og = OptimalGaussian(omega=1.0, t=t, g=1.0, lmbda=l, sigma=s)
                opt_trans = og.find_optimal_translation(verbose=False)
                db.add(
                    {'lmbda': l, 'sigma': s, 't': t},
                    {'F': F, 'v': v, 's': opt_trans},
                    save=False,
                )
    db.save_data()




if __name__ == '__main__':
    # initialize_db()
    main()
