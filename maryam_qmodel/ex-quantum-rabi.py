'''
Created on 10.08.2024

@author: mage

quantum Rabi model example
'''
from qmodel import *
from dft import *
import matplotlib.pyplot as plt
import numpy as np




oscillator_size = 30
b_oscillator = NumberBasis(oscillator_size)
b_spin = SpinBasis()
b = b_oscillator.tensor(b_spin)

num_op = b_oscillator.diag().extend(b) # number operator for oscillator
x_op = b_oscillator.x_operator().extend(b)
p_op = -1j*b_oscillator.dx_operator().extend(b)
sigma_z = b_spin.sigma_z().extend(b)
sigma_x = b_spin.sigma_x().extend(b)
sigma_y = b_spin.sigma_y().extend(b)

t = 1

H0_Rabi_KS = num_op + 1/2 - t*sigma_x # with 1/2 in harmonic oscillator
CouplingRabi = x_op*sigma_z

def check_sigma_x(lam, sigma, x, j):
    # sigma and x must be in a precise relation after d/d xi applied on displacement rule
    if abs(lam*sigma + j + x) > 10e-4:
        print(f'sigma--xi check: FAIL at lam={lam}, sigma={sigma}, xi={x}, j={j}! maybe increase oscillator_size value')

# test hyperviral expressions

lam=1.2
v=0.5
j=-0.5
H=H0_Rabi_KS + lam*CouplingRabi+ v*sigma_z+ j*x_op
sol=H.eig(hermitian=True)
Psi=sol['eigenvectors'][0]

#print((p_op**2).expval(Psi)-(x_op**2).expval(Psi)-lam*(x_op*sigma_z).expval(Psi)-j*x_op.expval(Psi))

def plot_functionals_in_sigma():
    sigma_space = np.linspace(-0.95,0.95,51)
    
    lam = 2
    x = 0.6
    #eps = 0.1
    
    E_full = EnergyFunctional(H0_Rabi_KS + lam*CouplingRabi, [sigma_z, x_op])
    E_KS = EnergyFunctional(H0_Rabi_KS, [sigma_z, x_op])
    F_full = []
    F_full_eps = []
    F_KS = []
    E_xc = []
    test_F_full = []
    test_F_KS = []
    F_approx = []
    v_full_eps = []
    v_full_eps_prox = []
    vx=[]
    vx_eta=[]
    vx_guess=[]
    vxc=[]
    
    for sigma in sigma_space:
        LT_full = E_full.legendre_transform([sigma, x])
        #LT_full_eps = E_full.legendre_transform([sigma, x], epsMY=eps)
        LT_KS = E_KS.legendre_transform([sigma, x])
        
        check_sigma_x(lam, sigma, x, LT_full['pot'][1])
        check_sigma_x(0, sigma, x, LT_KS['pot'][1])
        
        F_full.append(LT_full['F'])
        #F_full_eps.append(LT_full_eps['F'])
        #v_full_eps.append(LT_full_eps['pot'][0])
        #v_full_eps_prox.append(-1/eps * (sigma - E_full.prox([sigma, x], eps)[0])) # Th. 9 in paper
        F_KS.append(LT_KS['F'])
        E_xc.append(F_full[-1] - F_KS[-1])
        
        # solve KS and full system for test
        sol_full = E_full.solve(LT_full['pot'])
        sol_KS = E_KS.solve(LT_KS['pot'])
        test_F_full.append(sol_full['gs_energy'] - np.dot(LT_full['pot'], [sigma, x]))
        test_F_KS.append(sol_KS['gs_energy'] - np.dot(LT_KS['pot'], [sigma, x]))

        #vpx apporoimation
        ##KS should be solved w/o coupling
        eta=0.67
        vx.append(lam*x + lam**2*sigma)
        vx_eta.append(lam*x + lam**2*sigma*eta)
        vx_guess.append(lam*x + lam**2*sigma + 1/2*lam**2*(sigma_x*sigma_z*sigma_x).expval(sol_KS['gs_vector'])/sigma_x.expval(sol_KS['gs_vector'])**2)
        vxc.append(-t*sigma/sigma_x.expval(sol_KS['gs_vector'])+(t*sigma+lam*(x_op*sigma_x).expval(sol_full['gs_vector']))/sigma_x.expval(sol_full['gs_vector']))

  
        
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(r'Universal functional in $\sigma$ at $\xi$={}'.format(x))
    # ax.plot(sigma_space, F_full, 'b', label=r'$F^\lambda(\sigma,\xi)$, $\lambda={}$'.format(lam))
    # ax.plot(sigma_space, F_full_eps, 'b--', label=r'$F_\varepsilon^\lambda(\sigma,\xi)$, $\lambda={}$, $\varepsilon={}$'.format(lam, eps))
    # ax.plot(sigma_space, v_full_eps, 'k--', label=r'$v$ from $-\nabla F_\varepsilon^\lambda(\sigma,\xi)$'.format(lam, eps))
    # ax.plot(sigma_space, -np.gradient(F_full_eps, sigma_space[1]-sigma_space[0]), 'kx', label=r'$v$ from numeric differentiation')
    # ax.plot(sigma_space, v_full_eps_prox, 'k.', label=r'$v$ from proximal map')
    # ax.plot(sigma_space, F_KS, 'g', label=r'$F^0(\sigma,\xi)$')
    # ax.plot(sigma_space, test_F_full, 'yx', label=r'test for  $F^\lambda(\sigma,\xi)$')
    # ax.plot(sigma_space, test_F_KS, 'yx', label=r'test for $F^0(\sigma,\xi)$')

    ax.plot(sigma_space, np.gradient(E_xc)/(sigma_space[1]-sigma_space[0]), 'rx', label=r'$v_{xc}$ from numerical differentiation')
    ax.plot(sigma_space,vxc,'r-',label=r'$v_{Hxc}$ from FB_formula')
    ax.plot(sigma_space,vx,'g-',label=r'$v_{Hpx}$ xc approximation')
    ax.plot(sigma_space,vx_eta,'g--',label=r'$v_{Hpx}$ xc approximation with $\eta_c$')
    ax.plot(sigma_space,vx_guess,'g.',label=r'$v_{px}$ xc guess next order')
    
    ## approximations: correct 1/2 in HO
    #ax.plot(sigma_space, 1-t*np.sqrt(1-np.square(sigma_space))+x**2, 'g.', label=r'$1-t\sqrt{1-\sigma^2}+\xi^2$')
    #approx = lambda sigma: 1-t*sqrt(1-sigma**2)+(x+lam*sigma/2)**2 - lam**2/4
    #ax.plot(sigma_space, list(map(approx, sigma_space)), 'rx', label='approx') #label=r'$1-t\sqrt{1-\sigma^2}+\frac{1}{2}\xi^2+\lambda\sigma\xi-\frac{\lambda^2}{4}(1-\sigma^2)$')
    #approx_MY = moreau_envelope(sigma_space, approx, eps)
    #ax.plot(sigma_space, approx_MY, 'r--', label=r'(1d) MY of approx')
    
    ax.legend()
    
plot_functionals_in_sigma()

def plot_in_lambda():
    ## test lambda behavior
    ## almost degeneracy at lam = 8 if v=j=0
    ## for v \neq 0 a higher lambda leads to a full up filling
    v = -0.1 # fixed
    j = 0 # fixed
    lam_space = np.linspace(0,15,100)
    sigma_array = []
    x_array = []
    eig_array = []
    eig_diff = []
    for lam in lam_space:
        H0 = H0_Rabi_KS + lam*CouplingRabi
        sol = EnergyFunctional(H0, [sigma_z, x_op]).solve([v,j]) # Hamiltonian, with coupling and external potential
        Psi0 = sol['gs_vector']
        sigma_array.append(sigma_z.expval(Psi0, transform_real = True))
        x_array.append(x_op.expval(Psi0, transform_real = True))
        eig_array.append(np.real(sol['eigenvalues'][:10])) # lowest eigenvalues to see possible crossings
        eig_diff.append(sol['eigenvalues'][1].real - sol['eigenvalues'][0].real)
    
    fig, axs = plt.subplots(ncols=1, nrows=2)
    fig.suptitle(f'expectation values in $\\lambda$ ($t = {t}, v = {v}, j = {j}$)')
    axs[0].plot(lam_space, sigma_array)
    axs[0].set_xlabel(r'$\lambda$')
    axs[0].set_ylabel(r'$\sigma$')
    axs[1].plot(lam_space, x_array)
    axs[1].set_xlabel(r'$\lambda$')
    axs[1].set_ylabel(r'$x$')
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(f'eigenvalues in $\\lambda$ ($t = {t}, v = {v}, j = {j}$)')
    ax.plot(lam_space, eig_array)
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.suptitle(f'difference of two lowest eigenvalues in $\\lambda$ ($t = {t}, j = {j}, v = {v}$)')
    ax.plot(lam_space, eig_diff)
    ax.set_yscale('log')
    
#plot_in_lambda()

def plot_photon_filling():
    lam = 2
    v = -0.1 # fixed
    j = 0 # fixed
    Psi0 = EnergyFunctional(H0_Rabi_KS + lam*CouplingRabi, [sigma_z, x_op]).solve([v, j])['gs_vector']
    
    sigma_expval = sigma_z.expval(Psi0, transform_real = True)
    x_expval = x_op.expval(Psi0, transform_real = True)
    print('sigma_z expectation value = {}'.format(sigma_expval))
    print('x expectation value = {}'.format(x_expval))
    check_sigma_x(lam, sigma_expval, x_expval, j)
    
    rho0_up = [ b.hop({'n': n, 's': +1}).expval(Psi0, transform_real = True) for n in range(oscillator_size) ]
    rho0_down = [ b.hop({'n': n, 's': -1}).expval(Psi0, transform_real = True) for n in range(oscillator_size) ]

    fig, axs = plt.subplots(ncols=1, nrows=2)
    fig.suptitle(f'photon filling ($\\lambda = {lam}, t = {t}, j = {j}, v = {v}$)')
    ymax = max(max(rho0_up), max(rho0_down)) * 1.2
    print('sum of rho0 = {}'.format(sum(rho0_up)+sum(rho0_down)))
    
    axs[0].set_title('spin up', size=10)
    axs[0].bar(range(oscillator_size), rho0_up)
    axs[0].set_ylim(0, ymax)

    axs[1].set_title('spin down', size=10)
    axs[1].bar(range(oscillator_size), rho0_down)
    axs[1].set_ylim(0, ymax)
    
#plot_photon_filling()

def plot_functionals_in_sigma_for_paper():
    sigma_space = np.linspace(-1,1,301)
    #xi_space = np.linspace(-1,1,9)
    lam_space = np.linspace(0,2,5)
    #colors = plt.cm.twilight(np.linspace(0,1,len(xi_space)))
    colors = plt.cm.tab10(np.linspace(0,1,len(lam_space)))
    
    lam = 0
    x = 0
    eps = 0.1
    
    #E = EnergyFunctional(H0_Rabi_KS + lam*CouplingRabi, [sigma_z, x_op])
    
    fig, ax = plt.subplots(ncols=1, nrows=1)
    
    #for i,x in enumerate(xi_space):
    for i,lam in enumerate(lam_space):
        E = EnergyFunctional(H0_Rabi_KS + lam*CouplingRabi, [sigma_z, x_op])
        F = []
        
        for sigma in sigma_space:
            LT = E.legendre_transform([sigma, x])
            check_sigma_x(lam, sigma, x, LT['pot'][1])
            F.append(LT['F'])
        
        #ax.plot(sigma_space, F, color=colors[i], linewidth=(1 if x != 0 else 2), label=r'$\xi={:.2f}$'.format(x))
        ax.plot(sigma_space, F, color=colors[i], label=r'$\lambda={:.2f}$'.format(lam))
    
    ax.legend()
    ax.set_xlabel(r'$\sigma$')
    #ax.set_ylabel(r'$F_\mathrm{LL}(\sigma,\xi)$')
    ax.set_ylabel(r'$F_\mathrm{LL}(\sigma,0)$')

#plot_functionals_in_sigma_for_paper()
plt.show()


def check_sigma_x(lam, sigma, x, j):
    # Sigma and x must be in a precise relation after derivative applied on displacement rule
    if abs(lam * sigma + j + x) > 1e-4:
        print(f'sigma--xi check: FAIL at lam={lam}, sigma={sigma}, xi={x}, j={j}! '
              'Maybe increase oscillator_size value')

def is_close(a, b, tol=1e-8):
    return abs(a - b) < tol

def plot_functionals_in_sigma(lam_values, eta_values):
    sigma_space = np.linspace(-0.95, 0.95, 21)
    x = 0.6  # Fixed value of xi (expectation value of x)
    t = 1
    results = []
    optimal_eta_for_lambda = []

    # Define the Kohn-Sham energy functional outside the loops
    E_KS = EnergyFunctional(H0_Rabi_KS, [sigma_z, x_op])

    for lam in lam_values:
        E_full = EnergyFunctional(H0_Rabi_KS + lam * CouplingRabi, [sigma_z, x_op])
        lam_results = []

        for eta in eta_values:
            F_full = []
            F_KS = []
            E_xc = []
            vx_eta = []
            vxc = []

            for sigma in sigma_space:
                # Legendre transforms
                LT_full = E_full.legendre_transform([sigma, x])
                LT_KS = E_KS.legendre_transform([sigma, x])

                # Check sigma and x consistency
                check_sigma_x(lam, sigma, x, LT_full['pot'][1])
                check_sigma_x(0, sigma, x, LT_KS['pot'][1])

                # Compute universal functionals and exchange-correlation energy
                F_full.append(LT_full['F'])
                F_KS.append(LT_KS['F'])
                E_xc.append(F_full[-1] - F_KS[-1])

                # Solve for ground states
                sol_full = E_full.solve(LT_full['pot'])
                sol_KS = E_KS.solve(LT_KS['pot'])

                # Compute v_xc from force-balance equation
                sigma_x_expval_full = sigma_x.expval(sol_full['gs_vector'])
                sigma_x_expval_KS = sigma_x.expval(sol_KS['gs_vector'])
                numerator_full = t * sigma + lam * (x_op * sigma_x).expval(sol_full['gs_vector'])
                vxc_value = (-t * sigma / sigma_x_expval_KS + numerator_full / sigma_x_expval_full)
                vxc.append(vxc_value)

                # Compute v_eta approximation
                vx_eta_value = lam * x + lam**2 * sigma * eta
                vx_eta.append(vx_eta_value)

            # Compute differences
            differences = np.array(vx_eta) - np.array(vxc)
            mean_difference = np.mean(np.abs(differences))

            # Store lam_results for this lambda
            lam_results.append({
                'eta': eta,
                'mean_difference': mean_difference,
                'vx_eta': vx_eta.copy(),
                'vxc': vxc.copy(),
                'sigma_space': sigma_space.copy(),
            })

        # Find the eta with the lowest mean difference for this lambda
        lam_results_sorted = sorted(lam_results, key=lambda res: res['mean_difference'])
        optimal_eta_value = lam_results_sorted[0]['eta']
        min_difference = lam_results_sorted[0]['mean_difference']

        # Store the optimal eta and corresponding lambda
        optimal_eta_for_lambda.append({
            'lam': lam,
            'eta': optimal_eta_value,
            'mean_difference': min_difference
        })

        # Add lam_results to the overall results
        for res in lam_results:
            res['lam'] = lam  # Add lam to each result
            results.append(res)

    # Plotting the mean differences
    plt.figure()
    for lam in lam_values:
        lam_results = [res for res in results if is_close(res['lam'], lam)]
        plt.plot([res['eta'] for res in lam_results],
                 [res['mean_difference'] for res in lam_results],
                 label=f'λ = {lam}')
    plt.xlabel('η')
    plt.ylabel('Mean Absolute Difference')
    plt.title('Mean Absolute Difference between $v_\\eta$ and $v_{xc}$')
    plt.legend()
    plt.show()

    # Plotting the optimal eta against lambda
    optimal_lam = [item['lam'] for item in optimal_eta_for_lambda]
    optimal_eta = [item['eta'] for item in optimal_eta_for_lambda]

    plt.figure()
    plt.plot(optimal_lam, optimal_eta, 'bo-')
    plt.xlabel('λ')
    plt.ylabel('Optimal η')
    plt.title('Optimal η vs. λ for Minimum Mean Absolute Difference')
    plt.grid(True)
    plt.show()

    # Optionally, print the optimal eta values
    print("Optimal η values for each λ:")
    for item in optimal_eta_for_lambda:
        print(f"λ = {item['lam']}: Optimal η = {item['eta']}, Mean Difference = {item['mean_difference']}")

    # Optionally, plot v_eta and v_xc for the optimal eta at each lambda
    for item in optimal_eta_for_lambda:
        lam = item['lam']
        eta = item['eta']
        matching_results = [res for res in results if is_close(res['lam'], lam) and is_close(res['eta'], eta)]
        if matching_results:
            specific_results = matching_results[0]
            plt.figure()
            plt.plot(specific_results['sigma_space'], specific_results['vxc'], 'r-', label='$v_{xc}$')
            plt.plot(specific_results['sigma_space'], specific_results['vx_eta'], 'b--', label='$v_\\eta$')
            plt.xlabel('$\\sigma$')
            plt.ylabel('Potential')
            plt.title(f'Comparison of $v_\\eta$ and $v_{{xc}}$ for λ = {lam}, Optimal η = {eta}')
            plt.legend()
            plt.show()
        else:
            print(f"No results found for λ = {lam} and η = {eta}.")
            
#lam_values = [0.5, 1.0, 1.5, 2.0]
#eta_values = np.linspace(0, 1, 101)  # More points for better resolution

#plot_functionals_in_sigma(lam_values, eta_values)

