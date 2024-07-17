from . import jacobian_tlm
from .rhs_tlm import TLM
from ..analysis.plot import plot_distributions_tlm
from ..solver.total_current import TotalCurrent
from scipy.integrate import solve_ivp
from ..utils.radauDAE import RadauDAE
from matplotlib import pyplot as plt

import numpy as np
from ..utils import grid
from . import mass_tlm
from ..solver import initial_conditions

import time

def run(params, tf=1, show=True):
    start_time = time.time()
    
    grids = grid.Grid(params)
    ic = initial_conditions.Initial_Conditions(params, grids)
    mass = mass_tlm.Mass(params, grids)
    jac = jacobian_tlm.Jac(params)
    total_current = TotalCurrent(params, grids)

    sol_init = ic.sol_init_tlm
    mass = mass.M
    jac = jac.jac

    method = RadauDAE
    rtol = params.rtol
    atol = params.atol

    tlm_func = TLM(params)
    dae_fun = lambda t, u: tlm_func(t, u, 'pbi')

    sol = solve_ivp(fun=dae_fun, t_span=(0.0, tf), y0=sol_init,
                    rtol=rtol, atol=atol, jac_sparsity=jac,
                    method=method, vectorized=False, dense_output=False,
                    bPrintProgress=True, mass_matrix=mass
                    )
    t = sol.t
    dae_fun = lambda t, u: tlm_func(t, u, 'test1')
    sol_init = sol.y[:,-1]

    sol = solve_ivp(fun=dae_fun, t_span=(0.0, 10*tf), y0=sol_init,
                    rtol=rtol, atol=atol, jac_sparsity=jac,
                    method=method, vectorized=False, dense_output=True,
                    bPrintProgress=True, mass_matrix=mass
                    )

    sol_init = sol.y[:,-1]
    tf_reverse_scan = params.tf_scan
    dae_fun = lambda t, u: tlm_func(t, u, 'reverse_scan')

    sol = solve_ivp(fun=dae_fun, t_span=(0.0, tf_reverse_scan), y0=sol_init,
                    rtol=rtol, atol=atol, jac_sparsity=jac,
                    method=method, vectorized=False, dense_output=True,
                    bPrintProgress=True, mass_matrix=mass
                    )
    t_reverse = sol.t
    u_reverse = sol.y

    psi_applied_r = params.phi_precondition * np.ones(t_reverse.size) - params.psi_scan_rate*t_reverse
    J_reverse = total_current(t_reverse, u_reverse, "tlm")[0]

    sol_init = sol.y[:,-1]
    tf_forward_scan = params.tf_scan
    dae_fun = lambda t, u: tlm_func(t, u, 'forward_scan')

    sol = solve_ivp(fun=dae_fun, t_span=(0.0, tf_forward_scan), y0=sol_init,
                    rtol=rtol, atol=atol, jac_sparsity=jac,
                    method=method, vectorized=False, dense_output=True,
                    bPrintProgress=True, mass_matrix=mass
                    )
    t_forward = sol.t
    u_forward = sol.y

    psi_applied_f = params.phi_f*np.ones(t_forward.size) + params.psi_scan_rate*t_forward
    J_forward = total_current(t_forward, u_forward, "tlm")[0]

    J_nondim = np.concatenate((J_reverse, J_forward), 0)
    psi_scan = np.concatenate((psi_applied_r, psi_applied_f), 0)

    u_matrix = np.concatenate((u_reverse, u_forward), 1)
    t_vector = np.concatenate((t_reverse, t_forward), 0)

    u = sol.y
    t = sol.t
    x = grids.x
    xE = grids.xE
    xH = grids.xH

    plot_distributions_tlm(x, xE, xH, u, params)

    N = params.N
    x = grids.x

    P = u[0:N+1,-1]
    phi = u[N+1:2*N+2,-1]
    n = u[2*N+2:3*N+3,-1]
    p = u[3*N+3:4*N+4,-1]

    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    
    axes[0].plot(x, P)
    axes[1].plot(x, phi)
    axes[2].plot(x, n)
    axes[3].plot(x, p)

    axes[4].plot(psi_applied_r, J_reverse)
    axes[4].plot(psi_applied_f, J_forward)
    axes[4].set_xlim([0,40])
    axes[4].set_ylim([-0.2,0.8])
    
    if show:
        plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))









