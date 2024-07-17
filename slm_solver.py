from . import jacobian_slm
from .rhs_slm import SLM
from scipy.integrate import solve_ivp
from ..utils.radauDAE import RadauDAE
from ..analysis.plot import plot_distributions_slm
from ..solver.total_current import TotalCurrent
import matplotlib.pyplot as plt

from ..utils import grid
from . import mass_slm
from ..solver import initial_conditions

import time
import numpy as np

def run(params, tf=1, show=True):
    
    start_time = time.time()
    
    grids = grid.Grid(params)
    ic = initial_conditions.Initial_Conditions(params, grids)
    mass = mass_slm.Mass(params, grids)
    jac = jacobian_slm.Jac(params)

    sol_init = ic.sol_init_slm
    mass = mass.M
    jac = jac.jac

    method = RadauDAE
    rtol = params.rtol
    atol = params.atol

    slm_func = SLM(params)
    dae_fun = lambda t, u: slm_func(t, u, 'pbi')

    sol = solve_ivp(fun=dae_fun, t_span=(0.0, tf), y0=sol_init,
                    rtol=rtol, atol=atol, jac_sparsity=jac,
                    method=method, vectorized=False, dense_output=False,
                    bPrintProgress=True, mass_matrix=mass
                    )

    # """
    # dae_fun = lambda t, u: SLM(t, u, 'precondition')
    # sol_init = sol.y[:,-1]
    # sol = solve_ivp(fun=dae_fun, t_span=(0.0, tf), y0=sol_init,
    # rtol=rtol, atol=atol, jac_sparsity=jac,
    # method=method, vectorized=False, dense_output=True, mass=mass)
    # """

    sol_init = sol.y[:,-1]
    dae_fun = lambda t, u: slm_func(t, u, 'test1')

    sol = solve_ivp(fun=dae_fun, t_span=(0.0, 1), y0=sol_init,
                    rtol=rtol, atol=atol, jac_sparsity=jac,
                    method=method, vectorized=False, dense_output=True,
                    bPrintProgress=True, mass_matrix=mass
                    )
    t = sol.t
    u = sol.y

    plot_distributions_slm(grids.x, u, params)
    total_current = TotalCurrent(params, grids)
    J = total_current(t, u, "slm")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(t, J[0])
    axes[0].set_xlim([0, 5*10e-7])
    axes[0].set_ylim([0, 1.5])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Current Density')
    axes[0].set_title("Time-Current Density")

    n = u[2*params.N+2:3*params.N+3,-1]

    axes[1].plot(grids.x, n)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 0.06])
    axes[1].set_xlabel('Distance, x')
    axes[1].set_ylabel('Electron Concentration, n')
    axes[1].set_title("Ref Figure 6")

    if show:
        plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))






