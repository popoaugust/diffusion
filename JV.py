"""
@author: Nathan
Last modified 3/16/2021
This script defines the solution procedures for JV scans. The function
takes the input "single-layer" or "three-layer" to define which model
is used.
Both functions output the non-dimensional solution matrices: t_non_dim,
u_matrix, psi
and dimensional solution matrices: x, t, P, phi, n, p , J_total, V_applied
The non-dimensional JV scan is automatically plotted. The remaining
solutions can be dimensionalized and plotted using the dimensionalize
and plot scripts in the command window.
The solutions can be accessed through the variable explorer.
"""
from .total_current import TotalCurrent
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from ..utils.radauDAE import RadauDAE
from ..analysis.plot import plot_JV
from ..analysis.dimensionalize import Dimensionalizer

import numpy as np
import matplotlib.pyplot as plt

from ..utils import grid
from . import initial_conditions

import time

class JV():
    
    def __init__(self, params, tf = 1):
        ###""" Common class objects used by both JV functions. """###
        self.params = params
        self.grid = grid.Grid(self.params)
        self.ic = initial_conditions.Initial_Conditions(self.params, self.grid)
        self.total_current = TotalCurrent(self.params, self.grid)
        self.dimensionalizer = Dimensionalizer(self.params)

        ###""" Common solver settings used by both JV functions. """###

        self.tf = tf

        self.method = RadauDAE
        self.rtol = self.params.rtol
        self.atol = self.params.atol

    def __call__(self, *mode):
        if ("slm" in mode):
            from ..slm import jacobian_slm
            from ..slm.rhs_slm import SLM
            from ..slm import mass_slm
            
            ###""" Create class objects, mass matrix, and jacobian under standard
            ###conditions. """
            mass = mass_slm.Mass(self.params, self.grid)
            jac = jacobian_slm.Jac(self.params)
            
            mass = mass.M
            jac = jac.jac
            
            ##""" Solution process for JV scan.
            ###- Dense output is only used for the scanning solution procedures.
            ##"""
            sol_init = self.ic.sol_init_slm
            slm_func = SLM(self.params)
            dae_fun = lambda t, u: slm_func(t, u, "pbi")
            
            print("Eliminating transient behavior at built-in voltage.")
            start_time = time.time()
            sol = solve_ivp(fun=dae_fun, t_span=(0.0, self.tf), y0=sol_init,
                            rtol=self.rtol, atol=self.atol, jac_sparsity=jac,
                            method=self.method, vectorized=False, dense_output=False, 
                            bPrintProgress=True, mass_matrix=mass
                            )
            
            print("Preconditioning device to Vi.")
            dae_fun = lambda t, u: slm_func(t, u, "precondition")
            sol_init = sol.y[:,-1]
            
            sol = solve_ivp(fun=dae_fun, t_span=(0.0, 10*self.tf), y0=sol_init,
                            rtol=self.rtol, atol=self.atol, jac_sparsity=jac,
                            method=self.method, vectorized=False, dense_output=False, 
                            bPrintProgress=True, mass_matrix=mass
                            )
            
            sol_init = sol.y[:,-1]
            tf_scan = self.params.tf_scan
            dae_fun = lambda t, u: slm_func(t, u, "scan_1")
            
            print("Beginning JV scan.")
            
            sol = solve_ivp(fun=dae_fun, t_span=(0.0, tf_scan), y0=sol_init,
                            rtol=self.rtol, atol=self.atol, jac_sparsity=jac,
                            method=self.method, vectorized=False, dense_output=True,
                            bPrintProgress=True, mass_matrix=mass
                            )
            
            t_vector_1 = sol.t
            u_matrix_1 = sol.y
            
            ##""" These conditions allow for the correct calculations for psi
            ###depending if the scan begins reverse or forward. """
            
            if self.params.Vi < self.params.Vf:
                psi_1 = self.params.phi_precondition*np.ones(t_vector_1.size) + self.params.psi_scan_rate*t_vector_1
            elif self.params.Vi > self.params.Vf:
                psi_1 = self.params.phi_precondition*np.ones(t_vector_1.size) - self.params.psi_scan_rate*t_vector_1
            
            J_reverse = self.total_current(t_vector_1, u_matrix_1, "slm")[0]
            
            sol_init = sol.y[:,-1]
            dae_fun = lambda t, u: slm_func(t, u, "scan_2")
            print("Scanning opposite direction.")
            
            sol = solve_ivp(fun=dae_fun, t_span=(0.0, tf_scan), y0=sol_init,
                            rtol=self.rtol, atol=self.atol, jac_sparsity=jac,
                            method=self.method, vectorized=False, dense_output=True,
                            bPrintProgress=True, mass_matrix=mass
                            )
            
            print("Scan Complete")
            
            t_vector_2 = sol.t
            u_matrix_2 = sol.y
            
            if self.params.Vi < self.params.Vf:
                psi_2 = self.params.phi_f*np.ones(t_vector_2.size) - self.params.psi_scan_rate*t_vector_2
            elif self.params.Vi > self.params.Vf:
                psi_2 = self.params.phi_f*np.ones(t_vector_2.size) + self.params.psi_scan_rate*t_vector_2
            
            J_forward = self.total_current(t_vector_2, u_matrix_2, "slm")[0]
            
            ### """ Concatenate t_vectors and u_matrices into complete data
            ####     structures for both forward and backward scans. """

            t_non_dim = np.concatenate((t_vector_1, t_vector_1[-1] + t_vector_2), 0)
            u_matrix = np.concatenate((u_matrix_1, u_matrix_2), 1)
            psi = np.concatenate((psi_1, psi_2), 0)
            
            ###""" Dimensionalize results. """####
            dimensional_sol = self.dimensionalizer(self.grid.x, t_non_dim, u_matrix, psi, mode="slm")
            
            x = dimensional_sol[0]
            t = dimensional_sol[1]
            P = dimensional_sol[2]
            phi = dimensional_sol[3]
            n = dimensional_sol[4]
            p = dimensional_sol[5]
            J_total = dimensional_sol[6]
            V_applied = dimensional_sol[7]
            
            ##""" Plot results. """###
            plot_JV(V_applied, J_total, "dimensional", params=self.params)
            
            scan1_size = psi_1.size
            scan2_size = psi_2.size
            
            """""
            plt.plot(psi_1, J_reverse)
            plt.plot(psi_2, J_forward)
            plt.xlim([0,40])
            plt.ylim([0,0.8])
            """
            print("--- %s seconds ---" % (time.time() - start_time))

        elif ("tlm" in mode):
            from ..tlm import jacobian_tlm
            from ..tlm.rhs_tlm import TLM
            from ..tlm import mass_tlm
        
            # """ Create class objects, mass matrix, and jacobian under standard
            # conditions. """
            mass = mass_tlm.Mass(self.params, self.grid, mode=None)
            jac = jacobian_tlm.Jac(self.params, mode=None)

            mass = mass.M
            jac = jac.jac

            # """ Solution process for JV scan.
            # - Dense output is only used for the scanning solution procedures.
            # """
            sol_init = self.ic.sol_init_tlm
            tlm_func = TLM(self.params)
            dae_fun = lambda t, u: tlm_func(t, u, "pbi", mode=None)
            
            start_time = time.time()
            print("Eliminating transient behavior at built-in voltage.")
            sol = solve_ivp(fun=dae_fun, t_span=(0.0, self.tf), y0=sol_init,
                            rtol=self.rtol, atol=self.atol, jac_sparsity=jac,
                            method=self.method, vectorized=False, dense_output=False,
                            bPrintProgress=True, mass_matrix=mass
                            )

            dae_fun = lambda t, u: tlm_func(t, u, "precondition", mode=None)
            sol_init = sol.y[:,-1]
            
            # """ Adjust class objects to increase ion mobility to precondition at
            #     a voltage other than Vbi. """
            
            mass = mass_tlm.Mass(self.params, self.grid, mode="precondition")
            jac = jacobian_tlm.Jac(self.params, mode=None)
            
            mass = mass.M
            jac = jac.jac
            
            print("Preconditioning device to Vi.")
            sol = solve_ivp(fun=dae_fun, t_span=(0.0, 10*self.tf), y0=sol_init,
                            rtol=self.rtol, atol=self.atol, jac_sparsity=jac,
                            method=self.method, vectorized=False, dense_output=False,
                            bPrintProgress=True, mass_matrix=mass
                            )
            
            sol_precondition = sol.y[:,-1]
            tf_precondition = sol.t[-1]
            
            # """ Redefine jacobian to ensure conservation of ions for steady
            #     state solution. """
            jac = jacobian_tlm.Jac(self.params, mode="precondition")
            jac = jac.jac
            
            if self.params.Vi < self.params.Vbi:
                sol_init = least_squares(lambda u: TLM(tf_precondition, u, 
                    "precondition", mode="precondition"), sol_precondition,
                    jac_sparsity=jac).x

            else:
                sol_init = sol.y[:,-1]
            
            # """ Initial conditions for the JV scan are now calculated. Calculate
            #     non-dimensional scan time and reset class objects to normal
            #     conditions. """
            tf_scan = self.params.tf_scan
            dae_fun = lambda t, u: tlm_func(t, u, "scan_1", mode=None)
            
            mass = mass_tlm.Mass(self.params, self.grid, mode=None)
            jac = jacobian_tlm.Jac(self.params, mode=None)
            
            mass = mass.M
            jac = jac.jac
            
            print("Beginning JV scan.")
            sol = solve_ivp(fun=dae_fun, t_span=(0.0, tf_scan), y0=sol_init,
                            rtol=self.rtol, atol=self.atol, jac_sparsity=jac,
                            method=self.method, vectorized=False, dense_output=True,
                            bPrintProgress=True, mass_matrix=mass
                            )
        
            t_vector_1 = sol.t
            u_matrix_1 = sol.y
            
            # """ These conditions allow for the correct calculations for psi
            #     depending if the scan begins reverse or forward. """
            if self.params.Vi < self.params.Vf:
                psi_1 = self.params.phi_precondition*np.ones(t_vector_1.size) + self.params.psi_scan_rate*t_vector_1
            elif self.params.Vi > self.params.Vf:
                psi_1 = self.params.phi_precondition*np.ones(t_vector_1.size) - self.params.psi_scan_rate*t_vector_1
            
            sol_init = sol.y[:,-1]
            dae_fun = lambda t, u: tlm_func(t, u, "scan_2", mode=None)
            print("Scanning opposite direction.")
            sol = solve_ivp(fun=dae_fun, t_span=(0.0, tf_scan), y0=sol_init,
                            rtol=self.rtol, atol=self.atol, jac_sparsity=jac,
                            method=self.method, vectorized=False, dense_output=True,
                            bPrintProgress=True, mass_matrix=mass
                            )
            print("Scan Complete")
            print("--- %s seconds ---" % (time.time() - start_time))

            t_vector_2 = sol.t
            u_matrix_2 = sol.y
            
            if self.params.Vi < self.params.Vf:
                psi_2 = self.params.phi_f*np.ones(t_vector_2.size) - self.params.psi_scan_rate*t_vector_2
            elif self.params.Vi > self.params.Vf:
                psi_2 = self.params.phi_f*np.ones(t_vector_2.size) + self.params.psi_scan_rate*t_vector_2
            
            ### """ Concatenate t_vectors and u_matrices into complete data
            ###     structures for both forward and backward scans. """
            t_non_dim = np.concatenate((t_vector_1, t_vector_1[-1] + t_vector_2)
                , 0)
            u_matrix = np.concatenate((u_matrix_1, u_matrix_2), 1)
            psi = np.concatenate((psi_1, psi_2), 0)
            
            ###""" Dimensionalize results. """####
            dimensional_sol = self.dimensionalizer(self.grid.x, self.grid.xE, self.grid.xH,
                t_non_dim, u_matrix, psi, mode="tlm")
        
            x = dimensional_sol[0]
            t = dimensional_sol[1]
            P = dimensional_sol[2]
            phi = dimensional_sol[3]
            n = dimensional_sol[4]
            p = dimensional_sol[5]
            J_total = dimensional_sol[6]
            V_applied = dimensional_sol[7]
            
            ###""" Plot results. """####
            _, ax = plot_JV(V_applied, J_total, "dimensional", params=self.params)
            ax.set_ylim([-5,25])
                
            scan1_size = psi_1.size
            scan2_size = psi_2.size
            
        return t_non_dim, u_matrix, psi, x, t, P, phi, n, p , J_total, V_applied, scan1_size, scan2_size















