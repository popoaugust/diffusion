# -*- coding: utf-8 -*-
"""
# @author: Nathan
# Last modified 3/3/2021
# Initial conditions class used for the single-layer model. A 4*N+4 column
# vector is calculated.
# The class uses parameters and spatial grid class objects to define the
# vector.
# Citations:
# """

import numpy as np

from . import generation_recombination
from ..tlm import jacobian_tlm
from ..tlm.rhs_tlm import TLM
from scipy.integrate import solve_bvp
from scipy.optimize import least_squares

###""" Create class objects used in quasi-steady state functions. """

class Initial_Conditions():

    def __init__(self, params, grid):
        ##""" Define class variables from constructor class objects. """###
        self.params, self.grid = params, grid
        self.jac = jacobian_tlm.Jac(self.params)
        self.genrec = generation_recombination.GR(self.params, self.grid)
        
        chi = self.params.chi
        kE = self.params.kE
        kH = self.params.kH
        N = self.params.N
        NE = self.params.NE
        NH = self.params.NH
        n_bar = self.params.n_bar
        p_bar = self.params.p_bar
        x = self.grid.x
        
        ####""" Define initial charge densities and electric potential. """
        P0 = np.ones(N+1)
        phi0 = np.zeros(N+1)
        n0 = p_bar*x + n_bar*(1 - x)
        p0 = n0
        ###
        ####""" Define instance variable for single layer model. """
        self.sol_init_slm = np.concatenate((P0, phi0, n0, p0), axis=0)
        
        ####""" Compute profiles for carrier concentrations from quasi-steady
            ####state boundary value problem. """
        y_guess_eqn = lambda x: [kH*x+kE*(1-x)/chi, 0.0*x, chi*kH*x+kE*(1-x), 0.0*x]
        y_guess = y_guess_eqn(x)
        sol = solve_bvp(self.yode, self.ybcs, x, y_guess)
        p_init = sol.y[0]
        n_init = sol.y[2]
        ###
        ###""" Uniform carrier concentrations and electric potential for
            ##transport layers. """
        phiE_init = np.zeros(NE)
        nE_init = np.ones(NE)
        phiH_init = np.zeros(NH)
        pH_init = np.ones(NH)
        
        ###""" Concatenate initial concentrations. Create instance variables
            ###for three-layer model. """
        self.u0 = np.concatenate((P0, phi0, n_init, p_init, phiE_init,
            nE_init, phiH_init, pH_init), 0)
        
        ###""" Use fsolve to find steady-state solution at built in voltage.
            ###"""
        print('Calculating consistent initial conditions.')
        jac_sparsity = self.jac.jac
        tlm_func = TLM(self.params)
        self.sol_init_tlm = least_squares(lambda u: tlm_func(0, u, 'pbi'), self.u0, jac_sparsity = jac_sparsity).x
        print('Complete')

    ###""" Quasi-steady state functions used in the above class. """#####
    def yode(self, x, y):
        G = self.genrec.G_init
        R = self.genrec.R
        Kp = self.params.Kp
        Kn = self.params.Kn

        dpdx = [-y[1]/Kp, G(x)-R(y[2],y[0],1), y[3]/Kn, -(G(x)-R(y[2],y[0],1))]
        return dpdx
        
    def ybcs(self, ya, yb):
        kH = self.params.kH
        Rl = self.genrec.Rl
        kE = self.params.kE
        Rr = self.genrec.Rr
        
        res = [yb[0]-kH, ya[2]+Rl(1,ya[0]), ya[2]-kE, yb[3]+Rr(yb[2],1)]
        return res

































