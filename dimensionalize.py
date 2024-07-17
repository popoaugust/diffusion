"""
@author: Nathan
Last modified 3/13/2021 - stable with accurate results
A script that defines two functions (for single-layer and three-layer
models) that dimensionalize the solution outputs from the DAE solver.
Additionally, both functions dimensionalize
the current-densities used to plot the JV scans.
The function outputs are: x, t, P, phi, n, p, J_total, V_scan
Citations:
"""
from ..utils import grid
import numpy as np

from ..solver.total_current import TotalCurrent

###""" Define parameters class object and declare parameters. """
class Dimensionalizer():
    
    def __init__(self, params):
        self.params = params
        self.grid = grid.Grid(self.params)

        self.b = self.params.b
        self.dE = self.params.dE
        self.dH = self.params.dH
        self.DE = self.params.DE
        self.DI = self.params.DI

        self.epsp = self.params.epsp
        self.Fph = self.params.Fph
        self.G0 = self.params.G0
        self.LD = self.params.LD
        self.N = self.params.N
        self.NE = self.params.NE
        self.NH = self.params.NH
        self.N0 = self.params.N0
        self.q = self.params.q
        self.Tion = self.params.Tion
        self.Vt = self.params.Vt
        
        self.total_current = TotalCurrent(self.params, self.grid)

    def __call__(self, *vector, mode):
        if mode == "slm":
            assert len(vector) == 4
            return self.dimensionalize_slm(*vector)
        elif mode == "tlm":
            assert len(vector) == 6
            return self.dimensionalize_tlm(*vector)

    ###""" Single-layer model dimensionalization. """
    def dimensionalize_slm(self, x_vector, t_vector, u_matrix, psi_vector):
        ##""" The variable arrays are 2-dimensional to account for dense output.
            ##"""
        P = u_matrix[0:self.N+1,:]
        phi = u_matrix[self.N+1:2*self.N+2,:]
        n = u_matrix[2*self.N+2:3*self.N+3,:]
        p = u_matrix[3*self.N+3:4*self.N+4,:]
        
        x = x_vector*self.b
        t = t_vector*(self.LD*self.b/self.DI)
        
        P = self.N0*P
        phi = self.Vt*phi
        n = (self.Fph*self.b/self.DE)*n
        p = (self.Fph*self.b/self.DE)*p
        
        ###""" Non-Dimensional Current Density """###
        _, _, FP_data, dd_data, fn_data, fp_data = self.total_current(t_vector, u_matrix, "slm")
        
        J_dimensional = self.q*self.G0*self.b*(fn_data + fp_data) - (self.epsp*self.Vt/(self.b*self.Tion))*dd_data+ (self.q*self.DI*self.N0/self.b)*FP_data
        
        midpoint = int(np.ceil((self.N+1)/2))
        J_total = J_dimensional[midpoint,:]
        
        V_scan = psi_vector * self.Vt
        
        return x, t, P, phi, n, p, J_total, V_scan

    ###""" Three-layer model dimensionalization. """###
    def dimensionalize_tlm(self, x_vector, xE_vector, xH_vector, t_vector, u_matrix, psi_vector):
        ###""" The variable arrays are 2-dimensional to account for dense output.
            ####"""
        P = u_matrix[0:self.N+1,:]
        phi = u_matrix[self.N+1:2*self.N+2,:]
        n = u_matrix[2*self.N+2:3*self.N+3,:]
        p = u_matrix[3*self.N+3:4*self.N+4,:]
        phiE = u_matrix[4*self.N+4:4*self.N+self.NE+4,:]
        nE = u_matrix[4*self.N+self.NE+4:4*self.N+2*self.NE+4,:]
        phiH = u_matrix[4*self.N+2*self.NE+4:4*self.N+2*self.NE+self.NH+4,:]
        pH = u_matrix[4*self.N+2*self.NE+self.NH+4:4*self.N+2*self.NE+2*self.NH+4,:]
        
        phi = np.concatenate((phiE, phi, phiH), 0)
        n = np.concatenate((nE, n, n[-1]*np.ones((self.NH,t_vector.size))), 0)
        p = np.concatenate((p[0]*np.ones((self.NE,t_vector.size)), p, pH), 0)
        
        x = np.concatenate((-xE_vector[::-1], x_vector, xH_vector), 0)*self.b
        t = t_vector*self.params.LD*self.params.b/self.params.ion_mobility_factor
        P = self.N0*P
        phi = self.Vt*phi
        n = self.dE*n
        p = self.dH*p
        
        ###""" Non-Dimensional Current Density """#####
        J_total = self.q*self.G0*self.b*self.total_current(t_vector, u_matrix, "tlm")[0] / 10
        V_scan = psi_vector*self.Vt
        
        return x, t, P, phi, n, p, J_total, V_scan










