"""
@author: Nathan
Last modified 2/24/2021
This script defines two functions for the single-layer and three-layer
models that calculate the total current-density from the nondimensional
solutions.

Citations:
"""
import numpy as np
from ..utils import matrices

class TotalCurrent():
    
    def __init__(self, params, grid):
        self.params = params
        self.grid = grid
        self.mat = matrices.Matrices(self.params, grid)

        self.b = self.params.b
        self.delta = self.params.delta
        self.DI = self.params.DI
        self.epsp = self.params.epsp
        self.G0 = self.params.G0
        self.Kn = self.params.Kn
        self.Kp = self.params.Kp
        self.lam = self.params.lam
        self.lam2 = self.params.lam2
        self.N = self.params.N
        self.NE = self.params.NE
        self.NH = self.params.NH
        self.N0 = self.params.N0
        self.q = self.params.q
        self.sigma = self.params.sigma
        self.Tion = self.params.Tion
        self.Vt = self.params.Vt

        self.dx = self.grid.dx

        self.Av = self.mat.Av
        self.Dx = self.mat.Dx

    def __call__(self, t_vector, u_matrix, mode):
        if mode == "slm":
            return self.total_current_slm(t_vector, u_matrix)
        elif mode == "tlm":
            return self.total_current_tlm(t_vector, u_matrix)

    def total_current_slm(self, t_vector, u_matrix):
        P = u_matrix[0:self.N+1,:]
        phi = u_matrix[self.N+1:2*self.N+2,:]
        n = u_matrix[2*self.N+2:3*self.N+3,:]
        p = u_matrix[3*self.N+3:4*self.N+4,:]
        mE_data = np.zeros((self.N, t_vector.size))
        dd_data = np.zeros((self.N, t_vector.size))
        FP_data = np.zeros((self.N, t_vector.size))
        fn_data = np.zeros((self.N, t_vector.size))
        fp_data = np.zeros((self.N, t_vector.size))
        
        dt = np.append([0.0], np.diff(t_vector))
        
        for i in range(0, t_vector.size):
        
            mE = np.matmul(self.Dx, phi[:,i])
            dd = mE*dt[i]
            FP = self.lam * (np.matmul(self.Dx, P[:,i]) + mE * np.matmul(self.Av, P[:,i]))
            fn = (np.matmul(self.Dx, n[:,i]) - mE * np.matmul(self.Av, n[:,i]))
            fp = -(np.matmul(self.Dx, p[:,i]) + mE * np.matmul(self.Av, p[:,i]))

            mE_data[:,i] = mE
            dd_data[:,i] = dd
            FP_data[:,i] = FP
            fn_data[:,i] = fn
            fp_data[:,i] = fp
         
        J_data = fn_data + fp_data - (self.sigma*self.lam2/self.delta)*dd_data + (self.sigma*self.lam/self.delta)*FP_data
        
        midpoint = int(np.ceil((self.N+1)/2))
        J_total = J_data[midpoint,:]
        
        return J_total, J_data, FP_data, dd_data, fn_data, fp_data

    def total_current_tlm(self, t_vector, u_matrix):

        P = u_matrix[0:self.N+1,:]
        phi = u_matrix[self.N+1:2*self.N+2,:]
        n = u_matrix[2*self.N+2:3*self.N+3,:]
        p = u_matrix[3*self.N+3:4*self.N+4,:]
        
        mE_data = np.zeros((self.N, t_vector.size))
        dd_data = np.zeros((self.N, t_vector.size))
        FP_data = np.zeros((self.N, t_vector.size))
        fn_data = np.zeros((self.N, t_vector.size))
        fp_data = np.zeros((self.N, t_vector.size))
        
        dt = np.append([0.0], np.diff(t_vector))
        
        for i in range(0, t_vector.size):
            mE = np.matmul(self.Dx, phi[:,i])
            dd = mE*dt[i]
            FP = self.lam*(np.matmul(self.Dx, P[:,i]) + mE * np.matmul(self.Av, P[:,i]))
            fn = self.Kn*(np.matmul(self.Dx, n[:,i]) - mE * np.matmul(self.Av, n[:,i]))
            fp = -self.Kp*(np.matmul(self.Dx, p[:,i]) + mE * np.matmul(self.Av, p[:,i]))
            mE_data[:,i] = mE
            dd_data[:,i] = dd
            FP_data[:,i] = FP
            fn_data[:,i] = fn
            fp_data[:,i] = fp
        
        J_data = fn_data + fp_data - (self.epsp*self.Vt/(self.q*self.G0*self.b**2*self.Tion))*dd_data + (self.DI*self.N0/(self.G0*self.b**2))*FP_data
        
        midpoint = int(np.ceil((self.N+1)/2))
        J_total = J_data[midpoint,:]
        
        return J_total, J_data, FP_data, dd_data, fn_data, fp_data








