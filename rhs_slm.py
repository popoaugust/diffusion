#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:18:28 2023

@author: watcharanon
"""
import numpy as np

from ..solver import generation_recombination
from ..utils import grid, matrices

class SLM():
    
    def __init__(self, params):
        ##### Create class objects. #######
        self.params = params
        self.grid = grid.Grid(self.params)
        self.mat = matrices.Matrices(self.params, self.grid)
        self.genrec = generation_recombination.GR(self.params, self.grid)

        ##### Define parameters, grid, and matrices from class objects #########

        self.chi = self.params.chi
        self.delta = self.params.delta
        self.lam = self.params.lam
        self.lam2 = self.params.lam2
        self.N = self.params.N
        self.n_bar = self.params.n_bar
        self.p_bar = self.params.p_bar
        self.pbi = self.params.pbi
        self.phi_precondition = self.params.phi_precondition
        self.phi_f = self.params.phi_f
        self.psi_scan_rate = self.params.psi_scan_rate
        self.x = self.grid.x
        self.dx = self.grid.dx
        self.Av = self.mat.Av
        self.Dx = self.mat.Dx
        self.Lo = self.mat.Lo
        self.NN = self.mat.NN
        self.Rr = self.genrec.Rr
        self.Rl = self.genrec.Rl
        self.G = self.genrec.G
        self.R = self.genrec.mm_recomb


    def __call__(self, t, u, *psi):
        if ('pbi' in psi):
            psi = lambda t: self.pbi
        elif ('test1' in psi):
            psi = lambda t: self.pbi*(1 - (np.tanh(10**6*t)/np.tanh(10**6)))
        elif ('precondition' in psi):
            psi = lambda t: self.pbi*(1 - t/5) + self.phi_precondition*(t/5) if t<5 else self.phi_precondition
        elif ('reverse_scan' in psi):
            psi = lambda t: self.phi_precondition - self.psi_scan_rate*t 
        elif ('forward_scan' in psi):
            psi = lambda t: self.phi_f + self.psi_scan_rate*t
        elif ('scan_1' in psi):
            if self.params.Vi < self.params.Vf:
                psi = lambda t: self.phi_precondition + self.psi_scan_rate*t 
            elif self.params.Vi > self.params.Vf:
                psi = lambda t: self.phi_precondition - self.psi_scan_rate*t
        elif ('scan_2' in psi): 
            if self.params.Vi < self.params.Vf:
                psi = lambda t: self.phi_f - self.psi_scan_rate*t
            elif self.params.Vi > self.params.Vf:
                psi = lambda t: self.phi_f + self.psi_scan_rate*t
                
        rhs = np.zeros(4*self.N+4)
        
        P = u[0:self.N+1]
        phi = u[self.N+1:2*self.N+2]
        n = u[2*self.N+2:3*self.N+3]
        p = u[3*self.N+3:4*self.N+4]
        GR = self.G(np.matmul(self.Av, self.x)) - self.R(np.matmul(self.Av, p))
        mE = np.matmul(self.Dx, phi)
        FP = self.lam*(np.matmul(self.Dx, P) + mE*(np.matmul(self.Av,P)))
        cd = self.NN - np.matmul(self.Lo, P) + self.delta*(np.matmul(self.Lo, n) - self.chi*np.matmul(self.Lo, p))
        fn = (np.matmul(self.Dx, n) - mE*(np.matmul(self.Av, n)))
        fp = -(np.matmul(self.Dx, p) + mE*(np.matmul(self.Av, p)))
        
        rhs[0] = FP[0]
        rhs[1:self.N] = FP[1:self.N] - FP[0:self.N-1]
        rhs[self.N] = -FP[self.N-1]
        rhs[self.N+1] = phi[0] + 0.5*(psi(t) - self.pbi)
        rhs[self.N+2:2*self.N+1] = mE[1:self.N] - mE[0:self.N-1] - cd/self.lam2
        rhs[2*self.N+1] = phi[-1] - 0.5*(psi(t) - self.pbi)
        rhs[2*self.N+2] = n[0] - self.n_bar
        rhs[2*self.N+3:3*self.N+2] = fn[1:self.N] - fn[0:self.N-1] + (self.dx[1:self.N]*GR[1:self.N]+self.dx[0:self.N-1]*GR[0:self.N-1])/2
        rhs[3*self.N+2] = -fn[-1] + self.dx[-1]*GR[-1]/2
        rhs[3*self.N+3] = -fp[0] + self.dx[0]*GR[0]/2
        rhs[3*self.N+4:4*self.N+3] = -(fp[1:self.N] - fp[0:self.N-1]) + (self.dx[1:self.N]*GR[1:self.N]+self.dx[0:self.N-1]*GR[0:self.N-1])/2
        rhs[4*self.N+3] = p[-1] - self.p_bar
        
        return rhs
            
        # print(rhs)     
        # print(P) 
        
                
        

    


















