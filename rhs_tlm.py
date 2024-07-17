#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:45:32 2023

@author: watcharanon
"""
import numpy as np
from ..solver import generation_recombination
from ..utils import parameters, grid, matrices

class TLM():
    
    def __init__(self, params):
        self.params = params
        self.grid = grid.Grid(self.params)
        self.mat = matrices.Matrices(self.params, self.grid)
        self.genrec = generation_recombination.GR(self.params, self.grid)
        self.chi = self.params.chi
        self.delta = self.params.delta
        self.kE = self.params.kE
        self.KE = self.params.KE
        self.kH = self.params.kH
        self.KH = self.params.KH
        self.Kn = self.params.Kn
        self.Kp = self.params.Kp
        self.lam = self.params.lam
        self.lam2 = self.params.lam2
        self.lamE = self.params.lamE
        self.lamE2 = self.params.lamE2
        self.lamH = self.params.lamH
        self.lamH2 = self.params.lamH2
        self.N = self.params.N
        self.NE = self.params.NE
        self.NH = self.params.NH
        self.pbi = self.params.pbi
        self.rE = self.params.rE
        self.rH = self.params.rH
        self.dx = self.grid.dx
        self.dxE = self.grid.dxE
        self.dxH = self.grid.dxH
        self.x = self.grid.x
        self.Av = self.mat.Av
        self.AvE = self.mat.AvE
        self.AvH = self.mat.AvH
        self.Dx = self.mat.Dx
        self.DxE = self.mat.DxE
        self.DxH = self.mat.DxH
        self.Lo = self.mat.Lo
        self.LoE = self.mat.LoE
        self.LoH = self.mat.LoH
        self.ddE = self.mat.ddE
        self.ddH = self.mat.ddH
        self.NN = self.mat.NN
        self.Rr = self.genrec.Rr
        self.Rl = self.genrec.Rl
        self.G = self.genrec.G
        self.R = self.genrec.R
        self.phi_precondition = self.params.phi_precondition
        self.psi_scan_rate = self.params.psi_scan_rate
        self.phi_f = self.params.phi_f


    def __call__(self, t, u, *psi, mode=None):
        
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
                
    #### Allocate the residual vector. ###################
        rhs = np.zeros(4*self.N+2*self.NE+2*self.NH+4)
        
        P = u[0:self.N+1]
        phi = u[self.N+1:2*self.N+2]
        n = u[2*self.N+2:3*self.N+3]
        p = u[3*self.N+3:4*self.N+4]
        phiE = u[4*self.N+4:4*self.N+self.NE+4]; phiE = np.append(phiE, phi[0])
        nE = u[4*self.N+self.NE+4:4*self.N+2*self.NE+4]; nE = np.append(nE, n[0]/self.kE)
        phiH = u[4*self.N+2*self.NE+4:4*self.N+2*self.NE+self.NH+4]; phiH = np.append(phi[-1], phiH)
        pH = u[4*self.N+2*self.NE+self.NH+4:4*self.N+2*self.NE+2*self.NH+4]; pH = np.append(p[-1]/self.kH, pH)
        mE = np.matmul(self.Dx, phi)
        mEE = np.matmul(self.DxE, phiE)
        mEH = np.matmul(self.DxH, phiH)
        FP = self.lam * (np.matmul(self.Dx, P) + mE * (np.matmul(self.Av, P)))
        cd = self.NN - np.matmul(self.Lo, P) + self.delta * (np.matmul(self.Lo, n) - self.chi* np.matmul(self.Lo, p))
        cdE = np.matmul(self.LoE, nE) - self.ddE
        cdH = self.ddH - np.matmul(self.LoH, pH)
        fn = self.Kn*(np.matmul(self.Dx, n) - mE*(np.matmul(self.Av, n)))
        fnE = self.KE*(np.matmul(self.DxE, nE) - mEE*(np.matmul(self.AvE, nE)))
        fp = self.Kp*(np.matmul(self.Dx, p) + mE*(np.matmul(self.Av, p)))
        fpH = self.KH*(np.matmul(self.DxH, pH) + mEH*(np.matmul(self.AvH, pH)))
        GR = self.G(np.matmul(self.Av, self.x)) - self.R(np.matmul(self.Av, n), np.matmul(self.Av, p), np.matmul(self.Av, P))
        
        ## P equation 
        rhs[0] = FP[0]
        rhs[1:self.N] = FP[1:self.N] - FP[0:self.N-1]
        rhs[self.N] = -FP[self.N-1]
        
        ### phi equation
        rhs[self.N+1] = mE[0] - self.rE*mEE[-1] - self.dx[0]*(1/2 - P[0]/3 - P[1]/6 + self.delta*(n[0]/3+n[1]/6 - self.chi*(p[0]/3 + p[1]/6)))/self.lam2 - self.rE*self.dxE[-1]*(nE[-2]/6+ nE[-1]/3 - 1/2)/self.lamE2
        rhs[self.N+2:2*self.N+1] = mE[1:self.N] - mE[0:self.N-1] - cd/self.lam2
        rhs[2*self.N+1] = self.rH*mEH[0] - self.dx[-1]*(1/2 - P[-2]/6 - P[-1]/3 + self.delta*(n[-2]/6 + n[-1]/3 - self.chi*(p[-2]/6 + p[-1]/3)))/self.lam2 - self.rH*self.dxH[0]*(1/2 -pH[0]/3 - pH[1]/6)/self.lamH2 
        
        #######   n equation
        rhs[2*self.N+2] = fn[0] - fnE[-1] - self.Rl(nE[-1],p[0]) + (self.dx[0]*GR[0])/2
        rhs[2*self.N+3:3*self.N+2] = fn[1:self.N] - fn[0:self.N-1] + (self.dx[1:self.N]*GR[1:self.N] + self.dx[0:self.N-1]*GR[0:self.N-1])/2 
        rhs[3*self.N+2] = -fn[self.N-1] - self.Rr(n[self.N], pH[0]) + self.dx[self.N-1]*GR[-1]/2
        
        ##p equation
        rhs[3*self.N+3] = fp[0] - self.Rl(nE[-1],p[0]) + self.dx[0]*GR[0]/2
        rhs[3*self.N+4:4*self.N+3] = fp[1:self.N] - fp[0:self.N-1] + (self.dx[1:self.N]*GR[1:self.N] + self.dx[0:self.N-1]*GR[0:self.N-1])/2
        rhs[4*self.N+3] = fpH[0] - fp[-1] - self.Rr(n[self.N],pH[0]) + (self.dx[-1]*GR[-1])/2
        
        #### phiE equation
        rhs[4*self.N+4] = phiE[0] + 0.5*( psi(t) - self.pbi) #####
        rhs[4*self.N+5:4*self.N+self.NE+4] = mEE[1:self.NE] - mEE[0:self.NE-1] - cdE/self.lamE2
        #### nE equation
        rhs[4*self.N+self.NE+4] = nE[0] - 1
        rhs[4*self.N+self.NE+5:4*self.N+2*self.NE+4] = fnE[1:self.NE] - fnE[0:self.NE-1]
        #### phiH equation
        rhs[4*self.N+2*self.NE+4:4*self.N+2*self.NE+self.NH+3] = mEH[1:self.NH] - mEH[0:self.NH-1] - cdH/self.lamH2
        rhs[4*self.N+2*self.NE+self.NH+3] = phiH[-1] - 0.5*(psi(t) - self.pbi) #####
        ##### pH equation
        rhs[4*self.N+2*self.NE+self.NH+4:4*self.N+2*self.NE+2*self.NH+3] = fpH[1:self.NH] - fpH[0:self.NH-1]
        rhs[4*self.N+2*self.NE+2*self.NH+3] = pH[-1] - 1
        
        if mode == 'precondition':
            rhs[self.N] = np.trapz(P, self.x) - 1
            
        return rhs

                                   
                                           
    
    
































