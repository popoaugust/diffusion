#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 08:56:12 2023

@author: watcharanon
"""
import numpy as np
import math
from scipy.optimize import fsolve
import json

class Params():
    
    @classmethod
    def from_json(cls, json_path):
        with open(json_path, "r") as json_file:
            kwargs = json.load(json_file)
        
        return Params(**kwargs)
    
    def __init__(self, 
                 model      = 'beer-lambert',
                 slm        = True,
                 test_case_slm = True,
                 N          = 400,
                 rtol       = 1e-6,
                 atol       = 1e-10,
                 Vi         = 1.2,
                 Vf         = 0,
                 scan_rate  = 0.5,
                 T          = 298.0,
                 Fph        = 1.4e21,
                 b          = 600e-9,
                 epsp_coef  = 24.1,
                 alpha      = 1.3e7,
                 Ec         = -3.7,
                 Ev         = -5.4,
                 Dn         = 1.7e-4,
                 Dp         = 1.7e-4,
                 gc         = 8.1e24,
                 gv         = 5.8e24,
                 # ion parameter inputs
                 N0         = 1.6e25,
                 DIinf      = 6.5e-8,
                 EAI        = 0.58,
                 DI         = 2e-7,
                 # Three-Layer Model Parameter Inputs
                 dE         = 1e24,
                 gcE        = 5e25,
                 EcE        = -4.0,
                 bE         = 100e-9,
                 epsE_coef  = 10,
                 DE         = 1e-5,
                 # HTL Parameter Inputs
                 dH          = 1e24,
                 gvH         = 5e25,
                 EvH         = -5.1,
                 bH          = 200e-9,
                 epsH_coef   = 3,
                 DH          = 1e-6,
                 # Beer-Lambert Law Generation
                 Y           = 3.7,
                 w           = 2.4,
                 # Bulk recombination parameters
                 tn          = 3e-9,
                 tp          = 3e-7,
                 beta        = 0.0,
                 # Interface recombination parameters
                 betaE       = 0.0,
                 betaH       = 0.0,
                 vnE         = 1e5,
                 vpE         = 10,
                 vnH         = 0.1,
                 vpH         = 1e5,
                 ):
        
        self.model       = model
        self.N           = N # 400
        self.rtol        = rtol # 1e-6
        self.atol        = atol # 1e-10
        self.Vi          = Vi # 1.2
        self.Vf          = Vf # 0
        self.scan_rate   = scan_rate # 0.3
        self.q           = 1.60217646e-19
        self.kB          = 8.61733035e-5
        self.T           = T # 298.0
        self.eps0        = 8.85418782e-12
        self.Fph         = Fph # 1.4e21
        self.Vt          = self.kB * self.T
        self.b           = b # 600e-9
        self.epsp        = epsp_coef * self.eps0 # 24.1 * self.eps0
        self.alpha       = alpha # 1.3e7
        self.Ec          = Ec # -3.7
        self.Ev          = Ev # -5.4
        self.Dn          = Dn # 1.7e-4
        self.Dp          = Dp # 1.7e-4
        self.gc          = gc # 8.1e24
        self.gv          = gv # 5.8e24
        
        ######### ion parameter inputs#####################
        self.N0          = N0 # 1.6e25
        self.D = lambda Dinf, EA: Dinf * np.exp(-EA / (self.kB * self.T)) 
        self.DIinf       = DIinf # 6.5e-8
        self.EAI         = EAI # 0.58
        if DI is None: 
            self.DI      = self.D(self.DIinf, self.EAI)
        else: 
            self.DI      = DI # 0   ##self.D(self.DIinf, self.EAI)
        
        ######### Three-Layer Model Parameter Inputs#############
        self.dE          = dE # 1e24
        self.gcE         = gcE # 5e25
        self.EcE         = EcE # -4.0
        self.bE          = bE # 100e-9
        self.epsE        = epsE_coef * self.eps0 # 10 * self.eps0
        self.DE          = DE # 1e-5
        
        ############## HTL Parameter Inputs  #####################
        self.dH          = dH # 1e24
        self.gvH         = gvH # 5e25
        self.EvH         = EvH # -5.1
        self.bH          = bH # 200e-9
        self.epsH        = epsH_coef * self.eps0 # 3 * self.eps0
        self.DH          = DH # 1e-6
        
        #############  Beer-Lambert Law Generation #################
        self.Y           = Y # 3.7
        self.w           = w # 2.4
        
        ################# Bulk recombination parameters  ##############
        self.tn          = tn # 3e-9
        self.tp          = tp # 3e-7
        self.beta        = beta # 0.0
        
        ################## Interface recombination parameters
        self.betaE       = betaE # 0.0
        self.betaH       = betaH # 0.0
        self.vnE         = vnE # 1e5
        self.vpE         = vpE # 10
        self.vnH         = vnH # 0.1
        self.vpH         = vpH # 1e5
        
        ############## perovskite calculated parameter and non-dimensionalization#############
        self.Eg          = self.Ec - self.Ev
        self.LD          = np.sqrt(self.Vt * self.epsp / (self.q * self.N0))
        self.lam         = self.LD / self.b
        self.lam2        = self.lam**2
        self.ni          = np.sqrt(self.gc * self.gv) * np.exp(-self.Eg / (2*self.Vt)) 
        self.delta       = self.dE / self.N0
        self.chi         = self.dH / self.dE
        self.G0 = np.array((self.Fph / self.b)) * (1 - np.exp(-self.alpha * self.b))
        
        if self.DI != 0:
            self.Tion = self.b/self.DI*np.sqrt(self.Vt*self.epsp/(self.q*self.N0))
        else:
            self.Tion = self.dE/self.G0
            
        self.sigma = self.dE/(self.G0*self.Tion)
        self.Kp = self.Dp*self.dH/(self.G0*self.b**2)
        self.Kn = self.Dn*self.dE/(self.G0*self.b**2)
        self.Upsilon = self.alpha*self.b
            
        ##################### energy level parameter######################
        self.EfE = self.EcE - self.Vt*np.log(self.gcE/self.dE)
        self.EfH = self.EvH + self.Vt*np.log(self.gvH/self.dH)
        self.Vbi = self.EfE - self.EfH
        self.pbi = self.Vbi/self.Vt
        
        ################### interface parameters ############################
        self.kE = self.gc/self.gcE*np.exp((self.EcE-self.Ec)/self.Vt)
        self.kH = self.gv/self.gvH*np.exp((self.Ev-self.EvH)/self.Vt)
        self.n0 = self.kE*self.dE
        self.p0 = self.kH*self.dH
        
        ########  Single-Layer Model Conditions  #############
        
        if slm:
            self.delta = self.Fph*self.b/(self.DE*self.N0)
            self.sigma = self.DI*self.b/(self.DE*self.LD)
            self.k_p = self.Dp/self.DE
            self.k_n = self.Dn/self.DE
            
        ##### Dirichlet Boundary Condition s#######
            
        self.n_bar = self.n0*self.DE/(self.Fph*self.b) 
        self.p_bar = self.p0*self.DE/(self.Fph*self.b)
        
        #### Scan-rate scaling#####
        
        self.ion_mobility_factor= 2.4025533333333337e-16
        
        #####  Non-Dimensional Scan Parameters #############
        self.psi_scan_rate = self.scan_rate*(self.LD*self.b/self.ion_mobility_factor)/self.Vt
        self.phi_i = self.Vi/self.Vt
        self.phi_f = self.Vf/self.Vt
        self.phi_precondition = self.phi_i
        self.tf_scan = np.abs((self.phi_i - self.phi_f)/self.psi_scan_rate)
        
        ####### test Case Conditions ###########
        if test_case_slm: 
            self.pbi = 40
            self.DI = 2.4025533333333337e-16
            self.delta = 2.1*10e-7
            self.sigma = 5.8*10e-10
            self.n_bar = 20
            self.p_bar = 0.3
            self.lam = 2.4 * 10e-4
            self.lam2 = self.lam**2
            self.phi_precondition = 46.729618965853994
            self.phi_f = 0.0
            self.psi_scan_rate = 14.2
            self.tf_scan = np.abs((self.phi_precondition - self.phi_f)/self.psi_scan_rate)
            
        ################## ETL AND HTL #########################
        self.wE = self.bE/self.b
        self.wH = self.bH/self.b
        self.KE = self.DE*self.Kn/self.Dn
        self.KH = self.DH*self.Kp/self.Dp
        self.rE = self.epsE/self.epsp
        self.rH = self.epsH/self.epsp
        self.lamE2 = self.rE*self.N0/self.dE*self.lam2
        self.lamE = np.sqrt(self.lamE2)
        self.lamH2 = self.rH*self.N0/self.dH*self.lam2
        self.lamH = np.sqrt(self.lamH2)
        self.OmegaE = np.sqrt(self.N0/(self.rE*self.dE))
        self.OmegaH = np.sqrt(self.N0/(self.rH*self.dH))
        
        ############# BULK RECOMBINATION ###################
        self.ni2 = self.ni**2/(self.dE*self.dH)
        self.brate = self.beta*self.dE*self.dH/self.G0
        if self.tp>0 and self.tn>0:
            self.gamma = self.dH/(self.tp*self.G0)
            self.tor = self.tn*self.dH/(self.tp*self.dE)
            self.tor3 = (self.tn+self.tp)*self.ni/(self.tp*self.dE)
        else:
            [self.gamma, self.tor, self.tor3] = [0,0,0]
            
        ################### Interface Recombination Parameters ###################
        self.brateE = self.betaE*self.dE*self.dH/(self.b*self.G0)
        self.brateH = self.betaH*self.dE*self.dH/(self.b*self.G0) 
        if self.vpE>0 and self.vnE>0:
            self.gammaE = self.dH*self.vpE/(self.b*self.G0)
            self.torE = self.dH*self.vpE/(self.dE*self.vnE)
            self.torE3 = (1/self.kE+self.vpE/self.vnE)*self.ni/self.dE
        else:
            [self.gammaE, self.torE, self.torE3] = [0,0,0]
            
        if self.vnH>0 and self.vpH>0:
            self.gammaH = self.dE*self.vnH/(self.b*self.G0)
            self.torH = self.dE*self.vnH/(self.dH*self.vpH)
            self.torH3 = (1/self.kH+self.vnH/self.vpH)*self.ni/self.dH
        else:
            [self.gammaH, self.torH, self.torH3] = [0,0,0]
        
        ################### spatial Discretization Parameters  ###################    
        self.X = 0.2
        self.tanhfun = lambda x, st: (math.tanh(st*(2*x-1))/math.tanh(st)+1) /2
        
        def func(st):
            return self.lam - self.tanhfun(self.X, st)
        
        self.st = float(fsolve(func, 2))
        self.A = lambda b: (math.tanh(self.st*(1-2/self.N))-(1-np.double(b))*math.tanh(self.st))/np.double(b)
        self.NE = int(np.round(2/(1-math.atanh(self.A(self.wE))/self.st)))
        self.NH = int(np.round(2/(1-math.atanh(self.A(self.wH))/self.st)))
        

            
            
            
            
            
            


        
        
        

        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        

        
        
        
        
        

