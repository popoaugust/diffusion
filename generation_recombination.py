#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:22:00 2023

@author: watcharanon
"""

import numpy as np

class GR():
    
    def __init__(self, params, grid):
        
        brate = params.brate
        brateE = params.brateE
        brateH = params.brateH
        gamma = params.gamma
        gammaE = params.gammaE
        gammaH = params.gammaH
        kE = params.kE
        kH = params.kH
        ni2 = params.ni2
        tor = params.tor
        torE = params.torE
        torH = params.torH
        tor3 = params.tor3
        torE3 = params.torE3
        torH3 = params.torH3
        Y = params.Y
        w = params.w
        x = grid.x
        
        model = params.model
        
        ###""" Generation Rates"""#####
        ###""" Beer-Lambert Law """######
        self.beer = lambda x: Y*np.exp(-Y*x) 
        if model == 'beer-lambert':
            self.G = self.beer
            self.G_init = self.G
        
      
        # ########""" Transfer Matrix Optical Model """############
        from ..utils.transfer_matrix import transfer_matrix 
        from scipy.optimize import least_squares
        
        if model == 'transfer-matrix':
            self.G_dim, self.Jsc = transfer_matrix()
            self.G = lambda x: np.interp(x, grid.x, self.G_dim/params.G0) 
            Y_func = lambda Y: Y*np.exp(-Y*grid.x) - self.G_dim/params.G0
            self.Y_init = least_squares(Y_func, 5).x
            self.G_init = lambda x: self.Y_init*np.exp(-self.Y_init*x)
            
        ###""" Recombination Rates"""######
        ####""" Mono-molecular Recombination """ ########
        self.mm_recomb = lambda p: w*p
        ########""" Bi-molecular Recombination """#####
        self.bm_recomb = lambda n, p: brate*(n*p - ni2)    
        
        #####""" Schockley-Read-Hall (SRH) Recombination """######
        self.SRH = lambda n, p, gamma, ni2, tor, tor3: \
            gamma*(p - ni2/n)/(1 + tor*p/n + tor3/n)*(n>=tor*p)*(n>=tor3) \
                + gamma*(p - ni2/p)/(n/p + tor + tor3/p)*(tor*p>n)*(tor*p>
                  tor3) \
                  + gamma*(p*n - ni2)/(n + tor*p + tor3)*(tor3>n)*(tor3>=tor*p)
                  
        ############ Total Recombination #############
        self.R = lambda n, p, P: self.bm_recomb(n, p) + self.SRH(n, p, gamma , ni2, tor, tor3)
        
        #########  Total Interfacial Recombination
        self.Rl = lambda nE, p: brateE*(nE*p - ni2/kE) + self.SRH(nE, p, gammaE, ni2/kE, torE, torE3)
        self.Rr = lambda n, pH: brateH*(n*pH - ni2/kH) + self.SRH(pH, n, gammaH, ni2/kH, torH, torH3)
        self.GR = lambda n, p, P: self.G(x) - self.R(n, p, P)
        
        
        
        

        

       

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        

        
    