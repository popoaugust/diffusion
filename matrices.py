#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:54:42 2023

@author: watcharanon
"""

from scipy.sparse import diags 
import numpy as np

class Matrices():
    
    def __init__(self, params, grid):
        
        #################  Define class variables from constructor class objects.##############
        N = params.N
        NE = params.NE
        NH = params.NH
        dx = grid.dx
        dxE = grid.dxE
        dxH = grid.dxH
 
        ################   Arrays used to define the sub-diagonals [-1], main-diagonals #######
        ########  [0], and upper-diagonals [1] used in the instance variables. ################
        
        diagonals_Av = np.array([np.zeros(N), np.ones(N+1), np.ones(N)], dtype=object)
        diagonals_AvE = np.array([np.zeros(int(round(NE))), np.ones(int( round(NE))+1), np.ones(int(round(NE)))], dtype=object)
        diagonals_AvH = np.array([np.zeros(int(round(NH))), np.ones(int( round(NH+1))), np.ones(int(round(NH)))], dtype=object)
        diagonals_Lo = np.array([np.append(dx[0:-1]/6, [0.0]), np.append(np. append([0.0], [(dx[0:-1]+dx[1:])/3]), [0.0]), np.append([0.0], dx[1:]/6)], dtype=object)
        diagonals_LoE = np.array([np.append(dxE[0:-1]/6, [0.0]), np.append( np.append([0.0], [(dxE[0:-1]+dxE[1:])/3]), [0.0]), np.append ([0.0], dxE[1:]/6)], dtype=object)
        diagonals_LoH = np.array([np.append(dxH[0:-1]/6, [0.0]), np.append( np.append([0.0], [(dxH[0:-1]+dxH[1:])/3]), [0.0]), np.append ([0.0], dxH[1:]/6)], dtype=object)   

        diagonals_Dx = np.array([0./dx, np.append([-1./dx], [0.0]), 1./dx], dtype=object)
        diagonals_DxE = np.array([0./dxE, np.append([-1./dxE], [0.0]), 1./ dxE], dtype=object)
        diagonals_DxH = np.array([0./dxH, np.append([-1./dxH], [0.0]), 1./ dxH], dtype=object)
        
        
        ########## Averaging matrices.  #########
        self.Av = diags(diagonals_Av, [-1,0,1]).toarray() / 2; self.Av = self.Av[0:N, 0:N+1]
        self.AvE = diags(diagonals_AvE, [-1,0,1]).toarray() / 2; self.AvE = self.AvE[0:int(round(NE)), 0:int(round(NE))+1]
        self.AvH = diags(diagonals_AvH, [-1,0,1]).toarray() / 2; self.AvH = self.AvH[0:int(round(NH)), 0:int(round(NH))+1]
        
        
        self.Lo = diags(diagonals_Lo, [-1,0,1]).toarray(); self.Lo = self.Lo[1:N,0:N+1]
        self.LoE = diags(diagonals_LoE, [-1,0,1]).toarray(); self.LoE = self .LoE[1:int(round(NE)),0:int(round(NE))+1]
        self.LoH = diags(diagonals_LoH, [-1,0,1]).toarray(); self.LoH = self .LoH[1:int(round(NH)),0:int(round(NH))+1]
        
        ########### Differencing matrices.##########
        self.Dx = diags(diagonals_Dx, [-1,0,1]).toarray(); self.Dx = self.Dx[0:N,0:N+1];
        self.DxE = diags(diagonals_DxE, [-1,0,1]).toarray(); self.DxE = self .DxE[0:int(round(NE)),0:int(round(NE))+1]
        self.DxH = diags(diagonals_DxH, [-1,0,1]).toarray(); self.DxH = self .DxH[0:int(round(NH)),0:int(round(NH))+1]
        
        ########### Vectors for constant cation vacancy and doping densities. ################
        self.NN = (dx[1:]+dx[0:-1])/2
        self.ddE = (dxE[1:]+dxE[0:-1])/2
        self.ddH = (dxH[1:]+dxH[0:-1])/2
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
         
    