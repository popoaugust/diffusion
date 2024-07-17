#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:38:53 2023

@author: watcharanon
"""
import numpy as np 

class Grid():
        
    def __init__(self, params):
        
        # Define class variables from constructor class objects.#
        N = params.N
        NE = params.NE
        NH = params.NH
        st = params.st
        wE = params.wE
        wH = params.wH
        
        # Calculate spatial mesh for single-layer and three-layer models.#
        x = np.linspace(0, 1, N+1)
        self.x = (np.tanh(st*(2*x-1))/np.tanh(st)+1)/2
        xE = np.linspace(-wE, 0, NE+1)
        self.xE = wE*(np.tanh(st*(2*xE/wE+1))/np.tanh(st)+1)/2
        xH = np.linspace(1, 1+wH, NH+1)
        self.xH = 1 + wH*(np.tanh(st*(2*(xH-1)/wH-1))/np.tanh(st)+1)/2
        
        # Differencing vectors. #
        self.dx = np.diff(self.x)
        self.dxE = np.diff(self.xE)
        self.dxH = np.diff(self.xH)

        # The grid points at the interfaces are already included in the perovskite layer grid.#
        self.xE = self.xE[0:-1]
        self.xH = self.xH[1:]
        

        

        
    
