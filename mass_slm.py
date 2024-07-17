#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:21:18 2023

@author: watcharanon
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix

class Mass():
    
    def __init__(self, params, grid):
        N = params.N
        sigma = params.sigma
        dx = grid.dx
        ############## Arrays used to define the sub-diagonals [-1], main-diagonals #######
        ##########  [0], and upper-diagonals [1] used in the instance variables. #########
        
        diagonal_M11 = np.array([np.append(dx[0:-1]/6, dx[-1]/6), np.append( dx[0]/3, np.append((dx[0:-1]+dx[1:])/3, dx[-1]/3)), np.append(dx [0]/6, dx[1:]/6)], dtype=object)
        diagonal_M33 = sigma*np.array([np.append(dx[0:-1]/6, dx[-1]/6), np. append(dx[0]/3, np.append((dx[0:-1]+dx[1:])/3, dx[-1]/3)), np. append(dx[0]/6, dx[1:]/6)], dtype=object)
        diagonal_M44 = diagonal_M33
        
        
        ############  Piece by piece construction and horizontal concatenation of zero ########
        ###########   and non-zero arrays.#    #########
        
        # P equation
        M11 = diags(diagonal_M11, [-1,0,1]).toarray()
        M1 = np.concatenate((M11, np.zeros((N+1, 3*N+3))), 1)
        
        M2 = np.zeros((N+1,4*N+4))
        
        # n equation
        M33 = diags(diagonal_M33, [-1,0,1]).toarray(); M33[0,0] = 0; M33[0,1] = 0
        M312 = np.zeros((N+1,2*N+2))
        M34 = np.zeros((N+1,N+1))
        M3 = np.concatenate((M312, M33, M34), 1)
        
        # p equation
        M44 = diags(diagonal_M44, [-1,0,1]).toarray(); M44[-1,-1] = 0; M44[-1,-2] = 0
        M4123 = np.zeros((N+1,3*N+3))
        M4 = np.concatenate((M4123, M44), 1)
        
        ######## Define instance variables #######################
        M = np.concatenate((M1, M2, M3, M4)) 
        self.M = csr_matrix(M)
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        

    