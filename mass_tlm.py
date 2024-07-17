#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:32:57 2023

@author: watcharanon
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix

class Mass():
    def __init__(self, params, grid, mode=None):
        
        chi = params.chi
        dx = grid.dx
        dxE = grid.dxE
        dxH = grid.dxH
        kE = params.kE
        kH = params.kH
        N = params.N
        NE = params.NE
        NH = params.NH
        sigma = params.sigma

        ###################  Arrays used to define the sub-diagonals [-1], main-diagonals#######
        ######  [0], and upper-diagonals [1] used in the instance variables. ##################
        diagonal_M11 = np.array([np.append(dx[0:-1]/6, dx[-1]/6), np.append( dx[0]/3, np.append((dx[0:-1]+dx[1:])/3, dx[-1]/3)), np.append(dx [0]/6, dx[1:]/6)], dtype=object)
        diagonal_M33 = np.array([np.append(dx[0:-1]/6, dx[-1]/6), np.append( dx[0]/3, np.append((dx[0:-1]+dx[1:])/3, dx[-1]/3)), np.append(dx [0]/6, dx[1:]/6)], dtype=object)
        diagonal_M44 = np.array([np.append(dx[0:-1]/6, dx[-1]/6), np.append( dx[0]/3, np.append((dx[0:-1]+dx[1:])/3, dx[-1]/3)), np.append(dx [0]/6, dx[1:]/6)], dtype=object)
        diagonal_M66 = np.array([dxE[0:-1]/6, np.append([0.0], (dxE[0:-1] + dxE[1:])/3), np.append([0.0], dxE[1:-1]/6)], dtype=object)
        diagonal_M88 = np.array([np.append(dxH[1:-1]/6, [0.0]), np.append(( dxH[0:-1] + dxH[1:])/3, [0.0]), dxH[1:]/6], dtype=object)   

        ##################  Piece by piece construction and horizontal concatenation of zero
        #########   and non-zero arrays.
        
        M11 = diags(diagonal_M11, [-1,0,1]).toarray()
        M12 = np.zeros((N+1, N+1))
        M15 = np.zeros((N+1, NE))
        M17 = np.zeros((N+1, NH))
        M33 = sigma*diags(diagonal_M33, [-1,0,1]).toarray(); M33[0,0] = sigma*(dxE[-1]/kE + dx[0])/3
        M36 = np.zeros((N+1, NE)); M36[0,-1] = sigma*dxE[-1]/6
        M44 = sigma*chi*diags(diagonal_M44, [-1,0,1]).toarray(); 
        M44[-1,-1]= sigma*chi*(dx[-1] + dxH[0]/kH)/3
        M48 = np.zeros((N+1, NH)); M48[-1,0] = sigma*chi*dxH[0]/6
        M51 = np.zeros((NE, N+1))
        M55 = np.zeros((NE, NE))
        M57 = np.zeros((NE, NH))
        M63 = np.zeros((NE, N+1)); M63[-1,0] = sigma*(dxE[-1]/kE)/6
        M66 = sigma*diags(diagonal_M66, [-1,0,1]).toarray()
        M71 = np.zeros((NH, N+1))
        M75 = np.zeros((NH, NE))
        M77 = np.zeros((NH, NH))
        M84 = np.zeros((NH, N+1)); M84[0,-1] = sigma*chi*(dxH[0]/kH)/6
        M88 = sigma*chi*diags(diagonal_M88, [-1,0,1]).toarray()
        
        ##################  Define the sparsity structure of each row. a-b-c-d-e-f-g-h are
        #############   the blocks corresponding to each variable in the mass matrix.
        
        
        ### row 1 ######
        a = M11; bcd = np.tile(M12, (3)); ef = np.tile(M15, (2)); gh = np.tile(M17, (2))
        row1 = np.concatenate((a, bcd, ef, gh), 1)
        
        #### row 2 #####
        abcd = np.tile(M12, (4)); ef = np.tile(M15, (2)); gh = np.tile(M17,(2))
        row2 = np.concatenate((abcd, ef, gh), 1)
        
        ###### row 3 ######
        
        ab = np.tile(M12, (2)); c = M33; d = M12; e = M15; f = M36; gh = np.tile(M17, (2))
        row3 = np.concatenate((ab, c, d, e, f, gh), 1)
        
        ###### row 4 ######
        abc = np.tile(M12, (3)); d = M44; ef = np.tile(M15, (2)); g = M17; h= M48
        row4 = np.concatenate((abc, d, ef, g, h), 1)
        
        ####### row 5#######
        
        abcd = np.tile(M51, (4)); ef = np.tile(M55, (2)); gh = np.tile(M57,(2))
        row5 = np.concatenate((abcd, ef, gh), 1)
        
        ######### row 6 ########
        
        ab = np.tile(M51, (2)); c = M63; d = M51; e = M55; f = M66; gh = np.tile(M57, (2))
        row6 = np.concatenate((ab, c, d, e, f, gh), 1)
        
        ######### row 7 ##############
        abcd = np.tile(M71, (4)); ef = np.tile(M75, (2)); gh = np.tile(M77,(2))
        row7 = np.concatenate((abcd, ef, gh), 1)
        
        
        
        ########### row 8 ###########
        abc = np.tile(M71, (3)); d = M84; ef = np.tile(M75, (2)); g = M77; h= M88
        row8 = np.concatenate((abc, d, ef, g, h), 1)
        
        ########## vertical concatenate all rows. #############
        M = np.concatenate((row1, row2, row3, row4, row5, row6, row7, row8),0)
        
        ######## special condition ##########
        if mode == 'precondition': 
            M[0:N+1,:] = sigma*M[0:N+1,:]
        
        ######### define instance variable.###########
        self.M = csr_matrix(M)
        
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


                         
         
        
    