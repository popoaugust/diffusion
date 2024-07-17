#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:05:34 2023

@author: watcharanon
"""


import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix


class Jac():
    def __init__(self, params, mode=None):
        
       ##### Define class variable from constructor class objects. #######
        N = params.N
        NE = params.NE
        NH = params.NH
        
        ##############  Piece by piece construction and horizontal concatenation of zero  #######
        ####    and non-zero arrays. ######
        J11 = diags((np.ones(N), np.ones(N+1), np.ones(N)), [-1,0,1]).toarray()
        J13 = np.zeros((N+1, N+1))
        J15 = np.zeros((N+1, NE))
        J17 = np.zeros((N+1, NH))
        J25 = np.zeros((N+1, NE)); J25[0, NE-1] = 1
        J27 = np.zeros((N+1, NH)); J27[N, 0] = 1
        J51 = np.zeros((NE, N+1))
        J52 = np.zeros((NE, N+1)); J52[NE-1, 0] = 1
        J55 = diags((np.ones(NE-1), np.ones(NE), np.ones(NE-1)), [-1,0,1]).toarray(); J55[0,1] = 0
        J56 = diags((np.ones(NE-1), np.ones(NE), np.ones(NE-1)), [-1,0,1]).toarray(); J56[0,1] = 0; J56[0,0] = 0
        J57 = np.zeros((NE, NH))
        J71 = np.zeros((NH, N+1))
        J72 = np.zeros((NH, N+1)); J72[0, N] = 1
        J75 = np.zeros((NH, NE))
        J77 = diags((np.ones(NH-1), np.ones(NH), np.ones(NH-1)), [-1,0,1]).toarray(); J77[NH-1, NH-2] = 0
        J78 = diags((np.ones(NH-1), np.ones(NH), np.ones(NH-1)), [-1,0,1]).toarray(); J78[NH-1, NH-2] = 0; J78[NH-1, NH-1] = 0      
        
        ############   Define the sparsity structure of each row. a-b-c-d-e-f-g-h are the blocks corresponding to each variable in the jacobian matrix ####### 
        ab = np.tile(J11, (2)); cd = np.tile(J13, (2)); ef = np.tile(J15,(2)); gh = np.tile(J17, (2))
        row1 = np.concatenate((ab, cd, ef, gh), 1)
        ####  Row 2 ######
        abcd = np.tile(J11, (4)); ef = np.tile(J25, (2)); gh = np.tile(J27,(2))
        row2 = np.concatenate((abcd, ef, gh), 1)
        ###Row 3####
        abcd = np.tile(J11, (4)); ef = np.tile(J25, (2)); gh = np.tile(J17,(2))
        row3 = np.concatenate((abcd, ef, gh), 1)
        ####" Row 4 #####
        abcd = np.tile(J11, (4)); ef = np.tile(J15, (2)); gh = np.tile(J27,(2))
        row4 = np.concatenate((abcd, ef, gh), 1)
        ### Row 5 #######
        a = J51; bc = np.tile(J52, (2)); d = J51; e = J55; f = J56; gh = np.tile(J57, (2))
        row5 = np.concatenate((a, bc, d, e, f, gh), 1)
        ### Row 6 ######
        a = J51; bc = np.tile(J52, (2)); d = J51; e = J56; f = J55; gh = np.tile(J57, (2))
        row6 = np.concatenate((a, bc, d, e, f, gh), 1)
        ##### Row 7 #######
        a = J71; b = J72; c = J71; d = J72; ef = np.tile(J75, (2)); g = J77;h = J78
        row7 = np.concatenate((a, b, c, d, ef, g, h), 1)
        #### Row 8 ########
        a = J71; b = J72; c = J71; d = J72; ef = np.tile(J75, (2)); g = J78;h = J77  
        row8 = np.concatenate((a, b, c, d, ef, g, h), 1)
        
       ####### Vertically concatenate all rows. #########
        jac = np.concatenate((row1, row2, row3, row4, row5, row6, row7, row8), 0)
        if mode == 'precondition':
            jac[N,:] = 0.0; jac[N,0:N+1] = 1.0
        
        #####Define instance variables. ########
        self.jac = csr_matrix(jac)

                                    
        
        
                     
        
        
        