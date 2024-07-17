#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:00:48 2023

@author: watcharanon
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix

class Jac(): 
    
    def __init__(self, params):
        N = params.N
       #### Piece by piece construction and horizontal concatenation of zero
        ####    and non-zero arrays. #######
       ##### J11 can be used to define all non-zero matrices in the sparsity structure.
        J11 = diags((np.ones(N), np.ones(N+1), np.ones(N)), [-1,0,1]).toarray()
       #### First row, column 3 and 4.
        J13 = np.zeros((N+1, N+1))
       #### First row, column 1 and 2.
        J_top_left = np.tile(J11, (2))
       #### First row, column 3 and 4.
        J_top_right = np.tile(J13, (2))
       #### Horizontally concatenate the top row.
        J_top = np.concatenate((J_top_left, J_top_right), 1)
       #### Bottom three rows.
        J_bottom_3_rows = np.tile(J11, (3,4))
       #### Vertically concatenate the top row with the bottom three rows.
        jac = np.concatenate((J_top, J_bottom_3_rows), 0)
       #####Define instance variables. 
        self.jac = csr_matrix(jac)
        
    