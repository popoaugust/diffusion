# -*- coding: utf-8 -*-
"""
Created on Sat May 13 00:02:05 2023

@author: waran
"""

"""
This script defines the transfer matrix function. It also uses the Pandas
library to import complex refractive index data used in the calculation
.

The function outputs are:
Jsc: short-circuit current
gen: generation rate profile

This code was adapted from the Matlab code by George F. Burkhard, Eric T.
Hoke, Stanford University. 2010

Note: This was my first attempt at trying to port code from Matlab, so it
may seem disorganized. However, about 3/4 of the way through finishing
it I found that it had already been done!

The reference is by Kamil Mielczarek, University of Texas at Dallas.

Citations:
"""

import numpy as np
import pandas as pd

from . import parameters 
from . import grid

params = parameters.Params()
grid = grid.Grid(params)

def transfer_matrix():
    thickness_ito = 80
    thickness_ETL = params.bE*1e9
    thickness_perovskite = params.b*1e9
    thickness_HTL = params.bH*1e9

    # Wavelength range inputs.
    lower_bound = 350
    upper_bound = 800
    wavelength = np.array(range(lower_bound,upper_bound + 1))
    # Electric field is calculated in discrete positions.
    step_size = 1
    layers = ['ITO', 'TiO2', 'perovskite', 'Spiro-OMeTAD' ]
    thicknesses = [thickness_ito, thickness_ETL, thickness_perovskite, thickness_HTL]
    activelayer = 3
    
    h = 6.62606957e-34
    c = 2.99792458e8
    
    # -----------------------Data Input
    
    # Use pandas to obtain data from Excel file.
    datafile = pd.read_excel("Complex_Refractive_Index_Data.xlsx")
    
    # Add data designation here from the Excel file. The column names must be input to the bracketed portion of datafile.
    # The program will fail if column names are mispelled.
    data_wl_air = datafile["Air_wavelength"].values
    data_n_air = datafile["Air_n"].values
    data_k_air = datafile["Air_k"].values
   
    data_wl_ITO = datafile['ITO_wavelength']
    data_n_ITO = datafile["ITO_n"]
    data_k_ITO = datafile["ITO_k"]
   
    data_wl_TiO2 = datafile["TiO2_wavelength"]
    data_n_TiO2 = datafile["TiO2_n"]
    data_k_TiO2 = datafile["TiO2_k"]
   
    data_wl_MAPbI3 = datafile["MAPbI3_wavelength"].values
    data_n_MAPbI3 = datafile["MAPbI3_n"].values
    data_k_MAPbI3 = datafile["MAPbI3_k"].values
  
    data_wl_spiro = datafile["Spiro_wavelength"]
    data_n_spiro = datafile["Spiro_n"]
    data_k_spiro = datafile["Spiro_k"]
    
    #------------------Data Processing
    # Function to interpolate n and k data. Adds both values to produce total complex index of reflection.
    # Uses raw data input from Excel file.
    def interp_nk(wavelength_input, wavelength_data, n_data, k_data):
        wl_int = wavelength_data
        n_int = n_data
        k_int = k_data
        n = np.interp(wavelength_input, wl_int, n_int)
        k = np.interp(wavelength_input, wl_int, k_int)
        
        return n + 1j*k

    # Complex refractive index for each material at the discrete values given in the wavelength range.
    n_air = interp_nk(wavelength, data_wl_air, data_n_air, data_k_air)
    n_ITO = interp_nk(wavelength, data_wl_ITO, data_n_ITO, data_k_ITO)
    n_TiO2 = interp_nk(wavelength, data_wl_TiO2, data_n_TiO2, data_k_TiO2)
    n_MAPbI3 = interp_nk(wavelength, data_wl_MAPbI3, data_n_MAPbI3, data_k_MAPbI3)
    n_spiro = interp_nk(wavelength, data_wl_spiro, data_n_spiro, data_k_spiro)

    n_matrix_dimensions = (len(layers), len(wavelength))
    n = np.zeros(n_matrix_dimensions, dtype=complex)
    
    # Input data into empty n matrix.
    for index in range(0, len(n_ITO)):
        n[0,index] = n_ITO[index]
    for index in range(0, len(n_TiO2)):
        n[1,index] = n_TiO2[index]
    for index in range(0, len(n_MAPbI3)):
        n[2,index] = n_MAPbI3[index]
    for index in range(0, len(n_spiro)):
        n[3,index] = n_spiro[index]
        
    """ Transfer Matrix Optical Model """
    """ Calculate incoherent power transmission through ITO substrate. """
    T_ITO = np.abs(np.divide(4*n[0,:],(np.power((1 + n[0,:]),2))))
    R_ITO = np.power(np.abs(np.divide(1 - n[0,:],1 + n[0,:])),2)

    t = thicknesses
    t[0] = 0
    t_cumsum = np.cumsum(t)

    x_pos = np.arange(step_size/2, sum(t), step_size)
    #x_mat uses an inequality to produce boolean values. The remaining operations create a matrix that describes what layer number the corresponding point in x_pos is.
    x_mat = sum(
        np.tile(x_pos,[len(t),1]) > np.transpose(np.tile(np.transpose(t_cumsum), [len(x_pos), 1])), 1
        )
    R = wavelength * 0.0
    T = np.ones(len(wavelength)) * 0.0
    E_mat_dimensions = (len(x_pos), len(wavelength))
    E = np.zeros(E_mat_dimensions, dtype=complex)


    """ Define functions for I and L matrices. """
    def I(n1,n2):
        r = (n1 - n2)/(n1 + n2)
        t = 2*n1/(n1 + n2)
        return [[1,r],[r,1]]/t
    
    def L(n,d,wavelength):
        xi = 2*np.pi*n/wavelength
        return [[np.exp(-1j*xi*d),0],[0,np.exp(1j*xi*d)]]


    #-----------Construction of transfer matrices
    print('Calculating Transfer Matrices')
    for l in range(0,len(wavelength)):
        S = I(n[0,l],n[1,l])
        for matindex in range(2,len(t)):
            S = np.asarray(np.mat(S)*np.mat(L(n[matindex-1,l],t[matindex-1],wavelength[l]))*np.mat(I(n[matindex-1,l],n[matindex,l])))
        R[l] = np.abs(S[1,0]/S[0,0])**2
        T[l] = np.abs(2/(1+n[0,l]))/np.sqrt(1-R_ITO[l]*R[l])
        
        for material in range(1,len(t)):
            xi = 2*np.pi*n[material,l]/wavelength[l]
            dj = t[material]
            x_indices = np.argwhere(x_mat == material+1)
            x = x_pos[x_indices] - t_cumsum[material-1]
            S_prime = I(n[0,l],n[1,l])
            
            for matindex in range(2,material+1):
                S_prime = np.asarray(np.mat(S_prime)*np.mat(L(n[matindex-1,l],t[matindex-1],wavelength[l]))*np.mat((I(n[matindex-1,l],n[matindex,l]))))
            S_double_prime = np.eye(2)
            
            for matindex in range(material,len(t)-1):
                S_double_prime = np.asarray(np.mat(S_double_prime)*np.mat(I(n[matindex,l],n[matindex+1,l]))*np.mat(L(n[matindex+1,l],t[matindex+1],wavelength[l])))
            
            E[x_indices,l] = T[l]*(np.divide((S_double_prime[0,0]*np.exp(
                complex(0,-1.0)*xi*(dj - x)) + S_double_prime[1,0]*np.exp(
                complex(0,1.0)*xi*(dj - x))), S_prime[0,0]*S_double_prime
                [0,0]*np.exp(complex(0,-1.0)*xi*dj) + S_prime[0,1]*
                S_double_prime[1,0]*np.exp(complex(0,1.0)*xi*dj)))
   
    #----------Generation Equations
    def interp_1sun(wavelength_input, wavelength_data, sun_data):
        wl_int = wavelength_data
        sun_int = sun_data
        solar_spectrum = np.interp(wavelength_input, wl_int, sun_int)
        return solar_spectrum

    sun_datafile = pd.read_excel("Sun_Spectrum.xlsx")
    data_wl_sun = sun_datafile["Wavelength (nm)"].values
    data_sun_spectrum = sun_datafile["Global tilt  mW*cm-2*nm-1 (1sun AM 1.5)"]
    
    sun_mat = interp_1sun(wavelength, data_wl_sun, data_sun_spectrum)

    a_mat_dimensions = (len(t),len(wavelength))
    a = np.zeros(a_mat_dimensions)

    for i in range(1,len(t)):
        a[i,:] = (4*np.pi*np.imag(n[i,:]))/(wavelength*1.0e-7)
   
    activepos = np.argwhere(x_mat == activelayer)

    Q = np.tile(a[activelayer-1,:]*np.real(n[activelayer-1,:])*sun_mat,(np.
        size(activepos),1))*(abs(E[activepos,:])**2)

    G = (Q*1.0e-3)*np.tile(wavelength*1.0e-9,(np.size(activepos),1))/(h*c)

    if len(wavelength) == 1:
        wl_step = 1

    else:
        wl_step = (sorted(wavelength)[-1] - sorted(wavelength)[0])/(len(wavelength)-1)
    
    Gx = np.sum(G,2)*wl_step
    
    x_non_dim = np.append(
        np.append([0.0], (x_pos[activepos] - thickness_ETL)*1e-9/params.b), 
        [1.0]
        )
    
    G_total = np.append(np.append(Gx[0,0], Gx[:,0]), Gx[-1,0])*1e6 # m3s-1
    gen = np.interp(grid.x, x_non_dim, G_total)

    Jsc = sum(Gx)*step_size*1e-7*params.q*1e3; Jsc = Jsc[0]
    
    return gen, Jsc
    
    
