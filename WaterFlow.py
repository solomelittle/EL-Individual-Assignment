#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:44:34 2021

@author: emmalittle
"""
# %% Import packages and set constants; soil parameters and matrix dimensions/distances
import numpy as np
import pandas as pd
import scipy.integrate as spi
import scipy.sparse as sp
import MyTicToc as mt
import matplotlib.pyplot as plt

# %% Defining variables as functions

def Seff(hw, sPar):  # Effective saturation
    hc = -hw
    Se = (1 + (hc * (hc > 0) * sPar.alpha)**sPar.n)**(-sPar.m)
    return Se

def thetaw(hw, sPar):    # Water content
    Se = Seff(hw, sPar)
    theta_wat = sPar.theta_res + (sPar.theta_sat - sPar.theta_res) * Se
    return theta_wat

def diffWaterCapP(hw, sPar, mDim):    # Differential water capacity = Mass matrix in water flux function
    dh = np.sqrt(np.finfo(float).eps)
    
    if np.iscomplexobj(hw):
        hcmplx = hw.real + 1j*dh
    else:
        hcmplx = hw.real + 1j*dh
        
    thc = thetaw(hcmplx, sPar)
    thr = thetaw(hw,sPar)
    Cw = thc.imag / dh
    Sw = thr / sPar.theta_sat
    Ssw = 1000 * 9.81 * (sPar.Cv + sPar.theta_sat * sPar.beta_wat)
    Cwprime = Cw + Ssw * Sw 
    
    Mass = Cwprime
    
    #Accounting for ponding
    Mass[mDim.nN-1] = 1/mDim.dzIN[mDim.nIN-2] * (hw[mDim.nN-1]>0) \
        + Mass[mDim.nN-1] * (hw[mDim.nN-1]<=0)
    return Mass

def krw(hw, sPar): # Relative permeability
    Se = Seff(hw, sPar)
    relpw = Se**3
    return relpw

def K(hw, sPar, mDim):  # Kskr used in waterflux equation
    nr,nc = hw.shape
    nIN = mDim.nIN
    k = krw(hw,sPar)
    
    k_N  =  sPar.Ksat*k
    k_IN = np.zeros([nIN,nc], dtype=hw.dtype)
    k_IN[0] = k_N[0]
    ii = np.arange(1, nIN - 1)
    k_IN[ii] = np.minimum(k_N[ii - 1], k_N[ii])
    k_IN[nIN - 1] = k_N[nIN - 2]
    
    return k_IN
# %% Roots
def alpharoot(hw, sP):
    h1 = sP.h1
    h2 = sP.h2
    h3 = sP.h3
    h4 = sP.h4
    a=h1*(hw<=h4)+h1*(hw>=h1)+(hw-h4)/(h3-h4)*(hw>h4)*(hw<h3)+(hw>h3)*(hw<h2)+(hw-h1)/(h2-h1)*(hw<h1)*(hw>h2)
  
    return a

def betaroot(t,hw,mDim,sP):
    zN = mDim.zN
    dzN = mDim.dzN
    nN = mDim.nN
    dz = dzN[0]
    zetaL= 7500 # m/m3 empirical
    rhoL = 20.15 # empirical
    
    b = np.ones(np.shape(zN),dtype=hw.dtype)
    Lrv = zetaL*np.exp(rhoL*zN)*dz
    
    ii = np.arange(0, nN)
    b[ii] = Lrv[ii]/np.sum(Lrv)
    
    return b

def s_root (t,hw,sP,mDim, bPar):
    pEv = bPar.potEv(t, bPar)
#     nIN = mDim.nIN
#     nr,nc = hw.shape

    alpha = alpharoot(hw, sP)
    beta = betaroot(t,hw,mDim,sP)
    S=alpha*beta*pEv
#     S = np.zeros([nN,nc],dtype=hw.dtype)
#     S=alpha*beta*potEv
  
    return S

# %% Flux & Divergence Functions

def WaterFlux(t, hw, sPar, mDim, bPar):   # dYdt in the solver. Water flux defined at internodes
    nIN = mDim.nIN
    dzN = mDim.dzN
    nr, nc = hw.shape
    
    Kskr = K(hw, sPar, mDim)
    qw = np.zeros([nIN,nc],dtype=hw.dtype) # Initialize flow vector
    
    qw[nIN - 1] = bPar.TopBd(t,bPar) # because you start from the bottom
    # Internode fluxes
    ii = np.arange(1, nIN-1)
    qw[ii] = -Kskr[ii] * ((hw[ii] - hw[ii-1])/ dzN[ii-1]+1)
    
    if bPar.bottomTypeWat.lower() == 'gravity':
        qw[0] = -Kskr[0]
    else:
        qw[0] = -bPar.kRobBotWat * (hw[0]- bPar.hwBotBnd)
    # plt.plot(qw)    
    return qw

def DivWaterFlux(t, hw, sPar, mDim, bPar): # Divergent water flux defined at nodes
    nN = mDim.nN
    dzIN = mDim.dzIN
    nr,nc = hw.shape
    
   # Calculate water fluxes accross all internodes
    qw = WaterFlux(t, hw, sPar, mDim, bPar)
    S = s_root(t,hw,sPar, mDim, bPar)
    mass = diffWaterCapP(hw, sPar, mDim)
    
    divqw = np.zeros([nN, nc],dtype=hw.dtype)
    rateWF = np.zeros([nN,nc]).astype(hw.dtype)
    # Calculate divergence of flux for all nodes
    ii = np.arange(0, nN)
    divqw[ii] = -(qw[ii + 1] - qw[ii]) \
                   / (dzIN[ii])
    rateWF[ii]=(divqw[ii]-S[ii])/mass[ii]
    
    
    return divqw


# %% Solver (Integration Function)

def IntegrateWF(tRange, hwIni, sPar, mDim, bPar): # need tRange and iniSt
    
    def dYdt(t, hw):
        if len(hw.shape)==1:
            hw = hw.reshape(mDim.nN,1)
        rates = DivWaterFlux(t, hw, sPar, mDim, bPar)
        return rates

    def jacFun(t,y):
       if len(y.shape)==1:
           y = y.reshape (mDim.nN, 1)
       nr,nc = y.shape
       dh = np.sqrt(np.finfo(float).eps)
       ycmplx = np.repeat(y,nr,axis=1).astype(complex)
       c_ex = np.eye(nr)*1j*dh
       ycmplx = ycmplx + c_ex
       dfdy = dYdt(t,ycmplx).imag/dh
       return sp.coo_matrix(dfdy)

    t_span = [tRange[0],tRange[-1]]
    int_result = spi.solve_ivp(dYdt, t_span, hwIni.squeeze(), 
                                t_eval=tRange, 
                                method='BDF', vectorized=True, jac=jacFun, 
                                rtol=1e-8)
    return int_result




