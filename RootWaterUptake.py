#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:17:49 2017
@author: theimovaara
"""
import numpy as np
import scipy.integrate as spi
import scipy.sparse as sp


def Seff(hw, sP): # Effective Saturation
    hc = -hw
    Se = (1 + ((hc * (hc > 0)) * sP.a) ** sP.n) ** (-sP.m)
    return Se


def theta(hw, sP): # Water Content
    Se = Seff(hw, sP)
    th = sP.thetar + (sP.thetas - sP.thetar) * Se
    return th

def CPrimeFun(hw, sP, mDim): # Calculation of Richards' Mass Matrix
    th = theta(hw, sP)  # volumetric water content
    Sw = th / sP.thetas  # water saturation
    # Calculation of differential water capacity (dtheta/dhw)
    dh = np.sqrt(np.finfo(float).eps)
    if np.iscomplexobj(hw):
        hcmplx = hw.real + 1j*dh
    else:
        hcmplx = hw.real + 1j*dh
    th = theta(hcmplx, sP)
    C = th.imag / dh
    
    Ssw = sP.rhoW * sP.g * (sP.Cv + sP.thetas * sP.betaW)

    cPrime = C + Sw * Ssw
    Mass = cPrime
    #  Ponding water condition when hw(-1) >= 0
    # if head at top is larger or equal to 0, ponding water table
    # which is implementd by changing the differential water
    # capacity in order to make the equation: dhw/dt = qbnd - q(n-1/2)
    Mass[mDim.nN-1] = 1/mDim.dzIN[mDim.nIN-2] * (hw[mDim.nN-1]>0) \
        + Mass[mDim.nN-1] * (hw[mDim.nN-1]<=0)

    return Mass

def K(hw, sP, mDim): #Unsaturated hydraulic conductivity
    nr,nc = hw.shape
    nIN = mDim.nIN
    Se = Seff(hw, sP)
    kN = sP.KSat * Se ** 3

    kIN = np.zeros([nIN,nc], dtype=hw.dtype)
    kIN[0] = kN[0]
    ii = np.arange(1, nIN - 1)
    kIN[ii] = np.minimum(kN[ii - 1], kN[ii])
    kIN[nIN - 1] = kN[nIN - 2]
    return kIN

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
    alpha = alpharoot(hw, sP)
    beta = betaroot(t,hw,mDim,sP)
    S=alpha*beta*pEv
  
    return S

def WaterFlux(t, hw, sP, mDim, bPar): #Water Flow (m3/m2d)
    nr,nc = hw.shape
    nIN = mDim.nIN
    dzN = mDim.dzN

    # Calculate inter nodal permeabilities
    kIN = K(hw, sP, mDim)
    qw = np.zeros([nIN,nc], dtype=hw.dtype)

    # Top boundary flux (Wieringermeer Rainfall)
    qBnd = bPar.topBndFuncWat(t, bPar)
    qw[nIN - 1] = qBnd

    # Flux in all intermediate nodes (not including last element)
    ii = np.arange(1, nIN - 1)
    qw[ii] = -kIN[ii] * ((hw[ii] - hw[ii - 1]) / dzN[ii - 1] + 1)

    if bPar.bottomTypeWat.lower() == 'Gravity': # Decided by input in bPar
        qw[0] = -kIN[0]
    else:
        qw[0] = -bPar.kRobBotWat * (hw[0]- bPar.hwBotBnd)
    return qw


def DivWaterFlux(t, hw, sP, mDim, bPar): # RHS: f(t,y)
    nr,nc = hw.shape
    nN = mDim.nN
    dzIN = mDim.dzIN

    divqW = np.zeros([nN,nc]).astype(hw.dtype)
    rateWF = np.zeros([nN,nc]).astype(hw.dtype)
   
    S = s_root(t,hw,sP, mDim, bPar)
    Mass = CPrimeFun(hw, sP, mDim)
    qW = WaterFlux(t, hw, sP, mDim, bPar)
    
    ii = np.arange(0,nN)
    divqW[ii] = -(qW[ii + 1] - qW[ii]) / (dzIN[ii]) # Calculating flux divergence
    
    if bPar.rootpresence == 'affirmative':
        rateWF= (divqW[ii]-S[ii])/ Mass[ii] # Considering root water uptake term
    else: 
        rateWF= (divqW[ii])/ Mass[ii]
    return rateWF


def IntegrateWF(tRange, iniSt, sPar, mDim, bPar):

    def dYdt(t, hW):
        # Solver switches between 0-D and matrix-shaped states, to account for this:
        if len(hW.shape)==1:
            hW = hW.reshape(mDim.nN,1)
        rates = DivWaterFlux(t, hW, sPar, mDim, bPar)
        return rates

    def jacFun(t,y): # Complex Jacobian
       if len(y.shape)==1:
           y = y.reshape (mDim.nN, 1)
       nr,nc = y.shape
       dh = np.sqrt(np.finfo(float).eps)
       ycmplx = np.repeat(y,nr,axis=1).astype(complex)
       c_ex = np.eye(nr)*1j*dh
       ycmplx = ycmplx + c_ex
       dfdy = dYdt(t,ycmplx).imag/dh
       return sp.coo_matrix(dfdy)
    
    # Solving the rate equation:
    t_span = [tRange[0],tRange[-1]]
    int_result = spi.solve_ivp(dYdt, t_span, iniSt.squeeze(),
                               method='BDF', vectorized=True, jac=jacFun, 
                               t_eval=tRange,
                               rtol=1e-6)

    return int_result



