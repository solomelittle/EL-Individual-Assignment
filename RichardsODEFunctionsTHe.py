#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:17:49 2017
@author: theimovaara
"""
import numpy as np
import scipy.integrate as spi
import scipy.sparse as sp


def SeFun(hw, sP):
    #Effective saturation
    hc = -hw
    Se = (1 + ((hc * (hc > 0)) * sP.vGA) ** sP.vGN) ** (-sP.vGM)
    return Se


def thFun(hw, sP):
    #Water content
    Se = SeFun(hw, sP)
    th = sP.thR + (sP.thS - sP.thR) * Se
    return th


def CFunCmplx(hw, sP):
    #Differential water capacity, dtheta/dhw (complex derivative)
    dh = np.sqrt(np.finfo(float).eps)
    if np.iscomplexobj(hw):
        hcmplx = hw.real + 1j*dh
    else:
        hcmplx = hw.real + 1j*dh

    th = thFun(hcmplx, sP)
    C = th.imag / dh
    return C


def CPrimeFun(hw, sP, mDim):
    # Function for calculating the MassMatrix of the Richards Equation
    # including compression
    th = thFun(hw, sP)  # volumetric water content
    Sw = th / sP.thS  # water saturation
    Chw = CFunCmplx(hw, sP)
    rhoW = 1000  # density of water kg/m3
    gConst = 9.82  # gravitational constant m/s2
    betaW = 4.5e-10  # compressibility of water 1/Pa
    Ssw = rhoW * gConst * (sP.Cv + sP.thS * betaW)

    cPrime = Chw + Sw * Ssw

    #  Ponding water condition when hw(-1) >= 0
    # if head at top is larger or equal to 0, ponding water table
    # which is implementd by changing the differential water
    # capacity in order to make the equation:
    # dhw/dt = qbnd - q(n-1/2)
    cPrime[mDim.nN-1] = 1/mDim.dzIN[mDim.nIN-2] * (hw[mDim.nN-1]>0) \
        + cPrime[mDim.nN-1] * (hw[mDim.nN-1]<=0)

    return cPrime


def KFun(hw, sP, mDim):
    #Unsaturated hydraulic conductivity
    nr,nc = hw.shape
    nIN = mDim.nIN
    Se = SeFun(hw, sP)
    # kVal = sP.KSat * Se ** sP.vGE
    # * (1 - (1 - Se ** (1 / sP.vGM)) ** sP.vGM) ** 2
    kN = sP.KSat * Se ** 3  #nodal conductivity

    kIN = np.zeros([nIN,nc], dtype=hw.dtype)
    kIN[0] = kN[0]
    #print(kIN.shape)
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
#     nIN = mDim.nIN
#     nr,nc = hw.shape

    alpha = alpharoot(hw, sP)
    beta = betaroot(t,hw,mDim,sP)
    S=alpha*beta*pEv
#     S = np.zeros([nN,nc],dtype=hw.dtype)
#     S=alpha*beta*potEv
  
    return S

def WatFlux(t, hw, sP, mDim, bPar):
    #Waterflow (m3/m2d)
    nr,nc = hw.shape
    nIN = mDim.nIN
    dzN = mDim.dzN

    # Calculate inter nodal permeabilities
    kIN = KFun(hw, sP, mDim)
    qw = np.zeros([nIN,nc], dtype=hw.dtype)

    # Top boundary flux (Neumann flux)
    # Neumann at the top
    qBnd = bPar.topBndFuncWat(t, bPar)
    qw[nIN - 1] = qBnd

    # Flux in all intermediate nodes
    ii = np.arange(1, nIN - 1)  # does not include last element
    qw[ii] = -kIN[ii] * ((hw[ii] - hw[ii - 1]) / dzN[ii - 1] + 1)

    if bPar.bottomTypeWat.lower() == 'Gravity':
        qw[0] = -kIN[0]
    else:
        qw[0] = -bPar.kRobBotWat * (hw[0]- bPar.hwBotBnd)
    return qw


def DivWatFlux(t, hw, sP, mDim, bPar):
    #Right hand side f(t,y)
    nr,nc = hw.shape
    nN = mDim.nN
    dzIN = mDim.dzIN

    #lochw = hw.copy().reshape(mDim.zN.shape)
    divqW = np.zeros([nN,nc]).astype(hw.dtype)
    rateWF = np.zeros([nN,nc]).astype(hw.dtype)
    # Calculate heat fluxes accross all internodes
    S = s_root(t,hw,sP, mDim, bPar)
    massMD = CPrimeFun(hw, sP, mDim)

    qW = WatFlux(t, hw, sP, mDim, bPar)
    # Calculate divergence of flux for all nodes
    ii = np.arange(0,nN)
    divqW[ii] = -(qW[ii + 1] - qW[ii]) / (dzIN[ii])
    rateWF= (divqW[ii]-S[ii])/ massMD[ii]
    #Kmat = FillKMatWat(t, lochw, sP, mDim, bPar)
    #Yvec = FillYVecWat(t, lochw, sP, mDim, bPar)
    #divqW = (np.dot(Kmat,lochw) + Yvec)/massMD
    return rateWF


def IntegrateWF(tRange, iniSt, sPar, mDim, bPar):

    def dYdt(t, hW):

        # solver switches between zeroD and matrix shaped states
        # we need to take this into account to create a rate function that
        # works for every case...

        if len(hW.shape)==1:
            hW = hW.reshape(mDim.nN,1)
        rates = DivWatFlux(t, hW, sPar, mDim, bPar)

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
    

    # solve rate equatio
    t_span = [tRange[0],tRange[-1]]
    int_result = spi.solve_ivp(dYdt, t_span, iniSt.squeeze(),
                               method='BDF', vectorized=True, jac=jacFun, 
                               t_eval=tRange,
                               rtol=1e-7)

    return int_result



