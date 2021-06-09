#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:17:49 2017
@author: theimovaara
"""
import numpy as np
import scipy.integrate as spi
import scipy.interpolate as sint
import scipy.sparse as sp
import matplotlib.pyplot as plt


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

def CFun(hw, sP):
    #Differential water capacity, dtheta/dhw (analytical)
    hc = -hw
    Se = SeFun(hw, sP)
    dSedh = sP.vGA * sP.vGM / (1 - sP.vGM) * Se ** (1 / sP.vGM) * \
            (1 - Se ** (1 / sP.vGM)) ** sP.vGM * (hc > 0) + (hc <= 0) * 0
    return (sP.thS - sP.thR) * dSedh


def CmplxDerivative (hw, sP, fun):
    #Differential water capacity, dtheta/dhw (complex derivative)
    dh = np.sqrt(np.finfo(float).eps)
    hcmplx = hw.real + 1j*dh

    y = fun(hcmplx, sP)
    C = y.imag / dh
    return C


def CFunCmplx(hw, sP):
    #Differential water capacity, dtheta/dhw (complex derivative)
    C = CmplxDerivative(hw, sP, thFun)
    return C

def CFunDerivative(hw,sP):
    C = CmplxDerivative(hw, sP, CFun)
    return C


def CPrimeFun(hw, sP, mDim):
    # Function for calculating the MassMatrix of the Richards Equation
    # including compression
    th = thFun(hw, sP)  # volumetric water content
    Sw = th / sP.thS  # water saturation
    Chw = CFun(hw, sP)
    rhoW = 1000  # density of water kg/m3
    gConst = 9.82  # gravitational constant m/s2
    betaW = 4.5e-10  # compressibility of water 1/Pa
    Ssw = rhoW * gConst * (sP.Cv + sP.thS * betaW)

    cPrime = Chw + Sw * Ssw
    #dCdhw = CFunDerivative(hw,sP)
    mMat = cPrime #+ hw*dChdw

    #  Ponding water condition when hw(-1) >= 0
    # if head at top is larger or equal to 0, ponding water table
    # which is implementd by changing the differential water
    # capacity in order to make the equation:
    # dhw/dt = qbnd - q(n-1/2)
    mMat[mDim.nN-1] = 1/mDim.dzIN[mDim.nIN-2] * (hw[mDim.nN-1]>0) \
        + mMat[mDim.nN-1] * (hw[mDim.nN-1]<=0)

    return mMat


def ViscosityWaterT(T):

    # table gives viscosity in [mPa s] However we need it in units of days...
    # tempVal = np.array(
    #     [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    #       32, 33, 34, \
    #       35, 36, 37, 38, 39, 40, 45, 50, 55, 60, 65, 70, 75, 80])
    # viscVal = np.array(
    #     [1.6735, 1.619, 1.5673, 1.5182, 1.4715, 1.4271, 1.3847, 1.3444, 1.3059, 1.2692, 1.234, 1.2005, 1.1683, \
    #       1.1375, 1.1081, 1.0798, 1.0526, 1.0266, 1.0016, 0.9775, 0.9544, 0.9321, 0.9107, 0.89, 0.8701, 0.8509, \
    #       0.8324, 0.8145, 0.7972, 0.7805, 0.7644, 0.7488, 0.7337, 0.7191, 0.705, 0.6913, 0.678, 0.6652, 0.6527, \
    #       0.5958, 0.5465, 0.5036, 0.466, 0.4329, 0.4035, 0.3774, 0.354])
    # #f = sint.UnivariateSpline(tempVal, viscVal)
    # f = sint.interp1d(tempVal, viscVal, fill_value = 'extrapolate')

    # if (type(T) == float):
    #     #T is a scalar array
    #     vT = f(T-273.15)* 0.001 /(24*3600)
    # else:
    #     nr,nc = T.shape
    #     vT = np.zeros((nr,nc)).astype(T.dtype)
    #     ii = np.arange(0,nr)
    #     for jj in range(nc):
    #         vT[ii,jj] = f(T[ii,jj].squeeze()-273.15)* 0.001 / (24*3600)

    # Approximation Patek et al (2009)
    # T in Kelvin!!
    a = np.array([280.68, 511.45, 61.131, 0.45903])
    b = np.array([-1.9, -7.7, -19.6, -40.0])

    if type(T)==float:
        vT = np.sum(a * (T / 300) ** b, axis=0) * 1e-6
    else:
        # type should be numpy.ndarray
        nr,nc = T.shape
        vT = np.zeros((nr,nc)).astype(T.dtype)
        ii = np.arange(0,nr)
        for jj in range(nc):
            Ttmp = T[:,jj].reshape([nr,1]) # to prevent errors if T < 0, which onlyh occurs if timestep is too big!
            vT[ii,jj] = (1e-6 * np.sum(a * (Ttmp / 300) ** b, axis=1)).squeeze()

    # Functionalapproximation:
    # https://www.fxsolver.com/browse/formulas/dynamic+viscosity+of+water+%28as+a+function+of+temperature+temperature%29

    # A = 2.414e-5 #Pa s
    # B = 247.8 #K
    # C = 140 #K
    # vT = A * 10 **(B/(T-C)) / (24*3600)

    return vT


def KFun(hw, T, sP, mDim):
    #Unsaturated hydraulic conductivity
    nr,nc = hw.shape
    nIN = mDim.nIN
    Se = SeFun(hw, sP)
    # kVal = sP.KSat * Se ** sP.vGE
    # * (1 - (1 - Se ** (1 / sP.vGM)) ** sP.vGM) ** 2
    viscRef = sP.viscRef
    viscN = ViscosityWaterT(T)

    kN = sP.KSat * Se ** 3  #nodal conductivity
    #temperature dependency
    kN = kN * viscRef / viscN

    kIN = np.zeros([nIN,nc], dtype=hw.dtype)
    kIN[0] = kN[0]
    ii = np.arange(1, nIN - 1)
    kIN[ii] = np.minimum(kN[ii - 1], kN[ii])
    kIN[nIN - 1] = kN[nIN - 2]
    return kIN


def WatFlux(t, hw, T, sP, mDim, bPar):
    #Waterflow (m3/m2d)
    nr,nc = hw.shape
    nIN = mDim.nIN
    dzN = mDim.dzN

    # Calculate inter nodal permeabilities
    kIN = KFun(hw, T, sP, mDim)
    qw = np.zeros([nIN,nc], dtype=hw.dtype)

    # Top boundary flux (Neumann flux)
    # Neumann at the top
    qBnd = bPar.topBndFuncWat(t, bPar)
    if (nc > 1) and (np.size(t)==1):
        # in jacobian, but we need nc values for qBnd
        qBnd = np.repeat(qBnd,nc)
    qw[nIN - 1] = qBnd

    # Flux in all intermediate nodes
    ii = np.arange(1, nIN - 1)  # does not include last element
    qw[ii] = -kIN[ii] * ((hw[ii] - hw[ii - 1]) / dzN[ii - 1] + 1)

    if bPar.bottomTypeWat.lower() == 'gravity':
        qw[0] = -kIN[0]
    else:
        qw[0] = -bPar.kRobBotWat * (hw[0]- bPar.hwBotBnd)
    return qw

def BulkHeatFun(hw, sP):
    # [J/(m3 K)] volumetric heat capacity of soil solids
    zetaSol = 2.235e6
    # [J/(m3 K)] volumetric heat capacity of water (Fredlund 2006)
    zetaWat = 4.154e6
    zetaAir = 1.2e3

    thW = thFun(hw, sP)
    zetaB = (1 - sP.thS) * zetaSol + thW * zetaWat +(sP.thS-thW)*zetaAir
    return zetaB


def ThermCondFun1(hw, sP, mDim):
    nr,nc = hw.shape
    lambdaWat = 0.58  # [W/(mK)] thermal conductivity of water (Remember W=J/s)
    lambdaQuartz = 6.5  # [W/(mK)] thermal conductivity of quartz
    lambdaOther = 2.0  # [W/(mK)] thermal conductivity of other minerals

    # heat conductivity of solids is a function of quartz content (sP.qCont)
    lambdaSolids = lambdaQuartz ** sP.qCont + lambdaOther ** (1 - sP.qCont)

    thetaN = thFun(hw, sP)
    lamN = (lambdaWat ** thetaN + lambdaSolids ** (1 - thetaN)) * 24 * 3600

    nIN = mDim.nIN
    lamIN = np.zeros((nIN,nc), dtype=hw.dtype)
    lamIN[0] = lamN[0]
    ii = np.arange(1, nIN - 1)
    lamIN[ii] = np.minimum(lamN[ii - 1], lamN[ii])
    lamIN[nIN - 1] = lamN[nIN - 2]
    return lamIN


def ThermCondFun2(hw, sP, mDim):
    nr,nc = hw.shape
    nIN = mDim.nIN
    nN = mDim.nN

    theta = thFun(hw, sP)

    lamDryAir = 0.025  # [W/(mK)] thermal conductivity of dry air
    lamVapour = 0.0736 * theta / sP.thS  # thermal conductivity of vapor
    lamAir = lamDryAir + lamVapour
    lamWat = 0.57  # [W/(mK)] thermal conductivity of water (Remember W = J/s)
    #lamQuartz = 6  # [W/(mK)] thermal conductivity of quartz
    #lamOther = 2.0  # [W/(mK)] thermal conductivity of other minerals

    lamSolids = 6 #(lamQuartz ** sP.qCont) * (lamOther ** (1 - sP.qCont))

    g1 = 0.015 + (0.333 - 0.015) * theta / sP.thS
    g = np.array([g1, g1, 1 - 2 * g1])
    Fw = 1
    Fa = np.sum(1 / (1 + (lamAir / lamWat - 1) * g),0) / 3
    Fs = np.sum(1 / (1 + (lamSolids / lamWat - 1) * g),0) / 3

    lamBulk = (Fs * lamSolids * (1 - sP.thS) + Fw * lamWat * theta + Fa * lamAir * (1 - sP.thS - theta)) / \
              (Fs * (1 - sP.thS) + Fw * theta + Fa * (1 - sP.thS - theta)) * (24 * 3600)

    lamIN = np.zeros((nIN, nc), dtype=hw.dtype)
    lamIN[0] = lamBulk[0]
    for ii in np.arange(1, nIN - 1):
        lamIN[ii] = np.minimum(lamBulk[ii - 1], lamBulk[ii])
    lamIN[nIN - 1] = lamBulk[nN - 1]
    return lamIN

def HeatFlux(t, T, hw, sP, mDim, bPar):
    # function for calculating the heat flux across the domain
    nr,nc = T.shape
    nIN = mDim.nIN
    nN = mDim.nN
    dzN = mDim.dzN

    zetaWat = 4.154e6  # [J/(m3 K)] volumetric heat capacity of water (Fredlund 2006)
    #hw = hw.reshape(mDim.zN.shape)
    #T = T.reshape(mDim.zN.shape)

    qW = WatFlux(t, hw, T, sP, mDim, bPar)

    lamRobTop = bPar.lambdaRobTop
    lamRobBot = bPar.lambdaRobBot

    lamIN = ThermCondFun2(hw, sP, mDim)

    # Temperature at top boundary is known (Dirichlet boundary condition)
    bndT = bPar.topBndFuncHeat(t, bPar)
    if (nc > 1) and (np.size(t)==1):
        # in jacobian, but we need nc values for qBnd
        bndT = np.repeat(bndT,nc)
    qD = np.zeros((nIN, nc)).astype(hw.dtype)
    qC = np.zeros((nIN, nc)).astype(hw.dtype)

    # Temperature at internode based on flow direction
    # Robin / Neumann condition for bottom boundary
    qD[0] = -lamRobBot * (T[0] - bPar.TBndBot)
    qC[0] = qW[0] * zetaWat \
               * (bPar.TBndBot * (qW[0] >= 0)
                  + T[0]*(qW[0] < 0))

    # Flux in all intermediate nodes
    ii = np.arange(1, nIN - 1)
    qD[ii] = -lamIN[ii] * (T[ii] - T[ii - 1]) / dzN[ii - 1]
    qC[ii] = qW[ii] * zetaWat \
                * (T[ii - 1] * (qW[ii] >= 0)
                   + T[ii] * (qW[ii] < 0))

    # Robin/ Neumann boundary at the top
    qD[nIN - 1] = - lamRobTop * (bndT - T[nN - 1])  # * (qw[nIN - 1, 0] >= 0)
    qC[nIN - 1] = qW[nIN - 1] * zetaWat \
                     * (T[nN-1] * (qW[nIN - 1] >=0)
                        + bndT * (qW[nIN - 1] < 0))
    return qD + qC


def DivCoupledFlux(t, hw,T, sP, mDim, bPar):
    nr,nc = hw.shape

    nN = mDim.nN
    dzIN = mDim.dzIN
    zetaWat = 4.154e6 # [J/(m3 K)] volumetric heat capacity of water (Fredlund 2006)
    zetaAir = 1.2e3

    rHSH = np.zeros([nN, nc], dtype=hw.dtype)
    rHSW = np.zeros([nN, nc], dtype=hw.dtype)
    rHSTot = np.zeros([2*nN, nc], dtype=hw.dtype)

    divqW = np.zeros([nN, nc], dtype=hw.dtype)
    divqH = np.zeros([nN, nc], dtype=hw.dtype)

    mWat = CPrimeFun(hw, sP, mDim)
    qW = WatFlux(t, hw, T, sP, mDim, bPar)
    # Calculate divergence of flux for all nodes
    ii = np.arange(0,nN)
    divqW[ii] = -(qW[ii + 1] - qW[ii]) / (dzIN[ii])
    rHSW = divqW / mWat

    mHeat = BulkHeatFun(hw, sP)
    qH = HeatFlux(t, T, hw, sP, mDim, bPar)
    # Cal4culate divergence of flux for all nodes
    ii = np.arange(0, nN)
    divqH[ii] = -(qH[ii + 1] - qH[ii]) / (dzIN[ii])
    rHSH = (divqH - (zetaWat-zetaAir)*T*divqW) / mHeat

    rHSTot = np.vstack([rHSW, rHSH])

    return rHSTot


def IntegrateCHWF(tRange, iniSt, sPar, mDim, bPar):

    def dYdt(t, sVec):

        # solver switches between zeroD and matrix shaped states
        # we need to take this into account to create a rate function that
        # works for every case...
        if len(sVec.shape)==1:
            sVec = sVec.reshape(2*mDim.nN,1)

        nN = mDim.nN
        # unpack states
        hW = sVec[0:nN]
        T = sVec[nN:2*nN]

        rates = DivCoupledFlux(t, hW, T, sPar, mDim, bPar)

        return rates

    def jacFun(t,y):
        if len(y.shape)==1:
            y = y.reshape(2 * mDim.nN,1)

        nr, nc = y.shape
        dh = np.sqrt(np.finfo(float).eps)
        jac = np.zeros((nr,nr))

        ycmplx = y.copy().astype(complex)
        ycmplx = np.repeat(ycmplx,nr,axis=1)
        c_ex = np.ones([nr,1])* 1j*dh
        ycmplx = ycmplx + np.diagflat(c_ex,0)
        dfdy = dYdt(t, ycmplx).imag
        #dfdy[np.abs(dfdy) < 20*dh] = 0
        jac = dfdy/dh
        return sp.coo_matrix(jac)
        #return jac
    def my_events(t,y):
        ret_val = 1
        if t in tRange:
            ret_val = 0;
        return ret_val

    # solve rate equatio
    t_span = [tRange[0],tRange[-1]]
    int_result = spi.solve_ivp(dYdt, t_span, iniSt.squeeze(),
                               method='BDF', vectorized=True, jac=jacFun,
                               t_eval=tRange,
                               rtol=1e-7)

    return int_result



