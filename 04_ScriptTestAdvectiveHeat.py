#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:17:49 2017
@author: theimovaara
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.sparse as sp

import CoupledHeatWaterFlowTHe as cfun
import MyTicToc as mt


def BndTTop(t, bPar):
    #bndT = bPar.avgT - bPar.rangeT * np.cos(2 * np.pi * (t - bPar.tMin)
    #                                        / 365.25)
    bndT = 300

    return bndT


def BndqWatTop(t, bPar):
    # if (t > bPar.tWMin) and (t < bPar.tWMax):
    #     qBnd = bPar.qTop
    # else:
    qBnd = bPar.qTop
    return qBnd



### MAIN PART OF MODEL
# Define model Domain
nIN = 301
# soil profile
zIN = np.linspace(-1.5, 0, num=nIN).reshape(nIN, 1)
# nIN = np.shape(zIN)[0]
zN = np.zeros(nIN - 1).reshape(nIN - 1, 1)
zN[0, 0] = zIN[0, 0]
zN[1:nIN - 2, 0] = (zIN[1:nIN - 2, 0] + zIN[2:nIN - 1, 0]) / 2
zN[nIN - 2, 0] = zIN[nIN - 1]
nN = np.shape(zN)[0]

ii = np.arange(0, nN - 1)
dzN = (zN[ii + 1, 0] - zN[ii, 0]).reshape(nN - 1, 1)
dzIN = (zIN[1:, 0] - zIN[0:-1, 0]).reshape(nIN - 1, 1)

# collect model dimensions in a namedtuple: modDim
mDim = {'zN' : zN,
        'zIN' : zIN,
        'dzN' : dzN,
        'dzIN' : dzIN,
        'nN' : nN,
        'nIN' : nIN
        }
mDim = pd.Series(mDim)

# allN = np.arange(0, nN)
# allIN = np.arange(0, nIN)

# rhoW = 1000  # [kg/m3] density of water
rhoS = 2650  # [kg/m3] density of solid phase
rhoB = 1700  # %[kg/m3] dry bulk density of soil
n = 1 - rhoB / rhoS  # [-] porosity of soil = saturated water content.
qCont = 0.75  # quartz content

# collect soil parameters in a namedtuple: soilPar

sPar = {'vGA': np.ones(np.shape(zN)) * 1 / 0.5,  # alpha[1/m]
        'vGN': np.ones(np.shape(zN)) * 4.0,  # n[-]
        'vGM': np.ones(np.shape(zN)) * (1 - 1 / 4.0),  # m = 1-1/n[-]
        'thS': np.ones(np.shape(zN)) * n,  # saturated water content
        'thR': np.ones(np.shape(zN)) * 0.03,  # residual water content
        'KSat': np.ones(np.shape(zN)) * 0.5,  # [m/day]
        'vGE': 0.5,  # power factor for Mualem-van Genuchten                      
        'Cv': 1.0e-8,  # compressibility of compact sand [1/Pa]
        'viscRef': cfun.ViscosityWaterT(283.15),
        'qCont': qCont, # quartz content
        }

sPar = pd.Series(sPar)
# Define Boundary parameters
# collect boundary parameters in a named tuple boundpar...

# Define top boundary condition function
bPar =  {'topBndFuncHeat': BndTTop,
     'avgT': 273.15 + 10,
     'rangeT': 0,
     'tMin': 46,
     'topCond': 'Robin',
     'lambdaRobTop': 2.16e9,
     'lambdaRobBot': 0,
     'TBndBot': 273.15 + 10,
     'topBndFuncWat': BndqWatTop, #topBndFuncWat(t,bPar)
     'qTop': -0.05,  # top flux
     'tWMin': 0,
     'tWMax': 1375,
     'bottomTypeWat': 'Robbin', # Robin condition or Gravity condition
     'kRobBotWat': 0.05,  # Robin resistance term for bottom
     'hwBotBnd': 1.5,  # pressure head at lower boundary       
    }

bPar = pd.Series(bPar)



# Define Initial Conditions
# TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K
zRef = 0 #-0.5 #-0.25  # depth of water table
#hwIni = zRef - zN
hwIni = np.zeros(zN.shape)

TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K
sVec = np.concatenate([hwIni, TIni], axis=0)

# Time Discretization
# tOut2 = np.logspace(-11, np.log10(2 * 365), num=365)  # time
tOut = np.linspace(0,10,num = 500)

mt.tic()
int_result = cfun.IntegrateCHWF(tOut, sVec, sPar, mDim, bPar)

mt.toc()

#qH = cfun.HeatFlux(tOut, int_result.y, sPar, mDim, bPar)
hWSim = int_result.y[0:nN]
TSim = int_result.y[nN:2*nN]

plt.close('all')
fig1, ax1 = plt.subplots(figsize=(7, 4))
ii = np.arange(nN-1, 0, -10)
ax1.plot(tOut, TSim[ii, :].T, '-')
ax1.set_title('Temperature (ODE)')
ax1.set_xlabel('time (days)')
ax1.set_ylabel('temperature [K]')


fig2, ax2 = plt.subplots(figsize=(4, 7))
jj = np.arange(0, len(tOut))
ax2.plot(TSim[:, jj], zN, '-')
ax2.set_title('Temperature vs. depth (ODE)')
ax2.set_ylabel('depth [m]')
ax2.set_xlabel('temperature [K]')

qH = cfun.HeatFlux(tOut, TSim, hWSim, sPar, mDim, bPar)
fig3, ax3 = plt.subplots(figsize=(4, 7))
# plot fluxes after 2nd output time (initial rate is extreme due to initial conditions)
ax3.plot(qH[:,1:], zIN, '-')

ax3.set_title('Heat Flux vs. depth (ODE)')
ax3.set_ylabel('depth [m]')
ax3.set_xlabel('temperature [J/m2]')


# plt.show()


# if __name__ == "__main__":
#    main()
