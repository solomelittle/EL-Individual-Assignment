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
import seaborn as sns

import CoupledHeatWaterFlowTHe as cfun
import MyTicToc as mt




def BndTTop(t, bPar):
    #bndT = bPar.avgT - bPar.rangeT * np.cos(2 * np.pi * (t - bPar.tMin)
    #                                        / 365.25)
    bndT = 300 - bPar.rangeT * np.sin(2*np.pi*t/4)

    return bndT


def BndqWatTop(t, bPar):

    # if (t > bPar.tWMin) and (t < bPar.tWMax):
    #     qBnd = bPar.qTop
    # else:
    #     qBnd = 0

    qBnd = (np.sin(2*np.pi*t/4)>=0)*bPar.qTop
    return qBnd



### MAIN PART OF MODEL
# Define model Domain
nIN = 151
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

sPar = {'vGA': np.ones(np.shape(zN)) * 1 / 1.5,  # alpha[1/m]
        'vGN': np.ones(np.shape(zN)) * 2.0,  # n[-]
        'vGM': np.ones(np.shape(zN)) * (1 - 1 / 2.0),  # m = 1-1/n[-]
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

# Define top boundary condition function
bPar =  {'topBndFuncHeat': BndTTop,
     'avgT': 273.15 + 10,
     'rangeT': 20,
     'tMin': 46,
     'topCond': 'Robin',
     'lambdaRobTop': 2.16e6,
     'lambdaRobBot': 0,
     'TBndBot': 273.15 + 10,
     'topBndFuncWat': BndqWatTop, #topBndFuncWat(t,bPar)
     'qTop': -0.25,  # top flux
     'tWMin': 0,
     'tWMax': 3*365,
     'bottomTypeWat': 'Gravity', # Robin condition or Gravity condition
     'kRobBotWat': 10,  # Robin resistance term for bottom
     'hwBotBnd': 0.75,  # pressure head at lower boundary
    }

bPar = pd.Series(bPar)

# Define Initial Conditions
# TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K
zRef = -0.25 #-0.5 #-0.25  # depth of water table
#hwIni = zRef - zN

hwIni = np.ones(np.shape(zN)) * zRef
TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K
sVec = np.concatenate([hwIni, TIni], axis=0)

# Time Discretization
# tOut2 = np.logspace(-11, np.log10(3 * 365), num=3*365)  # time
tOut = np.linspace(0,36,num = 360)
nOut = len(tOut)
# tOut = np.sort(np.hstack((tOut1, bTime)))  # time

# copy initial vector to hw0. Apply squeeze to compress it to one dimension

mt.tic()
int_result = cfun.IntegrateCHWF(tOut, sVec, sPar, mDim, bPar)
mt.toc()

hWSim = int_result.y[0:nN]
TSim = int_result.y[nN:2*nN]
thSim = cfun.thFun(hWSim,sPar)

#mt.tic()
#TOutPic, hwOutPic = himp.HalfImplicitPicar(tOut2, hw0, T0, sPar, mDim, bPar, tPar)
#mt.toc()

sns.set()
plt.close('all')
fig1, ax1 = plt.subplots(figsize=(7, 4))
ii = np.arange(nN-1, 0, -20)
ax1.plot(tOut, TSim[ii,:].T, '-')
ax1.set_title('Temperature (ODE)')
ax1.set_xlabel('time (days)')
ax1.set_ylabel('temperature [K]')
ax1.legend(zN[ii])

fig2, ax2 = plt.subplots(figsize=(7, 7))
jj = np.arange(0, nOut)
ax2.plot(TSim[:, jj], zN, '-')
ax2.set_title('Temperature vs. depth (ODE)')
ax2.set_ylabel('depth [m]')
ax2.set_xlabel('temperature [K]')

qH = cfun.HeatFlux(tOut, TSim, hWSim, sPar, mDim, bPar)
fig3, ax3 = plt.subplots(figsize=(7, 7))
# plot fluxes after 2nd output time (initial rate is extreme due to initial conditions)
ax3.plot(qH[:,1:], zIN, '-')

ax3.set_title('Heat Flux vs. depth (ODE)')
ax3.set_ylabel('depth [m]')
ax3.set_xlabel('temperature [J/m2]')


fig4, ax4 = plt.subplots(figsize=(7, 4))
# plot the pressure head for different depths as a function of time
# in this case we plot every 20th layer.
ax4.plot(tOut, hWSim[ii,:].T, '-')
ax4.set_ylabel('pressure head [m]')
ax4.set_xlabel('time [d]')
ax4.legend(zN[ii])
#plot pressure head as a function of depth. Here we plot every time step
fig5, ax5 = plt.subplots(figsize=(7, 7))
ax5.plot(hWSim, zN, '-')
ax5.grid(b=True)
ax5.set_xlabel('pressure head [m]')
ax5.set_ylabel('depth [m]')

# plt.savefig('myfig.png')

fig6, ax6 = plt.subplots(figsize=(7, 7))
ax6.plot(thSim, zN, '-')
ax6.grid(b=True)
ax6.set_xlabel('water content [-]')
ax6.set_ylabel('depth [m]')

fig7, ax7 = plt.subplots(figsize=(7, 4))
# plot the pressure head for different depths as a function of time
# in this case we plot every 20th layer.
ax7.plot(tOut, thSim[ii,:].T, '-')
ax7.set_ylabel('water content [-]')
ax7.set_xlabel('time [d]')
ax7.legend(zN[ii])
# save figures

# if __name__ == "__main__":
#    main()
