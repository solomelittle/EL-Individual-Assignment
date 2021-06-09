
# coding: utf-8

# # Heat Diffusion in Soils
#
# This Jupyter Notebook gives an example how to implement a 1D heat diffusion model in Python.
#
# First we need to import the packages which we will be using:
#

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import CoupledHeatWaterFlowTHe as cfun
import MyTicToc as mt

sns.set()


## Main 

# Domain
nIN = 301
# soil profile until 15 meters depth
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

# collect model dimensions in a pandas series: mDim
mDim = {'zN' : zN,
        'zIN' : zIN,
        'dzN' : dzN,
        'dzIN' : dzIN,
        'nN' : nN,
        'nIN' : nIN
        }
mDim = pd.Series(mDim)

# ## Definition of material properties
# In this section of the code we define the material properties

# Soil Properties
# [J/(m3 K)] volumetric heat capacity of soil solids
zetaSol = 2.235e6
# [J/(m3 K)] volumetric heat capacity of water (Fredlund 2006)
zetaWat = 4.154e6

# rhoW = 1000  # [kg/m3] density of water
rhoS = 2650  # [kg/m3] density of solid phase
rhoB = 1700  # %[kg/m3] dry bulk density of soil
n = 1 - rhoB / rhoS  # [-] porosity of soil = saturated water content.
qCont = 0.75  # quartz content


# collect soil parameters in a pandas Series: sPar
sPar = {'vGA': np.ones(np.shape(zN)) * 1 / 0.5,  # alpha[1/m]
        'vGN': np.ones(np.shape(zN)) * 3.0,  # n[-]
        'vGM': np.ones(np.shape(zN)) * (1 - 1 / 3.0),  # m = 1-1/n[-]
        'thS': np.ones(np.shape(zN)) * 0.4,  # saturated water content
        'thR': np.ones(np.shape(zN)) * 0.03,  # residual water content
        'KSat': np.ones(np.shape(zN)) * 0.05,  # [m/day]
        'vGE': 0.5,  # power factor for Mualem-van Genuchten                      
        'Cv': 1.0e-8,  # compressibility of compact sand [1/Pa]
        'viscRef': cfun.ViscosityWaterT(283.15),
        'qCont': qCont, # quartz content
        }
sPar = pd.Series(sPar)

# ## Definition of the Boundary Parameters
# boundary parameters
# collect boundary parameters in a named tuple boundpar...
def BndTTop(t, bPar):
    #bndT = bPar.avgT - bPar.rangeT * np.cos(2 * np.pi * (t - bPar.tMin)
    #                                        / 365.25)
    bndT = 300

    return bndT


def BndqWatTop(t, bPar):
    #qBnd = bPar.qTop * (t > bPar.tWMin) * (t < bPar.tWMax)
    qBnd = bPar.qTop
    return qBnd

#lamdbaRobTop for heat flow: heat resistance of a thin air layer above the sample
# lam air = 0.025 W/mK or J/msK  so for an air boundary of 1 mm lambdaRob is
# 0.025*3600*24/1000 = 2.16
# lamRobBot is similar to the soil and a distance to the reference tmep (i.e. 1m)

bPar = {'topBndFuncHeat': BndTTop,
        'topCond': 'Robin',
        'lambdaRobTop': 2.16*1e9, # heat electrode placed directly on top of the soil surface
        'lambdaRobBot': 0, # insulating bottom boundary
        'TBndBot': 273.15 + 10,
        'topBndFuncWat': BndqWatTop, #topBndFuncWat(t,bPar)
        'qTop': 0.0,  # top flux
        'tWMin': 50,
        'tWMax': 375,
        'bottomTypeWat': 'Robin', # Robin condition or Gravity condition
        'kRobBotWat': 0.10,  # Robin resistance term for bottom
        'hwBotBnd': -1.0,  # pressure head at lower boundary       
        }

bPar = pd.Series(bPar)


# In[3:] Define Initial Conditions
zRef = -2.5 # depth of water table
hwIni = zRef - zN

TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K

sVecIni = np.vstack([hwIni, TIni])

# Time Discretization
tOut = np.linspace(0, 31, 310)  # time
nOut = np.shape(tOut)[0]


mt.tic()
int_result = cfun.IntegrateCHWF(tOut, sVecIni, sPar, mDim, bPar)

mt.toc()

# Dirichlet boundary condition: write boundary temperature to output.
if int_result.success:
    print('Integration has been successful')

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
jj = np.arange(0, nOut)
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

plt.show()
# plt.savefig('myfig.png')

# if __name__ == "__main__":
# main()
