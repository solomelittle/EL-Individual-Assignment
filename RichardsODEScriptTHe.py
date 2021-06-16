#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:17:49 2017
@author: theimovaara
"""
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


# RichardODEFunctionsTHe contains all functions. I keep these separate from 
# the script so that I can reuse the code easily for different scenarios.
# You call the functions from this package using rfun.funcName
import RichardsODEFunctionsTHe as rfun
import MyTicToc as mt


# In[1:] Define model domain and soil properties
# Define model Domain
nIN = 351
# soil profile
zIN = np.linspace(-5, 0, num=nIN).reshape(nIN, 1)
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

# allN = np.arange(0, nN)
# allIN = np.arange(0, nIN)

# Define Soil Properties
# rhoW = 1000  # [kg/m3] density of water
rhoS = 2650  # [kg/m3] density of solid phase
rhoB = 1700  # %[kg/m3] dry bulk density of soil
n = 1 - rhoB / rhoS  # [-] porosity of soil = saturated water content.


# collect soil parameters in a pandas Series: sPar
sPar = {'vGA': np.ones(np.shape(zN)) * 0.019*100,  # alpha[1/m]
        'vGN': np.ones(np.shape(zN)) * 3.0,  # n[-]
        'vGM': np.ones(np.shape(zN)) * (1 - 1 / 3.0),  # m = 1-1/n[-]
        'thS': np.ones(np.shape(zN)) * 0.41,  # saturated water content = phi,porosity
        'thR': np.ones(np.shape(zN)) * 0.095,  # residual water content
        'KSat': np.ones(np.shape(zN)) * 1, # 6.2,  # [m/day]
        'vGE': 0.5,  # power factor for Mualem-van Genuchten                      
        'Cv': 1.0e-8,  # compressibility of compact sand [1/Pa]
        'h1': np.ones(np.shape(zN))*0,
        'h2': np.ones(np.shape(zN))*-1,
        'h3': np.ones(np.shape(zN))*-8.5,
        'h4': np.ones(np.shape(zN))*-15,
        }
sPar = pd.Series(sPar)

# In[2:] Define Boundary parameters
# collect boundary parameters in a named tuple boundpar...
# I define the boundary functions here in the script because you cannot know
# all the different scenarios you would like to use...
# Read meteodata
meteo_data  = pd.read_excel('WieringermeerData_Meteo.xlsx')
meteo_data['num_date'] = meteo_data['datetime'].astype(np.int64)/(1e9*3600*24)
meteo_data.set_index('datetime',inplace=True)

# set simulation time to numeric dates from boudary data...
t_range = meteo_data['num_date'][:-1]
taxis = meteo_data.index[:-1]

def BndqWatTop(t, bPar):
    if np.size(t)==1:
        t = np.array([t])
    qBnd = np.zeros(len(t))

    for ii in range(len(t)):
        xy, md_ind, t_ind = np.intersect1d(bPar.meteo_data['num_date'], np.ceil(t[ii]), return_indices=True)
        rf = bPar.meteo_data['rain_station'].iloc[md_ind].values
        qBnd[ii] = -rf
    return qBnd

def pEV(t, bPar):
    if np.size(t)==1:
        t = np.array([t])
    potEv = np.zeros(len(t))

    for ii in range(len(t)):
        xy, md_ind, t_ind = np.intersect1d(bPar.meteo_data['num_date'], np.ceil(t[ii]), return_indices=True)
        evap = bPar.meteo_data['pEV'].iloc[md_ind].values
        potEv[ii] = evap
    return potEv


# Define top boundary condition function
bPar = {'topBndFuncWat': BndqWatTop, #topBndFuncWat(t,bPar)
        'potEv': pEV,
        'meteo_data': meteo_data,
        'qTop': -0.01,  # top flux
        'tWMin': 50,
        'tWMax': 375,
        'bottomTypeWat': 'Robin', # Robin condition or Gravity condition
        'kRobBotWat': 0.10,  # Robin resistance term for bottom
        'hwBotBnd': -1.0,  # pressure head at lower boundary
        }
bPar = pd.Series(bPar)

# In[3:] Define Initial Conditions
zRef = -4 # depth of water table
hwIni = zRef - zN

# In[4:] Solv problem
# Time Discretization
tOut2 = np.linspace(t_range[0],t_range[100],2*365)  # time
#tOut2 = np.linspace(0, 10*365, num=365*5)

print('Solving unsaturated water flow problem')
mt.tic()
hwODE = rfun.IntegrateWF(tOut2, hwIni, sPar, mDim, bPar)
mt.toc()

tOut2 = hwODE.t

# In[5:] Plot results...
plt.close('all')
fig1, ax1 = plt.subplots(figsize=(7, 4))
# plot the pressure head for different depths as a function of time
# in this case we plot every 20th layer.
for ii in np.arange(0, nN, 20):
    ax1.plot(hwODE.t, hwODE.y[ii, :], '-')

ax1.grid(b=True)
ax1.set_ylabel('pressure head [m]')
ax1.set_xlabel('time [d]') 

#plot pressure head as a function of depth. Here we plot every time step
fig2, ax2 = plt.subplots(figsize=(7, 7))
for ii in np.arange(0, hwODE.t.size, 1):
    ax2.plot(hwODE.y[:, ii], zN[:, 0], '-')

ax2.grid(b=True)
ax2.set_xlabel('pressure head [m]')
ax2.set_ylabel('depth [m]')

# plt.savefig('myfig.png')
# calculate water contents (using function) and plot results as a function of depth
thODE = np.zeros(np.shape(hwODE.y))
for ii in np.arange(0, hwODE.t.size, 1):
    hwTmp = hwODE.y[:, ii].reshape(zN.shape)
    thODE[:, ii] = rfun.thFun(hwTmp, sPar).reshape(1, nN)

fig3, ax3 = plt.subplots(figsize=(7, 7))
for ii in np.arange(0, hwODE.t.size, 1):
    ax3.plot(thODE[:, ii], zN[:, 0], '-')

ax3.grid(b=True)
ax3.set_xlabel('water content [-]')
ax3.set_ylabel('depth [m]')

# SODE = np.zeros(np.shape(hwODE.y))
# for ii in np.arange(0, hwODE.t.size, 1):
#     hwTmp = hwODE.y[:, ii].reshape(zN.shape)
#     SODE[:,ii] = rfun.s_root(tOut2,hwTmp, sPar,mDim,bPar)
    
# fig4, ax4 = plt.subplots(figsize=(7, 7))
# for ii in np.arange(0, hwODE.t.size, 1):
#     ax3.plot(SODE[ii,:], zN[ii,0], '-')

# ax4.grid(b=True)
# ax4.set_xlabel('Root Sink [m/s]')
# ax4.set_ylabel('depth [m]')

# plt.show()

# if __name__ == "__main__":
#    main()
