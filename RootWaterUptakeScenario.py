#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
June 2021
@author: eslittle
Based on excerpts provided by T. Heimovaara and prior group work.
To model a root-free scenario, replace 'affirmative' with 'no' in bPar.rootpresence 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import RootWaterUptake as RWU
import MyTicToc as mt

# %% Domain
nIN = 101
zIN = np.linspace(-5, 0, num=nIN).reshape(nIN, 1)
zN = np.zeros(nIN - 1).reshape(nIN - 1, 1)
zN[0, 0] = zIN[0, 0]
zN[1:nIN - 2, 0] = (zIN[1:nIN - 2, 0] + zIN[2:nIN - 1, 0]) / 2
zN[nIN - 2, 0] = zIN[nIN - 1]
nN = np.shape(zN)[0]

ii = np.arange(0, nN - 1)
dzN = (zN[ii + 1, 0] - zN[ii, 0]).reshape(nN - 1, 1)
dzIN = (zIN[1:, 0] - zIN[0:-1, 0]).reshape(nIN - 1, 1)

# Model Dimensions Series
mDim = {'zN' : zN,
        'zIN' : zIN,
        'dzN' : dzN,
        'dzIN' : dzIN,
        'nN' : nN,
        'nIN' : nIN
        }
mDim = pd.Series(mDim)

# %% Read Rainfall and Evap Data and set up Boundary Functions

# read meteo data
meteo_data  = pd.read_excel('WieringermeerData_Meteo.xlsx')
meteo_data['num_date'] = meteo_data['datetime'].astype(np.int64)/(1e9*3600*24)
meteo_data.set_index('datetime',inplace=True)

# set simulation time to numeric dates from boudary data...
t_range = meteo_data['num_date'][5844:-1] #5479 for 2018
taxis = meteo_data.index[5844:-1]

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

# %% Series including Soil and Boundary Parameters
    # Clay/Loam Soil. Properties from Chapter 2 M&H Textbook

# Soil Parameters
sPar = {'a': np.ones(np.shape(zN)) * 0.019*100,  # alpha[1/m]
        'n': np.ones(np.shape(zN)) * 3.0,  # n[-]
        'm': np.ones(np.shape(zN)) * (1 - 1 / 3.0),  # m = 1-1/n[-]
        'thetas': np.ones(np.shape(zN)) * 0.41,  # saturated water content = phi,porosity
        'thetar': np.ones(np.shape(zN)) * 0.095,  # residual water content
        'KSat': np.ones(np.shape(zN)) * 2, # 6.2,  # [m/day]                 
        'Cv': 1.0e-8,  # compressibility of compact sand [1/Pa]
        'betaW': 4.5e-10,  # compressibility of water 1/Pa
        'rhoW': 1000,  # density of water kg/m3
        'g': 9.81,  # gravitational constant m/s2
        'h1': np.ones(np.shape(zN))*0,
        'h2': np.ones(np.shape(zN))*-1,
        'h3': np.ones(np.shape(zN))*-8.5,
        'h4': np.ones(np.shape(zN))*-15,
        }
sPar = pd.Series(sPar)

# Boundary Parameters
bPar = {'topBndFuncWat': BndqWatTop, #topBndFuncWat(t,bPar)
        # ARE THERE ROOTS PRESENT? Yes roots: enter 'affirmative' /  No roots: enter 'negative'
        'rootpresence': 'affirmative',
        'potEv': pEV,
        'meteo_data': meteo_data,
        'qTop': -0.01,  # top flux
        'bottomTypeWat': 'Robin', # Robin condition or Gravity condition
        'kRobBotWat': 0.10,  # Robin resistance term for bottom
        'hwBotBnd': -1.0,  # pressure head at lower boundary
        }
bPar = pd.Series(bPar)

# %%Solving

# Initialize
zRef = -3 # depth of water table
hwIni = zRef - zN

# Time Discretization
tOut2 = np.linspace(t_range[0],t_range[364],365)  # time
tOut = tOut2

print('Solving unsaturated water flow problem')
mt.tic()
hwODE = RWU.IntegrateWF(tOut2, hwIni, sPar, mDim, bPar)
mt.toc()

tOut2 = hwODE.t
#theta_sim = RWU.thFun(hwODE,sPar)
qw_sim = RWU.WaterFlux(tOut,hwODE.y,sPar,mDim,bPar)
S_sim = RWU.s_root(tOut,hwODE.y, sPar, mDim, bPar)

# %% Plot
plt.close('all')
# Pressure head for different depths as a function of time. Every 10th layer
fig1, ax1 = plt.subplots(figsize=(7, 4))

for ii in np.arange(0, nN, 2):
    ax1.plot(taxis, hwODE.y[ii, :], '-')

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
#qwODE = np.zeros(np.shape(hwODE.y))
for ii in np.arange(0, hwODE.t.size, 1):
    hwTmp = hwODE.y[:, ii].reshape(zN.shape)
    thODE[:, ii] = RWU.theta(hwTmp, sPar).reshape(1, nN)
  #  qwODE[:, ii] = RWU.DivWaterFlux(ii, hwTmp, sPar, mDim, bPar).reshape(1, nN)

fig3, ax3 = plt.subplots(figsize=(7, 7))
for ii in np.arange(0, hwODE.t.size, 1):
    ax3.plot(thODE[:, ii], zN[:, 0], '-')

ax3.grid(b=True)
ax3.set_xlabel('water content [-]')
ax3.set_ylabel('depth [m]')
    
fig4, ax4 = plt.subplots(figsize=(7, 4))
#for ii in np.arange(0, hwODE.t.size, 1):
   # ax4.plot(qwODE[:, ii], zN[:, 0], '-')
# plot the flux as a function of time for diff depths
ax4.plot(zN,qw_sim[0:-1], '-')
ax4.set_ylabel('water flux [m/d]')
ax4.set_xlabel('depth [m]')

# fig5, ax5 = plt.subplots(figsize=(7, 4))

# # plot the flux as a function of time for diff depths
# ax5.plot(S_sim, '-')
# ax5.set_ylabel('water uptake by roots [m/d]')
# ax5.set_xlabel('depth [m]')

plt.show()

