# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:11:54 2021

@author: emmalittle
"""

# %% 
import numpy as np
import pandas as pd
import MyTicToc as mt
import matplotlib.pyplot as plt
import WaterFlow as WF

# Definition of the notes and internodes:
# Domain
nIN = 101   # Number of internodes.
# soil profile of one meter (note: original soil profile was 15 m for the heat flow problem)
zIN = np.linspace(-1, 0, num=nIN).reshape(nIN, 1)   # defining internodes
# nIN = np.shape(zIN)[0]
zN = np.zeros(nIN - 1).reshape(nIN - 1, 1)
zN[0, 0] = zIN[0, 0]                                # Shape = number of nodes, empty arrays
zN[1:nIN - 2, 0] = (zIN[1:nIN - 2, 0] + zIN[2:nIN - 1, 0]) / 2
zN[nIN - 2, 0] = zIN[nIN - 1]     # Defining the height of each node
nN = np.shape(zN)[0]                # number of nodes

ii = np.arange(0, nN - 1)
dzN = (zN[ii + 1, 0] - zN[ii, 0]).reshape(nN - 1, 1)    # height of each node (like dx in dy/dx)
dzIN = (zIN[1:, 0] - zIN[0:-1, 0]).reshape(nIN - 1, 1)  # distance between internodes

mDim = {'zN' : zN,
        'zIN' : zIN,
        'dzN' : dzN,
        'dzIN' : dzIN,
        'nN' : nN,
        'nIN' : nIN
        }
mDim = pd.Series(mDim)

#read meteo data
meteo_data  = pd.read_excel('WieringermeerData_Meteo.xlsx')
meteo_data['num_date'] = meteo_data['datetime'].astype(np.int64)/(1e9*3600*24)
meteo_data.set_index('datetime',inplace=True)

# set simulation time to numeric dates from boudary data...
t_range = meteo_data['num_date'][:-1]
taxis = meteo_data.index[:-1]

def BndQTop(t, bPar): 
    #bndTop = bPar.Topflow + 0.1 * bPar.Topflow * np.cos(t*0.1)     # Varying inflow between 0.1 +/- Topflow m^3/s ? (not sure about units)
    #bndTop = bPar.Topflow * (t > bPar.tWMin) * (t < bPar.tWMax) * (0.1 * np.abs( np.cos(t*.1)))
    bndTop = bPar.Topflow 
   
    # if np.size(t)==1:
    #     t = np.array([t])
    # bndTop = np.zeros((len(t)))
    
    # for ii in range(len(t)):
    #     xy, md_ind, t_ind = np.intersect1d(bPar.meteo_data['num_date'], np.ceil(t[ii]), return_indices=True)
    #     rf = bPar.meteo_data['rain_station'].iloc[md_ind].values
    #     bndTop[ii] = -rf
    return bndTop

def pEV(t, bPar,sPar):   # Potential Evaporation for use in Root Sink Equation
    if np.size(t)==1:
        t = np.array([t])
    potEv = np.zeros((len(t)))
    
    for ii in range(len(t)):
        xy, md_ind, t_ind = np.intersect1d(bPar.meteo_data['num_date'], np.ceil(t[ii]), return_indices=True)
        evap = bPar.meteo_data['pEV'].iloc[md_ind].values
        potEv[ii] = evap
    return potEv

rhow = 1000     # [kg/m3] density of water
g = 9.81        # [m/s/s] gravitational constant

# Definition of the Boundary Parameters: collect boundary parameters in a named tuple boundpar. Has to have something to do with flow rate
bPar = {'Topflow': -0.01,    # Top, placeholder value    #no flow = 0, otherwise e.g. -0.1
        'meteo_data': meteo_data,
        'bottomTypeWat': 'Gravity',    #Robin condition or Gravity condition
        'kRobBotWat': 0.1, # [1/d] Robin resistance term for bottom
        'hwBotBnd': -1.0,  # pressure head at lower boundary
        'tWMin': 50,  # For testing, decide ourselves what we want this to be ToDo
        'tWMax': 375,
        'TopBd': BndQTop,
        }
bPar = pd.Series(bPar)

# Soil parameters - fixed. Sandy clay loam properties from Chap 2 M&H
sPar = {'n': np.ones(np.shape(zN))*3 ,# *np.ones(np.shape(zN)),              # Do we need the np.ones(np.shape(zN)) term? Doesn't seem like it does much
        'm': np.ones(np.shape(zN))*2/3 ,# *np.ones(np.shape(zN)),
        'alpha': np.ones(np.shape(zN))*2 ,# *np.ones(np.shape(zN)),     # recommended change by Timo (before 0.059 which he said was way too low)
        'theta_res': np.ones(np.shape(zN))*0.03 ,# *np.ones(np.shape(zN)),   # theta_wir, M&H Chapter 2
        'theta_sat':np.ones(np.shape(zN))*0.4 ,# *np.ones(np.shape(zN)), # phi, M&H Chapter 2
        'phi': 0.39 ,# *np.ones(np.shape(zN)), # porosity (same as theta_saturated)
        'Ksat': np.ones(np.shape(zN))*0.05 ,# *np.ones(np.shape(zN)), # m/d - Timo says this value is high (31.4), adjusted. Next assignment: Ksat = ksat*rhow*g/viscosity ,and viscosity will be function of T
        #'S_Sw': 4e-10*rhow*g,# *np.ones(np.shape(zN)), # compressibility of water
        'beta_wat': 4.5e-10,
        'Cv': 1.0e-8,  # compressibility of compact sand [1/Pa],
        'potEv': pEV
        }
sPar = pd.Series(sPar)


#%%  

# Define Initial Conditions
zRef = -0.5 # depth of water table
hwIni = zRef - zN


# %% Solve problem
# Time Discretization
# t_out = np.logspace(-14, np.log10(5*365), num=365)  # time
#t_out = np.logspace(-14, np.log10(5*365),num=365) # Logarithmic! If we want days
# t_out = np.linspace(0, 10*120, num=120*5)
t_out = np.linspace(t_range[0],t_range[365],5*365)

print('Solving unsaturated water flow problem')
mt.tic()
hwODE = WF.IntegrateWF(t_out, hwIni, sPar, mDim, bPar)
mt.toc()

t_out = hwODE.t

# In[5:] Plot results...
plt.close('all')
fig1, ax1 = plt.subplots(figsize=(7, 4))
# plot the pressure head for different depths as a function of time
# in this case we plot every layer.
for ii in np.arange(0, nN, 20):
   ax1.plot(hwODE.t, hwODE.y[ii, :], '-')

ax1.grid(b=True)
ax1.set_ylabel('pressure head [m]')
ax1.set_xlabel('time [d]') 
ax1.set_title('Water pressure head vs time, time dependend top flux and Robin boundary condition')


#plot pressure head as a function of depth. Here we plot every time step
fig2, ax2 = plt.subplots(figsize=(7, 7))
for ii in np.arange(0, hwODE.t.size, 1):
    ax2.plot(hwODE.y[:, ii], zN[:, 0], '-')

ax2.grid(b=True)
ax2.set_xlabel('pressure head [m]')
ax2.set_ylabel('depth [m]')
ax2.set_title('Water pressure head vs depth')

#plot water content as a function of depth over time
thODE = np.zeros(np.shape(hwODE.y))
for ii in np.arange(0, hwODE.t.size, 1):
    hwTmp = hwODE.y[:, ii].reshape(zN.shape)
    thODE[:, ii] = WF.thetaw(hwTmp, sPar).reshape(1, nN)

fig3, ax3 = plt.subplots(figsize=(7, 7))
for ii in np.arange(0, hwODE.t.size, 1):
    ax3.plot(thODE[:, ii], zN[:, 0], '-')

ax3.grid(b=True)
ax3.set_xlabel('water content [-]')
ax3.set_ylabel('depth [m]')
ax3.set_title('Water content vs depth, time dependend top flux and Robin boundary condition')

fig4, ax4 = plt.subplots(figsize=(7, 7))
for ii in np.arange(0, hwODE.t.size, 1):
    ax4.plot(pEV(ii,bPar), '-')

ax4.grid(b=True)
ax4.set_xlabel('water content [-]')
ax4.set_ylabel('depth [m]')
ax4.set_title('W')

# Possibly plotting the water flux
# fig4, ax4 = plt.subplots(figsize=(7, 4))
# qODE = np.zeros(np.shape(hwODE.y))
# for ii in np.arange(0, hwODE.t.size, 1):
#     qODE[:,ii] = WF.WaterFlux(t_out[ii],hwTmp,sPar,mDim,bPar)
# # plot fluxes after 2nd output time (initial rate is extreme due to initial conditions)
# ax4.plot(t_out, qODE[ii,:], '-')
# ax4.set_title('Water Flux ')
# ax4.set_ylabel('depth [m]')
# ax4.set_xlabel('water flow [m/d]')
# ax4.legend(zN[ii])
# TRYING AGAIN!!
# qODE = np.zeros([51, 51])
# for ii in np.arange(0, hwODE.t.size, 1):
#     hwTmp = hwODE.y[:, ii].reshape(zN.shape)
#     qODE[:, ii] = WF.WaterFlux(t_out[ii], hwTmp, sPar, mDim, bPar).reshape(365, nIN) #def WaterFlux(t, hw, sPar, mDim, bPar)


# fig4, ax4 = plt.subplots(figsize=(7, 7))
# for ii in np.arange(0, hwODE.t.size, 1):
#     ax4.plot(qODE[:, ii], zIN[:, 0], '-')

# ax4.grid(b=True)
# ax4.set_xlabel('Water Flux [m/d]')
# ax4.set_ylabel('depth [m]')
# ax4.set_title('Water Flux with Flow and Robin boundary')

# fig5, ax5 = plt.subplots(figsize=(7, 7))
# for ii in np.arange(0, hwODE.t.size, 1):
#     ax4.plot(BndQTop[:, ii], zIN[:, 0], '-')

# ax5.grid(b=True)
# ax5.set_xlabel('Water Flux [m/d]')
# ax5.set_ylabel('depth [m]')
# ax5.set_title('Rainfall')

# fig6, ax6 = plt.subplots(figsize=(7, 7))
# for ii in np.arange(0, hwODE.t.size, 1):
#     ax4.plot(pEV[ii], zIN[:, 0], '-')

# ax6.grid(b=True)
# ax6.set_xlabel('Evap Water Flux [m/d]')
# ax6.set_ylabel('depth [m]')
# ax6.set_title('Rainfall')
# plt.show()


# # %% Testing Flux Sections
# fig5, ax5 = plt.subplots(figsize=(7, 7))
# for ii in np.arange(0, hwODE.t.size, 1):
#     ax5.plot(qODE[:, ii], zIN[:, 0], '-')

# ax5.grid(b=True)
# ax5.set_xlabel('Water Flux [m/d]')
# ax5.set_ylabel('depth [m]')
# ax5.set_title('Water Flux')



