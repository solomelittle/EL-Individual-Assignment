
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

# In[0:] Domain & Soil properties
nIN = 51
# soil profile until 15 meters depth
zIN = np.linspace(-2.0, 0, num=nIN).reshape(nIN, 1)
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
sPar = {'vGA': np.ones(np.shape(zN)) * 1 / 2.0,  # alpha[1/m]
        'vGN': np.ones(np.shape(zN)) * 2.0,  # n[-]
        'vGM': np.ones(np.shape(zN)) * (1 - 1 / 2.0),  # m = 1-1/n[-]
        'thS': np.ones(np.shape(zN)) * 0.4,  # saturated water content
        'thR': np.ones(np.shape(zN)) * 0.03,  # residual water content
        'KSat': np.ones(np.shape(zN)) * 0.25,  # [m/day]
        'vGE': 0.5,  # power factor for Mualem-van Genuchten
        'Cv': 1.0e-8,  # compressibility of compact sand [1/Pa]
        'viscRef': cfun.ViscosityWaterT(283.15),
        'qCont': qCont, # quartz content
        }
sPar = pd.Series(sPar)





# In[1:] Definition of the Boundary Parameters

# Read meteodata
meteo_data  = pd.read_excel('WieringermeerData_Meteo.xlsx')
meteo_data['num_date'] = meteo_data['datetime'].astype(np.int64)/(1e9*3600*24)
meteo_data.set_index('datetime',inplace=True)

# set simulation time to numeric dates from boudary data...
t_range = meteo_data['num_date'][:-1]
taxis = meteo_data.index[:-1]

# collect boundary parameters in a named tuple boundpar...
def BndTTop(t, bPar):
    if np.size(t)==1:
        t = np.array([t])
    bndT = np.zeros(len(t))
    for ii in range(len(t)):
        xy, md_ind, t_ind = np.intersect1d(bPar.meteo_data['num_date'], np.ceil(t[ii]), return_indices=True)
        topT = bPar.meteo_data['temp'].iloc[md_ind].values

        bndT[ii] = 273.15 + topT

    return bndT


def BndqWatTop(t, bPar):
    if np.size(t)==1:
        t = np.array([t])
    qBnd = np.zeros(len(t))

    for ii in range(len(t)):
        xy, md_ind, t_ind = np.intersect1d(bPar.meteo_data['num_date'], np.ceil(t[ii]), return_indices=True)
        rf = bPar.meteo_data['rain_station'].iloc[md_ind].values

        qBnd[ii] = -rf
    return qBnd



bPar = {'topBndFuncHeat': BndTTop,
        'meteo_data': meteo_data,
        'topCond': 'Robin',
        'lambdaRobTop': 1e9,
        'lambdaRobBot': 0,
        'TBndBot': 273.15 + 10,
        'topBndFuncWat': BndqWatTop, #topBndFuncWat(t,bPar)
        'bottomTypeWat': 'Robin', # Robin condition or Gravity condition
        'kRobBotWat': 0.05,  # Robin resistance term for bottom
        'hwBotBnd': 1.0,  # pressure head at lower boundary
        }

bPar = pd.Series(bPar)


# In[3:] Define Initial Conditions
zRef = -1.0 # depth of water table
hwIni = zRef - zN

TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K

sVecIni = np.concatenate([hwIni, TIni], axis=0)

# Time Discretization
tOut = np.linspace(t_range[0],t_range[365],365*5)
#tplot = taxis[0:50]
nOut = np.shape(tOut)[0]


nOut = len(tOut)
# tOut = np.sort(np.hstack((tOut1, bTime)))  # time

# copy initial vector to hw0. Apply squeeze to compress it to one dimension

mt.tic()
int_result = cfun.IntegrateCHWF(tOut, sVecIni, sPar, mDim, bPar)
mt.toc()

hWSim = int_result.y[0:nN]
TSim = int_result.y[nN:2*nN]
thSim = cfun.thFun(hWSim,sPar)
qWSim = cfun.WatFlux(tOut,hWSim,TSim,sPar,mDim,bPar)
qHSim = cfun.HeatFlux(tOut, TSim, hWSim, sPar, mDim, bPar)

#mt.tic()
#TOutPic, hwOutPic = himp.HalfImplicitPicar(tOut2, hw0, T0, sPar, mDim, bPar, tPar)
#mt.toc()

sns.set()
plt.close('all')
fig1, ax1 = plt.subplots(figsize=(7, 4))
ii = np.arange(nN-1, 0, -10)
ax1.plot(tOut, TSim[ii,].T, '-')
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

fig3, ax3 = plt.subplots(figsize=(7, 4))
# plot fluxes after 2nd output time (initial rate is extreme due to initial conditions)
ax3.plot(tOut, qHSim[ii,:].T, '-')
ax3.set_title('Heat Flux vs. depth (ODE)')
ax3.set_ylabel('depth [m]')
ax3.set_xlabel('temperature [J/m2]')
ax3.legend(zN[ii])

fig4, ax4 = plt.subplots(figsize=(7, 4))
# plot the pressure head for different depths as a function of time
# in this case we plot every 20th layer.
ax4.plot(tOut, hWSim[ii,:].T, '-')
ax4.set_ylabel('pressure head [m]')
ax4.set_xlabel('time [d]')

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

fig8, ax8 = plt.subplots(figsize=(7, 4))
# plot fluxes after 2nd output time (initial rate is extreme due to initial conditions)
ax8.plot(tOut, qWSim[ii,:].T, '-')
ax8.set_title('Water Flux ')
ax8.set_ylabel('depth [m]')
ax8.set_xlabel('water flow [m/d]')
ax8.legend(zN[ii])

fig1.savefig('./figures_scenarios/3_figure1.png')
fig2.savefig('./figures_scenarios/3_figure2.png')
fig3.savefig('./figures_scenarios/3_figure3.png')
fig4.savefig('./figures_scenarios/3_figure4.png')
fig5.savefig('./figures_scenarios/3_figure5.png')
fig6.savefig('./figures_scenarios/3_figure6.png')
fig7.savefig('./figures_scenarios/3_figure7.png')
fig8.savefig('./figures_scenarios/3_figure8.png')
# import shelve

# filename='/tmp/shelve.out'
# my_shelf = shelve.open(filename,'n') # 'n' for new

# for key in dir():
#     try:
#         my_shelf[key] = globals()[key]
#     except TypeError:
#         #
#         # __builtins__, my_shelf, and imported modules can not be shelved.
#         #
#         print('ERROR shelving: {0}'.format(key))
# my_shelf.close()

