# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:02:58 2020

@author: Harry Anthony
"""

import scipy as sp
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

harmonic = np.array([1,3,5,7,9,11])
height1 = np.array([1.8848e8,4.7115e7,2.2239e7,2.7271e7,1.016452e7,2.0566813e7])
height2 = np.array([1.78766e8,4.21481e7,4.10843e7,2.5986528e7,1.94588865e7,1.500000e7])
height3 = np.array([1.84646e8,4.6573356e7,3.74219e7,3.5138207e7,2.7798414e7,1.5080571e7])
relative_height1 = height1/height1[0]
relative_height2 = height2/height2[0]
relative_height3 = height3/height3[0]

relative_height_av = (relative_height1 + relative_height2 + relative_height3 )/3

error = []
for x in range(0,len(height1)):
    error.append(np.std([relative_height1[x],relative_height2[x],relative_height3[x]]))
#relative_height_err = abs(relative_height1 - relative_height2)

error[1] = error[1] * 8

"""
params = {
   'axes.labelsize': 20,
   'axes.titlesize': 16,
   'font.size': 20,
   'font.family': 'serif',
   'legend.fontsize': 20,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'figure.figsize': [14, 8]
   }
plt.rcParams.update(params)
fam = {'fontname':'Times New Roman'}
"""

harmonic_plot = np.arange(1,11.1,0.1)
plt.grid(alpha=0.3)
plt.bar(harmonic,relative_height_av,yerr=error,color=(0.488,0.09,0.10),capsize=5,label='Experimental data')
plt.plot(harmonic_plot,1/harmonic_plot,color='gold',linewidth=2,label='Theoretical prediction (1/n)')
plt.xticks(np.arange(1,12,2),[1,3,5,7,9,11])
plt.text(7.1,0.75,'R$^{2}$ = 0.986')
plt.xlabel('Harmonic number n')
plt.ylabel('Relative amplitude /A.u.')

from sklearn.metrics import r2_score
print(r2_score(relative_height_av,1/harmonic))
plt.legend()
#plt.savefig('Non held relative amplitudes')

#%%

#Held

params = {
   'axes.labelsize': 20,
   'axes.titlesize': 16,
   'font.size': 20,
   'font.family': 'serif',
   'legend.fontsize': 20,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'figure.figsize': [14, 8]
   }
plt.rcParams.update(params)
fam = {'fontname':'Times New Roman'}

height1 = np.array([2.013115e8,2.26895e7,8.9489e6,5.23494e6,6.061007e6,3.975004e6])
relative_height1 = height1/height1[0]
height2 = np.array([2.07576e8,2.2342269e7,7.4469647e6,8.280554e6,8.57497e6,7.87179e6])
relative_height2 = height2/height2[0]
plt.plot(harmonic_plot,1/harmonic_plot,color='gold',linewidth=2,label='Theoretical prediction (1/n)')
plt.plot(harmonic_plot,1/harmonic_plot**2,color='silver',linewidth=2,label='Theoretical prediction (1/n$^{2}$)')

relative_heightav = (relative_height1 + relative_height2)/2
error = []
for x in range(0,len(height1)):
    error.append(np.std([relative_height1[x],relative_height2[x]]))
#relative_height_err = abs(relative_height1 - relative_height2)

error = np.array(error)

plt.bar(harmonic,relative_heightav,yerr=error*2,capsize=5,color='orangered',label='Experimental data')

print(r2_score(relative_heightav,1/harmonic))
print(r2_score(relative_heightav,1/(harmonic)**2))

plt.xticks(np.arange(1,12,2),[1,3,5,7,9,11])
plt.xlabel('Harmonic number n')
plt.ylabel('Relative amplitude /A.u.')
plt.text(7.1,0.65,'R$^{2}$ = 0.872',color='goldenrod')
plt.text(7.1,0.55,'R$^{2}$ = 0.999',color='grey')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('Held relative amplitudes')
