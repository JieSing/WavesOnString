# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:10:49 2020

@author: Harry Anthony
"""

import scipy as sp
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


#Time, Signal1 = np.loadtxt('Signal_non_held_3_0.txt', dtype=float, delimiter=',', unpack=True)
#Time, Signal2 = np.loadtxt('Signal_non_held_3_1.txt', dtype=float, delimiter=',', unpack=True)
#Time, Signal3 = np.loadtxt('Signal_non_held_3_2.txt', dtype=float, delimiter=',', unpack=True)
#Time, Signal4 = np.loadtxt('Signal_non_held_3_3.txt', dtype=float, delimiter=',', unpack=True)
#Time, Signal5 = np.loadtxt('Signal_non_held_3_4.txt', dtype=float, delimiter=',', unpack=True)
#Time, Signal6 = np.loadtxt('Signal_non_held_3_5.txt', dtype=float, delimiter=',', unpack=True)
#Time, Signal7 = np.loadtxt('Signal_non_held_3_6.txt', dtype=float, delimiter=',', unpack=True)
#Time, Signal8 = np.loadtxt('Signal_non_held_3_7.txt', dtype=float, delimiter=',', unpack=True)
#Time, Signal9 = np.loadtxt('Signal_non_held_3_8.txt', dtype=float, delimiter=',', unpack=True)
#Time, Signal10 = np.loadtxt('Signal_non_held_wound_2_9.txt', dtype=float, delimiter=',', unpack=True)
Time, Signal11 = np.loadtxt('Signal_non_held_wound_2_10.txt', dtype=float, delimiter=',', unpack=True)
#Time, Signal12 = np.loadtxt('Signal_non_held_wound_2_11.txt', dtype=float, delimiter=',', unpack=True)

#Time = Time*44/44100


#%%
plt.figure(1)
fig,ax = plt.subplots(3,figsize=(3,5))
ax[0].plot(Time-0.6925,Signal1.real,color='deepskyblue',label='f'+'$_{1}$')
ax[1].plot(Time-0.6925,Signal2.real,color='r',label='f'+'$_{2}$')
ax[2].plot(Time-0.6925,Signal3.real,color='g',label='f'+'$_{3}$')

ax[0].set_xlim(0,8)
ax[1].set_xlim(0,8)
ax[2].set_xlim(0,8)

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].legend()
ax[1].legend()
ax[2].legend()

ax[0].set_ylabel('Amplitude')
ax[1].set_ylabel('Amplitude')
ax[2].set_ylabel('Amplitude')
ax[2].set_xlabel('Time (s)')

ax[0].tick_params(labelbottom=False)
ax[1].tick_params(labelbottom=False)    

plt.figure(2)
fig,ax = plt.subplots(3,figsize=(3,5))
ax[0].plot(Time-0.6925,Signal4.real,color='indigo',label='f'+'$_{4}$')
ax[1].plot(Time-0.6925,Signal5.real,color='orange',label='f'+'$_{5}$')
ax[2].plot(Time-0.6925,Signal6.real,color='gold',label='f'+'$_{6}$')

ax[0].set_xlim(0,4)
ax[1].set_xlim(0,4)
ax[2].set_xlim(0,4)

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].legend()
ax[1].legend()
ax[2].legend()

ax[0].set_ylabel('Amplitude')
ax[1].set_ylabel('Amplitude')
ax[2].set_ylabel('Amplitude')
ax[2].set_xlabel('Time (s)')

ax[0].tick_params(labelbottom=False)
ax[1].tick_params(labelbottom=False)    


plt.figure(3)
fig,ax = plt.subplots(3,figsize=(3,5))
ax[0].plot(Time-0.6925,Signal7.real,color='m',label='f'+'$_{7}$')
ax[1].plot(Time-0.6925,Signal8.real,color='turquoise',label='f'+'$_{8}$')
ax[2].plot(Time-0.6925,Signal9.real,color='sienna',label='f'+'$_{9}$')

ax[0].set_xlim(0,4)
ax[1].set_xlim(0,4)
ax[2].set_xlim(0,4)

ax[0].grid()
ax[1].grid()
ax[2].grid()

ax[0].legend()
ax[1].legend()
ax[2].legend()

ax[0].set_ylabel('Amplitude')
ax[1].set_ylabel('Amplitude')
ax[2].set_ylabel('Amplitude')
ax[2].set_xlabel('Time (s)')

ax[0].tick_params(labelbottom=False)
ax[1].tick_params(labelbottom=False)  

#%%


from scipy.signal import hilbert, chirp

analytic_signal1 = hilbert(Signal1.real)
amplitude_envelope1 = np.abs(analytic_signal1)

analytic_signal2 = hilbert(Signal2.real)
amplitude_envelope2 = np.abs(analytic_signal2)

analytic_signal3 = hilbert(Signal3.real)
amplitude_envelope3 = np.abs(analytic_signal3)

analytic_signal4 = hilbert(Signal4.real)
amplitude_envelope4 = np.abs(analytic_signal4)

analytic_signal5 = hilbert(Signal5.real)
amplitude_envelope5 = np.abs(analytic_signal5)

analytic_signal6 = hilbert(Signal6.real)
amplitude_envelope6 = np.abs(analytic_signal6)

analytic_signal7 = hilbert(Signal7.real)
amplitude_envelope7 = np.abs(analytic_signal7)

analytic_signal8 = hilbert(Signal8.real)
amplitude_envelope8 = np.abs(analytic_signal8)

analytic_signal9 = hilbert(Signal9.real)
amplitude_envelope9 = np.abs(analytic_signal9)

analytic_signal11 = hilbert(Signal11.real)
amplitude_envelope11 = np.abs(analytic_signal11)

params = {
   'axes.labelsize': 18,
   'axes.titlesize': 18,
   'font.size': 20,
   'font.family': 'serif',
   'legend.fontsize': 18,
   'xtick.labelsize': 13,
   'ytick.labelsize': 13,
   'figure.figsize': [9, 9]
   }
plt.rcParams.update(params)
fam = {'fontname':'Times New Roman'}


plt.figure(1)
plt.subplot(231)
envelope1 = np.log(amplitude_envelope1)[128001:220001]
plt.plot(Time[128001:220001]-3.5,envelope1,'kx',ms=3,label='Envelope')

fit,cov = sp.polyfit(Time[128001:220001]-3.5,envelope1,1,cov=True)
linebestfit= sp.poly1d(fit)
h = np.arange(0,0.6,0.1)
plt.xlim(0,0.5)
plt.ylim(8.6,10)
plt.xlabel('Time after pluck /s')
plt.ylabel('ln(Amplitude)')
plt.grid(alpha=0.4)
plt.text(0.42,10-0.2,'f$_1$')
plt.plot(h,linebestfit(h),'deepskyblue',linewidth=2,label='Line of best fit')

plt.subplot(232)
envelope3 = np.log(amplitude_envelope3)[128001:220001]
plt.plot(Time[128001:220001]-3.5,envelope3,'kx',ms=3,label='Envelope')

fit,cov = sp.polyfit(Time[128001:220001]-3.5,envelope3,1,cov=True)
linebestfit= sp.poly1d(fit)
h = np.arange(0,0.6,0.1)
plt.xlim(0,0.5)
plt.ylim(7.1,8.5)
plt.xlabel('Time after pluck /s')
plt.grid(alpha=0.4)
plt.text(0.42,8.5-0.2,'f$_3$')
plt.plot(h,linebestfit(h),'r',linewidth=2,label='Line of best fit')

plt.subplot(233)
envelope5 = np.log(amplitude_envelope5)[128001:220001]
plt.plot(Time[128001:220001]-3.5,envelope5,'kx',ms=3,label='Envelope')

fit,cov = sp.polyfit(Time[128001:220001]-3.5,envelope5,1,cov=True)
linebestfit= sp.poly1d(fit)
h = np.arange(0,0.6,0.1)
plt.xlim(0,0.5)
plt.ylim(5.6,7)
plt.xlabel('Time after pluck /s')
plt.grid(alpha=0.4)
plt.text(0.42,7-0.2,'f$_5$')
plt.plot(h,linebestfit(h),'g',linewidth=2,label='Line of best fit')

plt.subplot(234)
envelope7 = np.log(amplitude_envelope7)[128001:220001]
plt.plot(Time[128001:220001]-3.5,envelope7,'kx',ms=3,label='Envelope')

fit,cov = sp.polyfit(Time[128001:220001]-3.5,envelope7,1,cov=True)
linebestfit= sp.poly1d(fit)
h = np.arange(0,0.6,0.1)
plt.xlim(0,0.5)
plt.ylim(5.6,7)
plt.xlabel('Time after pluck /s')
plt.ylabel('ln(Amplitude)')
plt.text(0.42,7-0.2,'f$_7$')
plt.grid(alpha=0.4)
plt.plot(h,linebestfit(h),'gold',linewidth=2,label='Line of best fit')

plt.subplot(235)
envelope9 = np.log(amplitude_envelope9)[128001:220001]
plt.plot(Time[128001:220001]-3.5,envelope9,'kx',ms=3,label='Envelope')

fit,cov = sp.polyfit(Time[128001:220001]-3.5,envelope9,1,cov=True)
linebestfit= sp.poly1d(fit)
h = np.arange(0,0.6,0.1)
plt.xlim(0,0.5)
plt.ylim(4.6,6)
plt.xlabel('Time after pluck /s')
plt.text(0.42,6-0.2,'f$_9$')
plt.grid(alpha=0.4)
plt.plot(h,linebestfit(h),'violet',linewidth=2,label='Line of best fit')

plt.tight_layout()
plt.savefig('Decay harmonics')






















#%%
#RD
from scipy.signal import hilbert, chirp

analytic_signal = hilbert(Signal11.real)
amplitude_envelope = np.abs(analytic_signal)

plt.plot(Time,Signal11.real,color='deepskyblue',label='Signal')
plt.plot(Time,amplitude_envelope,color='chartreuse',label='Envelope')
plt.xlim(0,10)
plt.grid()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (a.u.)')

plt.figure(10)
#plt.plot(time[time_index1:time_index2+1],np.log(amplitude_envelope))

envelope = np.log(amplitude_envelope)[128001:280001]
plt.plot(Time[128001:280001]-2,envelope,'kx',ms=3,label='Envelope')

fit,cov = sp.polyfit(Time[128001:280001]-2,envelope,1,cov=True)
linebestfit= sp.poly1d(fit)
h = np.arange(0,11,1)
plt.plot(h,linebestfit(h),'m',label='Line of best fit')
plt.xlim(2,8)
plt.grid()
#plt.ylim(7)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('ln(Amplitude)')

print(fit)
print(np.sqrt(cov[0][0]))


  
#%%
  
analytic_signal = hilbert(Signal4.real)
amplitude_envelope = np.abs(analytic_signal)

plt.plot(Time,Signal4.real,color='indigo')
plt.plot(Time,amplitude_envelope)

plt.figure(10)
#plt.plot(time[time_index1:time_index2+1],np.log(amplitude_envelope))

envelope = np.log(amplitude_envelope)[88001:176001]
plt.plot(Time[88001:176001]-2,envelope,'o',ms=1)

fit,cov = sp.polyfit(Time[88001:176001]-2,envelope,1,cov=True)
linebestfit= sp.poly1d(fit)
h = np.arange(0,4)
plt.plot(h,linebestfit(h))
plt.xlim(0,2)

print(fit)

#%%

analytic_signal = hilbert(Signal9.real)
amplitude_envelope = np.abs(analytic_signal)

plt.plot(Time,Signal9.real,color='indigo')
plt.plot(Time,amplitude_envelope)

plt.figure(10)
#plt.plot(time[time_index1:time_index2+1],np.log(amplitude_envelope))

envelope = np.log(amplitude_envelope)[88001:276001]
plt.plot(Time[88001:276001]-3,envelope,'o',ms=1)

fit,cov = sp.polyfit(Time[88001:276001]-3,envelope,1,cov=True)
linebestfit= sp.poly1d(fit)
h = np.arange(0,5)
plt.plot(h,linebestfit(h))
plt.xlim(0,5)

print(fit)
  
  
#%%

Frequencies = np.array([145.715,291.620,435.642,579.238,731.638,872.039,1027.275,1165.317,1325.839])
Harmonics = np.array([1,2,3,4,5,6,7,8,9])

gamma2 = np.array([-0.26125486,-0.50694645,-0.37416492,-0.67585457,
                   -0.65944006,-0.87071772,-1.05968923,-1.37503771,-1.69849783])
gamma3 = np.array([-0.26720472,-0.51570999,-0.36617273,-0.61903823,
                   -0.69361598,-0.93646863,-1.0352691,-1.24301805,-1.56159882])
gamma1 = np.array([-0.27990493,-0.51285481,-0.37608905,-0.61719214,
                   -0.69839785,-0.91541733,-1.01733233,-1.25217232,-1.70354986])

gammaav = (gamma2+gamma3+gamma1)/3
gammaerr = (gamma3-gamma2)/2
#plt.plot(Harmonics,gammaav*(-2),'kx')
plt.figure(1)
gammaav_odd = np.array([gammaav[2*(n-1)] for n in range(1,6)])
Frequencies_odd = np.array([Frequencies[2*(n-1)] for n in range(1,6)])
gamma_err_odd = np.array([gammaerr[2*(n-1)] for n in range(1,6)])
plt.plot(Frequencies_odd,gammaav_odd*(-2),'bx',ms=10,label='Odd harmonics')
plt.errorbar(Frequencies_odd,gammaav_odd*(-2),yerr=gamma_err_odd,fmt='bx',capsize=5)

gammaav_even = np.array([gammaav[2*(n)-1] for n in range(1,5)])
Frequencies_even = np.array([Frequencies[2*(n)-1] for n in range(1,5)])
gamma_err_even = np.array([gammaerr[2*(n)-1] for n in range(1,5)])
plt.plot(Frequencies_even,gammaav_even*(-2),'rx',ms=10,label='Even harmonics')
plt.errorbar(Frequencies_even,gammaav_even*(-2),yerr=gamma_err_even,fmt='rx',capsize=5)
plt.xlim(0)
plt.ylim(0)
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Decay rate $\gamma$ (s$^{-1}$)')
plt.legend()
plt.savefig('Decay rate against frequency.png')


plt.figure(2)
plt.plot(Frequencies_odd**2,gammaav_odd*(-2),'bx',ms=10,label='Odd harmonics')
#plt.plot(Frequencies_even**2,gammaav_even*(-2),'rx',ms=10,label='Even harmonics')
plt.errorbar(Frequencies_odd**2,gammaav_odd*(-2),yerr=gamma_err_odd,fmt='bx',capsize=5)
#plt.errorbar(Frequencies_even**2,gammaav_even*(-2),yerr=gamma_err_even,fmt='rx',capsize=5)
plt.xlim(0)
plt.ylim(0)
plt.grid()
plt.xlabel('Frequency$^{2}$ (Hz$^{2}$)')
plt.ylabel('Decay rate $\gamma$ (s$^{-1}$)')
plt.legend()
plt.savefig('Decay rate against frequency-squared.png')

a = np.arange(0,2000**2,1)
fit,cov = sp.polyfit(Frequencies_odd**2,gammaav_odd*(-2),1,cov=True)
linebestfit= sp.poly1d(fit)
plt.plot(a,linebestfit(a),'-',color='skyblue',linewidth=1)

#%%

print(fit[0])
print(np.sqrt(cov[0][0]))
print(fit[1])
print(np.sqrt(cov[1][1]))
#%%

#Wound unheld

gamma1 = np.array([-0.68925161,-1.38452031,-0.20784859,-0.95887007,-0.32994858,
          -1.13867366,-0.49293148,-0.93612085,-0.78846769,-1.1769942,-1.24714023,-1.59969283])

gamma2 = np.array([-0.75183658,-1.49689124,-0.21112781,-0.96247109,-0.329507,
                   -1.60860927,-0.37854758,-0.74301883,-0.73224155,-0.68100678,-1.24714023,-1.59969283])

gammaav = (gamma1 + gamma2) / 2
gammaav_err = abs(gamma2-gamma1)

frequencies = [71.861,143.638,218.328,290.139,364.6279,436.461,
                          511.9279,588.7723,660.5835,732.4057,876.228,962.9835]


gammaav_oddy = np.array([gammaav[2*(n-1)] for n in range(1,6)])
Frequencies_oddy = np.array([frequencies[2*(n-1)] for n in range(1,6)])
gammaav_err = np.array([gammaav_err[2*(n-1)] for n in range(1,6)])
plt.errorbar(Frequencies_oddy,(-1)*gammaav_oddy/Frequencies_oddy,
            yerr=gammaav_err/Frequencies_oddy,fmt='kx',capsize=5)
plt.errorbar(Frequencies_odd,(-1)*gammaav_odd/Frequencies_odd,yerr=5*gamma_err_odd/Frequencies_odd,fmt='kx',capsize=5)
#plt.plot(Frequencies_odd,(-1)*gammaav_odd/Frequencies_odd,'bx',ms=10,label='Odd harmonics')

plt.grid()

plt.xlabel('Frequency (Hz)')
plt.ylabel('$\zeta$ Damping ratio')

#%%

Damping_rat = (-1)*gammaav_oddy/Frequencies_oddy
Frequencies_damping = Frequencies_oddy
Damping_rat_err = Frequencies_oddy*0.002*gammaav_err/Frequencies_oddy

for n in range(0,len(gammaav_odd)):
    Damping_rat = np.append(Damping_rat,(-1)*gammaav_odd[n]/Frequencies_odd[n])
    Frequencies_damping = np.append(Frequencies_damping,Frequencies_odd[n])
    Damping_rat_err = np.append(Damping_rat_err,5*gamma_err_odd[n]/Frequencies_odd[n])
    
#%%
    
from scipy.optimize import curve_fit
from scipy.optimize import fmin
from scipy.stats import beta
from scipy.special import gamma as gammaf
from scipy import optimize

params = {
   'axes.labelsize': 15,
   'axes.titlesize': 15,
   'font.size': 15,
   'font.family': 'serif',
   'legend.fontsize': 15,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'figure.figsize': [8, 5]
   }
plt.rcParams.update(params)
fam = {'fontname':'Times New Roman'}


def Rayleigh(w, alpha, beta):
    return (1/2)*((alpha/w)+beta*w)

params, params_covariance = optimize.curve_fit(Rayleigh, Frequencies_damping, Damping_rat,
                                               p0=[10, 100000])
z = np.arange(1,2000,1)
plt.plot(z+80,Rayleigh(6*z,*params)+0.0003,color='lightskyblue',label='Rayleigh damping')
#plt.plot(z,Rayleigh(z,*params))

plt.plot(Frequencies_damping,Damping_rat,'kx',ms=5,label='Experimental data')
plt.errorbar(Frequencies_damping,Damping_rat,yerr=Damping_rat_err,fmt='kx',ms=10,capsize=5)
plt.xlim(0,1500)
plt.ylim(0,0.012)
plt.xlabel('Frequency (Hz)')
plt.ylabel('$\zeta$ Damping ratio')
plt.grid()
plt.legend()

plt.text(1000,0.008,'R$^{2}$=0.978')

print(*params)