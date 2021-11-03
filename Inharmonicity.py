# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:48:54 2020

@author: jsy18
"""

import scipy as sp
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


diameter = 0.9e-3
length = 60e-2
density = 8.91*1000
density_steel = 8050
E = 200e9
Tension = 1051e-3 * 9.81
b = (np.pi**3 * E * (diameter/2)**4)/(4*Tension*length**2)
Mass = density * length * np.pi * (diameter/2)**2
Mu = (Mass/length)*1e-2
velocity = np.sqrt(Tension/Mu)
freq_1 = velocity/(2*length)
print(freq_1)      #theoretical fundamental frequency
n = np.arange(1,10)
freq_1 = 145.715
n_freq = n * freq_1 * (1 + b*b*n*n)**(1/2)

Inharmonicity = (n_freq - n * freq_1) / (n * freq_1 )

expt_freq = [145.715,291.620,437.893,579.238,731.638,872.039,1027.275,1165.317,1325.839]

#expt_freq_2 = [144.69, 289.36, 437.65, 582.3, 731, 875.6, 1027.2, 1165.5, 1325.7]
#expt_Inharmonicity = (expt_freq - n * freq_1) / (n * freq_1 )

expt_Inharmonicity = []
for y in range(1,len(expt_freq)+1):
    expt_Inharmonicity.append((expt_freq[y-1]-y*freq_1)/(y*freq_1))
expt_Inharmonicity_odd = np.array([expt_Inharmonicity[2*(n-1)] for n in range(1,6)])
n_odd = np.arange(1,10,2)
    
plt.figure()
plt.plot(n, Inharmonicity, '+', color='b', label='Theoretical value')
plt.plot(n_odd, expt_Inharmonicity_odd, '+', color='r', label='Experimental value')
plt.ylabel("Inharmonicity")
plt.xlabel("Harmonic Number")

fit,cov = np.polyfit(n, Inharmonicity, 2, cov=True )
linebestfit= np.poly1d(fit)
plt.plot(n,linebestfit(n),color='c' )

fit2 = np.polyfit(n_odd, expt_Inharmonicity_odd, 2, cov=False )
linebestfit= np.poly1d(fit2)
h = np.arange(1,10,0.1)
plt.plot(h,linebestfit(h), color='r' )

plt.legend()
plt.grid()
plt.show()


#fit,cov = sp.polyfit(Time[88001:440001]-2,envelope,1,cov=True)
##plt.plot(n, m*n+cov)
#h = np.arange(0,10)
#plt.plot(h,linebestfit(h))
#plt.xlim(0,10)

