# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:34:43 2020

@author: Harry Anthony
"""

import scipy as sp
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

filename = "19-11-2020-wound_non_held1.wav" #edit filename as appropriate

# read audio samples
input_data = read(filename)
signal = input_data[1]
sampling_freq = input_data[0]
sampling_freq = 44100
time = np.arange(len(signal))/sampling_freq


window = np.hamming(len(signal)*2)
window = window[int(len(signal)):len(signal)*2]
signal = signal * window

def plot_data(start_time,end_time):
    #function to plot data between start_time and end_time
    
    time_index1 = time.tolist().index(start_time)
    time_index2 = time.tolist().index(end_time)
    plt.figure()
    plt.plot(time[time_index1:time_index2+1],signal[time_index1:time_index2+1])
    plt.ylabel("Amplitude [a.u.]")
    plt.xlabel("Time (s)")
    plt.title("Recorded Signal")
    plt.show()

def FT_data(data,sampling_rate):
    global freq, FTdata, freq_index1, freq_index2
    #function to calcuate and display absolute value of Fourier Transform
    
    freq = 0.5 * sampling_rate * np.linspace(-1.0, 1.0, len(data))
    FTdata = np.fft.fftshift(np.fft.fft(np.fft.fftshift(data)))
    
    freq_index1 = np.amin(np.where(freq >= 0))
    freq_index2 = np.amin(np.where(freq >= 10000))
    print(freq_index2)
    
    plt.figure()
    plt.plot(freq[freq_index1:freq_index2+1],abs(FTdata[freq_index1:freq_index2+1]))
    plt.ylabel("Magnitude [a.u.]")
    plt.xlabel('Frequency (nx71.861)Hz')
    plt.title("Absolute Value of Fourier Transform")
    plt.xlim(0,5)
    plt.xticks(np.arange(0,1000,71.861),[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
    plt.savefig('Wound_wire_Fourier.png')
    plt.show()

plot_data(time[0],time[-1]) #plot signal in time window defined by 2 values
FT_data(signal,sampling_freq) #Fourier Transfomr and plot absolute value



#%%

freq = 0.5 * sampling_freq * np.linspace(-1.0, 1.0, len(signal))
FTdata = np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal)))

#Non-wound string
#frequencies_to_isolate = [145.715,291.620,435.642,579.238,731.638,872.039,1027.275,1165.317,1325.839]
#Wound string
frequencies_to_isolate = [71.861,143.638,218.328,290.139,364.6279,436.461,511.9279,588.7723,660.5835,732.4057,876.228,962.9835]

z = 0
while z<len(frequencies_to_isolate):
    print(z)
    
    frequency_to_isolate = frequencies_to_isolate[z]
    
    negative_freq = 0
    while freq[negative_freq] < -frequency_to_isolate:
        negative_freq = negative_freq + 1

    positive_freq = 0
    while freq[positive_freq] < frequency_to_isolate:
        positive_freq = positive_freq + 1

    #Width of Gaussian window    
    Gaussian_width = 10000
    
    Gaussian_window = sp.signal.windows.gaussian(Gaussian_width,int(Gaussian_width/5))
    Fourier = FTdata[int(negative_freq-Gaussian_width/2):int(negative_freq+Gaussian_width/2)]
    Fourier = Fourier * Gaussian_window
    Fourier_pad = np.pad(Fourier,(int(negative_freq-Gaussian_width/2),int(
        positive_freq-negative_freq-Gaussian_width)
                              ),'constant',constant_values=(0))
    Fourier = FTdata[int(positive_freq-Gaussian_width/2):int(positive_freq+Gaussian_width/2)]
    Fourier = Fourier * Gaussian_window

    for x in range(0,len(Fourier)):
        Fourier_pad = np.append(Fourier_pad,Fourier[x])

    Fourier_pad = np.pad(Fourier_pad,(0,int(len(FTdata)-positive_freq-Gaussian_width/2)
                              ),'constant',constant_values=(0))
    
    iFTdata = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(Fourier_pad)))
    window = np.hamming(len(signal)*2)
    iFTdata = iFTdata * window[int(len(signal)):len(signal)*2]
    time_index1 = time.tolist().index(time[0])
    time_index2 = time.tolist().index(time[-1])
    
    Signal_F = iFTdata
    
    File_name = 'Signal_non_held_wound_1_'+str(z)+'.txt'
    myfile=open(File_name,'a')
    for v in range(0,len(Signal_F[time_index1:time_index2+1])):
        myfile.write(str(time[time_index1:time_index2+1][v])+','+str(Signal_F.real[
                time_index1:time_index2+1][v])+'\n')
    
    z = z+1
    

