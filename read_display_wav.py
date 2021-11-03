#select .wav file, read and plot
#calculate Fourier Transform and display absolute value 0-500 Hz
import scipy as sp
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

filename = "19-11-2020-wound_non_held2.wav" #edit filename as appropriate

# read audio samples
input_data = read(filename)
signal = input_data[1]
print(signal)
sampling_freq = input_data[0]
sampling_freq = 44100
time = np.arange(len(signal))/sampling_freq
sampling_freq = 1/(time[1]-time[0])
#print(diff)

signal2 = np.array([])
time2 = np.array([])

#For isolating time
#for x in range(0,len(time)):
#    if time[x] >= 5.5 and time[x] < 6.5:
#        time2 = np.append(time2,time[x])
#        signal2 = np.append(signal2,signal[x])

#time = time2
#signal = signal2



#plt.figure(1)
window = np.hamming(len(signal)*2)
#a = np.arange(0,3969000,1)
#plt.plot(a,window[int(3969000):3969000*2])
window = window[int(len(signal)):len(signal)*2]

#plt.figure(2)
#plt.plot(time,signal*window)

#signal = signal * window

def plot_data(start_time,end_time):
    #function to plot data between start_time and end_time
    
    time_index1 = time.tolist().index(start_time)
    time_index2 = time.tolist().index(end_time)
    plt.figure()
    plt.plot(time[time_index1:time_index2+1]-1.1,signal[time_index1:time_index2+1],'x',color=(0.488,0.09,0.10))
    plt.xlim(0,10)
    plt.ylabel("Amplitude /a.u.")
    plt.xlabel("Time after pluck /s")
    #plt.title("Recorded Signal")
    plt.grid()
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
    plt.plot(freq[freq_index1:freq_index2+1],abs(FTdata[freq_index1:freq_index2+1]),color=(0.488,0.09,0.10))
    plt.ylabel("Magnitude /a.u.")
    #plt.xlabel("Frequency (n x 145.714) /Hz")
    plt.xlabel('Frequency (nx71.861) /Hz')
    plt.title("Absolute Value of Fourier Transform")
    
    plt.xticks(np.arange(0,1000,71.861),[0,1,2,3,4,5,6,7,8,9,10])
    plt.xlim(0,700)
    #plt.xticks(np.arange(0,2000,145.715),[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
    #plt.vlines(freq[freq_index1:freq_index2+1],0,abs(FTdata[freq_index1:freq_index2+1]))
    plt.savefig('Wound_wire_Fourier.png')
    plt.show()

plot_data(time[0],time[-1]) #plot signal in time window defined by 2 values
FT_data(signal,sampling_freq) #Fourier Transfomr and plot absolute value


#%%

freq = 0.5 * sampling_freq * np.linspace(-1.0, 1.0, len(signal))
FTdata = np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal)))

#Frequecy to isolate
frequency_to_isolate = 145.715
#1 = 145.715 (0.145715)
#2 = 291.620 (0.29162)
#3 = 435.642
#4 = 579.238
#5 = 731.638
#6 = 872.039

#Wound frequencies
# f =[71.861,143.638,218.328,290.139,364.6279,436.461,511.9279,588.7723,660.5835,732.4057,876.228,962.9835]

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

plt.plot(freq,Fourier_pad)

#%%

plt.figure(1)
#plt.plot(freq,FTdata.imag)
plt.plot(freq,abs(Fourier_pad))
plt.xlim(0,5)
plt.xticks(np.arange(0,2000,145.715),[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
plt.ylabel("Magnitude [a.u.]")
plt.xlabel("Frequency (n x 145.714)Hz")

#plt.figure(2)
#plt.plot(freq,FTdata.real)


#%%
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


iFTdata = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(Fourier_pad)))
window = np.hamming(len(signal)*2)
iFTdata = iFTdata * window[int(3969000):3969000*2]
time_index1 = time.tolist().index(time[0])
time_index2 = time.tolist().index(time[-1])
plt.plot(time[time_index1:time_index2+1],iFTdata[time_index1:time_index2+1])
plt.yscale('log')
plt.ylabel("Amplitude [a.u.]")
plt.xlabel("Time (s)")
plt.title("Recorded Signal")

Signal6 = iFTdata

#%%

myfile=open('Signal_6_2.txt','a')
for x in range(0,len(Signal6[time_index1:time_index2+1])):
    myfile.write(str(time[time_index1:time_index2+1][x])+','+str(Signal6.real[
                time_index1:time_index2+1][x])+'\n')

#%%

freq = 0.5 * sampling_freq * np.linspace(-1.0, 1.0, len(signal))
FTdata = np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal)))

frequencies_to_isolate = [145.715,291.620,435.642,579.238,731.638,872.039,1027.275,1165.317,1325.839]
#frequencies_to_isolate = [71.861,143.638,218.328,290.139,364.6279,436.461,511.9279,588.7723,660.5835,732.4057,876.228,962.9835]

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
    
    File_name = 'Signal_non_held_non_wound_horizontal_2_'+str(z)+'.txt'
    myfile=open(File_name,'a')
    for v in range(0,len(Signal_F[time_index1:time_index2+1])):
        myfile.write(str(time[time_index1:time_index2+1][v])+','+str(Signal_F.real[
                time_index1:time_index2+1][v])+'\n')
    
    z = z+1
    

#%%

time_index1 = time.tolist().index(time[0])
time_index2 = time.tolist().index(time[-1])

#plt.plot(time[time_index1:time_index2+1],Signal1[time_index1:time_index2+1])
#plt.plot(time[time_index1:time_index2+1],Signal2[time_index1:time_index2+1])
#plt.plot(time[time_index1:time_index2+1],Signal3[time_index1:time_index2+1])
#plt.plot(time[time_index1:time_index2+1],Signal4[time_index1:time_index2+1])
#plt.plot(time[time_index1:time_index2+1],Signal5[time_index1:time_index2+1])
plt.plot(time[time_index1:time_index2+1],Signal6[time_index1:time_index2+1])
plt.xlim(1550,7000)
plt.yscale('log')

#%%

filename = "signal_non_wound_horizontal2.wav" #edit filename as appropriate

# read audio samples
input_data = read(filename)
signal = input_data[1]
sampling_freq = input_data[0]
sampling_freq = 44100
time = np.arange(len(signal))/sampling_freq

signal1 = []
time1 = []
signal2 = []
time2 = []
signal3 = []
time3 = []
signal4 = []
time4 = []
signal5 = []
time5 = []

signal6 = []
time6 = []
signal7 = []
time7 = []
signal8 = []
time8 = []
signal9 = []
time9 = []
signal10 = []
time10 = []

signal11 = []
time11 = []
signal12 = []
time12 = []
signal13 = []
time13 = []
signal14 = []
time14 = []
signal15 = []
time15 = []

signal16 = []
time16 = []
signal17 = []
time17 = []
signal18 = []
time18 = []
signal19 = []
time19 = []
signal20 = []
time20 = []

signal21 = []
time21 = []
signal22 = []
time22 = []
signal23 = []
time23 = []
signal24 = []
time24 = []
signal25 = []
time25 = []

signal26 = []
time26 = []
signal27 = []
time27 = []
signal28 = []
time28 = []
signal29 = []
time29 = []
signal30 = []
time30 = []

signal31 = []
time31 = []
signal32 = []
time32 = []
signal33 = []
time33 = []
signal34 = []
time34 = []
signal35 = []
time35 = []

signal36 = []
time36 = []
signal37 = []
time37 = []
signal38 = []
time38 = []
signal39 = []
time39 = []
signal40 = []
time40 = []

signal41 = []
time41 = []
signal42 = []
time42 = []
signal43 = []
time43 = []
signal44 = []
time44 = []
signal45 = []
time45 = []

for y in range(1,46):
    time1 = 2*(y-1)
    time2 = time1+2
    print(y)
    signal_name = globals()['signal'+str(y)]
    #array_time = globals()['time'+str(y)]

    
    for x in range(0,len(time)):
        if time[x] >= time1 and time[x] < time2:
            signal_name.append(signal[x])
            #array_time.append(time[x])

#%%
            
print(len(signal1))
print(time[len(signal1)*16:len(signal1)*17][-1])
#print(len(signal2)

FTdata1 = []
FTdata2 = []
FTdata3 = []
FTdata4 = []
FTdata5 = []

FTdata6 = []
FTdata7 = []
FTdata8 = []
FTdata9 = []
FTdata10 = []

FTdata11 = []
FTdata12 = []
FTdata13 = []
FTdata14 = []
FTdata15 = []

FTdata16 = []
FTdata17 = []
FTdata18 = []
FTdata19 = []
FTdata20 = []

FTdataz = []

freq = 0.5 * sampling_freq * np.linspace(-1.0, 1.0, len(signal1))

freq_index1 = np.amin(np.where(freq >= 0))
freq_index2 = np.amin(np.where(freq >= 10000))


for n in range(1,21):
    signal_name = globals()['signal'+str(n)]
    FT_name = globals()['FTdata'+str(n)]
    FTdata = np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal_name)))
    FT_name.append(FTdata)
    FTdataz.append(abs(FTdata[freq_index1:freq_index2+1]))


#%%

from mpl_toolkits import mplot3d

freq = 0.5 * sampling_freq * np.linspace(-1.0, 1.0, len(signal1))

freq_index1 = np.amin(np.where(freq >= 0))
freq_index2 = np.amin(np.where(freq >= 10000))

print(len(FTdata1[0]))

plt.plot(freq[freq_index1:freq_index2+1],abs(FTdata1[0][freq_index1:freq_index2+1]))

#FTdata = np.fft.fftshift(np.fft.fft(np.fft.fftshift(data)))
    
#FT_data(signal1,sampling_freq)

#%%

print(len(FTdataz))
freqz = [freq[freq_index1:freq_index2+1]]*20

#X,Y = np.meshgrid(freqz,FTdataz,sparse=True)

#%%
from mpl_toolkits.mplot3d import axes3d

params = {
   'axes.labelsize': 10,
   'axes.titlesize': 15,
   'font.size': 10,
   'font.family': 'serif',
   'legend.fontsize': 15,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [13, 15]
   }


a = np.arange(0,20001,1)
a = np.array([a]*20)
print(a)

a = [[1]*20001,[3]*20001,[5]*20001,[7]*20001,[9]*20001,[11]*20001,[13]*20001,
     [15]*20001,[17]*20001,[19]*20001,[21]*20001,
     [23]*20001,[25]*20001,[27]*20001,[29]*20001,[31]*20001,[33]*20001,
     [35]*20001,[37]*20001,[39]*20001]
fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")


freqz = np.array([freq[freq_index1:freq_index2+1]]*20)
#FTdataz = np.array(FTdataz)
#ax.contour3D(freqz,a,FTdataz)
#ax.contour3D(freqz,FTdataz,a,60)
plot = ax.contour3D(a,freqz,FTdataz,100,cmap='cividis')
ax.view_init(elev=31., azim=-160)
#ax.dist=11




ax.xaxis.pane.fill = False
ax.xaxis.set_pane_color((0.5,1,1))
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.fill = False
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.fill = False
ax.zaxis.pane.set_edgecolor('white')
ax.grid(False)

# Remove z-axis
#ax.w_zaxis.line.set_lw(0.)
#ax.set_zticks([])



ax.set_yticks(np.arange(0,145.715*10+1,145.715))
ax.set_yticklabels(['0','f$_{1}$','f$_{2}$','f$_{3}$','f$_{4}$','f$_{5}$','f$_{6}$','f$_{7}$','f$_{8}$','f$_{9}$'])
ax.set_xlabel('Time /s', labelpad=20)
ax.set_ylabel('Frequency of mode n /Hz', labelpad=20)
ax.set_xticks(np.arange(0,26,5))
ax.set_zlabel('Amplitude /A.u.', labelpad=20)

ax.set_xlim(0,25)
ax.set_ylim(0,145.715*10+1)


cbar = fig.colorbar(plot, ax=ax, shrink=0.6)
cbar.set_label('Amplitude /A.u.')
cbar.set_ticks([0, 0.5e8,1.0e8,1.5e8,2e8,2.5e8,3e8,3.5e8,4e8,4.5e8])
#cbar.set_ticklabels(['0', '50', '100', '150', '200 nm'])
#ax.plot3D(freqz,a,FTdataz)

#%%


#ax = plt.axes(projection="3d")
#ax.contour3D(freqz,FTdataz,a,50)
#for ii in np.arange(0,360,1):
#    ax.view_init(elev=10., azim=ii)
#    ax.savefig("movie%d.png" % ii)
