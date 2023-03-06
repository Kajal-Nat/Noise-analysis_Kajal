# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 20:14:18 2022

@author: admin-nisel120

The dataset is located in the "folder" variable,
 and the parameters such as the filename,
  file prefix, delimiter, and sample rate are set.
   The temperature range is set to "AutoRange," 
   which means that the code will automatically detect the range of temperatures in the dataset.
    The method for analyzing the data is also specified as "psd_welch_mean." 
    The code also has some options for skipping the start or end of the data, 
    trimming the time, and using a rolling average. 
    However, these options are not currently being used in the code.

    Bug fix:V2.1
    If the we apply more than one data removal oparation only the last one is applied as all oparations take raw data to signal data variable
"""
import multiprocessing
from scipy.signal import decimate, firwin, kaiserord
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import welch 
#which system are you using
if os.path.exists("/Users/admin-nisem543"):
    print("running in Ajesh's Mac book pro")
    mac         = True
    lab_pc      = False
    kajal_pc    = False
elif os.path.exists("C:/Users/ktiwari"):
    print("running in Kajal's PC")
    mac         = False
    lab_pc      = False
    kajal_pc    = True
else  : 
    print("running in Ajesh's PC")
    kajal_pc    = False
    mac         = False
    lab_pc      = True
temperature_range = "AutoRange"
tosecond        = 1/104.6/573440 #It is constant no matter what, never change it

#this needs to be changed depending on what files to analyse
if mac: folder      = "/Users/admin-nisem543/Seafile/MAX PLANK/Data/PPMS/FGT3_S25_#047/D1/Combined"
if kajal_pc : folder = "D:\Data\Kajal\Seafile\PPMS\FGT3_S25_#47_9T_Noise\D5_2_Feb_2023_night"
if lab_pc  : folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT3_S25_#047\Data_combined"
fileprefix  = "K_5,3mS" #Also, change this depending on files to read
method          =  "MSA_n2_norm_Kajal_2"#"MSA_n2_norm_lowpass"#"MSA_n2_norm___f_scaled___" #"psd_welch_mean"#___skip_start_600s
#skip_few_minuts
skip_tail       = False
skip_start      = False
trim_time       = False
rollingavg      = True
if "skip_tail" in method : skip_tail        = True
if "skip_start" in method :skip_start       = True
if "trim_time" in method :trim_time         = True

    
filelist                = []
delimiter   = ","
lineterminator = "\n"

#fucntions for calculations of fft, PSD and other data editing tools

def fourier_transform_doubleside(signal,sample_rate,method):
    print("FFT on full time domain calculating....")
    # print(signal[:100])
    fourier     = np.fft.fft(signal) 
    N           = int(np.size(signal))
    frequency   = np.fft.fftfreq(N,1/sample_rate) #positive side fft of time domain
    #frequency = (k / N) * sample_rate
    #Assuming even number of data points
    magnitude   = np.abs(fourier)
    amplitude   = magnitude/N
    psd     = 2*amplitude**2
    psd[0]  = amplitude[0]**2
    # print(frequency[:100])
    # print("psd", psd[:100])
    return [frequency[:],psd]


def psd_welch(signal,sample_rate,method):
    print("welch method is running...")
    nperseg1     = int((1000-1)*sample_rate)
    f, Pxx_den  = welch(signal, sample_rate, nperseg=nperseg1)
    f_med, Pxx_den_med = welch(signal, sample_rate, nperseg=nperseg1, average='median')
    # plt.semilogy(f, Pxx_den, label='mean')
    # plt.semilogy(f_med, Pxx_den_med, label='median')
    # plt.ylim([0.5e-3, 1])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V**2/Hz]')
    # plt.legend()
    # plt.show()
    frequency   = f
    psd         = Pxx_den  
    return [frequency[:],psd]


def rolling_average(data):
    exponential = np.array([])
    i   = 0
    index = 0
    #Make a exponential sequence for rolling average
    while index <= np.size(data):
        power   = int(round(np.power(1.2,i)))
        if power == 0: power =1
        if power <100 and rollingavg: i   = i +1
        exponential = np.append(exponential,power)
        index   = power+index
    window_list = exponential
    result          = np.array([])
    stddev_list     = np.array([])
    if isinstance(window_list,np.ndarray):
        window_sum  = 0
        for window_size in window_list:
            # print("======window size", window_size)
            total   = 0
            total_list  = np.array([])
            window_size = int(window_size)
            if window_size ==0:
                window_size =1
            # if window_size >100:
            #     window_size = 100
            for index in np.arange(window_sum,window_sum+window_size):
                if window_sum+window_size >= np.size(data):
                    break
                total_list = np.append(total_list,data[index])
                
                # print("index,data[index],total",index,data[index],total)
            if window_sum+window_size >= np.size(data):
                    break
            total   = sum(total_list)
            stddev  = np.std(total_list)
            stddev_list = np.append(stddev_list,stddev)
            result  = np.append(result,total/window_size)
            # print("======result is", result)
            window_sum = window_sum + window_size

    return [result[:],stddev_list]

#######################
def lowpass(signal,fs,method):

    # Input signal
    x = signal

    # Decimation factors
    dec_factors = [4, 4, 2]

    # Stop-band frequency
    stop_freq = 15 # Hz

    width = 10

    # Attenuation at stop band
    attenuation = 100 # dB

    # Initialize output signal
    y = x

    # Pad the signal with zeros
    #y = np.pad(y, (taps, taps), mode='edge')

    # Loop over decimation stages
    for i, dec_factor in enumerate(dec_factors):
        # Calculate cutoff frequency
        cutoff = stop_freq * dec_factor**i / fs
        # Calculate number of taps
        if i == 0:
            taps = 35
        elif i == 1:
            taps = 25
        else:
            taps = 321
        # Design antialiasing filter
        n, beta = kaiserord(attenuation, width)
        b = firwin(n, cutoff, window=('kaiser', beta), scale=False)
        # Apply antialiasing filter
        y = np.convolve(y, b, mode='same')
        # Decimate signal
        y = decimate(y, dec_factor)

        # Pad the signal at both ends to reduce the sudden jumps
        if "padding" in method :
            padding_size = int(np.ceil(n / 2))
            y = np.pad(y, (taps, taps), mode='edge')
    # Final sampling rate
    fs_final = fs / np.prod(dec_factors)
    return (fs_final,y)


#### Data analysis function
def analyse_signal(    filename, folder, method, tosecond, lineterminator, delimiter, skip_tail, skip_start, trim_time, lowpass, psd_welch, fourier_transform_doubleside):
    print("analysing : ", filename)

    
    #### Reding the file into Data array
    filelocation    = os.path.join(folder , filename + ".txt")
    analysis_base   = os.path.join(folder,"Analyse")
    analysis_imagelocation = os.path.join(folder, "Analyse", f"_{method}",filename+f"{method}.png")
    analysis_filelocation2 = os.path.join(folder, "Analyse", f"_{method}",filename+ f"_{method}_analysed_reduced.txt")
    analysis_filelocation = os.path.join(folder, "Analyse", f"_{method}","full_data",filename + f"_{method}_analysed.txt")
    if not os.path.exists(analysis_base):
        print("making base analysis directory: \n", )
        os.mkdir(analysis_base)
    if not os.path.exists(os.path.dirname(analysis_imagelocation)):
        print("making analysis directory: \n", os.path.dirname(analysis_imagelocation))
        os.mkdir(os.path.dirname(analysis_imagelocation))
    if not os.path.exists(os.path.dirname(analysis_filelocation)):
        print("making analysis directory: \n", os.path.dirname(analysis_filelocation))
        os.mkdir(os.path.dirname(analysis_filelocation))

    with open(filelocation,"r") as f:
        data    = f.read()
        # Split the data into rows
        rows    = data.split(lineterminator)
        rows    = rows[0:-1]
        # Split each row into columns
        # Split each row into columns and convert the values to floats
        columns = [[float(x) for x in row.split(delimiter)] for row in rows]
        # Convert the columns to a numpy array
        data    = np.array(columns)[:]
    
    #calculating row sample rate using global constant "tosecond" and measurement_timeinterval from data file auomatically
    measurement_timeinterval=(data[1,0]-data[0,0])*tosecond
    row_sample_rate = 1/measurement_timeinterval
    
    
    skip_tail_raw   = int( 1000*row_sample_rate)    # Remove few data points from the end
    skip_start_raw  = int( 600*row_sample_rate)     # Remove few data points from the beginning
    trim_length     = int( 1000*row_sample_rate)     # Trimms the date. Keeps from endng till the trim_length towards the start eg:800seconds
    
    #### Skipping few minues
    if True: 
        data[:,0]       = (data[:,0])*tosecond
        if skip_tail: #Skipping end
            print(f"skipping tail of data {data[0,0]:.0f} to {data[0,0]:.0f}")
            data        = data[:-skip_tail_raw,:]
            print("remaining from {data[0,0]:.0f}s to {data[-1,0]:.0f}s")
        if skip_start: #Skip begining
            print(f"skipping start of data {data[0,0]:.0f}s to {data[0,0]:.0f}s")
            data        = data[skip_start_raw:,:]
            print(f"remaining from {data[0,0]:.0f}s to {data[-1,0]:.0f}seconds")
        if trim_time:
            print(f"trimming        :  data {data[0,0]:.0f}s to {data[0,0]:.0f}s")
            data        = data[-trim_length:,:]
            print(f"remaining from {data[0,0]:.0f}s to {data[-1,0]:.0f}seconds")
        
        time        = data[:,0]
        signal_x    = data[:,1]
        signal_y    = data[:,2]


#################################################
    #### Low pass filter 
    if "lowpass" in method:
        desimation_factor = [4,4,2]
        (sample_rate,signal_x) = lowpass(signal_x,row_sample_rate,method)
        print("low pass filter applied\n New sampling rate is:",row_sample_rate)
        (sample_rate,signal_y) = lowpass(signal_y,row_sample_rate,method)
        signal_length = signal_x.shape[0]
        time = np.linspace(time[0], time[-1], signal_length)
        print("low pass filter applied\n New sampling rate is:",sample_rate)
    else:
        sample_rate = row_sample_rate
########################################################

    
    #### Plotting the time domain signal
    plot_graph = True
    if plot_graph:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        ax1.plot(time,signal_x,label= "singal_x")
        ax1.set_title("signal_x")
        ax2.plot(time,signal_y,label= "singal_y")
        ax2.set_title("signal_y")
        #plt.savefig(os.path.join(os.path.dirname(analysis_filelocation),f"time_{filename[:-4]}_.png"))


    #### Calculating the PSD  
    if "MSA_n2_norm" in method:
        psd = fourier_transform_doubleside
    if "psd_welch_mean" in method:
        psd = psd_welch
    # acf = np.correlate(signal, signal, mode='full')[:np.size(signal)]
    # Use the Wiener-Khinchin theorem to convert the ACF to the PSD
    # Carry out the fourier_transform to optain the PSD
    [frequencypsd, MSA_norm_x]  =  psd(signal_x,sample_rate,method)
    [frequencypsd, MSA_norm_y]  =  psd(signal_y,sample_rate,method)
    # back ground substraction
    background_substracted      =  abs(MSA_norm_x-MSA_norm_y)
    if "f_scaled" in method:
        MSA_norm_x = MSA_norm_x*frequencypsd
        print("PSD Scaled")
    array   = np.vstack((frequencypsd,MSA_norm_x,MSA_norm_y,background_substracted))
    header = "Frequency,MSA_norm_x,MSA_norm_y,background_substracted"
    np.savetxt(analysis_filelocation,np.transpose(array),header = header, delimiter=",")
    

    #### Reduce the number of frequencis in PSD
    [reduced_frequency,stddev_list_frequency]   = rolling_average(frequencypsd)
    [reduced_MSA_norm_x,stddev_list_x]  = rolling_average(MSA_norm_x)
    [reduced_MSA_norm_y,stddev_list_y]  = rolling_average(MSA_norm_y)
    [reduced_bg_substracted,stddev_list_bg_substracted] = rolling_average(background_substracted)


    #### Save the PSD                    
    array   = np.vstack((reduced_frequency,reduced_MSA_norm_x,stddev_list_x,reduced_MSA_norm_y,stddev_list_y,reduced_bg_substracted,stddev_list_bg_substracted))
    header = "reduced_frequency,reduced_MSA_norm_x,stddev_list_x,reduced_MSA_norm_y,stddev_list_y,reduced_bg_substracted,stddev_list_bg_substracted"
    np.savetxt(analysis_filelocation2,np.transpose(array),header = header, delimiter=",")


    # Plot PSD
    if plot_graph: 
        ax3.loglog(reduced_frequency,reduced_MSA_norm_x)
        ax3.loglog(reduced_frequency,reduced_MSA_norm_y)
        ax3.set_title("Temperature is "+str(filename)[:-len(fileprefix)])
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("log PSD")
        ax4.loglog(frequencypsd,MSA_norm_x)
        ax4.loglog(frequencypsd,MSA_norm_y)
        ax4.set_title("Temperature is "+str(filename)[:-len(fileprefix)])
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("PSD")
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(os.path.join(os.path.dirname(analysis_filelocation),f"PSD_{filename[:-4]}_.png"))
        plt.savefig(analysis_imagelocation)

        # plt.loglog(frequencypsd,background_substracted)
        # plt.title("Temperature is "+str(filename)[:-len(fileprefix)])
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("PSD")
        # plt.show()
    return 0

if __name__=="__main__":


    #### Creating the file list
    filelist                = []
    if temperature_range    == "AutoRange":
        filelist = []
        temperature_range = np.array([])
        # Create a list of files 
        directory_filelist = os.listdir(folder)
        for filename in directory_filelist:
            if filename.endswith(fileprefix+'.txt'):
                temperature = float(filename[:filename.find(fileprefix)])
                temperature_range = np.append(temperature_range,temperature)
    temperature_range   = np.sort(temperature_range) 
    filelist    = []
    for temperature in temperature_range: 
        if temperature % 1 ==0 :
            temperature = int(temperature)
        filename    = str(temperature)+fileprefix
        filelist.append(filename)
    # filelist    = ['310K_10mS_3rd-Order']
    # filelist    = ['2K']
    print(filelist)

    #load and analyse all the files
    for filename in filelist:
        analyse_signal(filename, folder, method, tosecond, lineterminator, delimiter, skip_tail, skip_start, trim_time, lowpass, psd_welch, fourier_transform_doubleside)

