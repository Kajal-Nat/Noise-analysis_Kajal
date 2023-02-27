# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 20:14:18 2022


@author: admin-nisel120
"""
import numpy as np
import os
import matplotlib.pyplot as plt
def fourier_transform_one_side(signal,sample_rate,method):
    fourier     = np.fft.rfft(signal) 
    N           = int(np.size(signal))
    frequency   = np.fft.rfftfreq(N,1/sample_rate) #positive side fft of time domain
    #frequency = (k / N) * sample_rate
    #Assuming even number of data points
    psd_magnitude   = np.abs(fourier)**2
    if method is 'MSA_sum_norm':
        total    = np.sum(psd_magnitude)
        MSA_norm = psd_magnitude/total
    return [frequency[:],MSA_norm[:]]
def fourier_transform_doubleside(signal,sample_rate,method):
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
def rolling_average(data,window_list):
    result  = np.array([])
    if isinstance(window_list,np.ndarray):
        window_sum  = 0
        for window_size in window_list:
            # print("======window size", window_size)
            total   = 0
            window_size = int(window_size)
            if window_size ==0:
                window_size =1
            # if window_size >100:
            #     window_size = 100
            for index in np.arange(window_sum,window_sum+window_size):
                if window_sum+window_size >= np.size(data):
                    break
                total = total + data[index]
                # print("index,data[index],total",index,data[index],total)
            if window_sum+window_size >= np.size(data):
                    break
            result = np.append(result,total/window_size)
            # print("======result is", result)
            window_sum = window_sum + window_size

    return result[:]
fourier_transform = fourier_transform_doubleside
        
# folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT26052022\Device4\Device4_on-24-12-2022\Data"
# filename    ='10K_10mS_3rd-Order'
# fileprefix  = "K_10mS_3rd-Order"
# delimiter   = ","
#which system are you using
if os.path.exists("/Users/admin-nisem543"):
    print("running in Ajesh's Mac book pro")
    mac         = True
    lab_pc      = False
    kajal_pc    = False
elif os.path.exists(""):
    print("running in Kajal's PC")
    mac         = False
    lab_pc      = False
    kajal_pc    = True
else  : 
    print("running in Ajesh's PC")
    kajal_pc    = False
    mac         = False
    lab_pc      = True
if mac: folder      = "/Users/admin-nisem543/Seafile/MAX PLANK/Data/PPMS/FGT3_S25_#047/D1/Combined"
if kajal_pc : folder = "D:\Data\Kajal\Seafile\PPMS\FGT3_S25_#047\D1\Combined"
if lab_pc  : folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT3_S25_#047\Data_combined"
fileprefix    = "K"
delimiter   = "\t"
lineterminator = "\t\n"

filelist    = []
skipfewminut= True 
sample_rate = 104.6
tosecond    = 1/104.6/573440
method      = "MSA_sum_norm"
#Check 70K,110K data and edit it 
for temperature in np.arange(10,340,10): 
    filename    = str(temperature)+fileprefix
    filelist.append(filename)
# filelist    = ['310K_10mS_3rd-Order']
for filename in filelist:
    print("analysing : ",filename)
    filelocation    = folder + "\\" + filename + ".txt"
    analysis_filelocation = folder + "\\" + "Analyse" +"\\"+ filename + f"_{method}_analysed.txt"
    analysis_filelocation2 = folder + "\\" + "Analyse" +"\\"+ filename + f"_{method}_analysed_reduced.txt"
    with open(filelocation,"r") as f:
        data        = f.read()
        # Split the data into rows
        rows = data.split('\t\n')
        print(rows)
        # Split each row into columns
        columns = [row.split('\t') for row in rows]
        print(columns[0])
        # Convert the values in each column to the appropriate data type (e.g. float, int, etc.)
        for i in range(len(columns)):
            columns[i] = [float(x) for x in columns[i]]
        
    if skipfewminut:
            skip_raw      =int( -3000*sample_rate)
            time        = ((data[skip_raw:,0]-data[skip_raw,0])*tosecond)
            signal_x      = data[skip_raw:,1]-data[skip_raw,1]
            signal_y      = data[skip_raw:,2]-data[skip_raw,2]
    else:
        time        = (data[:,0]-data[0,0])*tosecond
        signal_x      = data[:,1]-data[0,1]
        signal_y      = data[:,2]-data[0,2]
    #data[:,0]   = (data[:,0]-data[0,0])*tosecond
    print("========singal_x")
    plt.plot(time,signal_x)
    plt.show()
    print("========singal_y")
    plt.plot(time,signal_y)
    plt.show()
    
    # acf = np.correlate(signal, signal, mode='full')[:np.size(signal)]
    # Use the Wiener-Khinchin theorem to convert the ACF to the PSD
    [frequencypsd, MSA_norm_x]  =  fourier_transform(signal_x,sample_rate,method)
    [frequencypsd, MSA_norm_y]  =  fourier_transform(signal_y,sample_rate,method)
    background_substracted      =  abs(MSA_norm_x-MSA_norm_y*MSA_norm_x[0]/MSA_norm_y[0] )
    array   = np.vstack((frequencypsd,MSA_norm_x,MSA_norm_y,background_substracted))
    header = "Frequency,MSA_norm_x,MSA_norm_y,background_substracted"
    np.savetxt(analysis_filelocation,np.transpose(array),header = header, delimiter=",")
    
    exponential = np.array([])
    i   = 0
    index = 0
    while index < np.size(signal_x):
        power   = int(round(np.power(1.025,i)))
        if power == 0: power =1
        exponential = np.append(exponential,power)
        index   = power+index
        i   = i +1
    reduced_frequency   = rolling_average(frequencypsd,exponential)
    reduced_MSA_norm_x  = rolling_average(MSA_norm_x,exponential)
    reduced_MSA_norm_y  = rolling_average(MSA_norm_y,exponential)
    reduced_background_substracted = rolling_average(background_substracted,exponential)
                        
    array   = np.vstack((reduced_frequency,reduced_MSA_norm_x,reduced_MSA_norm_y,reduced_background_substracted))
    
    
    np.savetxt(analysis_filelocation2,np.transpose(array),header = header, delimiter=",")
        
    plt.loglog(reduced_frequency,reduced_background_substracted)
    plt.title("Temperature is "+str(filename)[:-15])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.show()
    plt.loglog(frequencypsd,background_substracted)
    plt.title("Temperature is "+str(filename)[:-15])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.show()
