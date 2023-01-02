# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:09:53 2022

@author: admin-nisel120
"""
import numpy as np
import matplotlib.pyplot as plt
folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT26052022\Device4\Device4_on-24-12-2022\Data\Analyse"
method  = "MSA_sum_norm"
analysis_filelocation = folder +"\\" + f"final_results_{method}.txt"
filelist    = []
#Search a 2D array
def search(arr,search_column_index,value):
    arr     = np.array(arr)
    search_column   = arr[:,search_column_index]
    idx     = np.abs(search_column-value).argmin()
    raw     = arr[idx,:]
    return raw
#Check 70K,110K data and edit it 
for temperature in np.arange(10,340,10): 
    filename    = str(temperature)+f"K_10mS_3rd-Order_{method}_analysed_reduced"
    filelist.append(filename)
# filelist    = ['10K_10mS_3rd-Order_analysed_reduced']
frequencies_of_interest = [
    0.001,
    0.003,
    0.006,
    0.01,
    0.03,
    0.06,
    0.1,
    0.3,
    0.6,
    1,
    3,
    6,
    10
]
temperature_vs_psd = np.hstack(([0],frequencies_of_interest))
print(temperature_vs_psd)
for filename in filelist:
    background_corrected_psd = []
    print("analysing : ",filename)
    filelocation    = folder + "\\" + filename + ".txt"
    data = np.loadtxt(filelocation,skiprows=0, delimiter=",")
    for frequency in frequencies_of_interest:
        psd_value   = search(data,0,frequency)
        background_corrected_psd = np.append(background_corrected_psd,psd_value[3])
    new_raw     = np.hstack(([int(filename[:-33])],background_corrected_psd))
    temperature_vs_psd  = np.vstack((temperature_vs_psd, new_raw))
temperature_vs_psd  = temperature_vs_psd[:,:]
header  = "Temperature(K)"
for temp in frequencies_of_interest:
    header = header+","+str(temp)+" Hz"
with open(analysis_filelocation,"w") as f:
    f.write(header)
with open(analysis_filelocation,"a") as f:
    np.savetxt(f,temperature_vs_psd,delimiter=",",fmt="%s")
#scale the frequencies