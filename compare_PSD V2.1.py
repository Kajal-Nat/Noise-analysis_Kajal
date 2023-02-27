# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:09:53 2022
This program takes the analysed files from PSD.py
It finds PSD values in the data file belonging to specific frequencis given by frequencies_of_interest variable
function search performs the search of the data with is a 2D array along the specified column
psd colums values were wrong. it was all error values
psd_x                           = 2
psd_y                           = 4

@author: Ajesh
"""
import numpy as np
import matplotlib.pyplot as plt
import os

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
sub_folder  = "Analyse"
# import PyOrigin
# folder                  = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT26052022\Device4\Device4_on-24-12-2022\Data\Analyse"
# method                  = "MSA_n2_norm"
# fileprefix  = "K_10mS_3rd-Order"
# temperature_range       = "AutoRange"
# method                          = "MSA_n2_norm"
# bcakground_substracted          = 5
# std_dev_bcakground_substracted  = 6
# temperature_range               = "AutoRange"

# folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS9T\Ajesh\2022\GoldWire_Data_29-12-2022\Analyse"
# fileprefix  = "K_10mS_3rdOrder"
# temperature_range = [10,40,60,80,100,140,180,220,260,280,340]
# method      = "MSA_n2_norm"
# bcakground_substracted  = 3
# signal_x                = 1

# folder                          = r"C:\Data_analysis\old\New folder\Analyse\Reduced with exp(1.05)"
# folder                          = r"C:\Data_analysis\old\New folder\Analyse"
# fileprefix                      = "K"
# # temperature_range               = np.hstack(([2,10],np.arange(25,325,25)))
# method                          = "MSA_n2_norm"
# bcakground_substracted          = 5
# std_dev_bcakground_substracted  = 6
# temperature_range               = "AutoRange"

# folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT26052022\Device4_on_16-01-2023_5Prob\Data\Analyse-Fulltime"
# method                          = "MSA_n2_norm"
# bcakground_substracted          = 1
# std_dev_bcakground_substracted  = 2
# temperature_range               = "AutoRange"
# fileprefix                      = "K_10mS_K"

#### First measurement on FGT3-S25
# folder      = r'C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT3_S25_#047\Data\Analyse'
#### Second measurement on FGT3-S25
#folder      = r'C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT3_S25_#047\Data repeat on 20jan2023\Analyse'
#### Third measurement on FGT3-S25. Done on 20th jan night
# if mac: folder      = "FGT3_S25_#047/Data repeat on 20th night"
# else  : folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT3_S25_#047\Data reapeat on 20th night\Analyse"
#### Data taken from all 3 measuements are combined in Data_combined folder
#if mac: folder      = "/Users/admin-nisem543/Seafile/MAX PLANK/Data/PPMS14T/Ajesh_2022/FGT3_S25_#047/Combined"
#else  : folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT3_S25_#047\Data_combined"
#fileprefix  = "K_5mS"
#method          =  "MSA_n2_norm___f_scaled___round3"#"MSA_n2_norm_lowpass"#"MSA_n2_norm" #"psd_welch_mean"#___skip_start_600s
#### Testing old analysed data 19th Feb 2023
if mac: folder      = "/Users/admin-nisem543/Seafile/MAX PLANK/Data/PPMS/FGT3_S25_#047/D1/Combined"
if kajal_pc : folder = "D:\Data\Kajal\Seafile\PPMS\FGT3_S25_#047\D1\Combined"
if lab_pc  : folder      = r"C:\Users\admin-nisel120\ownCloud5\MAX PLANK\Data\Data\PPMS14T\Ajesh_2022\FGT3_S25_#047\Data_combined"
fileprefix  = "K_5mS"
method          =  "MSA_n2_norm"#"MSA_n2_norm_lowpass"#"MSA_n2_norm___f_scaled___" #"psd_welch_mean"#___skip_start_600s


#************************************************************* FGT3-S25_D5
#if mac: folder      = "/Users/admin-nisem543/Documents/FGT3_S25_#47_9T_Noise/D5_2-Feb_2023_night"
#else  : folder      = r""
#fileprefix  = "K_2,6mS"
#row_sample_rate = 837.1
#method                          = "MSA_n2_norm_1000s-lowpass-Data_Part3_skip_start_600s-trim_time_1000s-f_scaled-round2"#"MSA_n2_norm_lowpass" #,"psd_welch_mean" #
#method          =  "MSA_n2_norm___lowpass___skip_start_600s___f_scaled___round2"
#method          = "MSA_n2_norm___f_scaled___round3"
#method          = "MSA_n2_norm___lowpass___skip_start_600s___f_scaled___round3"


psd_average = False
temperature_range               = "AutoRange" # or [150,200,300]
folder = os.path.join(folder,sub_folder)
psd_x                           = 2
psd_y                           = 4
bg_substraction                 = False
scale_psd                       = False
frequencies_of_interest         = np.array([
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
    10,
    20
])
frequencies_of_interest     = frequencies_of_interest[:]


####Search a 2D array
def search(arr,search_column_index,value,average):
    arr             = np.array(arr)
    search_column   = arr[:,search_column_index]
    idx             = np.abs(search_column-value).argmin()
    raw             = arr[idx,:]
    if average:
        raw             = (arr[idx-1,:]+arr[idx,:]+arr[idx+1,:])/3
    return raw



if __name__=="__main__":


    #### Creating a file list either from the temperature list provided or from the folder
    filelist                = []
    if temperature_range    == "AutoRange":
        filelist = []
        temperature_range = np.array([])
        #### Create a list of files 
        all_filelist = os.listdir(folder)
        for filename in os.listdir(folder):
            endswich_value =f"{fileprefix}_{method}_analysed_reduced.txt"
            if filename.endswith(endswich_value):
                temperature = filename[:filename.find("K_")]
                temperature_range = np.append(temperature_range,float(temperature))
    temperature_range   = np.sort(temperature_range) 
    filelist    = []
    for temperature in temperature_range: 
        if float(temperature) % 1 ==0 :
            temperature = int(temperature)
        filename    = str(temperature)+f"{fileprefix}_{method}_analysed_reduced"
        filelist.append(filename)
    analysis_filelocation   = os.path.join(folder  , f"final_results_{fileprefix}_{method}.txt")
    # filelist    = ['10K_10mS_3rd-Order_analysed_reduced']


    #### Search and find the PSD values at the frequencies on interest
    frequencies_header  = frequencies_of_interest
    temperature_vs_psd  = np.hstack(([0],frequencies_header))
    for filename in filelist:
        temperature = float(filename[:filename.find("K_")])
        if float(temperature) % 1 ==0 :
            temperature = int(temperature)
        psd_list = []
        psd_list_bg_substract = []
        print("analysing : ",filename)
        filelocation    = os.path.join(folder ,  filename + ".txt")
        filepath     =os.path.abspath(filelocation)
        data = np.loadtxt(filepath,skiprows=0, delimiter=",")
        for frequency in frequencies_of_interest:
            psd_values      = search(data,0,frequency,psd_average)
            if bg_substraction:
                psd_list  = np.append(psd_list,psd_values[psd_x]-psd_values[psd_y])
            else:
                psd_list  = np.append(psd_list,psd_values[psd_x])
            #Append error 
            # psd_list  = np.append(psd_list,psd_values[std_dev_column_of_interest])
        new_raw         = np.hstack(([temperature],psd_list))
        temperature_vs_psd  = np.vstack((temperature_vs_psd, new_raw))
    temperature_vs_psd  = temperature_vs_psd[1:,:]





    #### Save the analised data
    header  = "Temperature(K)"
    for temp in frequencies_header:
        header = header+","+str(temp)+" Hz"
    if scale_psd:
        analysis_filelocation   = analysis_filelocation[:-4]+"_Scaled.txt"
    with open(analysis_filelocation,"w") as f:
        f.write(header)
        f.write("\n")
    with open(analysis_filelocation,"a") as f:
        np.savetxt(f,temperature_vs_psd,delimiter=",",fmt="%s")



    #### Plotting all the figures
    first_frequency = np.abs(frequencies_of_interest-0.006).argmin()
    last_frequency = np.abs(frequencies_of_interest-1).argmin()
    fig,(ax1,ax2) = plt.subplots(2,1)
    x       = temperature_vs_psd[:,0]
    xmin,xmax = min(x),max(x)
    for i, frequency in enumerate(frequencies_of_interest[first_frequency:last_frequency]):
        label   = str(frequency)+"Hz"
        y       = temperature_vs_psd[:,first_frequency+i]
        ax1.plot(x, y, label=label)
        ax2.semilogy(x, y, label=label)
        plt.legend()
        ax1.set_xlim((xmin,xmax))
        ax2.set_xlim((xmin,xmax))
        xmin_index = np.abs((x-xmin)).argmin()
        xmax_index = np.abs((x-xmax)).argmin()
        if i ==0:
            ymin, ymax = min(y[xmin_index:xmax_index]), max(y[xmin_index:xmax_index])
        if ymin>min(y[xmin_index:xmax_index]): ymin=min(y[xmin_index:xmax_index])
        if ymax<max(y[xmin_index:xmax_index]): ymax=max(y[xmin_index:xmax_index])
        ax1.set_ylim((ymin,ymax))
        ax2.set_ylim((ymin,ymax))
    plt.savefig(analysis_filelocation[:-4]+".png")
    plt.show()