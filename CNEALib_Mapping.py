# -*- coding: utf-8 -*-
"""
Add weighted STA on 20240113
Add Site_Screen_Num function on 20231226
Last updated on 12/08/2023
Created on Mon Oct 12 13:35:38 2020
@author: Jun Wang
"""

#%% Packages and functions

import peakutils
import numba as nb
#import matplotlib.ticker as plticker

from nptdms import TdmsFile
import numpy as np
import pylab
import numba
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import seaborn as sns
import time
import csv
import pandas as pd
import tables as tb
import os
import matplotlib as mpl

from tqdm import tqdm
import math

from scipy.signal import butter, lfilter, freqz
import scipy.stats as sci_stats

import h5py as h5
import glob
from pathlib import Path


import cv2  # run 'pip install opencv-python' to install
import os
from cv2 import VideoWriter, VideoWriter_fourcc
import seaborn as sns #sns.set_theme()


mpl.rcParams['agg.path.chunksize'] = 10000
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42

Gain_intra = 30
Image_id = 0
im_info_old =''

F_sample = 1205120/128


def TDMS_CONV(FILE_path_list, DAQ_num = [1,2,3], VC_mode = False):
    for FILE_handle in FILE_path_list:
        #FILE_path = Path(FILE_handle)
        # s_factor =0.000156252
        #Convert all DAQ1-2 files in the folder
        columns = 64 
        file_find = glob.glob(str(FILE_handle)+'/*DAQ*.tdms', recursive=True)
        file_find.sort()
        for filename in file_find: #FILE_handle[0:-6]+'*.tdms' if only convert the data recorded at same time
            print('Converting:')
            print(filename)
            DAQ_index = int(filename[filename.find('DAQ',-20,-1)+3])
            if (DAQ_index <3) and (DAQ_index in DAQ_num):
                with TdmsFile.open(filename) as tdms_file:
                    for group in tdms_file.groups():
                        channels = group.channels()
                        for ch_index in range(len(channels)):
                            data_temp = channels[ch_index][:] # /s_factor
                            data_reshape = data_temp.reshape((-1,128)) # reshape to 128 columns: 2 rows
                            data_reshape = data_reshape[:,::-1] #reverse the columns order
                            Row_odd = data_reshape[:,1::2] #0101 to 0164
                            Row_even = data_reshape[:,0::2] #0201 to 0264
                            new_path = filename[0:filename.find('DAQ',-20,-1)]+'Converted/' # new path to store the converted data
                            isdir = os.path.isdir(new_path)  
                            if not isdir:
                                os.mkdir(new_path)
                            f_name1 = new_path+'Row_'+str(2*ch_index+1+(DAQ_index-1)*32)+'.h5'
                            if os.path.isfile(f_name1):
                                f1 = tb.open_file(new_path+'Row_'+str(2*ch_index+1+(DAQ_index-1)*32)+'.h5',"a")
                                f2 = tb.open_file(new_path+'Row_'+str(2*ch_index+2+(DAQ_index-1)*32)+'.h5',"a")
                                try:
                                    R1 = f1.root.data
                                    R2 = f2.root.data
                                except:
                                    f1.close()
                                    f2.close()
                            else:
                                f1 = tb.open_file(new_path+'Row_'+str(2*ch_index+1+(DAQ_index-1)*32)+'.h5',"a")
                                f2 = tb.open_file(new_path+'Row_'+str(2*ch_index+2+(DAQ_index-1)*32)+'.h5',"a")
                                try:
                                    R1 = f1.create_earray(f1.root,'data',tb.Float16Atom(),shape=(0,columns))
                                    R2 = f2.create_earray(f2.root,'data',tb.Float16Atom(),shape=(0,columns))
                                except:
                                    f1.close()
                                    f2.close()
                            try:
                                R1.append(Row_odd)
                                R2.append(Row_even)
                                f1.close()
                                f2.close()
                            except:
                                f1.close()
                                f2.close()  
                                
            elif (DAQ_index ==3) and (DAQ_index in DAQ_num):
                with TdmsFile.open(filename) as tdms_file:
                    if VC_mode:
                        f_add = ['Vs1','Vref','Iref','Vtemp']
                    else:
                        f_add = ['Vs1','Vs4','Vref','Iref','Vtemp']
                    for group in tdms_file.groups():
                        channels = group.channels()
                        for ch_index in range(len(channels)):
                            data_temp = channels[ch_index][:] # /s_factor
                            data_temp = data_temp.reshape(-1,1)
                            new_path = filename[0:filename.find('DAQ',-20,-1)]+'Converted/' # new path to store the converted data
                            isdir = os.path.isdir(new_path)  
                            if not isdir:
                                os.mkdir(new_path)
                            f_name = new_path + f_add[ch_index]+'.h5'
                            f3 = tb.open_file(f_name,"a")
                            type(data_temp)
                        
                            try:
                                R3 = f3.create_earray(f3.root,'data',tb.Float16Atom(),shape=(0,1))
                                
                            except:
                                print('error: file with same name already exists')
                                f3.close()
                            try:
                                R3.append(data_temp)
                                f3.close()
                            except:
                                f3.close()                                
    return new_path



def TDMS_CONV_COMB(FILE_path_list, DAQ_num = [1,2,3], VC_mode = False):
    for FILE_handle in FILE_path_list:
        #FILE_path = Path(FILE_handle)
        # s_factor =0.000156252
        #Convert all DAQ1-2 files in the folder
        columns = 64 
        file_find = glob.glob(str(FILE_handle)+'/*DAQ*.tdms', recursive=True)
        file_find.sort()
        filename = file_find[0] # using the first file to create the folder
        new_path = filename[0:filename.find('DAQ',-20,-1)]+'Combined_Converted/' # new path to store the converted data
        isdir = os.path.isdir(new_path)  
        if not isdir:
            os.mkdir(new_path)
        
        for filename in file_find: #FILE_handle[0:-6]+'*.tdms' if only convert the data recorded at same time
            print('Converting:')
            print(filename)
            DAQ_index = int(filename[filename.find('DAQ',-20,-1)+3])
            if (DAQ_index <3) and (DAQ_index in DAQ_num):
                with TdmsFile.open(filename) as tdms_file:
                    for group in tdms_file.groups():
                        channels = group.channels()
                        for ch_index in range(len(channels)):
                            data_temp = channels[ch_index][:] # /s_factor
                            data_reshape = data_temp.reshape((-1,128)) # reshape to 128 columns: 2 rows
                            data_reshape = data_reshape[:,::-1] #reverse the columns order
                            Row_odd = data_reshape[:,1::2] #0101 to 0164
                            Row_even = data_reshape[:,0::2] #0201 to 0264
                            # new_path = filename[0:filename.find('DAQ',-20,-1)]+'Converted/' # new path to store the converted data
                            # isdir = os.path.isdir(new_path)  
                            # if not isdir:
                            #     os.mkdir(new_path)
                            f_name1 = new_path+'Row_'+str(2*ch_index+1+(DAQ_index-1)*32)+'.h5'
                            if os.path.isfile(f_name1):
                                f1 = tb.open_file(new_path+'Row_'+str(2*ch_index+1+(DAQ_index-1)*32)+'.h5',"a")
                                f2 = tb.open_file(new_path+'Row_'+str(2*ch_index+2+(DAQ_index-1)*32)+'.h5',"a")
                                try:
                                    R1 = f1.root.data
                                    R2 = f2.root.data
                                except:
                                    f1.close()
                                    f2.close()
                            else:
                                f1 = tb.open_file(new_path+'Row_'+str(2*ch_index+1+(DAQ_index-1)*32)+'.h5',"a")
                                f2 = tb.open_file(new_path+'Row_'+str(2*ch_index+2+(DAQ_index-1)*32)+'.h5',"a")
                                try:
                                    R1 = f1.create_earray(f1.root,'data',tb.Float16Atom(),shape=(0,columns))
                                    R2 = f2.create_earray(f2.root,'data',tb.Float16Atom(),shape=(0,columns))
                                except:
                                    f1.close()
                                    f2.close()
                            try:
                                R1.append(Row_odd)
                                R2.append(Row_even)
                                f1.close()
                                f2.close()
                            except:
                                f1.close()
                                f2.close()  
                                
            elif (DAQ_index ==3) and (DAQ_index in DAQ_num):
                with TdmsFile.open(filename) as tdms_file:
                    if VC_mode:
                        f_add = ['Vs1','Vref','Iref','Vtemp']
                    else:
                        f_add = ['Vs1','Vs4','Vref','Iref','Vtemp']
                    for group in tdms_file.groups():
                        channels = group.channels()
                        for ch_index in range(len(channels)):
                            data_temp = channels[ch_index][:] # /s_factor
                            data_temp = data_temp.reshape(-1,1)
                            # new_path = filename[0:filename.find('DAQ',-20,-1)]+'Converted/' # new path to store the converted data
                            # isdir = os.path.isdir(new_path)  
                            # if not isdir:
                            #     os.mkdir(new_path)
                            f_name = new_path + f_add[ch_index]+'.h5'
                            
                            type(data_temp)
                            
                            if os.path.isfile(f_name):
                                f3 = tb.open_file(f_name,"a")
                                try:
                                    R3 = f3.root.data
                                except:
                                    f3.close()
            
                            else:
                                f3 = tb.open_file(f_name,"a")
                                try:
                                    R3 = f3.create_earray(f3.root,'data',tb.Float16Atom(),shape=(0,1))
                                except:
                                    print('error: file with same name already exists')
                                    f3.close()
                                
                                                                
                            try:
                                R3.append(data_temp)
                                f3.close()
                            except:
                                f3.close()                                
    return new_path


def Site_Read(Root_folder,Site_name, Start_time=0, Length=1200, F_sample=1205120/128):
    Row_num=int(Site_name[0:2])
    Col_num=int(Site_name[2:4])-1
    
    h5_File=Root_folder + '/Row_' + str(Row_num) + '.h5'
    # s_factor =0.000156252
    with tb.open_file(h5_File) as site_data:
        try:
          return site_data.root.data[Start_time*F_sample:(Start_time + Length)*F_sample,Col_num] #*s_factor # add a minus sign to cancel the polarity of the inverting amplifer
        except:
         return site_data.root.data[Start_time*F_sample::,Col_num] #*s_factor # add a minus sign to cancel the polarity of the inverting amplifer   

def Row_Read(Root_folder,Row_num, Start_time=0, Length=1200, F_sample=1205120/128):
#changed h5_file to glob.glob to accommoedate filename_row1.h5  
    h5_File = glob.glob(Root_folder + '/*Row_' + str(Row_num) + '.h5')[0]
    with tb.open_file(h5_File, driver_core_backing_store=0) as site_data:
        try:
         return site_data.root.data[Start_time*F_sample:(Start_time + Length)*F_sample,:]
        except:
         return site_data.root.data[Start_time*F_sample::,:] 

def PCB_Read(Root_folder, Signal_name = 'Iref', Start_time=0, Length=1200, down_sample=100): #Signal_name = ['Vs1','Vs4','Vref','Iref','Vx']
 
# changed F_sample to localized variable. 
    F_sample =120512;
#changed h5_file to glob.glob to accommoedate filename_row1.h5    
    h5_File=glob.glob(Root_folder +'/'+ Signal_name + '.h5')[0]
    with tb.open_file(h5_File) as PCB_data:
        
        try:
         return PCB_data.root.data[Start_time*F_sample:(Start_time + Length)*F_sample][0::down_sample]
        except:
         return PCB_data.root.data[Start_time*F_sample::][0::down_sample] 


def Moving_Average(x, w):  # pad with zeros to make the resulting data same length
    return np.convolve(np.concatenate((x[0:int(w/2)], x, x[-w + int(w/2)+1 ::])), np.ones(w), 'valid') / w
         
def Highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)   
    y = lfilter(b, a, data)
    return y
def CI_Mask_Gen(Root_folder,Start_time=0, Ana_length = 2400, Mask_width = 2):

    F_sample = 1205120/128
    down_sample_pcb = 128
    F_sample_pcb = 120512/down_sample_pcb
    Signal_name = 'Iref'
    # Mask_width  # per unit second, 0.2 before, and 0.8 after
    
    Mask_pre = int(0.2*Mask_width*F_sample/(F_sample/F_sample_pcb))
    Mask_after = int(0.8*Mask_width*F_sample/(F_sample/F_sample_pcb))
    
    PCB_Data_Raw = PCB_Read(Root_folder,Signal_name = Signal_name, Start_time=Start_time, Length=Ana_length, down_sample=down_sample_pcb)
    
    PCB_Data = Highpass_filter(PCB_Data_Raw, 400, F_sample_pcb, order=3)
    PCB_Data = np.append(np.array([0]),np.diff(PCB_Data,axis = 0))
    
    CI_Mask = np.abs(PCB_Data) < 1e-4  
    CI_temp = np.copy(CI_Mask)
    
    for idx in range(len(CI_temp)):
        if CI_temp[idx]==False:
            try:
                CI_Mask[idx-Mask_pre:idx+Mask_after] = False
            except Exception as error:
                print('\nError happens when generating CI_Mask: ')
                print(error)
                continue
   
    return np.repeat(CI_Mask,F_sample/F_sample_pcb)


def plot_spikes(data, spike_time):
    plt.figure()
    plt.plot(data)
    plt.scatter(spike_time, data[spike_time], c = 'r')
    
    
def plot_active_sites(Root):
    sites = np.loadtxt(Root + '/Spike_Count.csv', dtype=float, delimiter=',')  
    sites = sites.reshape(64, 64)
    
    plt.close()
    plt.figure(1)

    ax = sns.heatmap(np.log10(sites),cmap="jet", xticklabels=9, yticklabels=9)
    plt.title('Active Sites')
    plt.pause(0.1)
    plt.savefig(Root + '/Active_sites' + '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required
    

def Dir_Site_Gen(Root_folder, num_of_sites=1024, site_string=['0101','0102'], Positive=True, STA_Mode=1):
   
    if STA_Mode == 1:
        
        if Positive:
            print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of positive spikes')
            sites = np.loadtxt(Root_folder + '/Spike_Count.csv', dtype=float, delimiter=',')
        else:
            pass
        
        sites = np.stack((np.arange(sites.size), sites), axis=1) # add row number before the counts
        sites = sites[sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
        sites = np.array(sites[::-1],dtype=int) # make it descending
        site_index = sites[0:num_of_sites, 0] # extract the sites to do STA
        site_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in site_index] # convert site_index to site_string in format of row+column
    # Use the top 50 sites with most spikes or use designated site list
    else:
        print('Extract the spike time of the assigned sites in the site_list')
        site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    
    
    if Positive: 
        with open(Root_folder + '/Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
            rows = list(csv.reader(spiketime_read))
            Spike_Time=[rows[index] for index in site_index]
    else:
        with open(Root_folder + '/Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
            rows = list(csv.reader(spiketime_read))
            Spike_Time=[rows[index] for index in site_index]     
        
    ############## generate the STA file for the sites listed site_string #####################  

    if len(site_string) == len(Spike_Time): # to confirm how many rows have been read
        num_sites = len(Spike_Time)
        print('Spiketime reading is correct!')
    else:
        num_sites = len(Spike_Time)
        print('Spiketime reading is not correct!')
    STA_file_dir = Root_folder + '/STA_Files_AP/'
    isdir = os.path.isdir(STA_file_dir)
    if not isdir:
        os.mkdir(STA_file_dir)

    return STA_file_dir, site_string



def Spike_index_single_shortened(data, sigma= 5, win_size= 100):
    """
    Optimized version to return positive / negative indeces from a single site.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    sigma : TYPE, optional
        DESCRIPTION. The default is 5.
    win_size : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    Positive index, negative index (could be NA if none)
    Spike_count

    """
    try:
        row_len, col_len= np.shape(data)
    except:
        row_len, = np.shape(data)
        col_len = row_len

# Upper round win_num. The original code will create one more window when col_len can be devided by win_size just right with no remainder
    win_num = int(math.ceil(col_len/win_size))
#    win_num = round(col_len/win_size + 0.5) # make sure include all the data, even last window is not full
    
    # reshape to be matrix, with each row as a window; using resize will make sure the discrepencies in size will be filled in with zero at the end of array.
    data_reshape = np.copy(data)
    data_reshape.resize((win_num, win_size))
    
    # take means/std of each window simultaneously.
    win_mean = np.mean(data_reshape, axis = 1)
    win_std = np.std(data_reshape, axis = 1)
    limit = sigma * win_std
    above_limit = win_mean + limit
    below_limit = win_mean - limit
    
    win_max = np.amax(data_reshape, axis = 1)
    win_min = np.amin(data_reshape, axis = 1)
    max_index =(np.argmax(data_reshape, axis = 1) + np.arange(win_num) * win_size)
    min_index =(np.argmin(data_reshape, axis = 1) + np.arange(win_num) * win_size)
    pos_spike = win_max > above_limit    
    neg_spike = win_min < below_limit
    
    # extract effective spike index of the windows with True
    pos_index = (max_index.T)[pos_spike.T]
    neg_index = (min_index.T)[neg_spike.T]
    
    # extract the absolute value of the windows with True
    pos_amplitude = ((win_max - win_mean).T)[pos_spike.T]
    neg_amplitude = ((win_mean - win_min).T)[neg_spike.T]
    
    pos_spike_count =  sum(pos_spike)
    neg_spike_count =  sum(neg_spike)
    
    return pos_index, neg_index, pos_spike_count, neg_spike_count, pos_amplitude, neg_amplitude


def Generate_Spike_Index (Root_folder, sigma=5, win_size = 100, Start_time = 000, Plot_length = 1200, F_sample = 1205120/128):
    
    with open(Root_folder + '/Positive_Spike_Index.csv', 'w') as pos_spike_index_write, open(Root_folder + '/Negative_Spike_Index.csv', 'w') as neg_spike_index_write,\
         open(Root_folder + '/Positive_Spike_Count.csv', 'w') as pos_spikecount_write, open(Root_folder + '/Negative_Spike_Count.csv', 'w') as neg_spikecount_write,\
         open(Root_folder + '/Positive_Spike_Amplitude.csv', 'w') as pos_spike_amplitude_write, open(Root_folder + '/Negative_Spike_Amplitude.csv', 'w') as neg_spike_amplitude_write:

        # tqdm is used to track progress
        for Row_num in tqdm(np.arange(1, 65)):
            # print('spike detect in row: ' + str(Row_num))
            data_row = Row_Read(Root_folder,Row_num = Row_num, Start_time=Start_time, Length=Plot_length, F_sample=F_sample).T  # read one-row data 
                        
            for Col_num in np.arange(64):     
                pos_index, neg_index, pos_spike_count, neg_spike_count, pos_amplitude, neg_amplitude= Spike_index_single_shortened(data_row[Col_num], sigma = sigma, win_size = win_size) # detect the spikes in the row
                # sitename = 'Site-%02d%02d,' % (Row_num, Col_num + 1)
                sitename = ''
                np.savetxt(pos_spike_index_write, pos_index.reshape(1,-1), delimiter = ',', header = sitename)
                np.savetxt(neg_spike_index_write, neg_index.reshape(1,-1), delimiter = ',', header = sitename)
                pos_spikecount_write.write(sitename + '%1.4e' % pos_spike_count + '\n')
                neg_spikecount_write.write(sitename + '%1.4e' % neg_spike_count + '\n')
                
                np.savetxt(pos_spike_amplitude_write, pos_amplitude.reshape(1,-1), delimiter = ',', header = sitename)
                np.savetxt(neg_spike_amplitude_write, neg_amplitude.reshape(1,-1), delimiter = ',', header = sitename)
    return


def Generate_STA(Root_folder, num_of_sites=50, site_string=['0101','0102'], win_size=282, F_sample = 1205120/128, Positive=True, STA_Mode=1):
   
    if STA_Mode == 1:
        
        if Positive:
            print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of positive spikes')
            sites = np.loadtxt(Root_folder + '/Positive_Spike_Count.csv', dtype=float, delimiter=',')
        else:
            print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of negative spikes')
            sites = np.loadtxt(Root_folder + '/Negative_Spike_Count.csv', dtype=float, delimiter=',')
        
        sites = np.stack((np.arange(sites.size), sites), axis=1) # add row number before the counts
        sites = sites[sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
        sites = np.array(sites[::-1],dtype=int) # make it descending
        site_index = sites[0:num_of_sites, 0] # extract the sites to do STA
        site_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in site_index] # convert site_index to site_string in format of row+column
    # Use the top 50 sites with most spikes or use designated site list
    else:
        print('Extract the spike time of the assigned sites in the site_list')
        site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    
    
    if Positive: 
        with open(Root_folder + '/Positive_Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
            rows = list(csv.reader(spiketime_read))
            Spike_Time=[rows[index] for index in site_index]
    else:
        with open(Root_folder + '/Negative_Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
            rows = list(csv.reader(spiketime_read))
            Spike_Time=[rows[index] for index in site_index]     
        
    ############## generate the STA file for the sites listed site_string #####################  

    if len(site_string) == len(Spike_Time): # to confirm how many rows have been read
        num_sites = len(Spike_Time)
        print('Spiketime reading is correct!')
    else:
        num_sites = len(Spike_Time)
        print('Spiketime reading is not correct!')
    STA_file_dir = Root_folder + '/STA_Files/'
    isdir = os.path.isdir(STA_file_dir)
    if not isdir:
        os.mkdir(STA_file_dir)
        
    
    STA_Array_temp = np.zeros((64,win_size))
    STA_Row = np.zeros((64,win_size))
    
    for Row_num in range(64): # do average row by row
        print('Doing window average of row: ' + str(Row_num))
        Row_Data=Row_Read(Root_folder,Row_num+1, Start_time=0, Length=1200, F_sample=1205120/128).T
        for site in range(num_sites): # do average site by site within one row raw data
            num_spikes = float(len(Spike_Time[site]))
            STA_Row = np.zeros((64,win_size)) # to initialize the data within the window
            for t in range(len(Spike_Time[site])):
                peak_index =  int(float(Spike_Time[site][t])) # seems unable to convert to scientific format string to int directly
                win_data = Row_Data[:, peak_index-46:peak_index+(win_size-46)] # around 5ms before peak and 25ms after peak, when sampling freq = 9415
#Changed the hard number 282 to win_size here 
                if np.shape(win_data) == (64, win_size): # only do average for a whole window --> change 282 to window size 
                    STA_Row = STA_Row + win_data 
                else:
                    num_spikes = num_spikes -1 #  skip the spikes at very beginning or end of the recording, do not have a whole window
            STA_Array_temp = STA_Row/num_spikes
            with open(STA_file_dir + 'STA_' + site_string[site] + '.csv', 'a') as STA_file_append:  # save the spike time(index) to csv file
                np.savetxt(STA_file_append, STA_Array_temp, delimiter=',', fmt='%1.6e' )
    return STA_file_dir, site_string



def STA_Plot_Peak(STA_file_dir, site_string):
    #######% reading the saved STA files, and plot the waveform and STA image of listed sites ################
    print('Start plotting the STA peak image')    
    Image_folder = STA_file_dir + 'STA_Images_Peak/'
    isdir = os.path.isdir(Image_folder)  
    if isdir==False:
        os.mkdir(Image_folder)
    
    Waveform_folder = STA_file_dir + 'STA_Waveforms/'
    isdir = os.path.isdir(Waveform_folder)  
    if isdir==False:
        os.mkdir(Waveform_folder)
    
    site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    sum_array = np.zeros((64,64))
    for site_idx in range(len(site_string)):
        site = site_string[site_idx]
        site_arr_index = site_index[site_idx]
        
        try:
            STA_file = STA_file_dir + 'STA_'+ site + '.csv'
            STA_Array = np.loadtxt(STA_file, dtype=float, delimiter=',')
            
            STA_max= np.amax(STA_Array, axis=1)
            STA_mean = np.mean(STA_Array[:,0:200],axis=1)
            STA_min= np.amin(STA_Array, axis=1)
            STA_std= np.std(STA_Array[:,-200::], axis=1)
            STA_PP = (STA_max - STA_min)/(STA_std)
            
            final_array = STA_PP.reshape(64, 64) + 1e-6 # add a small number to avoid divide by zero
            sum_array = sum_array + final_array*(final_array>10) + 1e-6
            
            
            num_of_sites=6  #select the peak_sites in the title
            peak_sites =np.copy(STA_PP)
            peak_sites = np.stack((np.arange(peak_sites.size), peak_sites), axis=1) # add row number before the counts
            peak_sites = peak_sites[peak_sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
            peak_sites = np.array(peak_sites[::-1],dtype=int) # make it descending
            title_site_index = peak_sites[0:num_of_sites, 0] # extract the sites to do STA
            title_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in title_site_index] # convert site_index to site_string in format of row+column
            title = str(title_string)
            
            
            plt.close()
            plt.figure(1)
            plt.ion()
    
            ax = sns.heatmap(np.log10(final_array),cmap="jet", xticklabels=9, yticklabels=9)
            plt.title(title)
            plt.ioff()
            # plt.pause(0.1)
            plt.savefig(Image_folder + site + '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required
        
            plt.close()
            plt.figure(2)
            plt.ion()
            plt.plot(STA_Array[site_arr_index,:]) # plot the averaged spike waveform
            plt.title(site)
            plt.ioff()
            # plt.pause(0.1)
            plt.savefig(Waveform_folder + site + '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required
        except:
            print('error happens in site: ' +str(site) )
    plt.close()
    plt.figure(1)
    plt.ion()
    ax = sns.heatmap(np.log10(sum_array/len(site_string)),cmap="jet", xticklabels=9, yticklabels=9)
    plt.title('Comb')
    plt.ioff()
    # plt.pause(0.1)
    plt.savefig(Image_folder +'total_av' + '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required    
    
    plt.close()
    plt.figure(1)
    plt.ion()
    ax = sns.heatmap(np.log10(sum_array),cmap="jet", xticklabels=9, yticklabels=9)
    plt.title('Comb')
    plt.ioff()
    # plt.pause(0.1)
    plt.savefig(Image_folder +'total' + '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required    


def STA_Plot_Peak_Sub_Sites(Root_folder, num_of_sites=1024, site_string=['0101','0102'], amp_limit = 30e-3, amp_limit_pre = 3e-3, F_sample = 1205120/128, Positive=True, STA_Mode=1,  col_per_figure = 16, save_pdf = False):
    if STA_Mode == 1:
    
        print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of spikes')
        sites = np.loadtxt(Root_folder + '/Spike_Count.csv', dtype=float, delimiter=',')
        
        sites = np.stack((np.arange(sites.size), sites), axis=1) # add row number before the counts
        sites = sites[sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
        sites = np.array(sites[::-1],dtype=int) # make it descending
        site_index = sites[0:num_of_sites, 0] # extract the sites to do STA
        site_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in site_index] # convert site_index to site_string in format of row+column
    # Use the sites with most spikes or use designated site list
    else:
        print('Extract the spike time of the assigned sites in the site_list')
        site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    
    
    with open(Root_folder + '/Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
        rows = list(csv.reader(spiketime_read))
        Spike_Time=[rows[index] for index in site_index]
      
        
    ############## generate the STA file for the sites listed site_string #####################  
    
    if len(site_string) == len(Spike_Time): # to confirm how many rows have been read
        num_sites = len(Spike_Time)
        print('Spiketime reading is correct!')
    else:
        num_sites = len(Spike_Time)
        print('Spiketime reading is not correct!')
    STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+ '/'
    
    isdir = os.path.isdir(STA_file_dir)  
    if isdir==False:
        os.mkdir(STA_file_dir)
    
    
    print('Start plotting subsite waveform')    
    
    
    Waveform_folder = STA_file_dir + 'STA_Waveforms' + '/'
    isdir = os.path.isdir(Waveform_folder)  
    if isdir==False:
        os.mkdir(Waveform_folder)
    
    site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    sum_array = np.zeros((64,64))
    for site_idx in tqdm(range(len(site_string))):
        site = site_string[site_idx]
        site_arr_index = site_index[site_idx]
        
        Waveform_sub_folder = Waveform_folder + site +'/'
        isdir = os.path.isdir(Waveform_sub_folder)  
        if isdir==False:
            os.mkdir(Waveform_sub_folder)


        try:
            STA_file = STA_file_dir + 'STA_'+ site + '.csv'
            STA_Array = np.loadtxt(STA_file, dtype=float, delimiter=',')
            
            # tqdm is used to track progress
            for Col_num in tqdm(np.arange(1, 65)):
                
                ax_index = int((Col_num-1)%col_per_figure)
                
                if (Col_num-1)%col_per_figure ==0:
                    plt.close('all')
                    plt.rcParams["font.family"] = "Arial"
                    Font_Size = 6

                    fig, ax_array = plt.subplots(nrows=1, ncols=col_per_figure, gridspec_kw = {'wspace':0, 'hspace':0})
                    fig.set_size_inches(18.5, 10.5, forward=True)

                    # flatten the array of axes, which makes them easier to iterate through and assign
                    ax_array = ax_array.flatten()
                    plt.ion()
                    #fig.patch.set_alpha(0.1)
                    plt.tight_layout()    
                    # ax.patch.set_facecolor('white')
                
                for Row_num in np.arange(1, 65):     
                    Site_name = str(Row_num).zfill(2) + str(Col_num).zfill(2)
                    site_idx = (Row_num - 1)*64 + Col_num -1 
                    Site_Data = STA_Array[site_idx,:]

                    Site_Data_MA = Site_Data - np.min(Site_Data)
                    Site_Data_MA = Site_Data_MA/np.max(Site_Data_MA)
                    
                    Offset = - (Row_num)
                    t = np.arange(0,len(Site_Data_MA))*1/F_sample

                    ax_array[ax_index].plot(t, Moving_Average(Site_Data_MA,10) + Offset, linewidth=0.5, label= Site_name)
                    ax_array[ax_index].set_title('Col: ' + str(Col_num).zfill(2)) 

                    if Row_num == 64:
                        ax_array[ax_index].axis(ymin=-65,ymax=0)
                        ax_array[ax_index].axvline(x=0.025, color='green', linestyle=':', linewidth=0.25, label='axvline - full height')
                    
                if (Col_num-1)%col_per_figure == (col_per_figure -1):
                    # plt.ylim([-65, 1])
                    plt.ioff()          
                    plt.savefig(Waveform_sub_folder +'Column_'+str(Col_num-col_per_figure) + ' to ' +str(Col_num) + '.png', bbox_inches = 'tight',dpi=220) # change '.png' to '.pdf' if a vector image is required
                    if save_pdf:
                        plt.savefig(Waveform_sub_folder +'Column_'+str(Col_num-col_per_figure) + ' to ' +str(Col_num) + '.pdf', bbox_inches = 'tight') # change '.png' to '.pdf' if a vector image is required
                            
        except:
            print('error happens in site: ' +str(site) )

    return STA_file_dir, site_string


def STA_Plot_Time(STA_file_dir, site_string):
    #######% reading the saved STA files, and plot the waveform and STA image of listed sites ################
    print('Start plotting the STA time image')    
    Image_folder = STA_file_dir + 'STA_Images_Time/'
    isdir = os.path.isdir(Image_folder)  
    if isdir==False:
        os.mkdir(Image_folder)
    
    Waveform_folder = STA_file_dir + 'STA_Waveforms/'
    isdir = os.path.isdir(Waveform_folder)  
    if isdir==False:
        os.mkdir(Waveform_folder)
    
    site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    sum_array = np.zeros((64,64))
    for site_idx in range(len(site_string)):
        site = site_string[site_idx]
        site_arr_index = site_index[site_idx]
        
        STA_file = STA_file_dir + 'STA_'+ site + '.csv'
        STA_Array = np.loadtxt(STA_file, dtype=float, delimiter=',')
        

        
        STA_max= np.amax(STA_Array, axis=1)
        STA_min= np.amin(STA_Array, axis=1)
        STA_std= np.std(STA_Array[:,-200::], axis=1) #only take the last 15ms of data to calculate STD
        # Only plot the time when there is a signal, log10(15) translate to ~1.2 in the peak amplitude plot
        STA_signal = (STA_max - STA_min)/(STA_std) > 10 # was 15
        
        STA_time = np.argmax(STA_Array,axis=1) 
        STA_time = STA_time * STA_signal.T  # set the time for all the rest sites to 0

        final_array = STA_time.reshape(64, 64)
        sum_array = sum_array + final_array
        
        plt.close()
        plt.figure(1)
        plt.ion()

        ax = sns.heatmap(final_array,cmap="jet", xticklabels=9, yticklabels=9)
        plt.title(site)
        plt.ioff()
        # plt.pause(0.1)
        plt.savefig(Image_folder + site + '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required
    
        plt.close()
        plt.figure(2)
        plt.ion()
        plt.plot(STA_Array[site_arr_index,:]) # plot the averaged spike waveform
        plt.title(site)
        plt.ioff()
        # plt.pause(0.1)
        plt.savefig(Waveform_folder + site + '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required
        
    plt.close()
    plt.figure(1)
    ax = sns.heatmap(np.log10(sum_array/len(site_string)),cmap="jet", xticklabels=9, yticklabels=9)
    plt.title('Comb')
    plt.pause(0.1)
    plt.savefig(Image_folder +'total_av' + '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required    
    
    plt.close()
    plt.figure(1)
    ax = sns.heatmap(np.log10(sum_array),cmap="jet", xticklabels=9, yticklabels=9)
    plt.title('Comb')
    plt.pause(0.1)
    plt.savefig(Image_folder +'total' + '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required   

#%
def Peak_Images_Gen(STA_file_dir, site_string):
    #######% reading the saved STA files, and plot the waveform and STA image of listed sites ######
    print('Start plotting the STA images')    
    
    site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    for site_idx in range(len(site_string)):
        
        site = site_string[site_idx]
        
        Image_folder = STA_file_dir + 'STA_' + site+ '_Images' + '/'
        isdir = os.path.isdir(Image_folder)  
        if isdir==False:
            os.mkdir(Image_folder)
        
        
        site_arr_index = site_index[site_idx]
        
        STA_file = STA_file_dir + 'STA_'+ site + '.csv'
        STA_Array = np.loadtxt(STA_file, dtype=float, delimiter=',')
        
        STA_max= np.amax(STA_Array, axis=1, keepdims=True)
        STA_mean = np.mean(STA_Array[:,0:10],axis=1, keepdims=True)
        STA_min= np.amin(STA_Array, axis=1, keepdims=True)
        STA_std= np.std(STA_Array[:,-141::], axis=1, keepdims=True)
        STA_peak = (STA_max - STA_min)/(STA_std) > 10
        
        STA_norm = (STA_Array - STA_mean)/(STA_std) - 5*STA_peak
        
        
        try:
            row_len, col_len= np.shape(STA_norm)
            print('columns num: '+str(col_len))
        except:
            row_len, = np.shape(STA_norm) # in case of 1 row
            col_len=1
        for i in range(col_len):
            single_array = STA_norm[:,i].reshape(64,64)
            plt.close()
            plt.figure(1)
            ax = sns.heatmap(single_array,vmin=-25, vmax=25, cmap="jet", xticklabels=9, yticklabels=9)
            plt.title(site)
            plt.pause(0.1)
            plt.savefig(Image_folder + site + '_' + str(i) + '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required
    return


def Peak_Images_Gen_Comb(STA_file_dir, site_string):
    #######% reading the saved STA files, and plot the waveform and STA image of listed sites ######
    print('Start plotting the STA images')    
    
    site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    sum_array = np.zeros((4096,282))
    for site_idx in range(len(site_string)):
        
        site = site_string[site_idx]
        
        Image_folder = STA_file_dir + 'Comb_Images' + '/'
        isdir = os.path.isdir(Image_folder)  
        if isdir==False:
            os.mkdir(Image_folder)
        
        
        site_arr_index = site_index[site_idx]
        
        STA_file = STA_file_dir + 'STA_'+ site + '.csv'
        STA_Array = np.loadtxt(STA_file, dtype=float, delimiter=',')
        
        STA_max= np.amax(STA_Array, axis=1, keepdims=True)
        STA_mean = np.mean(STA_Array[:,128::],axis=1, keepdims=True)
        STA_min= np.amin(STA_Array, axis=1, keepdims=True)
        STA_std= np.std(STA_Array[:,-141::], axis=1, keepdims=True)
        STA_peak = (STA_max - STA_min)/(STA_std) > 10
        
        STA_norm = (STA_Array - STA_mean)/(STA_std) + 2*STA_peak
        
        sum_array = sum_array + STA_norm
        
        
    try:
        row_len, col_len= np.shape(sum_array)
        print('columns num: '+str(col_len))
    except:
        row_len, = np.shape(sum_array) # in case of 1 row
        col_len=1
    for i in range(col_len):
        single_array = sum_array[:,i].reshape(64,64)
        plt.close()
        plt.figure(1)
        ax = sns.heatmap(single_array,vmin=-25, vmax=25, cmap="jet", xticklabels=9, yticklabels=9)
        plt.title('Comb')
        plt.pause(0.1)
        plt.savefig(Image_folder + 'Comb_' + str(i) + '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required
    return


def Movie_Gen_Single(STA_file_dir, site_string, start=0, end=282):    
    for site_name in site_string:
        image_folder = STA_file_dir + 'STA_' + site_name + '_Images' + '/'
        video_name = image_folder + site_name + '_video.avi'
        
        FPS =16
        fourcc = VideoWriter_fourcc(*'MP42') # originall it is 0
        
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        
        video = cv2.VideoWriter(video_name, 0, FPS, (width,height))
        
        for image_name in np.arange(start, end):
            frame=cv2.imread(os.path.join(image_folder, site_name + '_' + str(image_name) + '.png'))
            video.write(frame)
        
        cv2.destroyAllWindows()
        video.release()


def Movie_Gen_Comb(STA_file_dir, site_string = '', start=0, end=282):    
    for site_name in site_string:
        image_folder = STA_file_dir + 'Comb_Images' + '/'
        video_name = image_folder + 'Comb_video.avi'
        
        FPS =16
        fourcc = VideoWriter_fourcc(*'MP42') # originall it is 0
        
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        
        video = cv2.VideoWriter(video_name, 0, FPS, (width,height))
        
        for image_name in np.arange(start, end):
            frame=cv2.imread(os.path.join(image_folder, 'Comb_' + str(image_name) + '.png'))
            video.write(frame)
        
        cv2.destroyAllWindows()
        video.release()




def Spike_Video_Gen(Root_folder, Start_time = 0, Length = 10, Speed_factor = 5, upward_spke = True):

    ##### Please do not change the code below, unless you do know what you are doing!
    ##### Please do not change the code below, unless you do know what you are doing!
    File_path = Root_folder
    
    if upward_spke:
        File_name = 'Positive_Spike_Index.csv' #
    else:
        File_name = 'Negative_Spike_Index.csv' #
        
    Step = 282*Speed_factor
    Video_folder =File_path + '/Videos/'
    Video_name = 'from_' + str(Start_time) + 's_to_'+ str(Start_time+Length) + 's.mp4'
    isdir = os.path.isdir(Video_folder)  
    if not isdir:
        os.mkdir(Video_folder)  
    
    with open(File_path + '/'+File_name, 'r') as sp_index:  # read the spiketime of interested sites
        sp_rows = list(csv.reader(sp_index))
    
    Fs = 9415
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # FourCC code for MP4 codec
    out = cv2.VideoWriter(Video_folder + Video_name, fourcc, 33, (64, 64), isColor=False)  # Output video file name, codec, frames per second, frame size, and grayscale flag
    
    for frame_idx, frame_value in enumerate(tqdm(np.arange(Start_time*Fs,(Start_time+Length)*Fs,Step))):
        sp_temp =np.zeros(4096)
        for idx, item in enumerate(sp_rows): 
            item = np.array(item,dtype = float)
            sp_temp[idx] = np.sum((item >= frame_value) & (item < frame_value + Step))
        sp_temp = sp_temp*255/(Speed_factor/2)    
        out.write(sp_temp.reshape(64, 64).astype(np.uint8))
    
    out.release()
    print("Video generation complete.")   






def Plot_Save(Root_folder):
    # Root_folder = r'E:\CNEA_V2\2021_12_17_analyzed\WB23_4_03_03_37\Array Recording Intra\2021_12_17-03_54_08_Converted'  # intracellular recording
    
    Start_time = 000 # Unit second
    Plot_length = 2400 # Unit second
    
    F_sample = 1205120/128
    F_sample_PCB = 120512
    
    Image_folder = Root_folder + 'Images/'
    isdir = os.path.isdir(Image_folder)  
    if not isdir:
        os.mkdir(Image_folder)  
    
    
    for Row in tqdm(range(0,64)):
        
        Row_Data = -Row_Read(Root_folder, Row+1 ,Start_time=Start_time, Length=Plot_length)
    
        for Col in range(0,64):
            Site_name = str(Row+1).zfill(2) + str(Col+1).zfill(2)
            Site_Data = Row_Data[:,Col] # extract the specific column
            Site_Data = Site_Data - Moving_Average(Site_Data,2000)
            Mean_Data = np.mean(Site_Data)
            time = np.arange(Start_time,Start_time+len(Site_Data)/F_sample,1/F_sample)
            #print(Site_Data)
            
            #%
            #fig = plt.figure(figsize=(12.0, 10.0))
            plt.close('all')
            fig = plt.figure(1)
            plt.ion()
            #fig.patch.set_alpha(0.1)
            ax = fig.add_subplot(111)
            ax.patch.set_alpha(0.0)
            plt.tight_layout()    
            ax.patch.set_facecolor('white')
            Font_Size = 20
            plt.plot(time, Site_Data,color='black',linewidth=0.6,label=Site_name)
            
            plt.ylim([Mean_Data - 0.02, Mean_Data + 0.05])
            plt.title('Raw Data ',fontsize=Font_Size+8, fontweight='bold')
            plt.xlabel('Time $(s)$', fontsize=Font_Size, fontweight='bold',position=(0.5, 0.1))
            plt.ylabel('Amplitude ($V$)',  fontsize=Font_Size, fontweight='bold', position=(0, 0.5))
            ax.spines['left'].set_position(('outward',0))
            ax.spines['bottom'].set_position(('outward',0))
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            #ax.axhline(linewidth=2, color='black')
            #ax.axvline(linewidth=2, color='black') 
            # Eliminate upper and right axes
            #ax.spines['right'].set_color('none')
            #ax.spines['top'].set_color('none')
            # Show ticks in the left and lower axes only
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            plt.xticks(fontsize=Font_Size,fontweight='bold')
            plt.yticks(fontsize=Font_Size,fontweight='bold')
            plt.legend(loc='upper right',fontsize=Font_Size-4)
            #plt.text(x, y, 'add text here', fontsize=Font_Size, color='black')
            # plt.pause(0.001)
            plt.ioff()
            plt.savefig(Image_folder+Site_name+ '.png', bbox_inches = 'tight',dpi=200) # change '.png' to '.pdf' if a vector image is required





def Spike_Index_Gen_Intra_v2(Root_folder, Site_name, Start_time=0, Ana_length = 2400, CI_mask = np.array([True]),Threshold = 0.3):

    # Start_time = 0 # Unit second
    # Ana_length = 2400 # Unit second
    
    Gain_intra = 30
    F_sample = 1205120/128
    Site_Data = -Site_Read(Root_folder,Site_name,Start_time=Start_time, Length=Ana_length)
    # Site_Data_MA = Moving_Average(Site_Data - Moving_Average(Site_Data,2000), 4)
    Site_Data_MA = Site_Data - Moving_Average(Site_Data,2000)
    
    Site_Data_MA = (Site_Data_MA/Gain_intra*1e3)*CI_mask # unit is mV
    
    big_win_time = 20
    big_win_size = int(big_win_time*F_sample)
    win_size = int(0.2*F_sample)

    
    Spike_Win_Count = 0
    Spike_Count = 0
    Spike_Index = []
    Spike_Amp = []

    base_index = int(Start_time*F_sample)
    Ana_length = len(Site_Data_MA)/F_sample # real length
    
    for start_time in np.arange(0,Ana_length,big_win_time):
    
        start_index = int(start_time*F_sample)
        data = Site_Data_MA[start_index:start_index+big_win_size]
        
        try:
            row_len, col_len= np.shape(data)
        except:
            row_len, = np.shape(data)
            col_len = row_len
        
        
        
        # if current injection or Vs4 adjustion:
        # if current injection or Vs4 adjustion, skip the this time window
        win_cj_num = int(math.ceil(col_len/(win_size/10)))
        data_cj = np.copy(data)
        data_cj.resize((win_cj_num, int(win_size/10)))
        data_cj_diff2 = np.diff(data_cj, n =2, axis = 1)
        # print(np.shape(data_cj))
        # data_cj_diff2_max = np.amax(data_cj_diff2, axis = 1)
        # data_cj_diff2_min = np.amin(data_cj_diff2, axis = 1)
        data_cj_diff2_std = np.std(data_cj_diff2,axis=1)
        
        # print(data_cj)
        # print(data_cj_diff2_std)
        # print('max of diff3:')
        # print(np.min(data_cj_diff2_max - data_cj_diff2_min))
        # print(np.min(data_cj_diff2_std))
        
        try:
            if(np.min(data_cj_diff2_std) < 0.013): #amplifier get stuck because of current injection or change Vs4,2nd order ofsignal is flat
                continue
        except Exception as error:
            print('\nError happens in site: ' + Site_name + ', and the error message is: ')
            print(error)
            continue

        
        # Upper round win_num. The original code will create one more window when col_len can be devided by win_size just right with no remainder
        win_num = int(math.ceil(col_len/win_size))
        #    win_num = round(col_len/win_size + 0.5) # make sure include all the data, even last window is not full
        
        # reshape to be matrix, with each row as a window; using resize will make sure the discrepencies in size will be filled in with zero at the end of array.
        data_reshape = np.copy(data)
        data_reshape.resize((win_num, win_size))
        
        # take means/std of each window simultaneously.
        win_mean = np.mean(data_reshape, axis = 1)
        # win_mean = np.mean(data)
        # win_std = np.std(data_reshape, axis = 1)
        
        
        # above_limit = sigma * win_std
        above_limit = win_mean + Threshold  # 5*std, std = 0.025 for MH device, and 0.06 for SH device

        
        win_max = np.amax(data_reshape, axis = 1)
        win_min = np.amin(data_reshape, axis = 1)
        max_index = (np.argmax(data_reshape, axis = 1) + np.arange(win_num) * win_size)
        
        
        not_edge = ((max_index % win_size) > 15) & ((win_size - max_index % win_size) > 15)  # in case one spike being split into two small windows, use 15 data points to be conservative
        
        
        pos_spike = (win_max > above_limit) & (abs(win_min)<0.6*abs(win_max)) & (win_min<0) & (win_max >0) & not_edge # basic condition of a upward peak
        pos_spike_count =  sum(pos_spike)
        

        
        if pos_spike_count > 0:
            ## add differential to recognize steep rising noise ####
            data_diff = np.diff(data_reshape, axis =1)
            diff_max = np.amax(data_diff, axis = 1)
            diff_min = np.amin(data_diff, axis = 1)
            
            # diff_max_roll1 = np.roll(diff_max,1) # remove the bigger jump neighbour
            # diff_max_roll1[0] = 0
            # # diff_max_roll2 = np.roll(diff_max,2)
            
            # diff_min_roll1 = np.roll(diff_min,-1) # remove the bigger jump neighbour
            # diff_min_roll1[-1] = 0
            # # diff_min_roll2 = np.roll(diff_min,-2)
            
            
            data_diff2 = np.diff(data_diff, axis = 1)
            diff2_max = np.amax(data_diff2, axis = 1)
            diff2_min = np.amin(data_diff2, axis = 1)
            
            # print('max of diff2:')
            # print(np.min(diff2_max - diff2_min))
            # print(np.std(diff2_max - diff2_min))
            
            # if (np.min(diff2_max - diff2_min)<0.25) | (np.min(diff2_std) < 0.04): #amplifier get stuck because of current injection or change Vs4
            #     continue

        # pos_spike = (win_max > above_limit) & (win_max < noise_limit) & (abs(win_min)<0.6*abs(win_max)) & (win_min<0) # the conditions of being an intra spike
        # pos_spike = (win_max > above_limit) & (win_max < noise_limit) & (win_min<0) & (diff_max < 0.13) & (diff_min > - 0.13) & (diff_max_roll1 < 0.13) & (diff_max_roll2 < 0.13) & (diff_min_roll1 > - 0.13) & (diff_min_roll2 > - 0.13)   
        # pos_spike = (win_max > above_limit) & (win_min<0) & (diff_max < 0.13) & (diff_min > - 0.13) & (diff_max_roll1 < 0.13) & (diff_min_roll1 > - 0.13)   # the conditions of being an intra spike
            
            pos_spike = pos_spike & (abs(diff_max)< 0.28) & (abs(diff_min)< 0.15) # condition for most of slow intra spikes
            
            if np.sum(abs(diff_max)> 0.28):
                is_big_spike = (abs(diff_min)<0.35*abs(diff_max))&(abs(diff2_max) < 0.35*abs(diff_max))&(abs(diff2_min) < 0.35*abs(diff_max)) # add big intra spike
                pos_spike = pos_spike | is_big_spike
        
        
            
            pos_spike_count =  sum(pos_spike) # recount valid spikes
            if pos_spike_count > 3: # at least there are 4 spikes in the time window

                # extract effective spike index of the windows with True
                pos_index = (max_index.T)[pos_spike.T]
                
                # pos_index_temp = np.copy(pos_index)
                # for index in range(len(pos_index)-1): # remove the repeated spike because a spike can be splitted into two windows
                #     if pos_index_temp[index+1]-pos_index_temp[index] < int(win_size/4): # means very close
                #         if data[pos_index_temp[index+1]] < data[pos_index_temp[index]]:
                #             pos_index = np.delete(pos_index,np.where(pos_index == pos_index_temp[index+1]))
                #             # pos_spike[np.where(max_index == pos_index_temp[index+1])] = False
                #         else:
                #             pos_index = np.delete(pos_index,np.where(pos_index == pos_index_temp[index]))
                #             # pos_spike[np.where(max_index == pos_index_temp[index])] = False
                            
                
                # extract the absolute value of the windows with True
                # pos_amplitude = 1.3*((win_max - win_mean).T)[pos_spike.T] # add 30% negtive part
                pos_amplitude = 1.3* data[pos_index] # add 30% negtive part
                pos_index = pos_index + start_index + base_index # calculate the absolute index value
    
                Spike_Index = Spike_Index + pos_index.tolist()
                Spike_Amp = Spike_Amp + pos_amplitude.tolist()
                Spike_Count = Spike_Count + pos_spike_count
                Spike_Win_Count = Spike_Win_Count + 1
    
    Spike_Index = np.array(Spike_Index, dtype=int)
    Spike_Amp = np.array(Spike_Amp)
    # print(Spike_Index[0:10])
    # print(Spike_Amp)
    # print(Spike_Index)
    # print(Spike_Count)
    return Spike_Index, Spike_Amp, Spike_Count, Spike_Win_Count, Site_Data_MA


def Plot_Save_Labelled_v2(Root_folder,Start_time=0, Ana_length= 2400, Save_fig= False, Row_range= [1, 64], Threshold = 0.3, Diff_order = [0], Sparse= 10):
    F_sample = 1205120/128
    
    file_path = Path(Root_folder)
    parent_path= str(file_path.parent.absolute())
    
    # parent_path = 'D:\\Data_Process' + parent_path[2::] # Comment this line if you don't want to change the processing results to Disk D
    
    # time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")  + '/' 
    # ana_path = parent_path +'/Analysis_from'+str(Start_time)+'to'+str(Start_time + Ana_length) +'_' + time_stamp # uncommnet to save the spike detction results in specific folder with time stamp
    
    # Image_folder = ana_path  + '/Images_Labelled/'
    # Stat_folder = ana_path + '/Stat_Results/'

    # isdir = os.path.isdir(Stat_folder)  
    # if not isdir:
    #     os.makedirs(Stat_folder)  

    Image_folder = Root_folder  + '/Images_Labelled/'
    Stat_folder = Root_folder
    isdir = os.path.isdir(Image_folder)  
    if not isdir:
        os.makedirs(Image_folder)  
    
    CI_mask = CI_Mask_Gen(Root_folder, Start_time, Ana_length, Mask_width=2)
    
    with open(Stat_folder + 'Spike_Index.csv', 'w') as spike_index_write,\
          open(Stat_folder + 'Spike_Count.csv', 'w') as spike_count_write,\
          open(Stat_folder + 'Spike_Win_Count.csv', 'w') as spike_win_count_write,\
          open(Stat_folder + 'Spike_Amplitude.csv', 'w') as spike_amplitude_write:
    
        # tqdm is used to track progress
        for Row_num in tqdm(np.arange(Row_range[0], Row_range[1]+1)):
            for Col_num in np.arange(1, 65):     
                Site_name = str(Row_num).zfill(2) + str(Col_num).zfill(2)
                Spike_Index, Spike_Amp, Spike_Count, Spike_Win_Count, Site_Data_MA = Spike_Index_Gen_Intra_v2(Root_folder, Site_name, Start_time, Ana_length, CI_mask = CI_mask, Threshold = Threshold)
                sitename = ''
                np.savetxt(spike_index_write, Spike_Index.reshape(1,-1), delimiter = ',', header = sitename)
                spike_count_write.write(sitename + '%1.4e' % Spike_Count + '\n')
                spike_win_count_write.write(sitename + '%1.4e' % Spike_Win_Count + '\n')
                np.savetxt(spike_amplitude_write, Spike_Amp.reshape(1,-1), delimiter = ',', header = sitename)
                
                if Save_fig:
                    plt.close('all')
                    fig = plt.figure(1)
                    plt.ion()
                    #fig.patch.set_alpha(0.1)
                    ax = fig.add_subplot(111)
                    ax.patch.set_alpha(0.0)
                    plt.tight_layout()    
                    ax.patch.set_facecolor('white')
                    Font_Size = 20
                    
                    plt.scatter(Spike_Index/F_sample, Site_Data_MA[Spike_Index - int(Start_time*F_sample)], c = 'r')
                    
                    t_xaxis = np.arange(Start_time,len(Site_Data_MA)/F_sample + Start_time,1/F_sample)
                    plt.plot(t_xaxis[::Sparse], Site_Data_MA[::Sparse], color ='darkblue', linewidth=0.6,label=Site_name)    
                    
                    
                    
                    diff_order =1
                    if diff_order in Diff_order:
                        data_diff = np.diff(Site_Data_MA, diff_order)
                        t_diff = np.arange(Start_time, len(data_diff)/F_sample + Start_time, 1/F_sample)
                        
                        point_error = int(len(t_diff) - len(data_diff))
                        
                        if point_error ==0:
                            plt.plot(t_diff, data_diff-1, color ='pink', linewidth=0.6,label=Site_name + '_1st order')
                        elif point_error >0:
                            plt.plot(t_diff[0:-point_error], data_diff- 1, color ='pink', linewidth=0.6,label=Site_name + '_1st order')
                        elif point_error <0:
                            plt.plot(t_diff, data_diff[0:point_error] -1, color ='pink', linewidth=0.6,label=Site_name + '_1st order')
                        
                        
                    
                    diff_order =2
                    if diff_order in Diff_order:
                        data_diff = np.diff(Site_Data_MA, diff_order)
                        t_diff = np.arange(Start_time, len(data_diff)/F_sample + Start_time, 1/F_sample)
                        
                        point_error = int(len(t_diff) - len(data_diff))
                        
                        if point_error ==0:
                            plt.plot(t_diff, data_diff-2, color ='green', linewidth=0.6,label=Site_name + '_2nd order')
                        elif point_error >0:
                            plt.plot(t_diff[0:-point_error], data_diff-2, color ='green', linewidth=0.6,label=Site_name + '_2nd order')
                        elif point_error <0:
                            plt.plot(t_diff, data_diff[0:point_error] -2, color ='green', linewidth=0.6,label=Site_name + '_2nd order')
                        
                    
                    
                    offset = 0
                    if len(Spike_Amp) > 0:
                        offset = max(Spike_Amp)
                    
                    plt.ylim([-0.3*offset-2.3, offset+0.5])
                    plt.title('Raw Data ',fontsize=Font_Size+8, fontweight='bold')
                    plt.xlabel('Time $(s)$', fontsize=Font_Size, fontweight='bold',position=(0.5, 0.1))
                    plt.ylabel('Amplitude ($mV$)',  fontsize=Font_Size, fontweight='bold', position=(0, 0.5))
                    ax.spines['left'].set_position(('outward',0))
                    ax.spines['bottom'].set_position(('outward',0))
                    ax.spines['bottom'].set_linewidth(2)
                    ax.spines['left'].set_linewidth(2)
                    #ax.axhline(linewidth=2, color='black')
                    #ax.axvline(linewidth=2, color='black') 
                    # Eliminate upper and right axes
                    #ax.spines['right'].set_color('none')
                    #ax.spines['top'].set_color('none')
                    # Show ticks in the left and lower axes only
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    plt.xticks(fontsize=Font_Size,fontweight='bold')
                    plt.yticks(fontsize=Font_Size,fontweight='bold')
                    plt.legend(loc='upper right',fontsize=Font_Size-4)
                    #plt.text(x, y, 'add text here', fontsize=Font_Size, color='black')
                    # plt.pause(0.001)
                    plt.ioff()
                    plt.savefig(Image_folder +Site_name+ '.png', bbox_inches = 'tight',dpi=100) # change '.png' to '.pdf' if a vector image is required  
                    plt.close(fig)
                    
                    
def Spike_Index_Gen_Intra_VC(Root_folder, Site_name, Start_time=0, Ana_length = 2400, Threshold = 0.3, Vs1= 2.6, TGain= 700, F_sample= 1205120/128 ):

    # TGain # transimpedance gain, unit: Mega ohm
    # F_sample = 1205120/128
    Site_Data = Site_Read(Root_folder,Site_name,Start_time=Start_time, Length=Ana_length)

    Site_Data_MA = Vs1 - Site_Data 
    
    Site_Data_MA = (Site_Data_MA/TGain)*1e3 # unit is nA
    
    big_win_time = 20
    big_win_size = int(big_win_time*F_sample)
    win_size = int(0.2*F_sample)

    
    Spike_Win_Count = 0
    Spike_Count = 0
    Spike_Index = []
    Spike_Amp = []
    Site_DC = np.mean(Site_Data_MA)

    base_index = int(Start_time*F_sample)
    Ana_length = len(Site_Data_MA)/F_sample # real length
    
    for start_time in np.arange(0,Ana_length,big_win_time):
    
        start_index = int(start_time*F_sample)
        data = Site_Data_MA[start_index:start_index+big_win_size]
        
        try:
            row_len, col_len= np.shape(data)
        except:
            row_len, = np.shape(data)
            col_len = row_len
        
        
        
        # if current injection or Vs4 adjustion:
        # if current injection or Vs4 adjustion, skip the this time window
        win_cj_num = int(math.ceil(col_len/(win_size/10)))
        data_cj = np.copy(data)
        data_cj.resize((win_cj_num, int(win_size/10)))
        data_cj_diff2 = np.diff(data_cj, n =2, axis = 1)
        # print(np.shape(data_cj))
        # data_cj_diff2_max = np.amax(data_cj_diff2, axis = 1)
        # data_cj_diff2_min = np.amin(data_cj_diff2, axis = 1)
        data_cj_diff2_std = np.std(data_cj_diff2,axis=1)
        
        # print(data_cj)
        # print(data_cj_diff2_std)
        # print('max of diff3:')
        # print(np.min(data_cj_diff2_max - data_cj_diff2_min))
        # print(np.min(data_cj_diff2_std))
        
        try:
            if(np.min(data_cj_diff2_std) < 0.013): #amplifier get stuck because of current injection or change Vs4,2nd order ofsignal is flat
                continue
        except Exception as error:
            print('\nError happens in site: ' + Site_name + ', and the error message is: ')
            print(error)
            continue

        
        # Upper round win_num. The original code will create one more window when col_len can be devided by win_size just right with no remainder
        win_num = int(math.ceil(col_len/win_size))
        #    win_num = round(col_len/win_size + 0.5) # make sure include all the data, even last window is not full
        
        # reshape to be matrix, with each row as a window; using resize will make sure the discrepencies in size will be filled in with zero at the end of array.
        data_reshape = np.copy(data)
        data_reshape.resize((win_num, win_size))
        
        # take means/std of each window simultaneously.
        win_mean = np.mean(data_reshape, axis = 1)
        # win_mean = np.mean(data)
        # win_std = np.std(data_reshape, axis = 1)
        
        
        # above_limit = sigma * win_std
        above_limit = win_mean + Threshold  # 5*std, std = 0.025 for MH device, and 0.06 for SH device

        
        win_max = np.amax(data_reshape, axis = 1)
        win_min = np.amin(data_reshape, axis = 1)
        max_index = (np.argmax(data_reshape, axis = 1) + np.arange(win_num) * win_size)
        
        
        not_edge = ((max_index % win_size) > 15) & ((win_size - max_index % win_size) > 15)  # in case one spike being split into two small windows, use 15 data points to be conservative
        
        
        pos_spike = (win_max > above_limit) & (abs(win_min)<0.6*abs(win_max)) & (win_min<0) & (win_max >0) & not_edge # basic condition of a upward peak
        pos_spike_count =  sum(pos_spike)
        

        
        if pos_spike_count > 0:
            ## add differential to recognize steep rising noise ####
            data_diff = np.diff(data_reshape, axis =1)
            diff_max = np.amax(data_diff, axis = 1)
            diff_min = np.amin(data_diff, axis = 1)
            
            # diff_max_roll1 = np.roll(diff_max,1) # remove the bigger jump neighbour
            # diff_max_roll1[0] = 0
            # # diff_max_roll2 = np.roll(diff_max,2)
            
            # diff_min_roll1 = np.roll(diff_min,-1) # remove the bigger jump neighbour
            # diff_min_roll1[-1] = 0
            # # diff_min_roll2 = np.roll(diff_min,-2)
            
            
            data_diff2 = np.diff(data_diff, axis = 1)
            diff2_max = np.amax(data_diff2, axis = 1)
            diff2_min = np.amin(data_diff2, axis = 1)
            
            # print('max of diff2:')
            # print(np.min(diff2_max - diff2_min))
            # print(np.std(diff2_max - diff2_min))
            
            # if (np.min(diff2_max - diff2_min)<0.25) | (np.min(diff2_std) < 0.04): #amplifier get stuck because of current injection or change Vs4
            #     continue

        # pos_spike = (win_max > above_limit) & (win_max < noise_limit) & (abs(win_min)<0.6*abs(win_max)) & (win_min<0) # the conditions of being an intra spike
        # pos_spike = (win_max > above_limit) & (win_max < noise_limit) & (win_min<0) & (diff_max < 0.13) & (diff_min > - 0.13) & (diff_max_roll1 < 0.13) & (diff_max_roll2 < 0.13) & (diff_min_roll1 > - 0.13) & (diff_min_roll2 > - 0.13)   
        # pos_spike = (win_max > above_limit) & (win_min<0) & (diff_max < 0.13) & (diff_min > - 0.13) & (diff_max_roll1 < 0.13) & (diff_min_roll1 > - 0.13)   # the conditions of being an intra spike
            
            pos_spike = pos_spike & (abs(diff_max)< 0.28) & (abs(diff_min)< 0.15) # condition for most of slow intra spikes
            
            if np.sum(abs(diff_max)> 0.28):
                is_big_spike = (abs(diff_min)<0.35*abs(diff_max))&(abs(diff2_max) < 0.35*abs(diff_max))&(abs(diff2_min) < 0.35*abs(diff_max)) # add big intra spike
                pos_spike = pos_spike | is_big_spike
        
        
            
            pos_spike_count =  sum(pos_spike) # recount valid spikes
            if pos_spike_count > 3: # at least there are 4 spikes in the time window

                # extract effective spike index of the windows with True
                pos_index = (max_index.T)[pos_spike.T]
                
                # pos_index_temp = np.copy(pos_index)
                # for index in range(len(pos_index)-1): # remove the repeated spike because a spike can be splitted into two windows
                #     if pos_index_temp[index+1]-pos_index_temp[index] < int(win_size/4): # means very close
                #         if data[pos_index_temp[index+1]] < data[pos_index_temp[index]]:
                #             pos_index = np.delete(pos_index,np.where(pos_index == pos_index_temp[index+1]))
                #             # pos_spike[np.where(max_index == pos_index_temp[index+1])] = False
                #         else:
                #             pos_index = np.delete(pos_index,np.where(pos_index == pos_index_temp[index]))
                #             # pos_spike[np.where(max_index == pos_index_temp[index])] = False
                            
                
                # extract the absolute value of the windows with True
                # pos_amplitude = 1.3*((win_max - win_mean).T)[pos_spike.T] # add 30% negtive part
                pos_amplitude = 1.3* data[pos_index] # add 30% negtive part
                pos_index = pos_index + start_index + base_index # calculate the absolute index value
    
                Spike_Index = Spike_Index + pos_index.tolist()
                Spike_Amp = Spike_Amp + pos_amplitude.tolist()
                Spike_Count = Spike_Count + pos_spike_count
                Spike_Win_Count = Spike_Win_Count + 1
    
    Spike_Index = np.array(Spike_Index, dtype=int)
    Spike_Amp = np.array(Spike_Amp)
    
    # print(Spike_Index[0:10])
    # print(Spike_Amp)
    # print(Spike_Index)
    # print(Spike_Count)
    return Spike_Index, Spike_Amp, Spike_Count, Spike_Win_Count, Site_Data_MA, Site_DC


def Plot_Save_Labelled_VC(Root_folder, Start_time=0, Ana_length= 2400, Save_fig= False, Row_range= [1, 64], Threshold= 0.3, Diff_order= [0], Sparse= 10, Vs1= 2.6, TGain= 700, F_sample= 1205120/128):
    # F_sample = 1205120/128
    
    file_path = Path(Root_folder)
    parent_path= str(file_path.parent.absolute())
    
    # parent_path = 'D:\\Data_Process' + parent_path[2::] # Comment this line if you don't want to change the processing results to Disk D
    
    time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")  + '/'
    
    ana_path = parent_path +'/Analysis_from'+str(Start_time)+'to'+str(Start_time + Ana_length) +'_'+ time_stamp
    
    Image_folder = ana_path  + '/Images_Labelled/'
    Stat_folder = ana_path + '/Stat_Results/'
    
    isdir = os.path.isdir(Image_folder)  
    if not isdir:
        os.makedirs(Image_folder)  
    
    isdir = os.path.isdir(Stat_folder)  
    if not isdir:
        os.makedirs(Stat_folder)  
    
    
    # CI_mask = CI_Mask_Gen(Root_folder, Start_time, Ana_length, Mask_width=2)
    
    with open(Stat_folder + 'Spike_Index.csv', 'w') as spike_index_write,\
          open(Stat_folder + 'Spike_Count.csv', 'w') as spike_count_write,\
          open(Stat_folder + 'Spike_Win_Count.csv', 'w') as spike_win_count_write,\
          open(Stat_folder + 'Spike_Amplitude.csv', 'w') as spike_amplitude_write,\
          open(Stat_folder + 'Site_DC.csv', 'w') as site_dc_write:
    
        # tqdm is used to track progress
        for Row_num in tqdm(np.arange(Row_range[0], Row_range[1]+1)):
            for Col_num in np.arange(1, 65):     
                Site_name = str(Row_num).zfill(2) + str(Col_num).zfill(2)
                Spike_Index, Spike_Amp, Spike_Count, Spike_Win_Count, Site_Data_MA, Site_DC = Spike_Index_Gen_Intra_VC(Root_folder, Site_name, Start_time, Ana_length, Threshold = Threshold, Vs1= 2.6, TGain= 700, F_sample= 1205120/128)
                sitename = ''
                np.savetxt(spike_index_write, Spike_Index.reshape(1,-1), delimiter = ',', header = sitename)
                spike_count_write.write(sitename + '%1.4e' % Spike_Count + '\n')
                spike_win_count_write.write(sitename + '%1.4e' % Spike_Win_Count + '\n')
                np.savetxt(spike_amplitude_write, Spike_Amp.reshape(1,-1), delimiter = ',', header = sitename)
                site_dc_write.write(sitename + '%1.4e' % Site_DC + '\n')
                
                if Save_fig:
                    plt.close('all')
                    fig = plt.figure(1)
                    plt.ion()
                    #fig.patch.set_alpha(0.1)
                    ax = fig.add_subplot(111)
                    ax.patch.set_alpha(0.0)
                    plt.tight_layout()    
                    ax.patch.set_facecolor('white')
                    Font_Size = 20
                    
                    plt.scatter(Spike_Index/F_sample, Site_Data_MA[Spike_Index - int(Start_time*F_sample)], c = 'r')
                    
                    t_xaxis = np.arange(Start_time,len(Site_Data_MA)/F_sample + Start_time,1/F_sample)
                    plt.plot(t_xaxis[::Sparse], Site_Data_MA[::Sparse], color ='darkblue', linewidth=0.6,label=Site_name)    
                    
                    
                    
                    diff_order =1
                    if diff_order in Diff_order:
                        data_diff = np.diff(Site_Data_MA, diff_order)
                        t_diff = np.arange(Start_time, len(data_diff)/F_sample + Start_time, 1/F_sample)
                        
                        point_error = int(len(t_diff) - len(data_diff))
                        
                        if point_error ==0:
                            plt.plot(t_diff, data_diff-1, color ='pink', linewidth=0.6,label=Site_name + '_1st order')
                        elif point_error >0:
                            plt.plot(t_diff[0:-point_error], data_diff- 1, color ='pink', linewidth=0.6,label=Site_name + '_1st order')
                        elif point_error <0:
                            plt.plot(t_diff, data_diff[0:point_error] -1, color ='pink', linewidth=0.6,label=Site_name + '_1st order')
                        
                        
                    
                    diff_order =2
                    if diff_order in Diff_order:
                        data_diff = np.diff(Site_Data_MA, diff_order)
                        t_diff = np.arange(Start_time, len(data_diff)/F_sample + Start_time, 1/F_sample)
                        
                        point_error = int(len(t_diff) - len(data_diff))
                        
                        if point_error ==0:
                            plt.plot(t_diff, data_diff-2, color ='green', linewidth=0.6,label=Site_name + '_2nd order')
                        elif point_error >0:
                            plt.plot(t_diff[0:-point_error], data_diff-2, color ='green', linewidth=0.6,label=Site_name + '_2nd order')
                        elif point_error <0:
                            plt.plot(t_diff, data_diff[0:point_error] -2, color ='green', linewidth=0.6,label=Site_name + '_2nd order')
                        
                    
                    
                    offset = 0
                    if len(Spike_Amp) > 0:
                        offset = max(Spike_Amp)
                    
                    # plt.ylim([-0.3*offset-2.3, offset+0.5])
                    plt.title('Raw Data',fontsize=Font_Size+8, fontweight='bold')
                    plt.xlabel('Time $(s)$', fontsize=Font_Size, fontweight='bold',position=(0.5, 0.1))
                    plt.ylabel('Amplitude ($nA$)',  fontsize=Font_Size, fontweight='bold', position=(0, 0.5))
                    ax.spines['left'].set_position(('outward',0))
                    ax.spines['bottom'].set_position(('outward',0))
                    ax.spines['bottom'].set_linewidth(2)
                    ax.spines['left'].set_linewidth(2)
                    #ax.axhline(linewidth=2, color='black')
                    #ax.axvline(linewidth=2, color='black') 
                    # Eliminate upper and right axes
                    #ax.spines['right'].set_color('none')
                    #ax.spines['top'].set_color('none')
                    # Show ticks in the left and lower axes only
                    ax.xaxis.set_ticks_position('bottom')
                    ax.yaxis.set_ticks_position('left')
                    plt.xticks(fontsize=Font_Size,fontweight='bold')
                    plt.yticks(fontsize=Font_Size,fontweight='bold')
                    plt.legend(loc='upper right',fontsize=Font_Size-4)
                    #plt.text(x, y, 'add text here', fontsize=Font_Size, color='black')
                    # plt.pause(0.001)
                    plt.ioff()
                    plt.savefig(Image_folder +Site_name+ '.png', bbox_inches = 'tight',dpi=100) # change '.png' to '.pdf' if a vector image is required  
                    plt.close(fig)




def Generate_STA_Intra_PSP(Root_folder, Root_folder_target = '', num_of_sites=1024, site_string=['0101','0102'], win_size=705, amp_limit = 30e-3, amp_limit_pre = 6e-3, Start_time=0, Length=2400, F_sample = 1205120/128, Positive=True, STA_Mode=1, DC_flatten=False, Row_range= [1, 64], Time_shift= 0): # 

    if (Root_folder_target == '') or (Root_folder_target == Root_folder): # target folder is not provided, then will just perform typical STA
        print('Starting performing typical STA')
        Root_folder_target = Root_folder
        Cross_dish_label = ''
    else:
        S_id = Root_folder.find('Converted') # extract the recording time stamp
        Source_id = Root_folder[S_id-20:S_id-1]
        T_id = Root_folder_target.find('Converted')
        Target_id = Root_folder_target[T_id-20:T_id-1]
        Cross_dish_label = '_from' + Source_id + 'to' + Target_id + '_Cross_dish' 
        print('Starting performing cross-dish STA')

   
    if STA_Mode == 1:
        print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of spikes')
        sites = np.loadtxt(Root_folder + '/Spike_Count.csv', dtype=float, delimiter=',')
        
        sites = np.stack((np.arange(sites.size), sites), axis=1) # add row number before the counts
        sites = sites[sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
        sites = np.array(sites[::-1],dtype=int) # make it descending
        site_index = sites[0:num_of_sites, 0] # extract the sites to do STA
        site_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in site_index] # convert site_index to site_string in format of row+column
    # Use the top 50 sites with most spikes or use designated site list
    else:
        print('Extract the spike time of the assigned sites in the site_list')
        site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    
    
    with open(Root_folder + '/Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
        rows = list(csv.reader(spiketime_read))
        Spike_Time=[rows[index] for index in site_index]
      
        
    ############## generate the STA file for the sites listed site_string #####################  
    if len(site_string) == len(Spike_Time): # to confirm how many rows have been read
        num_sites = len(Spike_Time)
        print('Spiketime reading is correct!')
    else:
        num_sites = len(Spike_Time)
        print('Spiketime reading is not correct!')
        
    
    if DC_flatten:
        STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+ '_from' + str(Start_time)+'to'+str(Start_time+Length) + Cross_dish_label + '_DC_flatten/'
    else:
        STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+ '_from' + str(Start_time)+'to'+str(Start_time+Length) + Cross_dish_label +'/'
 
    isdir = os.path.isdir(STA_file_dir)
    if not isdir:
        os.mkdir(STA_file_dir)
        
    # write a log to save the parameters information
    with open(STA_file_dir + 'Log' + '.txt', 'a') as fp:
        l0 = 'Start time stamp: ' + time.strftime("%Y_%m_%d-%H_%M_%S")  + '\n'
        l1 = 'Source folder: ' + Root_folder + '\n'
        l2 = 'Target folder: ' + Root_folder_target + '\n'
        l3 = 'Sites: ' + str(num_sites) + '\n' + 'win_size: ' + str(win_size) + '\n' + 'amp_limit: ' + str(amp_limit) + '\n' + 'amp_limit_pre: ' + str(amp_limit_pre) + '\n' 
        l4 = 'From: ' + str(Start_time) + 's to ' + str(Start_time + Length)+' s\n'
        l5 = 'From: row ' + str(Row_range[0]) + ' to row ' + str(Row_range[1])+'\n'
        l6 = 'Time shift: ' + str(Time_shift)+'\n'  # Time_shift is used in the cross-dish to select different segments
        
        fp.writelines([l0,l1,l2,l3,l4,l5,l6])

   
    
    STA_Array_temp = np.zeros((64,win_size))
    STA_Row = np.zeros((64,win_size))
    
    for Row_num in tqdm(np.arange(Row_range[0], Row_range[1]+1)): # do average row by row
        # print('Doing window average of row: ' + str(Row_num))
        Row_Data=-Row_Read(Root_folder_target,Row_num, Start_time= Start_time + Time_shift, Length= Length, F_sample=1205120/128).T # add minus sign to recover the polarity
        
        if DC_flatten:
            for r_idx in range(len(Row_Data)):
                Row_Data[r_idx] = Row_Data[r_idx] - Moving_Average(Row_Data[r_idx], 2000)
        
        for site in range(num_sites): # do average site by site within one row raw data
            # num_spikes = float(len(Spike_Time[site]))
            num_average = np.zeros((64,1))
            STA_Row = np.zeros((64,win_size)) # to initialize the data within the window
            pre_size = int(win_size/3)
            
            for t in range(len(Spike_Time[site])):
                peak_index =  int(float(Spike_Time[site][t])) # seems unable to convert to scientific format string to int directly
                if (peak_index < (Start_time + Length)*F_sample) & (peak_index > (Start_time)*F_sample -1):
                    peak_index =  peak_index - int((Start_time)*F_sample) # If Start_time is not zero, then need to re-count the peak_index
                    win_data = Row_Data[:, peak_index-pre_size:peak_index + (win_size - pre_size)] # around 1/3 win_size = 25ms before peak and 2/3 after peak, when sampling freq = 9415
                    # win_data = win_data*33.33  ## this makes overflow of int
                    if np.shape(win_data) == (64, win_size): # only do average for a whole window 
                    
                    ######## remove a big action potential to reduce false results #########
                        win_mean = np.mean(win_data, axis = 1)
                        # win_std = np.std(win_data, axis = 1)
                        above_limit = win_mean + amp_limit  # It's 0.1mV, consider gain and unit, then it's 0.1/(1000/30), unit 30 is Intra Gain, 1000 is from V to mV.
                        below_limit = win_mean - amp_limit/4.0
                        win_max = np.amax(win_data, axis = 1)
                        win_min = np.amin(win_data, axis = 1)
                        
                        win_max_pre = np.amax(win_data[:,0:235], axis = 1)
                        win_min_pre = np.amin(win_data[:,0:235], axis = 1)
                        win_app_pre = (win_max_pre - win_min_pre)
                        no_spike = (win_max < above_limit) & (win_min > below_limit) & (win_app_pre < amp_limit_pre*2) # the conditions of remove an action potential, and no fluctuation before peak of pre-synaptic
                        win_data = win_data * no_spike[:,np.newaxis]
                    ######## remove a possible action potential #########
                        STA_Row = STA_Row + win_data 
                        num_average = num_average + np.array(no_spike[:,np.newaxis],dtype = int)
    
            STA_Array_temp = (STA_Row + 1e-3)/(num_average + 1e-3)  # add 1e-3 to avoid 0/0
            STA_Array_temp = STA_Array_temp*33.33 #  1000/30, 30 is Intra Gain, 1000 is from V to mV. 
            
            with open(STA_file_dir + 'STA_' + site_string[site] + '.csv', 'a') as STA_file_append:  # save the spike time(index) to csv file
                np.savetxt(STA_file_append, STA_Array_temp, delimiter=',', fmt='%1.6e' )
                
        # write the end time to the Log.txt file
    with open(STA_file_dir + 'Log' + '.txt', 'a') as fp:
        l0 = 'End time stamp: ' + time.strftime("%Y_%m_%d-%H_%M_%S")  + '\n'
        fp.writelines([l0])
        
    return STA_file_dir, site_string


# added 20230113
def Generate_STA_Intra_PSP_Weighted(Root_folder, Root_folder_target = '', num_of_sites=1024, site_string=['0101','0102'], win_size=705, amp_limit=30e-3, amp_limit_pre=6e-3, Start_time=0, Length=2400, F_sample=1205120/128, Positive=True, STA_Mode=1, DC_flatten=False, Row_range=[1, 64], Time_shift=0): # 
    if (Root_folder_target == '') or (Root_folder_target == Root_folder): # target folder is not provided, then will just perform typical STA
        print('Target folder is not provided, it will just perform typical STA')
        Root_folder_target = Root_folder
        Cross_dish_label = ''
    else:
        S_id = Root_folder.find('Converted') # extract the recording time stamp
        Source_id = Root_folder[S_id-20:S_id-1]
        T_id = Root_folder_target.find('Converted')
        Target_id = Root_folder_target[T_id-20:T_id-1]
        Cross_dish_label = '_from' + Source_id + 'to' + Target_id + '_Cross_dish'       

   
    if STA_Mode == 1:
        print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of spikes')
        sites = np.loadtxt(Root_folder + '/Spike_Count.csv', dtype=float, delimiter=',')
        
        sites = np.stack((np.arange(sites.size), sites), axis=1) # add row number before the counts
        sites = sites[sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
        sites = np.array(sites[::-1],dtype=int) # make it descending
        site_index = sites[0:num_of_sites, 0] # extract the sites to do STA
        site_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in site_index] # convert site_index to site_string in format of row+column
    # Use the top "num_of_sites" sites with most spikes or use designated site list
    else:
        print('Extract the spike time of the assigned sites in the site_list')
        site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    
    
    with open(Root_folder + '/Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
        rows = list(csv.reader(spiketime_read))
        Spike_Time=[rows[index] for index in site_index]
      
        
    ############## generate the STA file for the sites listed site_string #####################  
    if len(site_string) == len(Spike_Time): # to confirm how many rows have been read
        num_sites = len(Spike_Time)
        print('Spiketime reading is correct!')
    else:
        num_sites = len(Spike_Time)
        print('Spiketime reading is not correct!')
        
    
    if DC_flatten:
        STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+ '_from' + str(Start_time)+'to'+str(Start_time+Length) + Cross_dish_label + '_DC_flatten/'
    else:
        STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+ '_from' + str(Start_time)+'to'+str(Start_time+Length) + Cross_dish_label +'/'
 
    isdir = os.path.isdir(STA_file_dir)
    if not isdir:
        os.mkdir(STA_file_dir)
        
    # write a log to save the parameters information
    with open(STA_file_dir + 'Log' + '.txt', 'a') as fp:
        l0 = 'Start time stamp: ' + time.strftime("%Y_%m_%d-%H_%M_%S")  + '\n'
        l1 = 'Source folder: ' + Root_folder + '\n'
        l2 = 'Target folder: ' + Root_folder_target + '\n'
        l3 = 'Sites: ' + str(num_sites) + '\n' + 'win_size: ' + str(win_size) + '\n' + 'amp_limit: ' + str(amp_limit) + '\n' + 'amp_limit_pre: ' + str(amp_limit_pre) + '\n' 
        l4 = 'From: ' + str(Start_time) + 's to ' + str(Start_time + Length)+' s\n'
        l5 = 'From: row ' + str(Row_range[0]) + ' to row ' + str(Row_range[1])+'\n'
        l6 = 'Time shift: ' + str(Time_shift)+'\n'  # Time_shift is used in the cross-dish to select different segments
        
        fp.writelines([l0,l1,l2,l3,l4,l5,l6])

   
    
    STA_Array_temp = np.zeros((64,win_size))
    STA_Row = np.zeros((64,win_size))
    
    for Row_num in tqdm(np.arange(Row_range[0], Row_range[1]+1)): # do average row by row
        # print('Doing window average of row: ' + str(Row_num))
        Row_Data=-Row_Read(Root_folder_target,Row_num, Start_time= Start_time + Time_shift, Length= Length, F_sample=1205120/128).T # add minus sign to recover the polarity
        
        if DC_flatten:
            for r_idx in range(len(Row_Data)):
                Row_Data[r_idx] = Row_Data[r_idx] - Moving_Average(Row_Data[r_idx], 2000)
        
        
        for site in range(num_sites): # do average site by site within one row raw data
            # num_spikes = float(len(Spike_Time[site]))
            num_average = np.zeros((64,1))
            STA_Row = np.zeros((64,win_size)) # to initialize the data within the window
            pre_size =235 # int(win_size/3) 
            
            for t in range(len(Spike_Time[site])):
                peak_index =  int(float(Spike_Time[site][t])) # seems unable to convert to scientific format string to int directly
                if (peak_index < (Start_time + Length)*F_sample) & (peak_index > (Start_time)*F_sample -1):
                    peak_index =  peak_index - int((Start_time)*F_sample) # If Start_time is not zero, then need to re-count the peak_index
                    win_data = Row_Data[:, peak_index-pre_size:peak_index + (win_size - pre_size)] # around 1/3 win_size = 25ms before peak and 2/3 after peak, when sampling freq = 9415
                    # win_data = win_data*33.33  ## this makes overflow of int
                    if np.shape(win_data) == (64, win_size): # only do average for a whole window 
                    
                    ######## remove a big action potential to reduce false results #########
                        win_mean = np.mean(win_data, axis = 1)
                        # win_std = np.std(win_data, axis = 1)
                        above_limit = win_mean + amp_limit  # It's 0.1mV, consider gain and unit, then it's 0.1/(1000/30), unit 30 is Intra Gain, 1000 is from V to mV.
                        below_limit = win_mean - amp_limit/4.0
                        win_max = np.amax(win_data, axis = 1)
                        win_min = np.amin(win_data, axis = 1)
                        
                        win_max_pre = np.amax(win_data[:,0:235], axis = 1)
                        win_min_pre = np.amin(win_data[:,0:235], axis = 1)
                        win_app_pre = (win_max_pre - win_min_pre)
                        no_spike = (win_max < above_limit) & (win_min > below_limit) & (win_app_pre < amp_limit_pre*2) # the conditions of remove an action potential, and no fluctuation before peak of pre-synaptic
                        win_data = win_data * no_spike[:,np.newaxis]
                    ######## remove a possible action potential #########
                        STA_Row = STA_Row + win_data 
                        num_average = num_average + np.array(no_spike[:,np.newaxis],dtype = int)
    
            STA_Array_temp = (STA_Row + 1e-3)/(num_average + 1e-3)  # add 1e-3 to avoid 0/0
            STA_Array_temp = STA_Array_temp*33.33 #  1000/30, 30 is Intra Gain, 1000 is from V to mV. 
            
            with open(STA_file_dir + 'STA_' + site_string[site] + '.csv', 'a') as STA_file_append:  # save the spike time(index) to csv file
                np.savetxt(STA_file_append, STA_Array_temp, delimiter=',', fmt='%1.6e' )
                
        # write the end time to the Log.txt file
    with open(STA_file_dir + 'Log' + '.txt', 'a') as fp:
        l0 = 'End time stamp: ' + time.strftime("%Y_%m_%d-%H_%M_%S")  + '\n'
        fp.writelines([l0])
        
    return STA_file_dir, site_string




def Site_Read_Index(Root_folder,Site_name, Start_index=0, Length=470):
    global F_sample
    Row_num=int(Site_name[0:2])
    Col_num=int(Site_name[2:4])-1
    
    h5_File=Root_folder + '/Row_' + str(Row_num) + '.h5'
    
    if Start_index < 0:
        Start_index = 0
    
    with tb.open_file(h5_File) as site_data:
        try:
            return site_data.root.data[Start_index:Start_index + Length,Col_num] #*s_factor # add a minus sign to cancel the polarity of the inverting amplifer
        except:
            return site_data.root.data[-Length::,Col_num] 
        
            
def Template_Gen(width_total = 470, width_pre = 110, width_rise = 7, decay_factor = 150, offset = 0):
    t_decay = np.arange(0, width_total - width_pre - width_rise)
    template_decay = np.exp(-1*t_decay/decay_factor)
    template_rise = np.linspace(offset, 1, width_rise)
    template_pre = np.zeros(width_pre) + offset
    template = np.concatenate([template_pre,template_rise,template_decay])
    return template

def Template_AP(width_total = 705, width_pre = 35, width_rise = 200, rise_factor = 3, decay_factor = 100, offset = 0):
    t_decay = np.arange(0, width_total - width_pre - width_rise)
    template_decay = np.exp(-1*t_decay/decay_factor)
    t_pre = np.linspace(0,(1-offset)**(1/rise_factor),width_rise)
    template_rise = t_pre**3 + offset
    template_pre = np.zeros(width_pre) + offset
    template = np.concatenate([template_pre,template_rise,template_decay])
    return template

def Template_PSP(width_total = 705, width_pre = 235, width_rise = 110, decay_factor = 200, offset = 0.25):
    t_decay = np.arange(0, width_total - width_pre - width_rise)
    template_decay = np.exp(-1*t_decay/decay_factor)
    t = np.linspace(0,1,width_rise)
    template_rise = offset + np.sin(t*np.pi/2)*(1-offset)
    template_pre = np.zeros(width_pre) + offset
    template = np.concatenate([template_pre,template_rise,template_decay])
    return template



def Corr_Hist_Gen(File_path):
    
    File_name = 'Sum_Correlation_Info_v5.csv'
    
    try:
        Relation_Matrix_Corr = np.loadtxt(File_path + '/' +File_name, delimiter=",", dtype = float, skiprows=1)
    except Exception as error:
        print(error)
    
    try:
        Relation_Matrix_Header = np.loadtxt(File_path + '/' +File_name, delimiter=",", dtype = str, skiprows=0,max_rows=1)
    except Exception as error:
        print(error)
    
    
    # from matplotlib.gridspec import GridSpec
    # #%
    
    
    for template_id in tqdm(range(len(Relation_Matrix_Header))):
        plt.close('all')
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 14
        fig = plt.figure(figsize = (6.4,4.8), dpi =160,constrained_layout=True)
        gs = GridSpec(3, 2, figure=fig)
        plt.ion()
        plt.tight_layout()
    
        hist_full = fig.add_subplot(gs[0, :])
        hist_full.hist(Relation_Matrix_Corr[:,template_id], 200,color='purple')
        hist_full.set_xlabel('Correlation distribution')
        
        hist_zoom_l = fig.add_subplot(gs[1, 0])
        hist_zoom_l.hist(Relation_Matrix_Corr[:,template_id], 200,color='purple')
        hist_zoom_l.set_xlim(-1,-0.7)
        hist_zoom_l.set_ylim(0,1000)
        hist_zoom_l.set_xlabel('Zoom-in L')
        
        hist_zoom_r = fig.add_subplot(gs[1, 1])
        hist_zoom_r.hist(Relation_Matrix_Corr[:,template_id], 200,color='purple')
        hist_zoom_r.set_xlim(0.7,1)
        hist_zoom_r.set_ylim(0,1000)
        hist_zoom_r.set_xlabel('Zoom-in R')
        
        corr_ratio = np.zeros(100)
        corr_line = np.linspace(0,0.99,100,dtype='float16')
        
        for idx,corr in enumerate(corr_line):
            corr_ratio[idx] = np.sum(Relation_Matrix_Corr[:,template_id]< (-corr))/(np.sum(Relation_Matrix_Corr[:,template_id]> corr) + 1) # add 1 to make sure not divide by zero and avoid generating a big jump by adding a small number like 1e-3
        
        ax_ratio_full = fig.add_subplot(gs[2, 0])
        ax_ratio_full.plot(corr_line, corr_ratio,color='blue')
        ax_ratio_full.set_xlim(0,1)
        # ax_ratio.set_ylim(0,1)
        ax_ratio_full.set_xlabel('x')
        ax_ratio_full.set_ylabel('p(-x)/p(x)')
        
        ax_ratio_zoom_r = fig.add_subplot(gs[2, 1])
        ax_ratio_zoom_r.plot(corr_line[-50::], corr_ratio[-50::],color='blue')
        # ax_ratio_zoom_r.set_xlim(0,1)
        ax_ratio_zoom_r.set_xlabel('x: zoom-in R')
        ax_ratio_zoom_r.set_ylabel('p(-x)/p(x)')
        
        fig.suptitle('Stat report of template: '+ Relation_Matrix_Header[template_id])
        plt.ioff()
        plt.savefig(File_path + '/' + Relation_Matrix_Header[template_id] + '.png', bbox_inches = 'tight',dpi=220) # change '.png' to '.pdf' if a vector image is required
        plt.savefig(File_path + '/' + Relation_Matrix_Header[template_id] + '.pdf', bbox_inches = 'tight') # change '.png' to '.pdf' if a vector image is required


def Closest_value(input_list, input_value):
    arr = np.array(input_list).astype(float)
    i = (np.abs(arr - input_value)).argmin()
    return i, arr[i]



def Relation_Define_v3(Root_folder, num_of_sites=4096, site_string=['0101','0102'], win_size=1883, F_sample = 1205120/128, Positive=True, Seek_Mode=1, spike_num_limit =50, amplitude_limit=0.2, percentage_limit = 0.2, corr_limit = 0.6): # 

    # win_size=1883
    # F_sample = 1205120/128
    # # Positive=True
    template = Template_Gen()   
    
    # for visualizing #######################
    # plt.close('all')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # line1, = ax.plot(template, 'r-') # Returns a tuple of line objects, thus the comma
    # line2, = ax.plot(template, 'g-') # Returns a tuple of line objects, thus the comma
    # line3, = ax.plot(template, 'b-') # Returns a tuple of line objects, thus the comma
    # for visualizing #######################
    
    if Seek_Mode == 1:
        print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of spikes')
        sites = np.loadtxt(Root_folder + '/Spike_Count.csv', dtype=float, delimiter=',')
        
        sites = np.stack((np.arange(sites.size), sites), axis=1) # add row number before the counts
        sites = sites[sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
        sites = np.array(sites[::-1],dtype=int) # make it descending
        site_index = sites[0:num_of_sites, 0] # extract the sites to do STA
        site_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in site_index] # convert site_index to site_string in format of row+column
    else:
        print('Extract the spike time of the assigned sites in the site_list')
        site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    
    
    with open(Root_folder + '/Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
        rows = list(csv.reader(spiketime_read))
        Spike_Time=[rows[index] for index in site_index]
        
        
    with open(Root_folder + '/Spike_Amplitude.csv', 'r') as spikeamplitude_read:  # read the spikeamplitude of interested sites
        rows = list(csv.reader(spikeamplitude_read))
        Spike_Amplitude=[rows[index] for index in site_index]        
        
      
    ############## generate the STA file for the sites listed site_string #####################  
    
    if len(site_string) == len(Spike_Time): # to confirm how many rows have been read
        # num_sites = len(Spike_Time)
        print('Spiketime reading is correct!')
    else:
        # num_sites = len(Spike_Time)
        print('Spiketime reading is not correct!')
    Relation_file_dir = Root_folder + '/Relation_Files_Win_'+'{:.0f}'.format(win_size/5/F_sample*1000)+'_Amp_'+ str(int(amplitude_limit*1000))+'_Corr_'+ str(int(corr_limit*100))+'_Check_'+str(num_of_sites)+'sites'+'/'  # name the folder ending with the time window
    
    isdir = os.path.isdir(Relation_file_dir)
    if not isdir:
        os.mkdir(Relation_file_dir)
        
    # Relation_Array_temp = np.zeros((4096,9)) # to store the relation map: |Site|Relation|Num of spikes|Pre percentage|Post percentage|Time_start|Time_end
    
    
    for index in range(len(Spike_Time)):
        Spike_Time[index] = np.array(Spike_Time[index]).astype(float)
        Spike_Amplitude[index] = np.array(Spike_Amplitude[index]).astype(float)
    
    
    
    head_sum = ['PreSite', 'PostSite', 'Spikes_overlapped', 'Pre percentage', 'Post percentage','Time_start', 'Time_end','Spikes_from','Spikes_to']
    with open(Relation_file_dir + 'Sum_Relation_Info.csv', 'a', newline='') as Sum_file_append:  # save connection information to a csv file
        write = csv.writer(Sum_file_append) 
        write.writerow(head_sum)
    
    
    # 470 = 50ms
    connection_total = 0
    for index in tqdm(range(len(site_index))):
        connection_site = 0
        relation_info = np.zeros(9)
        site_str = str(int(site_index[index]/64)+1).zfill(2) + str(int(site_index[index]%64)+1).zfill(2)
        head = ['Site', 'Relation', 'Spikes_overlapped', 'Pre percentage', 'Post percentage','Time_start', 'Time_end','Spikes_from','Spikes_to']
        with open(Relation_file_dir + 'Relation_Info_' + site_str + '.csv', 'a', newline='') as Relation_file_append:  # save all information to a csv file
            write = csv.writer(Relation_file_append) 
            write.writerow(head)
        
        # find out the if overlap exists or not
        Spike_Time[index] = np.array(Spike_Time[index]).astype(float)
        lower_bound = min(Spike_Time[index], default =-1)
        upper_bound = max(Spike_Time[index], default =-2)
        
        if lower_bound < 0 or len(Spike_Time[index]) < spike_num_limit:  # no spike in this site
            break
        
        for index_to in range(len(site_index)): # compare site 'index' with all other sites
            post = 0
            pre = 0
            site_str_to = str(int(site_index[index_to]/64)+1).zfill(2) + str(int(site_index[index_to]%64)+1).zfill(2)
            site_str_from = site_str
            
            lower_bound_to = min(Spike_Time[index_to], default =-1)
            upper_bound_to = max(Spike_Time[index_to], default =-2)
            if lower_bound > upper_bound_to or upper_bound < lower_bound_to: # no overlap
                break
            else:  # there is overlap, then find the overlap
                lower_bound_ol = max(lower_bound, lower_bound_to) # ol is short for overlap
                upper_bound_ol = min(upper_bound, upper_bound_to)
                lower_index_ol, lower_value_ol = Closest_value(Spike_Time[index], lower_bound_ol) #find the element index range in site 'index'
                upper_index_ol, upper_value_ol = Closest_value(Spike_Time[index], upper_bound_ol)
                num_spikes_ol = upper_index_ol - lower_index_ol +1
                
                lower_index_to, lower_value_to = Closest_value(Spike_Time[index_to], lower_bound_ol) #find the element index range in site 'index_to'
                upper_index_to, upper_value_to = Closest_value(Spike_Time[index_to], upper_bound_ol)
                num_spikes_to = upper_index_to - lower_index_to
                
                
                if min(num_spikes_ol,num_spikes_to) < spike_num_limit: # at least certain number of spikes should exist
                    break
                
                if (upper_value_ol - lower_value_ol > F_sample*10): # overlap longer than 10s
                    for index_ol in np.arange(lower_index_ol, upper_index_ol+1):
                        time_value = Spike_Time[index][index_ol]
                        index_find, time_find = Closest_value(Spike_Time[index_to], time_value)
                        
                        if (abs(time_find - time_value) < win_size/5) and (Spike_Amplitude[index_to][index_find]< amplitude_limit): # within 40ms = win_time/, also check the amplitude of Site[index_to] 
                            post_data = -Site_Read_Index(Root_folder, site_str_to, Start_index= time_find-117, Length = 470) # reading the raw data 
                            post_data = post_data - np.min(post_data)
                            post_data = post_data/(np.max(abs(post_data)) + 1e-6) # normalize the data
                            
                            pre_data = -Site_Read_Index(Root_folder, site_str_from, Start_index= time_value-117, Length = 470) # reading the raw data 
                            pre_data = pre_data - np.min(pre_data)
                            pre_data = pre_data/(np.max(abs(pre_data)) + 1e-6) # normalize the data
                            
                            try:
                                corr, pValue = sci_stats.pearsonr(post_data,template)
                            except:
                                print('error happens when calculating correlation!')
                                corr = 0
    
                            
                            if corr > corr_limit:
                                if time_find > time_value: # the closest index found is after the original index
                                    post = post +1
                                else:
                                    pre  = pre + 1
                                
                                # # for visualizing the data   
                                # # line1.set_ydata(pre_data)
                                # line2.set_ydata(post_data)
                                # fig.canvas.draw()
                                # fig.canvas.flush_events()
                                # plt.title('Corr:'+'{:.2f}'.format(corr))
                                # # time.sleep(1)
                                #  # for visualizing the data
                                
            Pre_percent = max(pre/num_spikes_ol, pre/num_spikes_to)
            Post_percent = max(post/num_spikes_ol, post/num_spikes_to)
            
            if Post_percent > 0.95:
                relation = 2 # same neuron
            elif Post_percent > percentage_limit: # 10 percent
                relation = 1 # post synapse
            elif Pre_percent >0.95:
                relation = 2 # same neuron
            elif Pre_percent > percentage_limit:  # 10 percent
                relation =-1 # pre synapse
            else:
                relation = 0 # no relation
    
           
            relation_info[0] = site_str_to
            relation_info[1] = relation
            relation_info[2] = pre + post
            relation_info[3] = Pre_percent*100
            relation_info[4] = Post_percent*100
            relation_info[5] = lower_value_ol/F_sample
            relation_info[6] = upper_value_ol/F_sample
            relation_info[7] = num_spikes_ol
            relation_info[8] = num_spikes_to
            
            with open(Relation_file_dir + 'Relation_Info_' + site_str + '.csv', 'a', newline='') as Relation_file_append:  # save information to a csv file
                write = csv.writer(Relation_file_append) 
                write.writerow(relation_info)
            
            
            if relation ==-1:
                connection_total = connection_total +1
                connection_site = connection_site +1
                relation_info[1] = site_str # original site is as post synaptic neuron, relation_info[0] = site_str_to is pre synaptic neuron
                with open(Relation_file_dir + 'Sum_Relation_Info.csv', 'a', newline='') as Sum_file_append:  # save connection information to a csv file
                    write = csv.writer(Sum_file_append) 
                    write.writerow(relation_info)
                    
            if relation ==1:
                connection_total = connection_total +1
                connection_site = connection_site +1
                relation_info[0] = site_str # original site is presynaptic
                relation_info[1] = site_str_to # site_str_to is post synaptic neuron
                with open(Relation_file_dir + 'Sum_Relation_Info.csv', 'a', newline='') as Sum_file_append:  # save connection information to a csv file
                    write = csv.writer(Sum_file_append) 
                    write.writerow(relation_info)
                
        print('Found ' + str(connection_site) + ' new connections.')
        print('Found ' + str(connection_total) + ' connections in total.')
    print ('There are ' + str(connection_total) + ' connections in total')             
    
    with open(Relation_file_dir + 'Relation_sum_' + str(connection_total) + '.csv', 'a', newline='') as Relation_file_sum:  # save information to a csv file
        write = csv.writer(Relation_file_sum) 
        write.writerow([connection_total])


def Relation_Define_Extra(Root_folder, num_of_sites=4096, site_string=['0101','0102'], win_size=1883, F_sample = 1205120/128, Positive=True, Seek_Mode=1, spike_num_limit =50): # 

    # win_size=1883
    # F_sample = 1205120/128
    # # Positive=True
    
    if Seek_Mode == 1:
        print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of spikes')
        sites = np.loadtxt(Root_folder + '/Positive_Spike_Count.csv', dtype=float, delimiter=',')
        
        sites = np.stack((np.arange(sites.size), sites), axis=1) # add row number before the counts
        sites = sites[sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
        sites = np.array(sites[::-1],dtype=int) # make it descending
        site_index = sites[0:num_of_sites, 0] # extract the sites to do STA
        site_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in site_index] # convert site_index to site_string in format of row+column
    else:
        print('Extract the spike time of the assigned sites in the site_list')
        site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    
    
    with open(Root_folder + '/Positive_Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
        rows = list(csv.reader(spiketime_read))
        Spike_Time=[rows[index] for index in site_index]
      
    ############## generate the STA file for the sites listed site_string #####################  
    
    if len(site_string) == len(Spike_Time): # to confirm how many rows have been read
        # num_sites = len(Spike_Time)
        print('Spiketime reading is correct!')
    else:
        # num_sites = len(Spike_Time)
        print('Spiketime reading is not correct!')
    Relation_file_dir = Root_folder + '/Relation_Files_Win_'+'{:.0f}'.format(win_size/5/F_sample*1000)+ '_Check_'+str(num_of_sites)+'sites'+'/'  # name the folder ending with the time window
    
    isdir = os.path.isdir(Relation_file_dir)
    if not isdir:
        os.mkdir(Relation_file_dir)
        
    # Relation_Array_temp = np.zeros((4096,9)) # to store the relation map: |Site|Relation|Num of spikes|Pre percentage|Post percentage|Time_start|Time_end
    
    
    for index in range(len(Spike_Time)):
        Spike_Time[index] = np.array(Spike_Time[index]).astype(float)
    
    
    
    head_sum = ['PreSite', 'PostSite', 'Spikes_overlapped', 'Pre percentage', 'Post percentage','Time_start', 'Time_end','Spikes_from','Spikes_to']
    with open(Relation_file_dir + 'Sum_Relation_Info.csv', 'a', newline='') as Sum_file_append:  # save connection information to a csv file
        write = csv.writer(Sum_file_append) 
        write.writerow(head_sum)
    
    
    # 470 = 50ms
    connection_total = 0
    for index in tqdm(range(len(site_index))):
        connection_site = 0
        relation_info = np.zeros(9)
        site_str = str(int(site_index[index]/64)+1).zfill(2) + str(int(site_index[index]%64)+1).zfill(2)
        head = ['Site', 'Relation', 'Spikes_overlapped', 'Pre percentage', 'Post percentage','Time_start', 'Time_end','Spikes_from','Spikes_to']
        with open(Relation_file_dir + 'Relation_Info_' + site_str + '.csv', 'a', newline='') as Relation_file_append:  # save all information to a csv file
            write = csv.writer(Relation_file_append) 
            write.writerow(head)
        
    
        
        
        # find out the if overlap exists or not
        Spike_Time[index] = np.array(Spike_Time[index]).astype(float)
        lower_bound = min(Spike_Time[index], default =-1)
        upper_bound = max(Spike_Time[index], default =-2)
        
        if lower_bound < 0 or len(Spike_Time[index]) < spike_num_limit:  # no spike in this site
            break
        
        for index_to in range(len(site_index)): # compare site 'index' with all other sites
            post = 0
            pre = 0
            lower_bound_to = min(Spike_Time[index_to], default =-1)
            upper_bound_to = max(Spike_Time[index_to], default =-2)
            if lower_bound > upper_bound_to or upper_bound < lower_bound_to: # no overlap
                break
            else:  # there is overlap, then find the overlap
                lower_bound_ol = max(lower_bound, lower_bound_to) # ol is short for overlap
                upper_bound_ol = min(upper_bound, upper_bound_to)
                lower_index_ol, lower_value_ol = Closest_value(Spike_Time[index], lower_bound_ol) #find the element index range in site 'index'
                upper_index_ol, upper_value_ol = Closest_value(Spike_Time[index], upper_bound_ol)
                num_spikes_ol = upper_index_ol - lower_index_ol +1
                
                lower_index_to, lower_value_to = Closest_value(Spike_Time[index_to], lower_bound_ol) #find the element index range in site 'index_to'
                upper_index_to, upper_value_to = Closest_value(Spike_Time[index_to], upper_bound_ol)
                num_spikes_to = upper_index_to - lower_index_to
                
                
                if min(num_spikes_ol,num_spikes_to) < spike_num_limit: # at least certain number of spikes should exist
                    break
                
                if upper_value_ol - lower_value_ol > F_sample*10: # overlap longer than 10s
                    for index_ol in np.arange(lower_index_ol, upper_index_ol+1):
                        time_value = Spike_Time[index][index_ol]
                        index_find, time_find = Closest_value(Spike_Time[index_to], time_value)
                        if abs(time_find - time_value) < win_size/5: # within 40ms = win_time/5:
                            if time_find > time_value: # the closest index found is after the original index
                                post = post +1
                            else:
                                pre  = pre + 1
            
            
            Pre_percent = pre/len(Spike_Time[index]) # Pre_percent = max(pre/num_spikes_ol, pre/num_spikes_to)
            Post_percent = post/len(Spike_Time[index]) # Post_percent = max(post/num_spikes_ol, post/num_spikes_to)
            
            if Post_percent > 0.95:
                relation = 2 # same neuron
            elif Post_percent >0.3:
                relation = 1 # post synapse
            elif Pre_percent >0.95:
                relation = 2 # same neuron
            elif Pre_percent >0.3:
                relation =-1 # pre synapse
            else:
                relation = 0 # no relation
        
            
            site_str_to = str(int(site_index[index_to]/64)+1).zfill(2) + str(int(site_index[index_to]%64)+1).zfill(2)
            
            
           
            relation_info[0] = site_str_to
            relation_info[1] = relation
            relation_info[2] = pre + post
            relation_info[3] = Pre_percent*100
            relation_info[4] = Post_percent*100
            relation_info[5] = lower_value_ol/F_sample
            relation_info[6] = upper_value_ol/F_sample
            relation_info[7] = num_spikes_ol
            relation_info[8] = num_spikes_to
            
            with open(Relation_file_dir + 'Relation_Info_' + site_str + '.csv', 'a', newline='') as Relation_file_append:  # save information to a csv file
                write = csv.writer(Relation_file_append) 
                write.writerow(relation_info)
            
            
            if relation ==-1:
                connection_total = connection_total +1
                connection_site = connection_site +1
                relation_info[1] = site_str # original site is as post synaptic neuron, relation_info[0] = site_str_to is pre synaptic neuron
                with open(Relation_file_dir + 'Sum_Relation_Info.csv', 'a', newline='') as Sum_file_append:  # save connection information to a csv file
                    write = csv.writer(Sum_file_append) 
                    write.writerow(relation_info)
                    
            if relation ==1:
                connection_total = connection_total +1
                connection_site = connection_site +1
                relation_info[0] = site_str # original site is presynaptic
                relation_info[1] = site_str_to # site_str_to is post synaptic neuron
                with open(Relation_file_dir + 'Sum_Relation_Info.csv', 'a', newline='') as Sum_file_append:  # save connection information to a csv file
                    write = csv.writer(Sum_file_append) 
                    write.writerow(relation_info)
                
        print('Found ' + str(connection_site) + ' new connections.')
        print('Found ' + str(connection_total) + ' connections in total.')
    print ('There are ' + str(connection_total) + ' connections in total')             
    
    with open(Relation_file_dir + 'Relation_sum_' + str(connection_total) + '.csv', 'a', newline='') as Relation_file_sum:  # save information to a csv file
        write = csv.writer(Relation_file_sum) 
        write.writerow([connection_total])
        


def Relation_Define_STA_v5(Root_folder, Root_folder_target = '', num_of_sites= 1000, site_string= ['0101','0102'], amp_limit= 30e-3, amp_limit_pre= 6e-3, corr_limit= 0.7, Start_time= 0, Length= 2400, F_sample= 1205120/128, Positive= True, STA_Mode= 1,  col_per_figure= 16, save_png= True, save_pdf = False, IPSP= False, DC_flatten= False, Corr_Hist_Plot= True):

    if (Root_folder_target == '') or (Root_folder_target == Root_folder): # target folder is not provided, then will just perform typical STA
        print('Target folder is not provided, it will just perform typical STA')
        Root_folder_target = Root_folder
        Cross_dish_label = ''
    else:
        S_id = Root_folder.find('Converted') # extract the recording time stamp
        Source_id = Root_folder[S_id-20:S_id-1]
        T_id = Root_folder_target.find('Converted')
        Target_id = Root_folder_target[T_id-20:T_id-1]
        Cross_dish_label = '_from' + Source_id + 'to' + Target_id + '_Cross_dish'      
      


    template_ap_base = Template_AP(width_total = 705, width_pre = 35, width_rise = 200, rise_factor = 3, decay_factor = 100, offset = 0)
    template_epsp_base= Template_PSP(width_total = 705, width_pre = 235, width_rise = 110, decay_factor = 250, offset = 0.25)
    template_ipsp_base= 1-template_epsp_base
    
    if STA_Mode == 1:
    
        print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of spikes')
        sites = np.loadtxt(Root_folder + '/Spike_Count.csv', dtype=float, delimiter=',')
        
        sites = np.stack((np.arange(sites.size), sites), axis=1) # add row number before the counts
        sites = sites[sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
        sites = np.array(sites[::-1],dtype=int) # make it descending
        site_index = sites[0:num_of_sites, 0] # extract the sites to do STA
        site_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in site_index] # convert site_index to site_string in format of row+column
    # Use the top 50 sites with most spikes or use designated site list
    else:
        print('Extract the spike time of the assigned sites in the site_list')
        site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    
    
    with open(Root_folder + '/Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
        rows = list(csv.reader(spiketime_read))
        Spike_Time=[rows[index] for index in site_index]
    
    site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]    
        
    ############## generate the STA file for the sites listed site_string #####################  
    
    if len(site_string) == len(Spike_Time): # to confirm how many rows have been read
        num_sites = len(Spike_Time)
        print('Spiketime reading is correct!')
    else:
        num_sites = len(Spike_Time)
        print('Spiketime reading is not correct!')
        
    if DC_flatten:
        STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+ '_from' + str(Start_time)+'to'+str(Start_time+Length) + Cross_dish_label + '_DC_flatten/'
    else:
        STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+ '_from' + str(Start_time)+'to'+str(Start_time+Length) + Cross_dish_label +'/'
    
    # STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+'/'
    
    isdir = os.path.isdir(STA_file_dir)  
    if isdir==False:
        os.mkdir(STA_file_dir)
    
    
    print('Start plotting subsite waveform')    
    
    # now = datetime.now()
    # dt_string = now.strftime("%Y%m%d%H%M")
    
    # if IPSP:
    #     Waveform_folder = STA_file_dir + 'STA_Relation_Base_with_IPSP_' + dt_string +'/'
    # else:
    #     Waveform_folder = STA_file_dir + 'STA_Relation_Base_no_IPSP_' + dt_string +'/'
        
    if IPSP:
        Waveform_folder = STA_file_dir + 'STA_Relation_Base_with_IPSP'  +'/'
    else:
        Waveform_folder = STA_file_dir + 'STA_Relation_Base_no_IPSP'  +'/'        
        
        
    isdir = os.path.isdir(Waveform_folder)  
    if isdir==False:
        os.mkdir(Waveform_folder)
    
    
    
    #****** for relationship saving ***********
    
    head_sum = ['PreSite', 'PostSite', 'Correlation', 'SynapseType', 'Delay']
    with open(Waveform_folder + 'Sum_Relation_Info_v5.csv', 'w', newline='') as Sum_file_append:  # save connection information to a csv file
        write = csv.writer(Sum_file_append) 
        write.writerow(head_sum)
    
    head_type = ['AP', 'APEP', 'EPSP', 'IPSP']
    with open(Waveform_folder + 'Sum_Correlation_Info_v5.csv', 'w', newline='') as Sum_file_append:  # save correlation factors to a csv file
        write = csv.writer(Sum_file_append) 
        write.writerow(head_type)
    
    
    connection_total = 0
    connection_site = 0
    
    relation_info = np.zeros(5)
    relation_info_list = []
    
    correlation_info = np.zeros(4)
    correlation_info_list = []
    
    save_fig = save_png or save_pdf 
    
    #****** for relationship saving ***********
    
    
    # tqdm is used to track progress
    for site_idx in tqdm(range(len(site_string))):
        site = site_string[site_idx]
        site_arr_index = site_index[site_idx]
        
        if save_fig: # run this only when saving figures
            Waveform_sub_folder = Waveform_folder + site +'/'
            isdir = os.path.isdir(Waveform_sub_folder)  
            if isdir==False:
                os.mkdir(Waveform_sub_folder)
    
        try:
            STA_file = STA_file_dir + 'STA_'+ site + '.csv'
            STA_Array = np.loadtxt(STA_file, dtype=float, delimiter=',')
            t = np.arange(0,len(STA_Array[0,:]))*1/F_sample # randomly selct a site to calculate the length
            
            connection_site = 0 # reset the connection number of every site
            relation_info_list =[]
            correlation_info_list =[]
            
    
            for Col_num in np.arange(1, 65):
                
                if save_fig:
                    ax_index = int((Col_num-1)%col_per_figure)
                    if (Col_num-1)%col_per_figure ==0:
                        plt.close('all')
                        plt.rcParams["font.family"] = "Arial"
                        Font_Size = 6
                        fig, ax_array = plt.subplots(nrows=1, ncols=col_per_figure, gridspec_kw = {'wspace':0, 'hspace':0})
                        fig.set_size_inches(18.5, 10.5, forward=True)
                        ax_array = ax_array.flatten()
                        plt.ion()
                        plt.tight_layout()    
                    
                   
                for Row_num in np.arange(1, 65):   
                    site_idx = (Row_num - 1)*64 + Col_num -1 
                    Site_Data = STA_Array[site_idx,:]
                    Site_Data_MA = Site_Data - np.min(Site_Data)
                    max_amp = np.max(Site_Data_MA)
                    peak = Site_Data_MA.argmax()
                    base = Site_Data_MA.argmin()
                    corr_max = -1 # reset the value
                    
                    # print(peak)
                    
                    if max_amp !=0:
                        Site_Data_MA = Site_Data_MA/max_amp
                    Site_name = str(Row_num).zfill(2) + str(Col_num).zfill(2)
                    
                    if save_fig:
                        Offset = - (Row_num)
                        ax_array[ax_index].plot(t, Moving_Average(Site_Data_MA,10) + Offset, linewidth=0.5)
    
                        
                    #****** extracting AP/+ and EPSP information ***********    
                    if peak > 240 and peak < 470:
                        template_epsp_roll= Template_PSP(width_total = 705, width_pre = peak - 110, width_rise = 110, decay_factor = 250, offset = 0.25)
                        template_ap_roll= Template_PSP(width_total = 705, width_pre = peak - 40, width_rise = 40, decay_factor = 90, offset = 0.15)
                        # template_ap_roll = Template_AP(width_total = 705, width_pre = peak -200, width_rise = 200, rise_factor = 3, decay_factor = 100, offset = 0)
                        corr_epsp_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_epsp_roll)
                        corr_ap_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_ap_roll)
                        peak_roll_delay = peak
                        
                    else:
                        template_epsp_roll= Template_PSP(width_total = 705, width_pre = 355 - 110, width_rise = 110, decay_factor = 250, offset = 0.25)
                        template_ap_roll= Template_PSP(width_total = 705, width_pre = 355 - 40, width_rise = 40, decay_factor = 90, offset = 0.15)
                        corr_epsp_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_epsp_roll)
                        corr_ap_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_ap_roll)
                        peak_roll_delay = 355
                        
                    try:
                        corr_ap_base, pValue = sci_stats.pearsonr(Site_Data_MA,template_ap_base)
                        corr_epsp_base, pValue = sci_stats.pearsonr(Site_Data_MA,template_epsp_base)
                        corr_epsp = np.max([corr_epsp_roll, corr_epsp_base])
                        corr_ap = np.max([corr_ap_roll, corr_ap_base])
                        
    
                    except Exception as error:
                        print('\nError happens when calculating correlation at site: ' + Site_name+ ', and the error message is: ')
                        print(error) 
                        corr_epsp = 0
                        corr_ap = 0
                            
    
                        
                    #****** extracting IPSP information ***********
                    if IPSP:
                        if base > 350 and base < 470:
                            template_psp= Template_PSP(width_total = 705, width_pre = base -110, width_rise = 110, decay_factor = 250, offset = 0.25)
                            template_ipsp_roll= 1-template_psp
                            base_roll_delay = base
                        
                            try:
                                corr_ipsp_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_ipsp_roll)
                                corr_ipsp_base, pValue = sci_stats.pearsonr(Site_Data_MA,template_ipsp_base)
                                corr_ipsp = np.max([corr_ipsp_roll, corr_ipsp_base])
                                corr_max = np.max([corr_ap, corr_epsp, corr_ipsp])
                            
                            except Exception as error:
                                print('\nError happens when calculating correlation at site: ' + Site_name+ ', and the error message is: ')
                                print(error) 
                                corr_max = np.max([corr_ap, corr_epsp])
                                
                        else: # base is not valid,and use the middle of the range
                            template_psp= Template_PSP(width_total = 705, width_pre = 410 -110, width_rise = 110, decay_factor = 250, offset = 0.25)
                            template_ipsp_roll= 1-template_psp
                            base_roll_delay = 410
                            try:
                                corr_ipsp_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_ipsp_roll)
                                corr_ipsp_base, pValue = sci_stats.pearsonr(Site_Data_MA,template_ipsp_base)
                                corr_ipsp = np.max([corr_ipsp_roll, corr_ipsp_base])
                                corr_max = np.max([corr_ap, corr_epsp, corr_ipsp])
                            
                            except Exception as error:
                                print('\nError happens when calculating correlation at site: ' + Site_name+ ', and the error message is: ')
                                print(error) 
                                corr_max = np.max([corr_ap, corr_epsp])
                    else:
                        corr_max = np.max([corr_ap, corr_epsp])
                                # corr_max = corr_epsp

                    #****** for all correlationship saving ***********
                    correlation_info[0] = corr_ap_base # AP template
                    correlation_info[1] = corr_ap_roll # AP_EPSP template
                    correlation_info[2] = corr_epsp # EPSP template
                    correlation_info[3] = corr_ipsp  # IPSP template
                    correlation_info_list.append(np.copy(correlation_info)) # save to a list temporary to reduce access to .csv file
                    #****** for all correlationship saving ***********


                        
                    if corr_max > corr_limit: # regarded as effective connnection
                    
                        #****** for relationship saving ***********
                        if corr_max == corr_epsp_base:
                            syn_type = 1
                            textcolor = 'red'
                            template_plot = template_epsp_base
                            temp_delay = round((345 + peak)/2) # peak point of epsp_base:235 + 110, and the peak point of avaraged waveform
                        elif corr_max == corr_epsp_roll:
                            syn_type = 1
                            textcolor = 'red'
                            template_plot = template_epsp_roll
                            temp_delay = peak_roll_delay # peak
                            
                        elif corr_max == corr_ap_base:
                            syn_type = 0
                            textcolor = 'blue'
                            template_plot = template_ap_base
                            temp_delay = round((235 + peak)/2) # peak point of ap_base, 200 + 35
    
                        elif corr_max == corr_ap_roll:
                            syn_type = 0.5
                            textcolor = 'blue'
                            template_plot = template_ap_roll
                            temp_delay = peak_roll_delay # peak
                            
                        elif corr_max == corr_ipsp_base:
                            syn_type = -1
                            textcolor = 'purple'
                            template_plot = template_ipsp_base
                            temp_delay = round((345 + base)/2) # base point of ipsp_base:235 + 110, and the base/bottom point of avaraged waveform
                            
                        elif corr_max == corr_ipsp_roll:
                            syn_type = -1
                            textcolor = 'purple'
                            template_plot = template_ipsp_roll
                            temp_delay = base_roll_delay # base_roll
                        
                        connection_total = connection_total + 1
                        connection_site = connection_site +1
                        
                        relation_info[0] = int(site) # presite
                        relation_info[1] = int(Site_name) # postsite
                        relation_info[2] = corr_max
                        relation_info[3] = syn_type  # 1 for excitatory, -1 for inhibitory, 0 same site, 0.5, strong connection
                        relation_info[4] = temp_delay # delay
                        relation_info_list.append(np.copy(relation_info)) # save to a list temporary to reduce access to .csv file
                            

                        
                        #****** for relationship saving ***********
    
                        if save_fig:
                            ax_array[ax_index].plot(t, template_plot + Offset, linestyle=':', linewidth=0.5, color = 'black')
                            ax_array[ax_index].text(t[10], Offset + 0.5, "{:.2f}".format(corr_max), color= textcolor, fontsize= 12 )
                            
                            
                    if save_fig:
                        if Row_num == 64: # add title,and line for a column 
                            ax_array[ax_index].axis(ymin=-65,ymax=0)
                            # ax_array[ax_index].axvline(x=0.025, color='green', linestyle=':', linewidth=0.25, label='axvline - full height')
                            ax_array[ax_index].axvline(x=0.025, color='green', linestyle=':', linewidth=0.25)
                            ax_array[ax_index].set_title('Col: ' + str(Col_num).zfill(2))
                        
    
                if (Col_num-1)%col_per_figure == (col_per_figure -1):
                    if save_fig:
                        plt.ioff()
                    if save_png:
                        plt.savefig(Waveform_sub_folder +'Column_'+str(Col_num-col_per_figure) + ' to ' +str(Col_num) + '.png', bbox_inches = 'tight',dpi=220) # change '.png' to '.pdf' if a vector image is required
                    if save_pdf: # saving .pdf is slow. It is disabled by default
                        plt.savefig(Waveform_sub_folder +'Column_'+str(Col_num-col_per_figure) + ' to ' +str(Col_num) + '.pdf', bbox_inches = 'tight') # change '.png' to '.pdf' if a vector image is required
        
            print('Found ' + str(connection_site) + ' new connections.')
            print('Found ' + str(connection_total) + ' connections in total.')
            
            
            with open(Waveform_folder + 'Sum_Relation_Info_v5.csv', 'a', newline='') as Sum_file_append:  # save connection information of a single site from the temporaty list to a csv file
                write = csv.writer(Sum_file_append) 
                for relation_item in relation_info_list:
                    write.writerow(relation_item)
            
            
            with open(Waveform_folder + 'Sum_Correlation_Info_v5.csv', 'a', newline='') as Sum_file_append:  # save connection information of a single site from the temporaty list to a csv file
                write = csv.writer(Sum_file_append) 
                for correlation_item in correlation_info_list:
                    write.writerow(correlation_item)
                        
            
            
        except Exception as error:
            print('\nError happens in site: ' +str(site) + ', and the error message is: ')
            print(error)    
        
    
    print ('There are ' + str(connection_total) + ' connections in total')             
    with open(Waveform_folder + 'Relation_sum_' + str(connection_total) + '.csv', 'a', newline='') as Relation_file_sum:  # save information to a csv file
        write = csv.writer(Relation_file_sum) 
        write.writerow([connection_total])
    
    if Corr_Hist_Plot:
        print ('Start plotting the histogram report!')
        Corr_Hist_Gen(Waveform_folder)


def Relation_Define_STA_v6(Root_folder, Root_folder_target = '', num_of_sites= 1000, site_string= ['0101','0102'], amp_limit= 30e-3, amp_limit_pre= 6e-3, corr_limit= 0.7, Start_time= 0, Length= 2400, F_sample= 1205120/128, Positive= True, STA_Mode= 1,  col_per_figure= 16, save_png= True, save_pdf = False, IPSP= False, DC_flatten= False, Corr_Hist_Plot= True, win_size =705, Weighted=True):

    if (Root_folder_target == '') or (Root_folder_target == Root_folder): # target folder is not provided, then will just perform typical STA
        print('Target folder is not provided, it will just perform typical STA')
        Root_folder_target = Root_folder
        Cross_dish_label = ''
    else:
        S_id = Root_folder.find('Converted') # extract the recording time stamp
        Source_id = Root_folder[S_id-20:S_id-1]
        T_id = Root_folder_target.find('Converted')
        Target_id = Root_folder_target[T_id-20:T_id-1]
        Cross_dish_label = '_from' + Source_id + 'to' + Target_id + '_Cross_dish'      
      
    if Weighted:
        Weighted_label = '_Weighted'
        AP_amp =110
    else:
        Weighted_label = ''
        AP_amp =1

    template_ap_base = Template_AP(width_total = 705, width_pre = 35, width_rise = 200, rise_factor = 3, decay_factor = 100, offset = 0)
    template_epsp_base= Template_PSP(width_total = 705, width_pre = 235, width_rise = 110, decay_factor = 250, offset = 0.25)
    template_ipsp_base= 1-template_epsp_base
    
    if STA_Mode == 1:
    
        print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of spikes')
        sites = np.loadtxt(Root_folder + '/Spike_Count.csv', dtype=float, delimiter=',')
        
        sites = np.stack((np.arange(sites.size), sites), axis=1) # add row number before the counts
        sites = sites[sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
        sites = np.array(sites[::-1],dtype=int) # make it descending
        site_index = sites[0:num_of_sites, 0] # extract the sites to do STA
        site_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in site_index] # convert site_index to site_string in format of row+column
    # Use the top 50 sites with most spikes or use designated site list
    else:
        print('Extract the spike time of the assigned sites in the site_list')
        site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    
    
    with open(Root_folder + '/Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
        rows = list(csv.reader(spiketime_read))
        Spike_Time=[rows[index] for index in site_index]
    
    site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]    
        
    ############## generate the STA file for the sites listed site_string #####################  
    
    if len(site_string) == len(Spike_Time): # to confirm how many rows have been read
        num_sites = len(Spike_Time)
        print('Spiketime reading is correct!')
    else:
        num_sites = len(Spike_Time)
        print('Spiketime reading is not correct!')
        
    if DC_flatten:
        STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+ '_from' + str(Start_time)+'to'+str(Start_time+Length) + Cross_dish_label + Weighted_label + '_DC_flatten/'
    else:
        STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+ '_from' + str(Start_time)+'to'+str(Start_time+Length) + Cross_dish_label + Weighted_label +'/'
    
    # STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+'/'
    
    isdir = os.path.isdir(STA_file_dir)  
    if isdir==False:
        os.mkdir(STA_file_dir)
    
    
    print('Start plotting subsite waveform')    
    
    # now = datetime.now()
    # dt_string = now.strftime("%Y%m%d%H%M")
    
    # if IPSP:
    #     Waveform_folder = STA_file_dir + 'STA_Relation_Base_with_IPSP_' + dt_string +'/'
    # else:
    #     Waveform_folder = STA_file_dir + 'STA_Relation_Base_no_IPSP_' + dt_string +'/'
        
    if IPSP:
        Waveform_folder = STA_file_dir + 'STA_Relation_Base_with_IPSP'  +'/'
    else:
        Waveform_folder = STA_file_dir + 'STA_Relation_Base_no_IPSP'  +'/'        
        
        
    isdir = os.path.isdir(Waveform_folder)  
    if isdir==False:
        os.mkdir(Waveform_folder)
    
    
    
    #****** for relationship saving ***********
    
    head_sum = ['PreSite', 'PostSite', 'Correlation', 'SynapseType', 'Delay']
    with open(Waveform_folder + 'Sum_Relation_Info_v5.csv', 'w', newline='') as Sum_file_append:  # save connection information to a csv file
        write = csv.writer(Sum_file_append) 
        write.writerow(head_sum)
    
    head_type = ['AP', 'APEP', 'EPSP', 'IPSP']
    with open(Waveform_folder + 'Sum_Correlation_Info_v5.csv', 'w', newline='') as Sum_file_append:  # save correlation factors to a csv file
        write = csv.writer(Sum_file_append) 
        write.writerow(head_type)
    
    
    connection_total = 0
    connection_site = 0
    
    relation_info = np.zeros(5)
    relation_info_list = []
    
    correlation_info = np.zeros(4)
    correlation_info_list = []
    
    save_fig = save_png or save_pdf 
    
    #****** for relationship saving ***********
    
    
    # tqdm is used to track progress
    for site_idx in tqdm(range(len(site_string))):
        site = site_string[site_idx]
        site_arr_index = site_index[site_idx]
        
        if save_fig: # run this only when saving figures
            Waveform_sub_folder = Waveform_folder + site +'/'
            isdir = os.path.isdir(Waveform_sub_folder)  
            if isdir==False:
                os.mkdir(Waveform_sub_folder)
    
        try:
            STA_file = STA_file_dir + 'STA_'+ site + '.csv'
            STA_Array = np.loadtxt(STA_file, dtype=float, delimiter=',')
            t = np.arange(0,len(template_ap_base))*1/F_sample # randomly selct a template to calculate the length
            
            connection_site = 0 # reset the connection number of every site
            relation_info_list =[]
            correlation_info_list =[]
            
            
            if len(STA_Array[0]) > win_size:
                site_target_list = STA_Array[:,-2].astype(int).tolist()
            
    
            for Col_num in np.arange(1, 65):
                
                if save_fig:
                    ax_index = int((Col_num-1)%col_per_figure)
                    if (Col_num-1)%col_per_figure ==0:
                        plt.close('all')
                        plt.rcParams["font.family"] = "Arial"
                        Font_Size = 6
                        fig, ax_array = plt.subplots(nrows=1, ncols=col_per_figure, gridspec_kw = {'wspace':0, 'hspace':0})
                        fig.set_size_inches(18.5, 10.5, forward=True)
                        ax_array = ax_array.flatten()
                        plt.ion()
                        plt.tight_layout()    
                    
                   
                for Row_num in np.arange(1, 65):   
                    site_idx = (Row_num - 1)*64 + Col_num -1 
                    site_loc = int(Row_num*100+Col_num)
                    
                    
                    if len(STA_Array[0]) == win_size:
                        Site_Data = STA_Array[site_idx,:]
                    elif site_loc in site_target_list:
                        Site_Data = STA_Array[site_target_list.index(site_loc),0:win_size]
                    elif Row_num == 64:
                        if save_fig:
                            ax_array[ax_index].axis(ymin=-65,ymax=0)
                            # ax_array[ax_index].axvline(x=0.025, color='green', linestyle=':', linewidth=0.25, label='axvline - full height')
                            ax_array[ax_index].axvline(x=0.025, color='green', linestyle=':', linewidth=0.25)
                            ax_array[ax_index].set_title('Col: ' + str(Col_num).zfill(2))
                        
                        continue
                    else:
                        continue
                        
                    Site_Data_MA = Site_Data - np.min(Site_Data)
                    max_amp = np.max(Site_Data_MA)
                    peak = Site_Data_MA.argmax()
                    base = Site_Data_MA.argmin()
                    corr_max = -1 # reset the value
                    
                    # print(peak)
                    
                    if max_amp !=0:
                        Site_Data_MA = Site_Data_MA/max_amp
                    Site_name = str(Row_num).zfill(2) + str(Col_num).zfill(2)
                    
                    if save_fig:
                        Offset = - (Row_num)
                        ax_array[ax_index].plot(t, Moving_Average(Site_Data_MA,10) + Offset, linewidth=0.5)
    
                        
                    #****** extracting AP/+ and EPSP information ***********    
                    if peak > 240 and peak < 470:
                        template_epsp_roll= Template_PSP(width_total = 705, width_pre = peak - 110, width_rise = 110, decay_factor = 250, offset = 0.25)
                        template_ap_roll= Template_PSP(width_total = 705, width_pre = peak - 40, width_rise = 40, decay_factor = 90, offset = 0.15)
                        # template_ap_roll = Template_AP(width_total = 705, width_pre = peak -200, width_rise = 200, rise_factor = 3, decay_factor = 100, offset = 0)
                        corr_epsp_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_epsp_roll)
                        corr_ap_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_ap_roll)
                        peak_roll_delay = peak
                        
                    else:
                        template_epsp_roll= Template_PSP(width_total = 705, width_pre = 355 - 110, width_rise = 110, decay_factor = 250, offset = 0.25)
                        template_ap_roll= Template_PSP(width_total = 705, width_pre = 355 - 40, width_rise = 40, decay_factor = 90, offset = 0.15)
                        corr_epsp_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_epsp_roll)
                        corr_ap_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_ap_roll)
                        peak_roll_delay = 355
                        
                    try:
                        corr_ap_base, pValue = sci_stats.pearsonr(Site_Data_MA,template_ap_base)
                        corr_epsp_base, pValue = sci_stats.pearsonr(Site_Data_MA,template_epsp_base)
                        corr_epsp = np.max([corr_epsp_roll, corr_epsp_base])
                        corr_ap = np.max([corr_ap_roll, corr_ap_base])
                        
    
                    except Exception as error:
                        print('\nError happens when calculating correlation at site: ' + Site_name+ ', and the error message is: ')
                        print(error) 
                        corr_epsp = 0
                        corr_ap = 0
                            
    
                        
                    #****** extracting IPSP information ***********
                    if IPSP:
                        if base > 350 and base < 470:
                            template_psp= Template_PSP(width_total = 705, width_pre = base -110, width_rise = 110, decay_factor = 250, offset = 0.25)
                            template_ipsp_roll= 1-template_psp
                            base_roll_delay = base
                        
                            try:
                                corr_ipsp_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_ipsp_roll)
                                corr_ipsp_base, pValue = sci_stats.pearsonr(Site_Data_MA,template_ipsp_base)
                                corr_ipsp = np.max([corr_ipsp_roll, corr_ipsp_base])
                                corr_max = np.max([corr_ap, corr_epsp, corr_ipsp])
                            
                            except Exception as error:
                                print('\nError happens when calculating correlation at site: ' + Site_name+ ', and the error message is: ')
                                print(error) 
                                corr_max = np.max([corr_ap, corr_epsp])
                                
                        else: # base is not valid,and use the middle of the range
                            template_psp= Template_PSP(width_total = 705, width_pre = 410 -110, width_rise = 110, decay_factor = 250, offset = 0.25)
                            template_ipsp_roll= 1-template_psp
                            base_roll_delay = 410
                            try:
                                corr_ipsp_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_ipsp_roll)
                                corr_ipsp_base, pValue = sci_stats.pearsonr(Site_Data_MA,template_ipsp_base)
                                corr_ipsp = np.max([corr_ipsp_roll, corr_ipsp_base])
                                corr_max = np.max([corr_ap, corr_epsp, corr_ipsp])
                            
                            except Exception as error:
                                print('\nError happens when calculating correlation at site: ' + Site_name+ ', and the error message is: ')
                                print(error) 
                                corr_max = np.max([corr_ap, corr_epsp])
                    else:
                        corr_max = np.max([corr_ap, corr_epsp])
                                # corr_max = corr_epsp

                    #****** for all correlationship saving ***********
                    correlation_info[0] = corr_ap_base # AP template
                    correlation_info[1] = corr_ap_roll # AP_EPSP template
                    correlation_info[2] = corr_epsp # EPSP template
                    correlation_info[3] = corr_ipsp  # IPSP template
                    correlation_info_list.append(np.copy(correlation_info)) # save to a list temporary to reduce access to .csv file
                    #****** for all correlationship saving ***********


                        
                    if corr_max > corr_limit: # regarded as effective connnection
                    
                        #****** for relationship saving ***********
                        if corr_max == corr_epsp_base:
                            syn_type = 1
                            textcolor = 'red'
                            template_plot = template_epsp_base
                            temp_delay = round((345 + peak)/2) # peak point of epsp_base:235 + 110, and the peak point of avaraged waveform
                        elif corr_max == corr_epsp_roll:
                            syn_type = 1
                            textcolor = 'red'
                            template_plot = template_epsp_roll
                            temp_delay = peak_roll_delay # peak
                            
                        elif corr_max == corr_ap_base:
                            syn_type = 0
                            textcolor = 'blue'
                            template_plot = template_ap_base
                            temp_delay = round((235 + peak)/2) # peak point of ap_base, 200 + 35
    
                        elif corr_max == corr_ap_roll:
                            syn_type = 0.5
                            textcolor = 'blue'
                            template_plot = template_ap_roll
                            temp_delay = peak_roll_delay # peak
                            
                        elif corr_max == corr_ipsp_base:
                            syn_type = -1
                            textcolor = 'purple'
                            template_plot = template_ipsp_base
                            temp_delay = round((345 + base)/2) # base point of ipsp_base:235 + 110, and the base/bottom point of avaraged waveform
                            
                        elif corr_max == corr_ipsp_roll:
                            syn_type = -1
                            textcolor = 'purple'
                            template_plot = template_ipsp_roll
                            temp_delay = base_roll_delay # base_roll
                        
                        connection_total = connection_total + 1
                        connection_site = connection_site +1
                        
                        relation_info[0] = int(site) # presite
                        relation_info[1] = int(Site_name) # postsite
                        relation_info[2] = corr_max
                        relation_info[3] = syn_type  # 1 for excitatory, -1 for inhibitory, 0 same site, 0.5, strong connection
                        relation_info[4] = temp_delay # delay
                        relation_info_list.append(np.copy(relation_info)) # save to a list temporary to reduce access to .csv file
                            

                        
                        #****** for relationship saving ***********
    
                        if save_fig:
                            ax_array[ax_index].plot(t, template_plot + Offset, linestyle=':', linewidth=0.5, color = 'black')
                            ax_array[ax_index].text(t[10], Offset + 0.5, "{:.2f}".format(corr_max), color= textcolor, fontsize= 12 )
                            
                            
                    if save_fig:
                        if Row_num == 64: # add title,and line for a column 
                            ax_array[ax_index].axis(ymin=-65,ymax=0)
                            # ax_array[ax_index].axvline(x=0.025, color='green', linestyle=':', linewidth=0.25, label='axvline - full height')
                            ax_array[ax_index].axvline(x=0.025, color='green', linestyle=':', linewidth=0.25)
                            ax_array[ax_index].set_title('Col: ' + str(Col_num).zfill(2))
                        
    
                if (Col_num-1)%col_per_figure == (col_per_figure -1):
                    if save_fig:
                        plt.ioff()
                    if save_png:
                        plt.savefig(Waveform_sub_folder +'Column_'+str(Col_num-col_per_figure) + ' to ' +str(Col_num) + '.png', bbox_inches = 'tight',dpi=220) # change '.png' to '.pdf' if a vector image is required
                    if save_pdf: # saving .pdf is slow. It is disabled by default
                        plt.savefig(Waveform_sub_folder +'Column_'+str(Col_num-col_per_figure) + ' to ' +str(Col_num) + '.pdf', bbox_inches = 'tight') # change '.png' to '.pdf' if a vector image is required
        
            print('Found ' + str(connection_site) + ' new connections.')
            print('Found ' + str(connection_total) + ' connections in total.')
            
            
            with open(Waveform_folder + 'Sum_Relation_Info_v5.csv', 'a', newline='') as Sum_file_append:  # save connection information of a single site from the temporaty list to a csv file
                write = csv.writer(Sum_file_append) 
                for relation_item in relation_info_list:
                    write.writerow(relation_item)
            
            
            with open(Waveform_folder + 'Sum_Correlation_Info_v5.csv', 'a', newline='') as Sum_file_append:  # save connection information of a single site from the temporaty list to a csv file
                write = csv.writer(Sum_file_append) 
                for correlation_item in correlation_info_list:
                    write.writerow(correlation_item)
                        
            
            
        except Exception as error:
            print('\nError happens in site: ' +str(site) + ', and the error message is: ')
            print(error)    
        
    
    print ('There are ' + str(connection_total) + ' connections in total')             
    with open(Waveform_folder + 'Relation_sum_' + str(connection_total) + '.csv', 'a', newline='') as Relation_file_sum:  # save information to a csv file
        write = csv.writer(Relation_file_sum) 
        write.writerow([connection_total])
    
    if Corr_Hist_Plot:
        print ('Start plotting the histogram report!')
        Corr_Hist_Gen(Waveform_folder)



def Relation_Define_STA_hist(Root_folder, num_of_sites=1000, site_string=['0101','0102'], amp_limit = 30e-3, amp_limit_pre = 6e-3, corr_limit = 0.7, Start_time=0, Length=2400, F_sample = 1205120/128, Positive=True, STA_Mode=1,  col_per_figure = 16, save_png= False, save_pdf = False, IPSP= False):

    template_ap_base = Template_AP(width_total = 705, width_pre = 35, width_rise = 200, rise_factor = 3, decay_factor = 100, offset = 0)
    template_epsp_base= Template_PSP(width_total = 705, width_pre = 235, width_rise = 110, decay_factor = 250, offset = 0.25)
    template_ipsp_base= 1-template_epsp_base
    
    peak = 340 # (240+470)/2
    template_ap_roll= Template_PSP(width_total = 705, width_pre = peak - 40, width_rise = 40, decay_factor = 90, offset = 0.15)

    #if base > 350 and base < 470:
    base = 410 # (350+470)/2
    template_psp= Template_PSP(width_total = 705, width_pre = base -110, width_rise = 110, decay_factor = 250, offset = 0.25)
    template_ipsp_roll= 1-template_psp
    
    
    if STA_Mode == 1:
    
        print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of spikes')
        sites = np.loadtxt(Root_folder + '/Spike_Count.csv', dtype=float, delimiter=',')
        
        sites = np.stack((np.arange(sites.size), sites), axis=1) # add row number before the counts
        sites = sites[sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
        sites = np.array(sites[::-1],dtype=int) # make it descending
        site_index = sites[0:num_of_sites, 0] # extract the sites to do STA
        site_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in site_index] # convert site_index to site_string in format of row+column
    # Use the top 50 sites with most spikes or use designated site list
    else:
        print('Extract the spike time of the assigned sites in the site_list')
        site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    
    
    with open(Root_folder + '/Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
        rows = list(csv.reader(spiketime_read))
        Spike_Time=[rows[index] for index in site_index]
    
    site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]    
        
    ############## generate the STA file for the sites listed site_string #####################  
    
    if len(site_string) == len(Spike_Time): # to confirm how many rows have been read
        num_sites = len(Spike_Time)
        print('Spiketime reading is correct!')
    else:
        num_sites = len(Spike_Time)
        print('Spiketime reading is not correct!')
    STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+ '_from' + str(Start_time)+'to'+str(Start_time+Length) +'/'
    
    # STA_file_dir = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV'+'/'
    
    isdir = os.path.isdir(STA_file_dir)  
    if isdir==False:
        os.mkdir(STA_file_dir)
    
    
    print('Start plotting subsite waveform')    
    
    # now = datetime.now()
    # dt_string = now.strftime("%Y%m%d%H%M")
    
    # if IPSP:
    #     Waveform_folder = STA_file_dir + 'STA_Relation_Base_with_IPSP_' + dt_string +'/'
    # else:
    #     Waveform_folder = STA_file_dir + 'STA_Relation_Base_no_IPSP_' + dt_string +'/'
        
    if IPSP:
        Waveform_folder = STA_file_dir + 'STA_Relation_Base_with_IPSP'  +'/'
    else:
        Waveform_folder = STA_file_dir + 'STA_Relation_Base_no_IPSP'  +'/'        
        
        
    isdir = os.path.isdir(Waveform_folder)  
    if isdir==False:
        os.mkdir(Waveform_folder)
    
    
    
    #****** for relationship saving ***********
    
    # head_sum = ['PreSite', 'PostSite', 'Correlation', 'SynapseType']
    # with open(Waveform_folder + 'Sum_Relation_Info.csv', 'a', newline='') as Sum_file_append:  # save connection information to a csv file
    #     write = csv.writer(Sum_file_append) 
    #     write.writerow(head_sum)
    
    head_type = ['AP', 'APEP', 'EPSP', 'IPSP']
    with open(Waveform_folder + 'Sum_Correlation_Info.csv', 'w', newline='') as Sum_file_append:  # save correlation factors to a csv file
        write = csv.writer(Sum_file_append) 
        write.writerow(head_type)
    
    
    connection_total = 0
    connection_site = 0
    
    # relation_info = np.zeros(4)
    # relation_info_list = []
    
    correlation_info = np.zeros(4)
    correlation_info_list = []
    
    save_fig = save_png or save_pdf 
    
    #****** for relationship saving ***********
    
    
    # tqdm is used to track progress
    for site_idx in tqdm(range(len(site_string))):
        site = site_string[site_idx]
        site_arr_index = site_index[site_idx]
        
        if save_fig: # run this only when saving figures
            Waveform_sub_folder = Waveform_folder + site +'/'
            isdir = os.path.isdir(Waveform_sub_folder)  
            if isdir==False:
                os.mkdir(Waveform_sub_folder)
    
        try:
            STA_file = STA_file_dir + 'STA_'+ site + '.csv'
            STA_Array = np.loadtxt(STA_file, dtype=float, delimiter=',')
            t = np.arange(0,len(STA_Array[0,:]))*1/F_sample # randomly selct a site to calculate the length
            
            connection_site = 0 # reset the connection number of every site
            relation_info_list =[]
            correlation_info_list =[]
            
    
            for Col_num in np.arange(1, 65):
                
                if save_fig:
                    ax_index = int((Col_num-1)%col_per_figure)
                    if (Col_num-1)%col_per_figure ==0:
                        plt.close('all')
                        plt.rcParams["font.family"] = "Arial"
                        Font_Size = 6
                        fig, ax_array = plt.subplots(nrows=1, ncols=col_per_figure, gridspec_kw = {'wspace':0, 'hspace':0})
                        fig.set_size_inches(18.5, 10.5, forward=True)
                        ax_array = ax_array.flatten()
                        plt.ion()
                        plt.tight_layout()    
                    
                   
                for Row_num in np.arange(1, 65):   
                    site_idx = (Row_num - 1)*64 + Col_num -1 
                    Site_Data = STA_Array[site_idx,:]
                    Site_Data_MA = Site_Data - np.min(Site_Data)
                    max_amp = np.max(Site_Data_MA)
                    peak = Site_Data_MA.argmax()
                    base = Site_Data_MA.argmin()
                    corr_max = -1 # reset the value
                    
                    # print(peak)
                    
                    if max_amp !=0:
                        Site_Data_MA = Site_Data_MA/max_amp
                    Site_name = str(Row_num).zfill(2) + str(Col_num).zfill(2)
                    
                    #****** extracting AP/+ and EPSP information ***********    
                    try:
                        # corr_epsp_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_epsp_roll)
                        corr_ap_base, pValue = sci_stats.pearsonr(Site_Data_MA,template_ap_base)
                        corr_ap_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_ap_roll)
                        corr_epsp_base, pValue = sci_stats.pearsonr(Site_Data_MA,template_epsp_base)
                        corr_ipsp_roll, pValue = sci_stats.pearsonr(Site_Data_MA,template_ipsp_roll)

                    except Exception as error:
                        corr_ap_base = 0
                        corr_ap_roll = 0
                        corr_epsp_base = 0
                        corr_ipsp_roll = 0
                        print('\nError happens when calculating correlation at site: ' + Site_name+ ', and the error message is: ')
                        print(error) 

                    #****** for all correlationship saving ***********
                    correlation_info[0] = corr_ap_base # AP template
                    correlation_info[1] = corr_ap_roll # AP_EPSP template
                    correlation_info[2] = corr_epsp_base # EPSP template
                    correlation_info[3] = corr_ipsp_roll  # IPSP template
                    correlation_info_list.append(np.copy(correlation_info)) # save to a list temporary to reduce access to .csv file
                    #****** for all correlationship saving ***********

            
            with open(Waveform_folder + 'Sum_Correlation_Info.csv', 'a', newline='') as Sum_file_append:  # save connection information of a single site from the temporaty list to a csv file
                write = csv.writer(Sum_file_append) 
                for correlation_item in correlation_info_list:
                    write.writerow(correlation_item)
                        
            
        except Exception as error:
            print('\nError happens in site: ' +str(site) + ', and the error message is: ')
            print(error)   
    #****** plot the histgram of all the templates ***********    
    print('Start plotting histgram reports!') 
    Corr_Hist_Gen(Waveform_folder)
    
        
        
def Connection_Seek(Root_folder, num_of_sites=1000, site_string=['0101','0102'], Seek_Mode=1, spike_num_limit =50):    
 
    win_size=1883
    F_sample = 1205120/128
    # Positive=True
    
    if Seek_Mode == 1:
        print('Extract the spike time of the '+ str(num_of_sites) +' sites with most of number of spikes')
        sites = np.loadtxt(Root_folder + '/Spike_Count.csv', dtype=float, delimiter=',')
        
        sites = np.stack((np.arange(sites.size), sites), axis=1) # add row number before the counts
        sites = sites[sites[:,1].argsort(kind='mergesort')] # sort the array based on number of spikes
        sites = np.array(sites[::-1],dtype=int) # make it descending
        site_index = sites[0:num_of_sites, 0] # extract the sites to do STA
        site_string = [str(int(i/64)+1).zfill(2)+str(int(i%64) +1).zfill(2) for i in site_index] # convert site_index to site_string in format of row+column
    else:
        print('Extract the spike time of the assigned sites in the site_list')
        site_index = [64*(int(site[0:2])-1)+ int(site[2:4])-1 for site in site_string]
    
    
    with open(Root_folder + '/Spike_Index.csv', 'r') as spiketime_read:  # read the spiketime of interested sites
        rows = list(csv.reader(spiketime_read))
        Spike_Time=[rows[index] for index in site_index]
      
    ############## generate the STA file for the sites listed site_string #####################  
    
    if len(site_string) == len(Spike_Time): # to confirm how many rows have been read
        # num_sites = len(Spike_Time)
        print('Spiketime reading is correct!')
    else:
        # num_sites = len(Spike_Time)
        print('Spiketime reading is not correct!')
    Relation_file_dir = Root_folder + '/Relation_Files_Win_'+'{:.0f}'.format(win_size/5/F_sample*1000)+ '_Check_'+str(num_of_sites)+'sites'+'/'  # name the folder ending with the time window
    
    isdir = os.path.isdir(Relation_file_dir)
    if not isdir:
        os.mkdir(Relation_file_dir)
        
    # Relation_Array_temp = np.zeros((4096,9)) # to store the relation map: |Site|Relation|Num of spikes|Pre percentage|Post percentage|Time_start|Time_end
    
    
    for index in range(len(Spike_Time)):
        Spike_Time[index] = np.array(Spike_Time[index]).astype(float)
    
    
    
    head_sum = ['PreSite', 'PostSite', 'Spikes_overlapped', 'Pre percentage', 'Post percentage','Time_start', 'Time_end','Spikes_from','Spikes_to']
    with open(Relation_file_dir + 'Sum_Relation_Info.csv', 'a', newline='') as Sum_file_append:  # save connection information to a csv file
        write = csv.writer(Sum_file_append) 
        write.writerow(head_sum)
    
    
    # 470 = 50ms
    connection_total = 0
    for index in tqdm(range(len(site_index))):
        connection_site = 0
        relation_info = np.zeros(9)
        site_str = str(int(site_index[index]/64)+1).zfill(2) + str(int(site_index[index]%64)+1).zfill(2)
        head = ['Site', 'Relation', 'Spikes_overlapped', 'Pre percentage', 'Post percentage','Time_start', 'Time_end','Spikes_from','Spikes_to']
        with open(Relation_file_dir + 'Relation_Info_' + site_str + '.csv', 'a', newline='') as Relation_file_append:  # save all information to a csv file
            write = csv.writer(Relation_file_append) 
            write.writerow(head)
        
    
        
        
        # find out the if overlap exists or not
        Spike_Time[index] = np.array(Spike_Time[index]).astype(float)
        lower_bound = min(Spike_Time[index], default =-1)
        upper_bound = max(Spike_Time[index], default =-2)
        
        if lower_bound < 0 or len(Spike_Time[index]) < spike_num_limit:  # no spike in this site
            break
        
        for index_to in range(len(site_index)): # compare site 'index' with all other sites
            post = 0
            pre = 0
            lower_bound_to = min(Spike_Time[index_to], default =-1)
            upper_bound_to = max(Spike_Time[index_to], default =-2)
            if lower_bound > upper_bound_to or upper_bound < lower_bound_to: # no overlap
                break
            else:  # there is overlap, then find the overlap
                lower_bound_ol = max(lower_bound, lower_bound_to) # ol is short for overlap
                upper_bound_ol = min(upper_bound, upper_bound_to)
                lower_index_ol, lower_value_ol = Closest_value(Spike_Time[index], lower_bound_ol) #find the element index range in site 'index'
                upper_index_ol, upper_value_ol = Closest_value(Spike_Time[index], upper_bound_ol)
                num_spikes_ol = upper_index_ol - lower_index_ol +1
                
                lower_index_to, lower_value_to = Closest_value(Spike_Time[index_to], lower_bound_ol) #find the element index range in site 'index_to'
                upper_index_to, upper_value_to = Closest_value(Spike_Time[index_to], upper_bound_ol)
                num_spikes_to = upper_index_to - lower_index_to
                
                
                if min(num_spikes_ol,num_spikes_to) < spike_num_limit: # at least certain number of spikes should exist
                    break
                
                if upper_value_ol - lower_value_ol > F_sample*10: # overlap longer than 10s
                    for index_ol in np.arange(lower_index_ol, upper_index_ol+1):
                        time_value = Spike_Time[index][index_ol]
                        index_find, time_find = Closest_value(Spike_Time[index_to], time_value)
                        if abs(time_find - time_value) < win_size/5: # within 40ms = win_time/5:
                            if time_find > time_value: # the closest index found is after the original index
                                post = post +1
                            else:
                                pre  = pre + 1
            
            
            Pre_percent = max(pre/num_spikes_ol, pre/num_spikes_to)
            Post_percent = max(post/num_spikes_ol, post/num_spikes_to)
            
            if Post_percent > 0.95:
                relation = 2 # same neuron
            elif Post_percent >0.3:
                relation = 1 # post synapse
            elif Pre_percent >0.95:
                relation = 2 # same neuron
            elif Pre_percent >0.3:
                relation =-1 # pre synapse
            else:
                relation = 0 # no relation
        
            
            site_str_to = str(int(site_index[index_to]/64)+1).zfill(2) + str(int(site_index[index_to]%64)+1).zfill(2)
            
            
           
            relation_info[0] = site_str_to
            relation_info[1] = relation
            relation_info[2] = pre + post
            relation_info[3] = Pre_percent*100
            relation_info[4] = Post_percent*100
            relation_info[5] = lower_value_ol/F_sample
            relation_info[6] = upper_value_ol/F_sample
            relation_info[7] = num_spikes_ol
            relation_info[8] = num_spikes_to
            
            with open(Relation_file_dir + 'Relation_Info_' + site_str + '.csv', 'a', newline='') as Relation_file_append:  # save information to a csv file
                write = csv.writer(Relation_file_append) 
                write.writerow(relation_info)
            
            
            if relation ==-1:
                connection_total = connection_total +1
                connection_site = connection_site +1
                relation_info[1] = site_str # original site is as post synaptic neuron, relation_info[0] = site_str_to is pre synaptic neuron
                with open(Relation_file_dir + 'Sum_Relation_Info.csv', 'a', newline='') as Sum_file_append:  # save connection information to a csv file
                    write = csv.writer(Sum_file_append) 
                    write.writerow(relation_info)
                    
            if relation ==1:
                connection_total = connection_total +1
                connection_site = connection_site +1
                relation_info[0] = site_str # original site is presynaptic
                relation_info[1] = site_str_to # site_str_to is post synaptic neuron
                with open(Relation_file_dir + 'Sum_Relation_Info.csv', 'a', newline='') as Sum_file_append:  # save connection information to a csv file
                    write = csv.writer(Sum_file_append) 
                    write.writerow(relation_info)
                
        print('Found ' + str(connection_site) + ' new connections.')
        print('Found ' + str(connection_total) + ' connections in total.')
    print ('There are ' + str(connection_total) + ' connections in total')             
    
    with open(Relation_file_dir + 'Relation_sum_' + str(connection_total) + '.csv', 'a', newline='') as Relation_file_sum:  # save information to a csv file
        write = csv.writer(Relation_file_sum) 
        write.writerow([connection_total])



def Relation_Convert(Root_folder = '', window_list = [10,20,30,40]):
    File_name = 'Sum_Relation_Info' + '.csv'
    
    for win in window_list:
        FILE_path = Root_folder+ '/' + 'Relation_Files_Win_'+ str(win)+ '_Check_4096sites' 
    
        Connection_Pair = []
        
        try:
            Relation_Matrix = np.loadtxt(FILE_path + '/' +File_name, delimiter=",", dtype = int, skiprows=1)[:,0:2]
        except:
            print('win_'+str(win)+' does not exist!')
            continue
    
        for index in range(len(Relation_Matrix)):
            current_pair = [str(Relation_Matrix[index,0]).zfill(4), 'synapse', str(Relation_Matrix[index,1]).zfill(4)]
            if current_pair not in Connection_Pair:
                Connection_Pair.append(current_pair)
            
        with open(FILE_path + '/' +File_name[:-4]+ '_Win'+ str(win) + '.txt', 'w') as fp:
            for item in Connection_Pair:
                string_to_write = item[0]+' '+item[1]+ ' ' + item[2]
                fp.write("%s\n" % string_to_write) # write each item on a new line
            print('Done')




def Relation_Convert_Delta(Root_folder = '', window_list = [10,20,30,40]):
    File_name = 'Sum_Relation_Info' + '.csv'
    
    Connection_Pair_Previous = []
    for win in window_list:
        FILE_path = Root_folder+ '/' + 'Relation_Files_Win_'+ str(win)+ '_Check_4096sites' 
    
        Connection_Pair_Current = []
        
        try:
            Relation_Matrix = np.loadtxt(FILE_path + '/' +File_name, delimiter=",", dtype = int, skiprows=1)[:,0:2]
        except:
            print('win_'+str(win)+' does not exist!')
            continue
    
        for index in range(len(Relation_Matrix)):
            current_pair = [str(Relation_Matrix[index,0]).zfill(4), 'synapse', str(Relation_Matrix[index,1]).zfill(4)]
            if current_pair not in Connection_Pair_Current:
                Connection_Pair_Current.append(current_pair)
            
        with open(FILE_path + '/' +File_name[:-4]+ '_Delta_Win'+ str(win) + '.txt', 'w') as fp:
            for item in Connection_Pair_Current:
                if item not in Connection_Pair_Previous:
                    string_to_write = item[0]+' '+item[1]+ ' ' + item[2]
                    fp.write("%s\n" % string_to_write) # write each item on a new line
                else:
                    print (item)
            print('Done')
        Connection_Pair_Previous = Connection_Pair_Current
        
        
        
        
def Relation_Convert_Amp(Root_folder = '', window_list = [10,20,30,40], amp_list = [100,200,300,400]):
    File_name = 'Sum_Relation_Info' + '.csv'
    
    for win in window_list:
        for amp in amp_list:
            FILE_path = Root_folder+ '/' + 'Relation_Files_Win_'+ str(win) + '_Amp_' + str(amp) +'_Check_4096sites' 
        
            Connection_Pair = []
            
            try:
                Relation_Matrix = np.loadtxt(FILE_path + '/' +File_name, delimiter=",", dtype = int, skiprows=1)[:,0:2]
            
                for index in range(len(Relation_Matrix)):
                    Connection_Pair.append([str(Relation_Matrix[index,0]).zfill(4), 'synapse', str(Relation_Matrix[index,1]).zfill(4)])
                    
                with open(Root_folder + '/' +File_name[:-4]+ '_Win'+ str(win) + '_Amp' + str(amp) + '.txt', 'w') as fp:
                    for item in Connection_Pair:
                        string_to_write = item[0]+' '+item[1]+ ' ' + item[2]
                        fp.write("%s\n" % string_to_write) # write each item on a new line
                    print('Done')
            except:
                print ('Warning: no connection info or convertion is wrong!')
                
                
                
def Relation_Convert_Overlap(Root_folder_small = '', Root_folder_big = '', window_list = [10,20,30,40], amp_list = [100,200,300,400]):
    File_name = 'Sum_Relation_Info' + '.csv'
    
    for win in window_list:
        if amp_list == []:

            try:
                
                FILE_path_small = Root_folder_small+ '/' + 'Relation_Files_Win_'+ str(win) + '_Check_4096sites' 
                FILE_path_big = Root_folder_big+ '/' + 'Relation_Files_Win_'+ str(win) + '_Check_4096sites'
            
                Connection_Pair = []
                
                
                Relation_Matrix_small = np.loadtxt(FILE_path_small + '/' +File_name, delimiter=",", dtype = int, skiprows=1)[:,0:2]
                Relation_Matrix_big = np.loadtxt(FILE_path_big + '/' +File_name, delimiter=",", dtype = int, skiprows=1)[:,0:2]
            
                for index in range(len(Relation_Matrix_small)):
                    if (Relation_Matrix_small[index] in Relation_Matrix_big):
                        Connection_Pair.append([str(Relation_Matrix_small[index,0]).zfill(4), 'synapse', str(Relation_Matrix_small[index,1]).zfill(4)])
                        print(index)
                    
                with open(Root_folder_big + '/' +File_name[:-4]+ '_Win'+ str(win) + '_overlap.txt', 'w') as fp:
                    for item in Connection_Pair:
                        string_to_write = item[0]+' '+item[1]+ ' ' + item[2]
                        fp.write("%s\n" % string_to_write) # write each item on a new line
                    print('Done')
            except:
                print ('Warning: no connection info or convertion is wrong!')
            
            
        else:
            for amp in amp_list:
                FILE_path = Root_folder+ '/' + 'Relation_Files_Win_'+ str(win) + '_Amp_' + str(amp) +'_Check_4096sites' 
            
                Connection_Pair = []
                
                try:
                    Relation_Matrix = np.loadtxt(FILE_path + '/' +File_name, delimiter=",", dtype = int, skiprows=1)[:,0:2]
                
                    for index in range(len(Relation_Matrix)):
                        Connection_Pair.append([str(Relation_Matrix[index,0]).zfill(4), 'synapse', str(Relation_Matrix[index,1]).zfill(4)])
                        
                    with open(Root_folder + '/' +File_name[:-4]+ '_Win'+ str(win) + '_Amp' + str(amp) + '_overlap.txt', 'w') as fp:
                        for item in Connection_Pair:
                            string_to_write = item[0]+' '+item[1]+ ' ' + item[2]
                            fp.write("%s\n" % string_to_write) # write each item on a new line
                        print('Done')
                except:
                    print ('Warning: no connection info or convertion is wrong!')

def Relation_Convert_Overlap_v2(Root_folder_small = '', Root_folder_big = '', window_list = [10,20,30,40], amp_list = [100,200,300,400]):
    File_name = 'Sum_Relation_Info' + '.csv'
    
    for win in window_list:
        if amp_list == []:
            

            try:
                
                FILE_path_small = Root_folder_small+ '/' + 'Relation_Files_Win_'+ str(win) + '_Check_4096sites' 
                FILE_path_big = Root_folder_big+ '/' + 'Relation_Files_Win_'+ str(win) + '_Check_4096sites'
            
                Connection_Pair = []
                Connection_Pair_Info = []
                
                Relation_Matrix_small_full = np.loadtxt(FILE_path_small + '/' +File_name, delimiter=",", dtype = int, skiprows=1)
                Relation_Matrix_big_full = np.loadtxt(FILE_path_big + '/' +File_name, delimiter=",", dtype = int, skiprows=1)
                print('running')
                
                Relation_Matrix_small = Relation_Matrix_small_full[:,0:2]
                Relation_Matrix_big = Relation_Matrix_big_full[:,0:2]
                print('running2')
                
                
                for index in range(len(Relation_Matrix_small)):
                    if (Relation_Matrix_small[index] in Relation_Matrix_big):
                        Connection_Pair.append([str(Relation_Matrix_small[index,0]).zfill(4), 'synapse', str(Relation_Matrix_small[index,1]).zfill(4)])
                        
                        index_big = np.where(Relation_Matrix_big == Relation_Matrix_small[index])
                        
                        Connection_Pair_Info.append(np.concatenate(Relation_Matrix_small_full[index,0:5], Relation_Matrix_big_full[index_big,0:5]))
                        
                        print(index)
                    
                with open(Root_folder_big + '/' +File_name[:-4]+ '_Win'+ str(win) + '_overlap_v2.txt', 'w') as fp:
                    for item in Connection_Pair:
                        string_to_write = item[0]+' '+item[1]+ ' ' + item[2]
                        fp.write("%s\n" % string_to_write) # write each item on a new line
                    print('Done')
                    
                with open(Root_folder_big + '/' +File_name[:-4]+ '_Win'+ str(win) + '_overlap_info.csv', 'w') as fp:
                    np.savetxt(fp, Connection_Pair_Info, delimiter = ',')
                    print('Done')                    
                    
                    
            except:
                print ('Warning: no connection info or convertion is wrong!')
            
            
        else:
            for amp in amp_list:
                FILE_path = Root_folder+ '/' + 'Relation_Files_Win_'+ str(win) + '_Amp_' + str(amp) +'_Check_4096sites' 
            
                Connection_Pair = []
                
                try:
                    Relation_Matrix = np.loadtxt(FILE_path + '/' +File_name, delimiter=",", dtype = int, skiprows=1)[:,0:2]
                
                    for index in range(len(Relation_Matrix)):
                        Connection_Pair.append([str(Relation_Matrix[index,0]).zfill(4), 'synapse', str(Relation_Matrix[index,1]).zfill(4)])
                        
                    with open(Root_folder + '/' +File_name[:-4]+ '_Win'+ str(win) + '_Amp' + str(amp) + '_overlap.txt', 'w') as fp:
                        for item in Connection_Pair:
                            string_to_write = item[0]+' '+item[1]+ ' ' + item[2]
                            fp.write("%s\n" % string_to_write) # write each item on a new line
                        print('Done')
                except:
                    print ('Warning: no connection info or convertion is wrong!')
                
                

def Relation_Convert_STA(Root_folder = '', amp_limit = 30e-3, amp_limit_pre = 6e-3, corr_limit = 0.75, Spike_limit = 50, Start_time= 0, Length= 2400, IPSP = False):
    
    print('Convresion starts!')
    File_name = 'Sum_Relation_Info_v5.csv'
    
    if IPSP:
        File_path = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV' + '_from' + str(Start_time)+'to'+str(Start_time+Length) + '/' + 'STA_Relation_Base_with_IPSP'
        File_path_new = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV' + '_from' + str(Start_time)+'to'+str(Start_time+Length) + '/' + 'STA_Relation_Converted_with_IPSP_{:.2f}'.format(corr_limit)+ '_' + str(Spike_limit)
    else:
        File_path = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV' + '_from' + str(Start_time)+'to'+str(Start_time+Length) + '/' + 'STA_Relation_Base_no_IPSP'
        File_path_new = Root_folder + '/STA_Files' + '_{:.1f}'.format(amp_limit*1000/30)+ '_{:.1f}'.format(amp_limit_pre*1000/30)+ 'mV' + '_from' + str(Start_time)+'to'+str(Start_time+Length) + '/' + 'STA_Relation_Converted_no_IPSP_{:.2f}'.format(corr_limit)+ '_' + str(Spike_limit)
    
    
    Spike_num_file = Root_folder + '/' + 'Spike_Count.csv'
    
    
    isdir = os.path.isdir(File_path_new)
    if not isdir:
        os.mkdir(File_path_new)
    
    Connection_Pair = []
    
    
    try:
        Spike_num = np.loadtxt(Spike_num_file, delimiter=",", dtype = int, skiprows=0)
    except Exception as error:
        print(error)
    
    
    try:
        Relation_Matrix = np.loadtxt(File_path + '/' +File_name, delimiter=",", dtype = float, skiprows=1)    
    except Exception as error:
        print(error)
    
    for index in range(len(Relation_Matrix)):
        
        Pre_site = str(int(Relation_Matrix[index,0])).zfill(4)
        Post_site = str(int(Relation_Matrix[index,1])).zfill(4)
        Pre_site_index = (int(Pre_site[0:2])-1)*64 + int(Pre_site[2:4])-1
        Post_site_index = (int(Post_site[0:2])-1)*64 + int(Post_site[2:4])-1
        # print(Relation_Matrix[index,2])
        if (Pre_site !=Post_site)& (Spike_num[Pre_site_index]>Spike_limit)& (Spike_num[Post_site_index]>Spike_limit)& (Relation_Matrix[index,2]>corr_limit):
            current_pair = [Pre_site, 'synapse', Post_site]
            if current_pair not in Connection_Pair:
                Connection_Pair.append(current_pair)
                with open(File_path_new + '/Sum_Relation_Info_v5.csv', 'a', newline='') as Sum_file_append:  # save connection information to a csv file
                    write = csv.writer(Sum_file_append) 
                    write.writerow(Relation_Matrix[index])
    print('Find ' + str(len(Connection_Pair)) + ' connections.') 
    
    with open(File_path_new  + '/Relation_sum_' + str(len(Connection_Pair)) + '.csv', 'a', newline='') as Relation_file_sum:  # save information to a csv file
        write = csv.writer(Relation_file_sum) 
        write.writerow([len(Connection_Pair)])
    
       
    with open(File_path_new + '/' +File_name[:-4] + '.txt', 'w') as fp:
        for item in Connection_Pair:
            string_to_write = item[0]+' '+item[1]+ ' ' + item[2]
            fp.write("%s\n" % string_to_write) # write each item on a new line
        print('Convresion Done!')                
                
                
                
                
def Map_Plot(Root_folder = '', window_list = [10,20,30,40]):
    import matlab.engine
    Meng = matlab.engine.start_matlab()
    # Meng.addpath(Filepath) # Specify the dir of Connection_Map.m file, if it is not included in the path of Matlab
    
    for win in tqdm(window_list):
        File_name = 'Sum_Relation_Info_Win'+ str(win) + '.txt'
        File_path = Root_folder+ '/' + 'Relation_Files_Win_'+ str(win)+ '_Check_4096sites' 
        if os.path.isfile(File_path + '/' + File_name):
            Meng.Connection_Map(File_path, File_name, nargout=0)
        else:
            print ( 'Sum_Relation_Info_Win'+ str(win) + '.txt does not exist!')
    Meng.exit()

def Map_Plot_Delta(Root_folder = '', window_list = [10,20,30,40]):
    import matlab.engine
    Meng = matlab.engine.start_matlab()
    # Meng.addpath(Filepath) # Specify the dir of Connection_Map.m file, if it is not included in the path of Matlab
    
    for win in tqdm(window_list):
        File_name = 'Sum_Relation_Info_Delta_Win'+ str(win) + '.txt'
        File_path = Root_folder+ '/' + 'Relation_Files_Win_'+ str(win)+ '_Check_4096sites' 
        if os.path.isfile(File_path + '/' + File_name):
            Meng.Connection_Map_Delta(File_path, File_name, nargout=0)
        else:
            print ( 'Sum_Relation_Info_Delta_Win'+ str(win) + '.txt does not exist!')
    Meng.exit() 







# @numba.njit
def Spike_Detect(y, lag=20, threshold=5, influence=0):
    row_len, col_len= np.shape(y)
    signals = np.zeros((row_len,col_len))
    filteredY = y
    avgFilter = np.zeros((row_len,col_len))
    stdFilter = np.zeros((row_len,col_len))
    
    # for i in range(0, lag):
    #     avgFilter[:, i] = np.mean(y[:, 0: lag],axis=1)
    #     stdFilter[:, i] = np.std(y[:, 0: lag],axis=1)
    avgFilter[:, lag - 1] = np.mean(y[:, 0: lag],axis=1)
    stdFilter[:, lag - 1] = np.std(y[:, 0: lag],axis=1)

    for i in range(lag, col_len):

        limit = threshold * stdFilter [:, i-1] # range 
        above_limit = y[:, i] - (avgFilter[:, i-1]+ limit) - 1e-6 # upboundary
        below_limit = y[:, i] - (avgFilter[:, i-1]- limit) + 1e-6 #lowerboundary
        
        signals[:, i] = np.array((above_limit * below_limit)/(5**2)+1, dtype=np.uint8) # if the element >0, then that means data is above threshold
        #signals[:, i] =  np.floor((above_limit * below_limit)/(5**2)+1)
        #signals[:, i] =  (above_limit * below_limit)/(5**2)+1
        
        
        influence_column = influence * signals[:, i] + 1*(1 - signals[:, i]) # when signal is 0, influence doesn't apply.
        
        filteredY[:, i] = influence_column * y[:, i] + (1 - influence_column) * filteredY[:, i-1]
        avgFilter[:, i] = np.mean(filteredY[:, (i-lag+1):i+1], axis=1)
        stdFilter[:, i] = np.std(filteredY[:, (i-lag+1):i+1], axis=1)
    
    x = np.sum(signals[:,:-1] < signals[:,1:], axis=1) # calcualte the number of rising edge
    return x, signals




def Site_Read_h5(Root_folder,Site_name, Start_time=0, Length=1200):
    global F_sample
    Row_num=int(Site_name[0:2])
    Col_num=int(Site_name[2:4])-1
    
    h5_File=Root_folder + '/Row_' + str(Row_num) + '.h5'
    s_factor =0.000156252
    with tb.open_file(h5_File) as site_data:
        try:
         return pd.DataFrame(site_data.root.data[Start_time*F_sample:(Start_time + Length)*F_sample,Col_num]*s_factor*-1)  # add a minus sign to cancel the polarity of the inverting amplifer
        except:
         return pd.DataFrame(site_data.root.data[Start_time*F_sample::,Col_num]*s_factor*-1) # if Length is too big, then just read all  the data



def Data_Segment_h5(Root_folder, sites, start_time = None, end_time = None, background_correction=True, 
                 Rolling_window_size = None, F_sample = F_sample, F_sample_PCB = 120512, progress_bar = True):
    # Check for sites format to be list of ints or strings
    if type(sites) != list:
        if type(sites) == str:
            sites = [sites]
        else:
            print("Wrong sites")
            return
    # Whether we want progressbar?
    if progress_bar:
        df = pd.concat([Site_Read_h5(Root_folder, Site_name) for Site_name in tqdm(sites)], axis = 1)
    else:
        df = pd.concat([Site_Read_h5(Root_folder, Site_name) for Site_name in sites], axis = 1)
    ## Processing of the column names to be only site names, otherwise would also have group name
    #df.columns = df.columns.str.split('_').str[-1].str[:-1]
    
    ## Simple background correction based on average of all readings
    if background_correction:
        df = df - df.mean()
    
    ## Time stamps are determined based on the input frequencies. The query based on start/end time could either or both be None, which would output from very beginning or till very end
    df.index = np.arange(0,df.shape[0]/F_sample,1/F_sample)
    df.index.name = 'time'
    df = df.loc[start_time:end_time]
    
    ## If rolling window size is not None (default None), perform rolling window average
    if Rolling_window_size:
        df = df.rolling(Rolling_window_size).mean().iloc[Rolling_window_size:]
    
    return df




def Site_Read_TDMS(Root_folder,Site_name):

    Row_num=int(Site_name[0:2])
    
    # Caclulate data row and column
    Multiplexer_Row=int((Row_num-1)/2);
    
    Site_Num = 'Site_'+Site_name
    
    if Multiplexer_Row<16:
        Group_Name='Analog_Input_DAQ1';
    else:
        Group_Name='Analog_Input_DAQ2';
    
    
    TDMS_File=Root_folder + '/AI_' + str(Multiplexer_Row).zfill(2) + '.tdms'
    with TdmsFile.open(TDMS_File) as tdms_file:
        try:
            ## add a minus sign to cancel the polarity of the inverting amplifer
            return tdms_file[Group_Name][Site_Num].as_dataframe() 
            #return -tdms_file[Group_Name][Site_Num][:]
        except: 
            ## catch exception if one of the sites does not exist
            print("%s %s does not exist, move on to next site" % (Group_Name, Site_Num))
            return None


def Data_Segment_TDMS(Root_folder, sites, start_time = None, end_time = None, background_correction=True, 
                 Rolling_window_size = None, F_sample = F_sample, F_sample_PCB = 120512, progress_bar = True):
    # Check for sites format to be list of ints or strings
    if type(sites) != list:
        if type(sites) == str:
            sites = [sites]
        else:
            print("Wrong sites")
            return
    # Whether we want progressbar?
    if progress_bar:
        df = pd.concat([Site_Read_TDMS(Root_folder, Site_name) for Site_name in tqdm(sites)], axis = 1)
    else:
        df = pd.concat([Site_Read_TDMS(Root_folder, Site_name) for Site_name in sites], axis = 1)
    ## Processing of the column names to be only site names, otherwise would also have group name
    df.columns = df.columns.str.split('_').str[-1].str[:-1]
    
    ## Simple background correction based on average of all readings
    if background_correction:
        df = df - df.mean()
    
    ## Time stamps are determined based on the input frequencies. The query based on start/end time could either or both be None, which would output from very beginning or till very end
    df.index = np.arange(0,df.shape[0]/F_sample,1/F_sample)
    df.index.name = 'time'
    df = df.loc[start_time:end_time]
    
    ## If rolling window size is not None (default None), perform rolling window average
    if Rolling_window_size:
        df = df.rolling(Rolling_window_size).mean().iloc[Rolling_window_size:]
    
    return df



#% Find peaks using peakutils; assuming correction using polynomial fitting, 

def baseline_correction(data, deg = 3, thres = 0.5, min_dist = 0.05 * F_sample, plot = False):
    # Data should be panda series with timestamps as index
    
    # First, make everything positive to avoid math domain error.
    y_correction  = data.min()
    data_positive = data - y_correction
    baseline      = peakutils.baseline(data_positive, deg)
    baseline_corr = baseline + y_correction
    data_corr     = (data - baseline_corr)
    data_corr     /= data_corr.max()
    peaktime_pre     = data.index[peakutils.indexes(data.to_numpy(), thres, min_dist)]
    peaktime_after   = data_corr.index[peakutils.indexes(data_corr.to_numpy(), thres, min_dist)]
    
    if plot:
        figs, axes = plt.subplots(2,1, sharex = True)
        data.plot(ax = axes[0], label = data.name)
        axes[0].scatter(peaktime_pre, data.loc[peaktime_pre], c = 'red', label = '%d peaks' % len(peaktime_pre))
        axes[0].legend()
    
        data_corr.plot(ax = axes[1], label = data.name + '_corrected')
        axes[1].scatter(peaktime_after, data_corr.loc[peaktime_after], c = 'red', label = '%d peaks' % len(peaktime_after))
        axes[1].legend()
    return data_corr, peaktime_after

#% Average peaks
def average_spikes(data, peaktime, avg_window_size_ms = 20, plot = False):
    # Data should be panda series with timestamps as index
    
    # Get index number instead of absolute time stamp
    peakindex = np.array([data.index.get_loc(x) for x in peaktime])
    
    # normalize window size by frequency
    avg_window_size_half = int(avg_window_size_ms/1000 * F_sample / 2)
    # Taking half before, and half after the peak time point
    time_index = np.concatenate([range(-avg_window_size_half, avg_window_size_half)])
    
    avg_peak = np.array([data.iloc[peakindex + x].mean() for x in time_index])
    
    time_axis = time_index / F_sample
    
    if plot:
        plt.plot(time_axis, avg_peak)
        plt.xlabel ("Time (ms)")
        plt.title("Average peak from %d peaks identified from site %s" % (len(peaktime), data.name))
    return pd.Series(avg_peak, index = time_axis, name = data.name)


time_shift = 60 #unit is second for console 1-4 , v3 computer
# time_shift = 120 #unit is second for console 5-8 , v3 computer
# time_shift = 180 #unit is second for console 9-12 , v3 computer
# time_shift = 240 #unit is second for console 9-12 , v5 computer



def Site_Screen_Num(Root_folder, File_name ='/Spike_Index.csv', Start_time_list=[], End_time_list=[], Num_th = 500):
    Site_list = []
    Fs = 9415 
    with open(Root_folder + File_name, 'r') as spiketime_read:  # read the spiketime of interested sites
        rows = list(csv.reader(spiketime_read))
        # Spike_Time=[rows[index] for index in site_index]
    
    for idx, item in enumerate(rows):
        if len(item)> Num_th:
            item = np.array(item,dtype=float)/Fs
            qual_flag = False
            for tdx, stime in enumerate(Start_time_list):
                qual_flag =qual_flag | (np.sum((item > stime)&(item < End_time_list[tdx])) >= Num_th)
            if qual_flag:
                Site_list.append(str(int(idx/64)+1).zfill(2)+str(idx%64+1).zfill(2))
    return Site_list
            


def Mapping_Plot(File_path, File_name, Type = 2):
    
    if Type ==1:
        type_str = 'EPSP'
    elif Type == 0:
        type_str ='AP+'
    elif Type == -1:
        type_str = 'IPSP'
    else:
        type_str = 'All'
    
    # set the figure size to be square
    plt.close('all')
    fig = plt.figure(2,figsize = (8,6), dpi =180)
    ax = fig.add_subplot(111)

    # change the font to Arial with a size of 14
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.xticks([1,16,32,48,64])
    plt.yticks([1,16,32,48,64])
    plt.xlim([0,65])
    plt.ylim([0,65])
    
    
    Pairs_total = np.loadtxt(File_path + '/' +File_name, delimiter=",", dtype = int, skiprows=1)
    Pairs_subset = []
    
    if type_str == 'All':
        Pairs_subset = Pairs_total
    else: # extract the specific type
        for i in range(len(Pairs_total)):
            if Pairs_total[i][3] == Type:
                Pairs_subset.append(Pairs_total[i][0:2].tolist())
                
                
                
        
    Pairs_subset = np.array(Pairs_subset)
    Pre = Pairs_subset[:,0]
    Post = Pairs_subset[:,1]
    
    print('Calulating how many connected cells...')
    Cells = np.concatenate((Pre,Post))
    Unique_cells, counts = np.unique(Cells, return_counts=True)
    print('Find %d cells in %d connections.' %(len(Unique_cells),len(Pairs_subset)))
    
    
    Pre_x = Pre%100
    Post_x = Post%100
    Pre_y = (Pre/100).astype(int)
    Post_y = (Post/100).astype(int)
    
    plt.scatter(np.concatenate((Pre_x,Post_x)),np.concatenate((Pre_y,Post_y)), color='black', s=4)
    
    #%
    print('Start plotting the map...')
    
    Distance = []
    for i in tqdm(range(len(Pairs_subset))):
        x_values = [Pre[i]%100, Post[i]%100]
        y_values = [int(Pre[i]/100), int(Post[i]/100)]
        
        # calculate the number of segments based on the distance between points
        dist = np.hypot(x_values[1]-x_values[0], y_values[1]-y_values[0])
        Distance.append(dist)
        num_segments = int(dist) + 10
        
        # create an array of x and y values for each segment
        x_segments = np.linspace(x_values[0], x_values[1], num_segments)
        y_segments = np.linspace(y_values[0], y_values[1], num_segments)
        
        
        # define the color range from yellow to blue
        colors = plt.cm.coolwarm(np.linspace(0.1,1,num_segments))
        
        # plot each segment with a different color
        for j in range(num_segments-1):
            plt.plot([x_segments[j],x_segments[j+1]], [y_segments[j],y_segments[j+1]], color=colors[j])
    
    # add a color bar to show the range of colors used for the lines
    sm = plt.cm.ScalarMappable(cmap='coolwarm_r', norm=plt.Normalize(vmin=0.1,vmax=1))
    sm.set_array([])
    cbar=plt.colorbar(sm, ax=ax)
    
    # remove ticks from color bar and add custom labels at its ends 
    cbar.set_ticks([])
    
    cbar.ax.text(-0.2,1.01,'Pre-')
    cbar.ax.text(-0.2,0.06,'Post-')
    
    # # Add custom text labels at the ends of the colorbar
    # cbar.ax.text(-0.2, 1.05, 'Pre-', va='center', ha='right', transform=cbar.ax.transAxes)
    # cbar.ax.text(-0.2, 0.0, 'Post-', va='center', ha='right', transform=cbar.ax.transAxes)
    
    plt.title( type_str + '\n %d connections'%(len(Pairs_subset)) + ', %d cells' %(len(Unique_cells)))
    plt.show()
    
    
    Device = 'Mapping_' + File_name[0:-4] + '_' + type_str
    im_info = Device + ''
    Site_name = ''
    im_format = ['.pdf','.png'] # .pdf .eps .png
    
    Image_folder =os.path.dirname(os.path.realpath(__file__)) + '/Images/'
    Image_folder = File_path + '/Mapping_Images/'
    isdir = os.path.isdir(Image_folder)  
    if not isdir:
        os.mkdir(Image_folder)  
    
    global Image_id, im_info_old
    
    for f_index in range(len(im_format)):
    
        Image_name = Image_folder + Site_name + im_info +str(Image_id) + im_format[f_index]
        
        if os.path.isfile(Image_name):
            Image_id = Image_id + 1
            Image_name = Image_folder + Site_name + im_info +str(Image_id) + im_format[f_index] # update image name
        elif im_info_old != im_info:
            Image_id =0
            Image_name = Image_folder + Site_name + im_info +str(Image_id) + im_format[f_index] # update image name
            
        im_info_old = im_info
        plt.savefig(Image_name, bbox_inches = 'tight',dpi=200, transparent=False) 
        
        
        
    plt.close('all')
    fig = plt.figure(3,figsize = (8,6), dpi =180)
    #fig.patch.set_alpha(0.1)
    ax = fig.add_subplot(111)
    # ax.patch.set_alpha(0.0)
    # plt.tight_layout()    
    ax.patch.set_facecolor('white')
    Font_Size = 12
    Distance = np.array(Distance)*20
    ax.hist(Distance, 50, color='green',label = 'Distance')
    
    # plt.ylim([0,100])
    # plt.xlim([-0.2,30])
    
    # plt.title('65 min recording',fontsize=Font_Size+2, fontweight='bold')
    plt.xlabel('Distance (um)', fontsize=Font_Size,position=(0.5, 0.1))
    plt.ylabel('Sites (#)',  fontsize=Font_Size, position=(0, 0.5))
    ax.spines['left'].set_position(('outward',0))
    ax.spines['bottom'].set_position(('outward',0))
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xticks(fontsize=Font_Size)
    plt.yticks(fontsize=Font_Size)
    plt.show()
    
    # x_text = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.7 + ax.get_xlim()[0]
    # y_text = (ax.get_ylim()[1] - ax.get_ylim()[0])*0.8 + ax.get_ylim()[0]
    # y_step = (ax.get_ylim()[1] - ax.get_ylim()[0])*0.05
    
    # plt.legend(loc='upper right',fontsize=Font_Size-4)
    plt.title('Mean: {:.1f} um, median: {:.1f} um'.format(np.mean(Distance), np.median(Distance)), fontweight='bold')
    # plt.text(x_text, y_text, 'Total Sites: '+ "{:.0f}".format(np.sum(Site_coupled_2min)), fontsize=Font_Size, color='black')
    # plt.text(x_text, y_text - y_step, 'Maximum: '+ "{:.2f}".format(max(Amp_max.reshape(4096,1))[0]) +' mV', fontsize=Font_Size, color='black')
    # plt.text(x_text, y_text - 2*y_step, 'Mean: '+ "{:.2f}".format(np.mean(Amp_max.reshape(4096,1))) +' mV', fontsize=Font_Size, color='black')
    # plt.text(x_text, y_text - 3*y_step, 'STD: '+ "{:.2f}".format(np.std(Amp_max.reshape(4096,1))) +' mV', fontsize=Font_Size, color='black')
    plt.show()
    
    
    Device = 'Mapping_Distance_' + File_name[0:-4] + '_' + type_str
    im_info = Device + ''
    Site_name = ''
    im_format = ['.pdf','.png'] # .pdf .eps .png
    
    Image_folder =os.path.dirname(os.path.realpath(__file__)) + '/Images/'
    Image_folder = File_path + '/Mapping_Images/'
    isdir = os.path.isdir(Image_folder)  
    if not isdir:
        os.mkdir(Image_folder)  
    
    # global Image_id, im_info_old
    
    for f_index in range(len(im_format)):
    
        Image_name = Image_folder + Site_name + im_info +str(Image_id) + im_format[f_index]
        
        if os.path.isfile(Image_name):
            Image_id = Image_id + 1
            Image_name = Image_folder + Site_name + im_info +str(Image_id) + im_format[f_index] # update image name
        elif im_info_old != im_info:
            Image_id =0
            Image_name = Image_folder + Site_name + im_info +str(Image_id) + im_format[f_index] # update image name
            
        im_info_old = im_info
        plt.savefig(Image_name, bbox_inches = 'tight',dpi=200, transparent=False)     
        
        
        
        
#%%  Step 1: Data conversion and spike detection
#%%  Step 1: Data conversion and spike detection, a few more examples are listed in the following sections

FILE_path_list = {r'E:\2022_12_13\SHR3_1_10_08_09\Array Recording Intra Combined'}  

# FILE_path_list = {r'E:\2022_12_13\SHR3_1_10_08_09\Array Recording Intra Combined'}

Start_time=0 # start time, we can change the value to run spike detection for differnt time segments 
Ana_length =1200 # spike detection with 20 min recording


################ Normally we do not need to change the settings below ###################
################ Feel free to explore if you know what you are gonna change #############

Threshold = 0.3 #  0.125 for Multihole device, and 0.3 for single hole device
DAQ_num = [1,2,3] # It can be [1], [2], or [3], 1 for Row 1 to 32, 2 for Row 33 to 64, 3 for Vs1,2,3,4, and I

Root_folder = TDMS_CONV_COMB(FILE_path_list, DAQ_num, VC_mode = False) # Converted the TDMS files to .h5 files. The Root_folder returned by this function is the one with "xx_Converted". We only need run this function once.
Diff_order = [0] # can be [0,1,2], then it will also plot derivitive of recorded waveform, more info but much slower
Plot_Save_Labelled_v2(Root_folder, Start_time=Start_time, Ana_length=Ana_length, Save_fig=True, Row_range=[1,64], Threshold=Threshold, Diff_order=Diff_order) # perform spike detection, and generate four .csv files containing spike amplitude, count(how many), index(timing), 20-s windows with >= 4 spikes



#%% Step 2: Perform the STA and find the correlation and connection type of interested/selected sites
#%% Step 2: Perform the STA and find the correlation and connection type of interested/selected sites


Root_folder = r'E:\2022_12_13\SHR3_1_10_08_09\Array Recording Intra Combined\2022_12_13-10_19_59_Combined_Converted' 

num_of_sites=1000 # How many source sites
Start_time = 0 # start time, we can change the value to run STA and mapping for differnt time segments 
Length = 1200 #  # mapping with 20 min recording


################ Normally we do not need to change the settings below ###################
################ Feel free to explore if you know what you are gonna change #############

amp_limit = 30e-3 # define the maximum peak-peak within the 50 ms after source/pre spike to avoid big AP
amp_limit_pre = 6e-3 # define the maximum peak-peak within the 25 ms before source/pre spike to make sure stable baseline
corr_limit = 0.8 # threshold of the correlation between STA generated PSP waveforms and templates
Spike_limit = 500 # the pre/source site should at least have 500 spikes

STA_Mode = 1# There are two modes for performing STA, "1" is using the the most active sites (set by num_of_sites); "2" is using the selected sites defined by variable 'site_string'.
site_string = ['0101','1212'] # put the sites you want to do STA and mapping. This will only be used if STA_Mode = 2

IPSP= True # 'True' to include IPSP, 'False' to exclude IPSP
DC_flatten = False # remove the slow oscillation of baseline, but this seems having negligible effect
Time_shift = 0 # This is used in cross-dish to select different start-time from the source dish 

STA_file_dir,site_string = Generate_STA_Intra_PSP(Root_folder, Root_folder_target=Root_folder, num_of_sites=num_of_sites, site_string = site_string, amp_limit = amp_limit, amp_limit_pre=amp_limit_pre, Start_time=Start_time, Length=Length, Positive = True, STA_Mode= STA_Mode, DC_flatten= DC_flatten, Time_shift= Time_shift) # Generate the PSP waveforms for each source sites, saved as .csv file under the folder of 'STA_Files_XXXX'  

Relation_Define_STA_v5(Root_folder, Root_folder_target=Root_folder, num_of_sites=num_of_sites, site_string=site_string, amp_limit = amp_limit, amp_limit_pre = amp_limit_pre, corr_limit = corr_limit, Start_time=Start_time, Length=Length, F_sample = 1205120/128, Positive=True, STA_Mode=STA_Mode,  col_per_figure = 16, save_png= False, save_pdf = False, IPSP= IPSP, DC_flatten= DC_flatten, Corr_Hist_Plot= True) # Caculate the correlation values between the generated PSP waveforms and templates, then define the connection type. The resulted files are saved in the folder: STA_Relation_Base_with_IPSP

Relation_Convert_STA(Root_folder = Root_folder, amp_limit = amp_limit, amp_limit_pre = amp_limit_pre, corr_limit = corr_limit, Spike_limit = Spike_limit, Start_time=Start_time, Length=Length, IPSP = IPSP) # Perform the final screening of connections by applying the thresholds such as corr_limit, Spike_limit, etc. The generated files are placed in the folder: STA_Relation_Converted_with_XXXX



#%% Step 3: plot the map using the file 'Sum_Relation_Info_v5.csv' after screening

##%%%%%% change following variables to generate the mapping

File_path = r'E:\2022_12_13\SHR3_1_10_08_09\Array Recording Intra Combined\2022_12_13-10_19_59_Combined_Converted\STA_Files_1.0_0.2mV_from60to1260\STA_Relation_Converted_with_IPSP_0.80_500'
File_path = r'E:\2022_12_13\SHR3_1_10_08_09\Array Recording Intra Combined\2022_12_13-10_19_59_Combined_Converted\STA_Files_1.0_0.2mV_from660to1860\STA_Relation_Converted_with_IPSP_0.80_500'
File_name = 'Sum_Relation_Info_v5.csv' # connection information is saved in this file
Type = 2 #  1 for EPSP, 0 for AP, -1 for IPSP, 2 or any other number is to selelct all the connections

Mapping_Plot(File_path, File_name, Type= Type)

