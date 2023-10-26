# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 22:29:19 2022

NPM_OphysDataVis

For opto-stimulation-based sensor screening

@author: kenta.hagihara
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import csv
import glob
import json
from scipy.optimize import curve_fit
import re

import PreprocessingFunctions as pf

#3A8-BLA
#655422
#AnalDir0 = r"F:\Data_230504reorg\fpFIP_655422-p-15000_2023-03-24_14-30-08"
#AnalDir1 = r"F:\Data_230504reorg\fpFIP_655422-p-15000-GBR_2023-03-24_15-02-11"
#AnalDir2 = r"F:\Data_230504reorg\fpFIP_655422-q-15000_2023-03-15_13-34-12"
#AnalDir3 = r"F:\Data_230504reorg\fpFIP_655422-o-15000_2023-03-15_13-58-28"
#AnalDir4 = r"F:\Data_230504reorg\fpFIP_655422-p-15000_2023-03-15_14-22-02"

#655424
#AnalDir0 = r"F:\Data_230504reorg\fpFIP_655424-p-15000_2023-03-24_15-36-40"
#AnalDir1 = r"F:\Data_230504reorg\fpFIP_655424-p-15000-GBR_2023-03-24_16-06-59"
#AnalDir2 = r"F:\Data_230504reorg\fpFIP_655424-q-15000_2023-03-15_15-01-53"
#AnalDir3 = r"F:\Data_230504reorg\fpFIP_655424-o-15000_2023-03-15_15-25-39"
#AnalDir4 = r"F:\Data_230504reorg\fpFIP_655424-p-15000_2023-03-15_15-49-10"

#655425
#AnalDir0 = r"F:\Data_230504reorg\fpFIP_655425-p-15000_2023-03-24_16-57-20"
#AnalDir1 = r"F:\Data_230504reorg\fpFIP_655425-p-15000-GBR_2023-03-24_17-29-22"
#AnalDir2 = r"F:\Data_230504reorg\fpFIP_655425-q-15000_2023-03-15_16-35-56"
#AnalDir3 = r"F:\Data_230504reorg\fpFIP_655425-o-15000_2023-03-15_16-59-15"
#AnalDir4 = r"F:\Data_230504reorg\fpFIP_655425-p-15000_2023-03-15_17-22-18"

#669469 to fig
#AnalDir0 =r"F:\Data_230504reorg\fpFIP_669469_2023-04-11_16-35-33"
#AnalDir1 =r"F:\Data_230504reorg\fpFIP_669469-GBR-10mg_2023-04-11_17-21-21"
#AnalDir2 =r"F:\Data_230504reorg\fpFIP_669469-q-15000_2023-04-21_14-12-56"
#AnalDir3 =r"F:\Data_230504reorg\fpFIP_669469-o-15000_2023-04-21_14-39-37"
#AnalDir4 =r"F:\Data_230504reorg\fpFIP_669469-p-15000_2023-04-21_15-04-05"

#669472 (no 5/10/20Hz...)
#AnalDir0 =r"F:\Data_230504reorg\fpFIP_669472_2023-04-12_16-25-03"
#AnalDir1 =r"F:\Data_230504reorg\fpFIP_669472-GBR-10mg_2023-04-12_16-57-02"

#669473 (no 5/10/20Hz...)
#AnalDir0 =r"F:\Data_230504reorg\fpFIP_669473_2023-04-12_17-28-30"
#AnalDir1 =r"F:\Data_230504reorg\fpFIP_669473-GBR-10mg_2023-04-12_17-59-28"

#672636 23Aug30 added to the data set
AnalDir0 = r"F:\Data_230726reorg_1\fpFIP_FIP-672636-2023-05-30-18-54-03_2023-05-30_19-04-57"
AnalDir1 = r"F:\Data_230726reorg_1\fpFIP_FIP-672636-2023-05-30-19-24-56_2023-05-30_19-35-50"
AnalDir2 = r"F:\Data_230726reorg_1\fpFIP_FIP-672636-2023-05-30-18-08-14_2023-05-30_18-18-36"
AnalDir3 = r"F:\Data_230726reorg_1\fpFIP_FIP-672636-2023-05-30-18-30-53_2023-05-30_18-41-42"
AnalDir4 = r"F:\Data_230726reorg_1\fpFIP_FIP-672636-2023-05-30-18-54-03_2023-05-30_19-04-57"

#_______

#664627 3A8-NAc to fig
#AnalDir0 = r"F:\Data_230504reorg\fpFIP_664627_2023-04-28_15-26-57" #GBR- 20Hz 15ms
#AnalDir1 = r"F:\Data_230504reorg\fpFIP_664627-GBR_2023-04-28_16-00-21" #GBR+ 20Hz 15ms
#AnalDir2 = r"F:\Data_230504reorg\fpFIP_664627-q-15000_2023-05-01_16-01-49" #5Hz
#AnalDir3 = r"F:\Data_230504reorg\fpFIP_664627-o-15000_2023-05-01_16-31-08" #10Hz
#AnalDir4 = r"F:\Data_230504reorg\fpFIP_664627-p-15000_2023-05-01_17-19-18" #20Hz

#664629 3A8-NAc
#AnalDir0 = r"F:\Data_230504reorg\fpFIP_664629_2023-04-28_16-28-53"
#AnalDir1 = r"F:\Data_230504reorg\fpFIP_664629-GBR_2023-04-28_17-01-19"
#AnalDir2 = r"F:\Data_230504reorg\fpFIP_664629-q-15000_2023-05-01_18-08-22"
#AnalDir3 = r"F:\Data_230504reorg\fpFIP_664629-o-15000_2023-05-01_18-36-31"
#AnalDir4 = r"F:\Data_230504reorg\fpFIP_664629-p-15000_2023-05-01_18-59-45"

#669479 3A8-NAc
#AnalDir0 = r"F:\Data_230504reorg\fpFIP_669479_2023-04-28_14-28-14"
#AnalDir1 = r"F:\Data_230504reorg\fpFIP_669479-GBR_2023-04-28_14-57-41"
#AnalDir2 = r"F:\Data_230504reorg\fpFIP_669479-q-15000_2023-05-01_14-08-54"
#AnalDir3 = r"F:\Data_230504reorg\fpFIP_669479-o-15000_2023-05-01_14-45-18"
#AnalDir4 = r"F:\Data_230504reorg\fpFIP_669479-p-15000_2023-05-01_15-09-11"

#669485 3A8-NAc
#AnalDir0 = r"F:\Data_230504reorg\fpFIP_669485_2023-04-28_13-17-38"
#AnalDir1 = r"F:\Data_230504reorg\fpFIP_669485-GBR_2023-04-28_13-57-25"
#AnalDir2 = r"F:\Data_230504reorg\fpFIP_669485-q-15000_2023-05-01_12-27-10"
#AnalDir3 = r"F:\Data_230504reorg\fpFIP_669485-o-15000_2023-05-01_12-50-04"
#AnalDir4 = r"F:\Data_230504reorg\fpFIP_669485-p-15000_2023-05-01_13-12-56"

#_______

#673056 3A6-NAc to fig
#AnalDir0 = r"F:\Data_230516reorg\fpFIP_FIP-673056-2023-05-09-15-11-28_2023-05-09_15-18-05"
#AnalDir1 = r"F:\Data_230516reorg\fpFIP_FIP-673056-2023-05-09-15-39-38_2023-05-09_15-46-15"
#AnalDir2 = r"F:\Data_230516reorg\fpFIP_FIP-673056-2023-05-09-14-19-37_2023-05-09_14-26-25"
#AnalDir3 = r"F:\Data_230516reorg\fpFIP_FIP-673056-2023-05-09-14-47-14_2023-05-09_14-54-06"
#AnalDir4 = r"F:\Data_230516reorg\fpFIP_FIP-673056-2023-05-09-15-11-28_2023-05-09_15-18-05"

#673057 3A6-NAc
#AnalDir0 = r"F:\Data_230516reorg\fpFIP_FIP-673057-2023-05-12-14-43-43_2023-05-12_14-51-14"
#AnalDir1 = r"F:\Data_230516reorg\fpFIP_FIP-673057-2023-05-12-15-13-08_2023-05-12_15-20-40"
#AnalDir2 = r"F:\Data_230516reorg\fpFIP_FIP-673057-2023-05-12-13-38-03_2023-05-12_13-45-06"
#AnalDir3 = r"F:\Data_230516reorg\fpFIP_FIP-673057-2023-05-12-14-06-39_2023-05-12_14-14-07"
#AnalDir4 = r"F:\Data_230516reorg\fpFIP_FIP-673057-2023-05-12-14-43-43_2023-05-12_14-51-14"

#664630 3A6-NA/BLA Dual
#AnalDir0 = r"F:\Data_230516reorg\fpFIP_FIP-664630-2023-05-09-17-00-29_2023-05-09_17-07-04"
#AnalDir1 = r"F:\Data_230516reorg\fpFIP_FIP-664630-2023-05-09-17-30-14_2023-05-09_17-36-49"
#AnalDir2 = r"F:\Data_230516reorg\fpFIP_FIP-664630-2023-05-09-16-10-44_2023-05-09_16-15-46"
#AnalDir3 = r"F:\Data_230516reorg\fpFIP_FIP-664630-2023-05-09-16-36-42_2023-05-09_16-41-45"
#AnalDir4 = r"F:\Data_230516reorg\fpFIP_FIP-664630-2023-05-09-17-00-29_2023-05-09_17-07-04"

#664631 3A6-NA/BLA Dual
#AnalDir0 = r"F:\Data_230516reorg\fpFIP_FIP-664631-2023-05-09-19-03-32_2023-05-09_19-10-25"
#AnalDir1 = r"F:\Data_230516reorg\fpFIP_FIP-664631-2023-05-09-19-40-02_2023-05-09_19-46-59"
#AnalDir2 = r"F:\Data_230516reorg\fpFIP_FIP-664631-2023-05-09-18-16-06_2023-05-09_18-22-25"
#AnalDir3 = r"F:\Data_230516reorg\fpFIP_FIP-664631-2023-05-09-18-40-17_2023-05-09_18-47-21"
#AnalDir4 = r"F:\Data_230516reorg\fpFIP_FIP-664631-2023-05-09-19-03-32_2023-05-09_19-10-25"

#672639 3A6-NA/BLA Dual 
#AnalDir0 = r"F:\Data_230516reorg\fpFIP_FIP-672639-2023-05-12-18-47-00_2023-05-12_18-54-43"
#AnalDir1 = r"F:\Data_230516reorg\fpFIP_FIP-672639-2023-05-12-19-16-30_2023-05-12_19-23-40"
#AnalDir2 = r"F:\Data_230516reorg\fpFIP_FIP-672639-2023-05-12-18-00-38_2023-05-12_18-05-39"
#AnalDir3 = r"F:\Data_230516reorg\fpFIP_FIP-672639-2023-05-12-18-23-20_2023-05-12_18-30-59"
#AnalDir4 = r"F:\Data_230516reorg\fpFIP_FIP-672639-2023-05-12-18-47-00_2023-05-12_18-54-43"

#6726340 3A6-NA/BLA Dual to fig
#AnalDir0 = r"F:\Data_230516reorg\fpFIP_FIP-672640-2023-05-12-16-52-20_2023-05-12_16-59-56"
#AnalDir1 = r"F:\Data_230516reorg\fpFIP_FIP-672640-2023-05-12-17-28-44_2023-05-12_17-36-24"
#AnalDir2 = r"F:\Data_230516reorg\fpFIP_FIP-672640-2023-05-12-16-03-18_2023-05-12_16-06-34"
#AnalDir3 = r"F:\Data_230516reorg\fpFIP_FIP-672640-2023-05-12-16-28-46_2023-05-12_16-36-08"
#AnalDir4 = r"F:\Data_230516reorg\fpFIP_FIP-672640-2023-05-12-16-52-20_2023-05-12_16-59-56"

#679375 1.3b-NA/BLA to fig (NAc)
#AnalDir0 = r"F:\photometry_FIPopt\230623\FIP_679375_2023-06-23_15-00-01"
#AnalDir1 = r"F:\photometry_FIPopt\230623\FIP_679375_2023-06-23_15-36-30"
#AnalDir2 = r"F:\photometry_FIPopt\230623\FIP_679375_2023-06-23_14-09-00"
#AnalDir3 = r"F:\photometry_FIPopt\230623\FIP_679375_2023-06-23_14-32-13"
#AnalDir4 = r"F:\photometry_FIPopt\230623\FIP_679375_2023-06-23_15-00-01"

#680323 1.3b-NA/BLA
#AnalDir0 = r"F:\photometry_FIPopt\230623\FIP_680323_2023-06-23_16-59-55"
#AnalDir1 = r"F:\photometry_FIPopt\230623\FIP_680323_2023-06-23_17-32-35"
#AnalDir2 = r"F:\photometry_FIPopt\230623\FIP_680323_2023-06-23_16-11-48"
#AnalDir3 = r"F:\photometry_FIPopt\230623\FIP_680323_2023-06-23_16-36-38"
#AnalDir4 = r"F:\photometry_FIPopt\230623\FIP_680323_2023-06-23_16-59-55"

#680324 1.3b-NA/BLA
#AnalDir0 = r"F:\photometry_FIPopt\230630\FIP_680324_2023-06-30_13-44-06"
#AnalDir1 = r"F:\photometry_FIPopt\230630\FIP_680324_2023-06-30_14-17-10"
#AnalDir2 = r"F:\photometry_FIPopt\230630\FIP_680324_2023-06-30_12-57-35"
#AnalDir3 = r"F:\photometry_FIPopt\230630\FIP_680324_2023-06-30_13-20-48"
#AnalDir4 = r"F:\photometry_FIPopt\230630\FIP_680324_2023-06-30_13-44-06"

#682492 1.3b-NA/BLA
#AnalDir0 = r"F:\photometry_FIPopt\230630\FIP_682492_2023-06-30_15-40-30"
#AnalDir1 = r"F:\photometry_FIPopt\230630\FIP_682492_2023-06-30_16-12-30"
#AnalDir2 = r"F:\photometry_FIPopt\230630\FIP_682492_2023-06-30_14-53-33"
#AnalDir3 = r"F:\photometry_FIPopt\230630\FIP_682492_2023-06-30_15-16-35"
#AnalDir4 = r"F:\photometry_FIPopt\230630\FIP_682492_2023-06-30_15-40-30"

SaveDir = "F:\Analysis\FIP_opto\dLight3paper"

SaveFlag = 1
FiberROI = 1 #1:Fiber1, 2:Fiber2

nFrame2cut = 100  #crop initial n frames
sampling_rate = 20 #individual channel (not total)
kernelSize = 1 #median filter
degree = 4 #polyfit
b_percentile = 0.70 #To calculare F0, median of bottom x%

post = 15 #second (length after PSTH) 
#post = 30 #second (length after PSTH) #for visualization

#%%
LabtrackID = re.search(r"\d{6}", os.path.basename(AnalDir0)).group()

AnalDirAll = [AnalDir0, AnalDir1]

if 'AnalDir2' in locals():
    AnalDirAll = [AnalDir0, AnalDir1, AnalDir2, AnalDir3, AnalDir4]

Psth_G_baseAll = []
Psth_C_baseAll = []
DecayTauAll = []
dFFmaxAll = []
Ctrl_dF_FAll = []
G_dF_FAll = []
G_rawAll = []
Ctrl_rawAll = []


#%% main loop
for ii_dir in range(len(AnalDirAll)):

    AnalDir = AnalDirAll[ii_dir]
    
    # reading opto_stim.json for stimulation params
    if bool(glob.glob(AnalDir + os.sep + "*opto_stim.json")) == True:
        stimfile  = glob.glob(AnalDir + os.sep + "*opto_stim.json")[0]
        
        with open(stimfile) as file:
            dict = json.load(file)
        
        base = dict["baseline_duration"]
        trialN = dict["number_pulse_trains"]
        StimPeriod = dict["pulse_train_duration"] 
        ITI = dict["pulse_train_interval"]
        
    else:
        base = 120 #sec, set mannially if you do not have opto_stim.json
        trialN = 40 #
        StimPeriod = 2 #sec
        ITI = 28 #sec
    
    #%% read files
    file1  = glob.glob(AnalDir + os.sep + "FIP_DataIso*")[0]
    file2 = glob.glob(AnalDir + os.sep + "FIP_DataG*")[0]
    file3 = glob.glob(AnalDir + os.sep + "FIP_DataR*")[0]
    
    with open(file1) as f:
        reader = csv.reader(f)
        datatemp = np.array([row for row in reader])
        data1 = datatemp[1:,:].astype(np.float32)
        #del datatemp
        
    with open(file2) as f:
        reader = csv.reader(f)
        datatemp = np.array([row for row in reader])
        data2 = datatemp[1:,:].astype(np.float32)
        #del datatemp
        
    with open(file3) as f:
        reader = csv.reader(f)
        datatemp = np.array([row for row in reader])
        data3 = datatemp[1:,:].astype(np.float32)
        #del datatemp
            
    # in case acquisition halted accidentally
    Length = np.amin([len(data1),len(data2),len(data3)])
    
    data1 = data1[0:Length] 
    data2 = data2[0:Length]
    data3 = data3[0:Length]
    
    PMts= data2[:,0]
    Data_Fiber1iso = data1[:,1]
    Data_Fiber1G = data2[:,1]
    Data_Fiber1R = data3[:,1]
         
    Data_Fiber2iso = data1[:,2]
    Data_Fiber2G = data2[:,2]
    Data_Fiber2R = data3[:,2]
    
    #%% Preprocess
    Ctrl1_dF_F = pf.tc_preprocess(Data_Fiber1iso, nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    G1_dF_F = pf.tc_preprocess(Data_Fiber1G, nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    R1_dF_F = pf.tc_preprocess(Data_Fiber1R, nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    
    Ctrl2_dF_F = pf.tc_preprocess(Data_Fiber2iso, nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    G2_dF_F = pf.tc_preprocess(Data_Fiber2G, nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    R2_dF_F = pf.tc_preprocess(Data_Fiber2R, nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    
    if FiberROI == 1:
        G_raw = Data_Fiber1G
        Ctrl_raw = Data_Fiber1iso
        G_dF_F = G1_dF_F
        Ctrl_dF_F = Ctrl1_dF_F
    elif FiberROI == 2:
        G_raw = Data_Fiber2G
        Ctrl_raw = Data_Fiber2iso
        G_dF_F = G2_dF_F
        Ctrl_dF_F = Ctrl2_dF_F        
    
    '''
    tc_cropped = pf.tc_crop(Data_Fiber1G, nFrame2cut)
    tc_filtered = pf.medfilt(tc_cropped, kernel_size=kernelSize)
    tc_filtered = pf.tc_lowcut(tc_filtered, sampling_rate)
    tc_poly = pf.tc_polyfit(tc_filtered, sampling_rate, degree)
    tc_estim = tc_filtered - tc_poly
    tc_base = pf.tc_slidingbase(tc_filtered, sampling_rate)
    tc_dFoF = pf.tc_dFF(tc_filtered, tc_base, b_percentile)
    tc_dFoF = pf.tc_filling(tc_dFoF, nFrame2cut)
    '''
    #%% 
    time_seconds = np.arange(len(G_dF_F)) /sampling_rate
    
    OptoStim = np.arange(trialN) * (StimPeriod + ITI) + base
    OptoStim = OptoStim * sampling_rate  #sec * Hz
    
    #%%
    if bool(glob.glob(AnalDir + os.sep + "RunningSpeed*")) == True:
        print('loading RunningSpeed')
        file_RS  = glob.glob(AnalDir + os.sep + "RunningSpeed*")[0]
        
        with open(file_RS) as f:
            reader = csv.reader(f)
            datatemp = np.array([row for row in reader])
            data_RS = datatemp[1:,:].astype(np.float32)
            del datatemp
            
        data_RS_time_temp = data_RS[:,0]
        data_RS_time = (data_RS_time_temp - data_RS_time_temp[0])/1000 #ms to s 
        
    
    if bool(glob.glob(AnalDir + os.sep + "PupilTracking*")) == True:
        print('loading PupilTracking')
        file_Pupil = glob.glob(AnalDir + os.sep + "PupilTracking*")[0]
        file_EyeCam = glob.glob(AnalDir + os.sep + "FaceEyeCamera*.csv")[0]
        
        with open(file_Pupil) as f:
            reader = csv.reader(f)
            datatemp = np.array([row for row in reader])
            data_Pupil = datatemp[1:,:].astype(np.float32)
            del datatemp
    
        with open(file_EyeCam) as f:
            reader = csv.reader(f)
            datatemp = np.array([row for row in reader])
            data_EyeCam_time = datatemp[1:,:].astype(np.float32)
            del datatemp
            
        data_EyeCam_time = (data_EyeCam_time - data_EyeCam_time[0])/1000 #ms to s 
    
    #%%
    gs = gridspec.GridSpec(6,8, wspace=1, hspace=1)
    plt.figure(figsize=(20, 8))
    plt.subplot(gs[0:2, 0:8])
    
    plt.plot(time_seconds, Ctrl_dF_F*100, 'blue', label='Iso_Ctrl')
    plt.plot(time_seconds, G_dF_F*100, 'green', label='Green_Signal')
    #lt.plot(time_seconds, R1_dF_F*100, 'magenta', label='R_artifact')
    plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
    plt.xlabel('Time (seconds)')
    plt.ylabel('dF/F (%)')
    plt.title("SubjectID: " + LabtrackID)
    plt.legend()
    plt.xlim([0, time_seconds[-1]])
    plt.grid(True)
    
    for ii in range(trialN):
        plt.axvspan(base + (StimPeriod + ITI)*ii, base + StimPeriod + (StimPeriod + ITI)*ii, color = [1, 0, 1, 0.4])    
    
    if bool(glob.glob(AnalDir + os.sep + "RunningSpeed*")) == True:
        plt.subplot(gs[2, 0:8])
        plt.plot(data_RS_time, data_RS[:,1], color=[0.4, 0.4, 0.4])
        plt.ylabel('A.U.')
        plt.xlim([0, time_seconds[-1]])
        plt.title('Running Wheel Movement')
        
    if bool(glob.glob(AnalDir + os.sep + "PupilTracking*")) == True:
        if len(data_EyeCam_time) == len(data_Pupil):
            plt.subplot(gs[3, 0:8])
            plt.plot(data_EyeCam_time, data_Pupil, color=[0.4, 0.4, 0.4])
            plt.ylabel('pixel')
            plt.xlim([0, time_seconds[-1]])
            plt.title('Pupil Diam.')
            plt.xlabel('second')

    #%% PSTH functions
    def PSTHmaker(TC, Stims, preW, postW):
        
        cnt = 0
        
        for ii in range(len(Stims)):
            if Stims[ii] - preW >= 0 and  Stims[ii] + postW < len(TC):
                
                A = int(Stims[ii]-preW) 
                B = int(Stims[ii]+postW)
                
                if cnt == 0:
                    PSTHout = TC[A:B]
                    cnt = 1
                else:
                    PSTHout = np.vstack([PSTHout,TC[A:B]])
            else:
                PSTHout = np.vstack([PSTHout, np.zeros(preW+postW)])
            
        return PSTHout
    
    
    def PSTHplot(PSTH, MainColor, SubColor, LabelStr):
        plt.plot(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, np.mean(PSTH.T,axis=1),label=LabelStr,color = MainColor)
        #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
        #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
        y11 =  np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
        y22 =  np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
        plt.fill_between(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, y11, y22, facecolor=SubColor, alpha=0.5)
    
    
    #%% PSTH baseline subtraction
    
    def PSTH_baseline(PSTH, preW):
        
        for ii in range(np.shape(Psth_G)[0]):
            
            Trace_this = PSTH[ii, :]
            Trace_this_base = Trace_this[0:preW]
            Trace_this_subtracted = Trace_this - np.mean(Trace_this_base)        
            
            if ii == 0:
                PSTHbase = Trace_this_subtracted
            else:
                PSTHbase = np.vstack([PSTHbase,Trace_this_subtracted])
        
        return PSTHbase
    
    #%%
    Psth_G = PSTHmaker(G_dF_F*100, OptoStim, 100, sampling_rate * post)
    Psth_C = PSTHmaker(Ctrl_dF_F*100, OptoStim, 100, sampling_rate * post)
    Psth_G_base = PSTH_baseline(Psth_G, 100)
    Psth_C_base = PSTH_baseline(Psth_C, 100)    
    
    plt.subplot(gs[4:6, 0:2])
    preW=100
    sampling_rate=20
    
    PSTHplot(Psth_G, "g", "darkgreen", "Green_signal")
    PSTHplot(Psth_C, "b", "darkblue", "Iso_Ctrl")
    ymax = np.max([np.max(np.mean(Psth_G,axis=0))+1,5]) 
    plt.ylim([-5, np.max([ymax+1, 5])])
    plt.xlim([-5,post])
    plt.legend()
    plt.grid(True)
    plt.title("Trial Averaged, Mean+-SEM")
    plt.xlabel('Time (seconds) from StimOnsets')
    plt.ylabel('dF/F (%)')
    plt.axvspan(0, 2, color = [1, 0, 1, 0.4])
    
    #%%
    plt.subplot(gs[4:6, 2:4])
    
    PSTHplot(Psth_G_base, "g", "darkgreen", "Green_signal")
    PSTHplot(Psth_C_base, "b", "darkblue", "Iso_Ctrl")
    ymax = np.max([np.max(np.mean(Psth_G_base,axis=0))+1,5]) 
    plt.ylim([-5, np.max([ymax+5, 5])])
    plt.xlim([-5,post])
    plt.legend()
    plt.grid(True)
    plt.title("LocalBase_subtracted, Mean+-SEM")
    plt.xlabel('Time (seconds) from StimOnsets')
    plt.axvspan(0, 2, color = [1, 0, 1, 0.4])
    
    if SaveFlag == 1:
        plt.savefig(SaveDir + os.sep + LabtrackID + '_' + str(ii_dir) + "_Fiber_" + str(FiberROI) + ".pdf")
    
    #%% ExpFit
    
    # Define the exponential function
    def exponential_func(x, a, b, c):
        return -a * np.exp(-b * x) + c
    
    # Generate some data
    x_data_a = np.linspace(0, 2, 40)
    y_data_a = np.mean(Psth_G[:, 100:140].T,axis=1)
    
    x_data_d = np.linspace(0, 7, 140)
    y_data_d = np.mean(Psth_G[:, 140:280].T,axis=1)
    
    try:
        # Fit the data with the exponential function
        popt_a, pcov_a = curve_fit(exponential_func, x_data_a, y_data_a)
        popt_d, pcov_d = curve_fit(exponential_func, x_data_d, y_data_d)
    
        # Print the fitted parameters
        print('Fitted params: a = %f, b = %f, c = %f' % (popt_a[0], popt_a[1], popt_a[2]))
        print('Fitted params: a = %f, b = %f, c = %f' % (popt_d[0], popt_d[1], popt_d[2]))
    
    
        # Plot the data and the fitted function
        plt.subplot(gs[4:6, 5:6])
        plt.plot(x_data_a, y_data_a)
        plt.plot(x_data_a, exponential_func(x_data_a, *popt_a), 'r-')
        plt.title("Rise_ExpFit  b:" + "{:.2f}".format(popt_a[1]))
        plt.xlabel("s")
        plt.ylabel("dF/F")
    
        plt.subplot(gs[4:6, 6:8])
        plt.plot(x_data_d, y_data_d)
        plt.plot(x_data_d, exponential_func(x_data_d, *popt_d), 'r-')
        plt.title("Decay_ExpFit  b:" + "{:.2f}".format(popt_d[1]) + "  half-time:" + "{:.2f}".format(np.log(2)/popt_d[1]) + "s")
        plt.xlabel("s")
        plt.ylabel("dF/F")
    
        if SaveFlag == 1:
            plt.savefig(SaveDir + os.sep + LabtrackID + '_' + str(ii_dir) + "_Fiber_" + str(FiberROI) + ".pdf")
    
    except RuntimeError:
        print("exp curve fitting failed. Skipped")
        
    #%% For loop data pool

    Psth_G_baseAll.append(Psth_G_base)
    Psth_C_baseAll.append(Psth_C_base)
    DecayTauAll.append(np.log(2)/popt_d[1])
    dFFmaxAll.append(np.max(np.mean(Psth_G_base, axis=0)))
    Ctrl_dF_FAll.append(Ctrl_dF_F)
    G_dF_FAll.append(G_dF_F)
    Ctrl_rawAll.append(Ctrl_raw)
    G_rawAll.append(G_raw)

#%% single animal summary
gs = gridspec.GridSpec(6,8, wspace=1, hspace=1)
plt.figure(figsize=(20, 8))
plt.subplot(gs[0:4, 0:8])

plt.plot(time_seconds, Ctrl_dF_FAll[0]*100, 'blue', label='Iso_Ctrl')
plt.plot(time_seconds, G_dF_FAll[0]*100, 'green', label='Green_Signal')
#lt.plot(time_seconds, R_dF_F*100, 'magenta', label='R_artifact')
plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
plt.xlabel('Time (s)')
plt.ylabel('dF/F (%)')
plt.title("SubjectID: " + LabtrackID)
plt.legend()
plt.xlim([0, time_seconds[-1]])
plt.grid(True)

for ii in range(trialN):
    plt.axvspan(base + (StimPeriod + ITI)*ii, base + StimPeriod + (StimPeriod + ITI)*ii, color = [1, 0, 1, 0.4])


plt.subplot(gs[4:6, 0:2])
PSTHplot(Psth_G_baseAll[0], "g", "darkgreen", "GBR-")
PSTHplot(Psth_C_baseAll[0], "b", "darkblue", "Iso_Ctrl")
ymax = np.max([np.max(np.mean(Psth_G_baseAll[0],axis=0)),5]) 
ymin = np.min([np.min(np.mean(Psth_C_baseAll[0],axis=0)) -10,-2])
plt.legend()
plt.grid(True)
plt.title("Mean+-SEM")
plt.xlabel('Time (s) from StimOnsets')
plt.ylabel("dF/F")
plt.axvspan(0, 2, color = [1, 0, 1, 0.4])

PSTHplot(Psth_G_baseAll[1], "magenta", "magenta", "GBR+")
PSTHplot(Psth_C_baseAll[1], "k", "k", "Iso_Ctrl")
ymax = np.max([np.max(np.mean(Psth_G_baseAll[1],axis=0)),5]) 
ymin = np.min([np.min(np.mean(Psth_C_baseAll[1],axis=0)) -10,-2])
plt.ylim([ymin, ymax*1.2])
plt.xlim([-5,post])
plt.legend()


plt.subplot(gs[4:6, 2:3])
dFFtrials1 = np.max(Psth_G_baseAll[0][:,preW:preW + int(sampling_rate*StimPeriod)],axis=1)
dFFtrials2 = np.max(Psth_G_baseAll[1][:,preW:preW + int(sampling_rate*StimPeriod)],axis=1)
plt.scatter(np.ones([len(dFFtrials1),1]), dFFtrials1, c="green", s=3)
plt.scatter(np.ones([len(dFFtrials2),1])*2, dFFtrials2, c="magenta", s=3)
plt.errorbar(1.2, np.mean(dFFtrials1), yerr=np.std(dFFtrials1), fmt='o', color='green', ecolor='lightgray', elinewidth=3, capsize=2);
plt.errorbar(2.2, np.mean(dFFtrials2), yerr=np.std(dFFtrials2), fmt='o', color='magenta', ecolor='lightgray', elinewidth=3, capsize=2);
plt.xlim([0.5, 2.5])
plt.ylim([0, np.max([dFFtrials1, dFFtrials2])*1.1])
plt.ylabel("dF/F")
plt.title("IndTrials, SD")

if 'AnalDir2' in locals():
    plt.subplot(gs[4:6, 4:6])
    
    PSTHplot(Psth_G_baseAll[4], "g", "darkgreen", "20Hz")
    PSTHplot(Psth_G_baseAll[3], "darkorange", "darkorange", "10Hz")
    PSTHplot(Psth_G_baseAll[2], "olive", "olive", "5Hz")
    ymax = np.max([np.max(np.mean(Psth_G_baseAll[4],axis=0))+10,5]) 
    ymin = np.min([np.min(np.mean(Psth_C_baseAll[4],axis=0)) -10,-2])
    plt.ylim([ymin, ymax])
    plt.xlim([-5,post])
    plt.legend()
    plt.grid(True)
    plt.title("Mean+-SEM")
    plt.xlabel('Time (s) from StimOnsets')
    plt.ylabel("dF/F")
    plt.axvspan(0, 2, color = [1, 0, 1, 0.4])
    
    
    plt.subplot(gs[4:6, 6:7])
    dFFtrials5 = np.max(Psth_G_baseAll[4][:,preW:preW + int(sampling_rate*StimPeriod)],axis=1)
    dFFtrials4 = np.max(Psth_G_baseAll[3][:,preW:preW + int(sampling_rate*StimPeriod)],axis=1)
    dFFtrials3 = np.max(Psth_G_baseAll[2][:,preW:preW + int(sampling_rate*StimPeriod)],axis=1)
    plt.scatter(np.ones([len(dFFtrials3),1]), dFFtrials3, c="olive", s=3)
    plt.scatter(np.ones([len(dFFtrials4),1])*2, dFFtrials4, c="darkorange", s=3)
    plt.scatter(np.ones([len(dFFtrials5),1])*3, dFFtrials5, c="g", s=3)
    
    plt.errorbar(1.2, np.mean(dFFtrials3), yerr=np.std(dFFtrials3), fmt='o', color='olive', ecolor='lightgray', elinewidth=3, capsize=2);
    plt.errorbar(2.2, np.mean(dFFtrials4), yerr=np.std(dFFtrials4), fmt='o', color='darkorange', ecolor='lightgray', elinewidth=3, capsize=2);
    plt.errorbar(3.2, np.mean(dFFtrials5), yerr=np.std(dFFtrials5), fmt='o', color='g', ecolor='lightgray', elinewidth=3, capsize=2);
    plt.xlim([0.5, 3.5])
    plt.ylim([0, np.max([dFFtrials1, dFFtrials2])*1.1])
    plt.ylabel("dF/F")
    plt.title("IndTrials, SD")

plt.subplot(gs[4:6, 7:8])

plt.scatter(1, DecayTauAll[0], c="green", s=50)
plt.scatter(2, DecayTauAll[1], c="Magenta", s=50)

if 'AnalDir2' in locals():
    plt.scatter(3, DecayTauAll[2], c="olive", s=50)
    plt.scatter(4, DecayTauAll[3], c="darkorange", s=50)
    plt.scatter(5, DecayTauAll[4], c="green", s=50)

plt.xlim([0.5, 5.5])
plt.ylim([0, np.max(DecayTauAll)*1.1])
plt.ylabel("half-time(s)")
plt.title("Off-kinetics")
if SaveFlag == 1:
    plt.savefig(SaveDir + os.sep + LabtrackID + "_AnimalSum_Fiber_" + str(FiberROI) + ".pdf")

#To dataframe
df = pd.DataFrame([Psth_G_baseAll, Psth_C_baseAll, DecayTauAll, dFFmaxAll, Ctrl_dF_FAll, G_dF_FAll])
df = df.T
df = df.set_axis(['Psth_G', 'Psth_C', 'DecayTau', 'dFFmax','Ctrl_dF_F','G_dF_FAll'], axis='columns')
if SaveFlag == 1:
    df.to_pickle(SaveDir + os.sep + LabtrackID + "_Fiber_" + str(FiberROI) + '_df.pkl')
     
#%% for fig
#ylim_val1=[-5, 70]
#ylim_val2=[-2, 40]

ylim_val1=[-25, 200]
ylim_val2=[-5, 200]


gs = gridspec.GridSpec(1,12, wspace=1, hspace=1)
plt.figure(figsize=(20, 4))
plt.subplot(gs[0, 0:8])

plt.plot(time_seconds, Ctrl_dF_FAll[0]*100, 'blue', label='Iso_Ctrl')
plt.plot(time_seconds, G_dF_FAll[0]*100, 'green', label='Green_Signal')
#lt.plot(time_seconds, R_dF_F*100, 'magenta', label='R_artifact')
plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
plt.xlabel('Time (s)')
plt.ylabel('dF/F (%)')
plt.title("SubjectID: " + LabtrackID)
plt.legend()
plt.xlim([0, time_seconds[-1]])
plt.ylim(ylim_val1)

for ii in range(trialN):
    plt.axvspan(base + (StimPeriod + ITI)*ii, base + StimPeriod + (StimPeriod + ITI)*ii, color = [1, 0, 1, 0.4])

plt.subplot(gs[0, 8:10])
PSTHplot(Psth_G_baseAll[0], "g", "darkgreen", "GBR-")
PSTHplot(Psth_C_baseAll[0], "b", "darkblue", "Iso_Ctrl")
ymax = np.max([np.max(np.mean(Psth_G_baseAll[0],axis=0)),5]) 
ymin = np.min([np.min(np.mean(Psth_C_baseAll[0],axis=0)) -10,-2])
plt.legend()
plt.title("Mean+-SEM")
plt.xlabel('Time (s) from StimOnsets')
plt.ylabel("dF/F")
plt.axvspan(0, 2, color = [1, 0, 1, 0.4])

PSTHplot(Psth_G_baseAll[1], "magenta", "magenta", "GBR+")
PSTHplot(Psth_C_baseAll[1], "k", "k", "Iso_Ctrl")
plt.plot(np.linspace(-5,post,20), np.zeros(len(np.linspace(-5,post,20))),'--k')
ymax = np.max([np.max(np.mean(Psth_G_baseAll[1],axis=0)),5]) 
ymin = np.min([np.min(np.mean(Psth_C_baseAll[1],axis=0)) -10,-2])
plt.ylim(ylim_val2)
plt.xlim([-5,post])
plt.legend()

plt.subplot(gs[0, 10:12])
PSTHplot(Psth_G_baseAll[4], "g", "darkgreen", "20Hz")
PSTHplot(Psth_G_baseAll[3], "darkorange", "darkorange", "10Hz")
PSTHplot(Psth_G_baseAll[2], "olive", "olive", "5Hz")
plt.plot(np.linspace(-5,post,20), np.zeros(len(np.linspace(-5,post,20))),'--k')
ymax = np.max([np.max(np.mean(Psth_G_baseAll[4],axis=0))+10,5]) 
ymin = np.min([np.min(np.mean(Psth_C_baseAll[4],axis=0)) -10,-2])
plt.ylim(ylim_val2)
plt.xlim([-5,post])
plt.legend()
plt.title("Mean+-SEM")
plt.xlabel('Time (s) from StimOnsets')
plt.ylabel("dF/F")
plt.axvspan(0, 2, color = [1, 0, 1, 0.4])



