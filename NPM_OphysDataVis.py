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
import csv
import glob
from scipy.optimize import curve_fit
import json

import PreprocessingFunctions as pf

AnalDir = r"F:\photometry_FIPopt\230428\664627"
AnalDir = r"F:\photometry_FIPopt\230428\664629"
AnalDir = r"F:\photometry_FIPopt\230428\669479"
#AnalDir = r"F:\photometry_FIPopt\230428\669485"

FiberROI = 1 #1:Fiber1, 2:Fiber2

# params for pre-processing
nFrame2cut = 100  #crop initial n frames
sampling_rate = 20 #individual channel (not total)
kernelSize = 1 #median filter
degree = 4 #polyfit
b_percentile = 0.70 #To calculare F0, median of bottom x%

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
    G_dF_F = G1_dF_F
    Ctrl_dF_F = Ctrl1_dF_F
elif FiberROI == 2:
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
gs = gridspec.GridSpec(6,8)
plt.figure(figsize=(20, 8))
plt.subplot(gs[0:2, 0:8])

plt.plot(time_seconds, Ctrl_dF_F*100, 'blue', label='Iso_Ctrl')
plt.plot(time_seconds, G_dF_F*100, 'green', label='Green_Signal')
#lt.plot(time_seconds, R1_dF_F*100, 'magenta', label='R_artifact')
plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')
plt.title("Whole Trace   SubjectID: " + os.path.basename(AnalDir))
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
Psth_G = PSTHmaker(G_dF_F*100, OptoStim, 100, 300)
Psth_C = PSTHmaker(Ctrl_dF_F*100, OptoStim, 100, 300)
Psth_G_base = PSTH_baseline(Psth_G, 100)
Psth_C_base = PSTH_baseline(Psth_C, 100)    

plt.subplot(gs[4:6, 0:2])
preW=100
sampling_rate=20

PSTHplot(Psth_G, "g", "darkgreen", "Green_signal")
PSTHplot(Psth_C, "b", "darkblue", "Iso_Ctrl")
ymax = np.max([np.max(np.mean(Psth_G,axis=0))+1,5]) 
plt.ylim([-5, np.max([ymax+1, 5])])
plt.xlim([-5,15])
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
ymax = np.max([np.max(np.mean(Psth_G,axis=0))+1,5]) 
plt.ylim([-5, np.max([ymax+5, 5])])
plt.xlim([-5,15])
plt.legend()
plt.grid(True)
plt.title("LocalBase_subtracted, Mean+-SEM")
plt.xlabel('Time (seconds) from StimOnsets')
plt.axvspan(0, 2, color = [1, 0, 1, 0.4])

#plt.savefig(AnalDir + os.sep + "OptoResp_dFoF_" + os.path.basename(AnalDir) +".pdf")

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

    #plt.savefig(AnalDir + os.sep + "OptoResp_dFoF_" + os.path.basename(AnalDir) +".pdf")

except RuntimeError:
    print("exp curve fitting failed. Skipped")








