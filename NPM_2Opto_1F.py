# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 22:29:19 2022

NPM_2Opto

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

AnalDir = r"F:\Data\fpFIP_650120opto_2023-01-16_20-49-41"
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230228\650102"
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230228\650103"
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230228\650105" 
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230228\650119"
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230228\650120"

AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230301\655422"   
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230301\655424_2"
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230301\655425"
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230301\659080"
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230301\659081"

AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230308\655422"

AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230310\655425_p_15000"

AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230309\659081_p_15000"
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230309\659081_p_10000"
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230309\659081_p_5000"
AnalDir = r"N:\workgroups\discovery\KentaHagihara\DataTransfer_photometry_FIPopt\230310\655425_p_10000"

AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230315\655425_GBR_p_15000"
AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230315\655425_p_15000"
AnalDir = r"F:\photometry_FIPopt\230421\669486"


#AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230417\669472_560nm"
#AnalDir2 = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230417\669472_620nm"


#AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230315\655422_GBR_p_15000"
#AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230315\655422_p_15000"
#AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230315\655424_GBR_p_15000"
#AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230315\655424_p_15000"

#AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230315\655422_o_15000"
#AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230315\655425_q_15000"
#AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230315\655425_o_15000"

#AnalDir = r"N:\workgroups\discovery\KentaHagihara\photometry_FIPtemp_tower1\221221\650120_opto1"
#AnalDir = r"N:\workgroups\discovery\KentaHagihara\photometry_FIPtemp_tower1\221221\650120_opto2"

#AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230315\655425_GBR_p_15000"
#AnalDir = r"S:\KentaHagihara_InternalTransfer\DataTransfer_photometry_opto\230315\655425_p_15000"
AnalDir = r"F:\photometry_FIPopt\230329\655425_GBR"

sampling_rate = 20 #Hz
base = 120 #sec
trialN = 40 #
StimPeriod = 2 #sec
ITI = 28 #sec

flag_compare = 0;


Ctrl1_dF_F = np.load(glob.glob(AnalDir + os.sep + "Ctrl1_dF_F.npy")[0])
G1_dF_F = np.load(glob.glob(AnalDir + os.sep + "G1_dF_F.npy")[0])

if flag_compare == 1:
    Ctrl2_dF_F = np.load(glob.glob(AnalDir2 + os.sep + "Ctrl1_dF_F.npy")[0])
    G2_dF_F = np.load(glob.glob(AnalDir2 + os.sep + "G1_dF_F.npy")[0])


file_TS  = glob.glob(AnalDir + os.sep + "FIP_DataG_*")[0]
with open(file_TS) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data_TS = datatemp[1:,:].astype(np.float32)
    del datatemp
    
data_TS_temp = data_TS[:,0]
time_seconds = (data_TS_temp - data_TS_temp[0])/1000 #ms to s 
    
#time_seconds = np.arange(len(G1_dF_F)) /sampling_rate

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

plt.plot(time_seconds, Ctrl1_dF_F*100, 'blue', label='Iso_Ctrl')
plt.plot(time_seconds, G1_dF_F*100, 'green', label='Green_Signal')
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
Psth_G = PSTHmaker(G1_dF_F*100, OptoStim, 100, 300)
Psth_C = PSTHmaker(Ctrl1_dF_F*100, OptoStim, 100, 300)
Psth_G_base = PSTH_baseline(Psth_G, 100)
Psth_C_base = PSTH_baseline(Psth_C, 100)    

plt.subplot(gs[4:6, 0:2])
preW=100
sampling_rate=20

PSTHplot(Psth_G, "g", "darkgreen", "Green_signal")
PSTHplot(Psth_C, "b", "darkblue", "Iso_Ctrl")
ymax = np.max([np.max(np.mean(Psth_G,axis=0))+1,5]) 
plt.ylim([-1.5, np.max([ymax+1, 5])])
plt.xlim([-5,15])
plt.legend()
plt.grid(True)
plt.title("Trial Averaged, Mean+-SEM")
plt.xlabel('Time (seconds) from StimOnsets')
plt.ylabel('dF/F (%)')
plt.axvspan(0, 2, color = [1, 0, 1, 0.4])

if flag_compare == 1:
    Psth_G2 = PSTHmaker(G2_dF_F*100, OptoStim, 100, 300)
    Psth_G2_base = PSTH_baseline(Psth_G2, 100)
    
    PSTHplot(Psth_G2, "r", "Magenta", "GBR+")
    plt.legend()

#%%
plt.subplot(gs[4:6, 2:4])

PSTHplot(Psth_G_base, "g", "darkgreen", "Green_signal")
PSTHplot(Psth_C_base, "b", "darkblue", "Iso_Ctrl")
ymax = np.max([np.max(np.mean(Psth_G,axis=0))+1,5]) 
plt.ylim([-1.5, np.max([ymax+5, 5])])
plt.xlim([-5,15])
plt.legend()
plt.grid(True)
plt.title("LocalBase_subtracted, Mean+-SEM")
plt.xlabel('Time (seconds) from StimOnsets')
plt.axvspan(0, 2, color = [1, 0, 1, 0.4])

if flag_compare == 1:
    Psth_G2 = PSTHmaker(G2_dF_F*100, OptoStim, 100, 300)
    Psth_G2_base = PSTH_baseline(Psth_G2, 100)    
    PSTHplot(Psth_G2_base, "r", "darkred", "GBR+")
    plt.legend()

plt.savefig(AnalDir + os.sep + "OptoResp_dFoF_" + os.path.basename(AnalDir) +".pdf")

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

    plt.savefig(AnalDir + os.sep + "OptoResp_dFoF_" + os.path.basename(AnalDir) +".pdf")

except RuntimeError:
    print("exp curve fitting failed. Skipped")


#%% dataframe

















