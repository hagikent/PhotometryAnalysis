# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:29:29 2022

Run after "NPM_1preprocessing"
To align photometry data and TTLs generated with BPOD, logged with NI+Bonsai 

BPOD TTLs (designed by HH)
'barcode': 0.010
'go_cue': 0.001,      # Should be the shortest, otherwise will miss some really fast licks
'choice_L': 0.002,    # Relatively time-sensitive, should be shorter
'choice_R': 0.003,    # Relatively time-sensitive, should be shorter
'choice_M': 0.004,    # Relatively time-sensitive, should be shorter
'reward': 0.03,       # Not very time-sensitive
'iti_start': 0.04     # Not very time-sensitive

maxlength = 41 (@1000Hz logging)

@author: Kenta M. Hagihara @SvobodaLab

"""

# clear all
from IPython import get_ipython
get_ipython().magic("reset -sf")

#%%
import os
import numpy as np
import pylab as plt
import csv
import glob

#Mac
#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/xxxx"

#Win
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\xxxx"


#GCaMP_dF_F = np.load(glob.glob(AnalDir + os.sep + "GCaMP_dF_F.npy")[0])
GCaMP_dF_F = np.load(glob.glob(AnalDir + os.sep + "R2_dF_F.npy")[0])

TTLsignal = np.fromfile(glob.glob(AnalDir + os.sep + "TTL_20*")[0])
file_TS = glob.glob(AnalDir + os.sep + "TimeStamp*")[0]
file_TTLTS = glob.glob(AnalDir + os.sep + "TTL_TS*")[0]

#%%
with open(file_TS) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    PMts = datatemp[0:,:].astype(np.float32)

with open(file_TTLTS) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    TTLts = datatemp[0:,:].astype(np.float32)

print('number of excitation LEDs: '+ str(np.round(len(PMts)/len(GCaMP_dF_F))))


PMts2 = PMts[0:len(PMts):int(np.round(len(PMts)/len(GCaMP_dF_F))),:] #length of this must be the same as that of GCaMP_dF_F


#%%Sorting NIDAQ-AI channels
if (len(TTLsignal)/1000) / len(TTLts) == 1:
    plt.figure()
    plt.plot(TTLsignal)
    print("Num Analog Channel: 1")
    
elif (len(TTLsignal)/1000) / len(TTLts) == 2:
    TTLsignal2 = TTLsignal[1::2]
    TTLsignal = TTLsignal[0::2]
    plt.figure()
    plt.plot(TTLsignal)
    plt.plot(TTLsignal2)
    print("Num Analog Channel: 2")
        
elif (len(TTLsignal)/1000) / len(TTLts) == 3:
    TTLsignal2 = TTLsignal[1::3]
    TTLsignal3 = TTLsignal[2::3]
    TTLsignal = TTLsignal[0::3]
    plt.figure()
    plt.plot(TTLsignal,label='Events')
    plt.plot(TTLsignal2,label='LickL')
    plt.plot(TTLsignal3,label='LickR')
    plt.legend()
    print("Num Analog Channel: 3")
else:
    print("Something is wrong with TimeStamps or Analog Recording...")
    
#%% analoginputs binalize
TTLsignal[TTLsignal < 1] = 0
TTLsignal[TTLsignal >= 1] = 1
TTLsignal_shift = np.roll(TTLsignal, 1)
diff = TTLsignal - TTLsignal_shift

# Sorting
TTL_p = []
TTL_l = []

for ii in range(len(TTLsignal)):
    if diff[ii] == 1:
        for jj in range(120): #Max length:40
            if ii+jj > len(TTLsignal)-1:
                break
            
            if diff[ii+jj] == -1:
                TTL_p = np.append(TTL_p, ii) 
                TTL_l = np.append(TTL_l, jj)
                break

## binalize raw lick signals             
if 'TTLsignal2' in locals():
    TTLsignal2[TTLsignal2 < 0.5] = 0
    TTLsignal2[TTLsignal2 >= 0.5] = 1
    TTLsignal2_shift = np.roll(TTLsignal2, 1)
    diff2 = TTLsignal2 - TTLsignal2_shift
    
    TTL2_p = []
    for ii in range(len(TTLsignal2)):
        if diff2[ii] == 1:
            TTL2_p = np.append(TTL2_p, ii) 
            
if 'TTLsignal3' in locals():    
    TTLsignal3[TTLsignal3 < 0.5] = 0
    TTLsignal3[TTLsignal3 >= 0.5] = 1
    TTLsignal3_shift = np.roll(TTLsignal3, 1)
    diff3 = TTLsignal3 - TTLsignal3_shift
    
    TTL3_p = []
    for ii in range(len(TTLsignal3)):
        if diff3[ii] == 1:
            TTL3_p = np.append(TTL3_p, ii)
    
#%% Alignment between PMT and TTL
TTL_p_align = []

for ii in range(len(TTL_p)):
    ind_tmp = int(np.ceil(TTL_p[ii]/1000)-2)  #consider NIDAQ buffer 1s (1000samples@1kHz)
    dec_tmp = TTL_p[ii]/1000 + 1 - np.ceil(TTL_p[ii]/1000)
    if ind_tmp >= len(TTLts):
        break
    ms_target = TTLts[ind_tmp]
    idx = int(np.argmin(np.abs(np.array(PMts2[:,0]) - ms_target - dec_tmp*1000)))
    TTL_p_align = np.append(TTL_p_align, idx)
    
TTL_l_align = TTL_l[0:len(TTL_p_align)] 


if 'TTL2_p' in locals():
    TTL2_p_align = []
    for ii in range(len(TTL2_p)):
        ind_tmp = int(np.ceil(TTL2_p[ii]/1000)-2)  #consider NIDAQ buffer 1s (1000samples@1kHz)
        dec_tmp = TTL2_p[ii]/1000 + 1 - np.ceil(TTL2_p[ii]/1000)
        if ind_tmp >= len(TTLts):
            break
        ms_target = TTLts[ind_tmp]
        idx = int(np.argmin(np.abs(np.array(PMts2[:,0]) - ms_target - dec_tmp*1000)))
        TTL2_p_align = np.append(TTL2_p_align, idx)
    
if 'TTL3_p' in locals():
    TTL3_p_align = []
    for ii in range(len(TTL3_p)):
        ind_tmp = int(np.ceil(TTL3_p[ii]/1000)-2)  #consider NIDAQ buffer 1s (1000samples@1kHz)
        dec_tmp = TTL3_p[ii]/1000 + 1 - np.ceil(TTL3_p[ii]/1000)
        if ind_tmp >= len(TTLts):
            break
        ms_target = TTLts[ind_tmp]
        idx = int(np.argmin(np.abs(np.array(PMts2[:,0]) - ms_target - dec_tmp*1000)))
        TTL3_p_align = np.append(TTL3_p_align, idx)

#%% Rewarded Unrewarded L/R trials

RewardedL = []
UnRewardedL =[]
RewardedR = []
UnRewardedR = []

for ii in range(len(TTL_l_align)-1):
    if TTL_l_align[ii] == 2 and TTL_l_align[ii+1] == 30:
        RewardedL = np.append(RewardedL,ii)
    if TTL_l_align[ii] == 2 and TTL_l_align[ii+1] == 40:
         UnRewardedL = np.append(UnRewardedL,ii)   
    if TTL_l_align[ii] == 3 and TTL_l_align[ii+1] == 30:
        RewardedR = np.append(RewardedR,ii)
    if TTL_l_align[ii] == 3 and TTL_l_align[ii+1] == 40:
        UnRewardedR = np.append(UnRewardedR,ii)

UnRewarded = np.union1d(UnRewardedL, UnRewardedR)
Rewarded = np.union1d(RewardedL, RewardedR)

#%% Subselect only the 1st bit 
BarcodeP = TTL_p_align[TTL_l_align == 10]
BarcodeP_clean = BarcodeP[0]

for ii in range(1,len(BarcodeP)):
    if BarcodeP[ii] - BarcodeP[ii-1] > 100:
        BarcodeP_clean = np.append(BarcodeP_clean, BarcodeP[ii])
        
#%% ReactionTime
ReTiRaw = []
for ii in range(len(TTL_l)-1):
    if TTL_l[ii] == 1 and (TTL_l[ii+1] == 3 or TTL_l[ii+1] == 2):
        ReTiRaw = np.append(ReTiRaw, TTL_p[ii+1] - TTL_p[ii])

plt.figure()
plt.xlabel("msec")
plt.ylabel("#Trial")
plt.hist(ReTiRaw,50,range=[0, 1000])
plt.title("ReactionTime")


#%%
"""
ReTi = []
for ii in range(len(TTL_l_align)-1):
    if TTL_l_align[ii] == 1 and (TTL_l_align[ii+1] == 3 or TTL_l_align[ii+1] == 2):
        ReTi = np.append(ReTi, PMts2[int(TTL_p_align[ii+1])] - PMts2[int(TTL_p_align[ii])])

plt.figure()
plt.hist(ReTi)
"""
#%% Ca, Behavior Overview
time_seconds = np.arange(len(GCaMP_dF_F)) /20

plt.figure()
plt.plot(time_seconds, GCaMP_dF_F*100, 'g')
plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
plt.xlabel('Time (seconds)')
plt.ylabel('GCaMP dF/F (%)')
plt.title('GCaMP dF/F')
plt.grid(True)

plt.scatter(TTL_p_align[TTL_l_align == 1]/20,np.ones(len(TTL_p_align[TTL_l_align == 1]))*30,label='go cue')
plt.scatter(TTL_p_align[TTL_l_align == 2]/20,np.ones(len(TTL_p_align[TTL_l_align == 2]))*28,label='Choice L')
plt.scatter(TTL_p_align[TTL_l_align == 3]/20,np.ones(len(TTL_p_align[TTL_l_align == 3]))*26,label='Choice R')
plt.scatter(TTL_p_align[TTL_l_align == 30]/20,np.ones(len(TTL_p_align[TTL_l_align == 30]))*24,label='Reward')



if 'TTL3_p' and 'TTL2_p' in locals():
    plt.scatter(TTL2_p_align/20,np.ones(len(TTL2_p_align))*20,label='Lick L (raw)')
    plt.scatter(TTL3_p_align/20,np.ones(len(TTL3_p_align))*18,label='Lick R (raw)')

plt.legend()
    
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
    
    return PSTHout


def PSTHplot(PSTH, MainColor, SubColor, LabelStr):
    plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1),label=LabelStr,color = MainColor)
    plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
    plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")



#%%
PSTH_Rewarded = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[Rewarded.astype(int)], 100, 200)
PSTH_RewardedL = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[RewardedL.astype(int)], 100, 200)
PSTH_RewardedR = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[RewardedR.astype(int)], 100, 200)

PSTH_UnRewarded = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[UnRewarded.astype(int)], 100, 200)
PSTH_UnRewardedL = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[UnRewardedL.astype(int)], 100, 200)
PSTH_UnRewardedR = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[UnRewardedR.astype(int)], 100, 200)

PSTH_gocue = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[TTL_l_align == 1], 100, 200)
#%%
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
PSTHplot(PSTH_Rewarded, "b", "darkblue", "RewardedTrials")
PSTHplot(PSTH_UnRewarded, "r", "darkred", "UnRewardedTrials")
ymax = np.max([np.max(np.mean(PSTH_Rewarded,axis=0))+1,5]) 
plt.ylim([-1,ymax])
plt.xlim([-5,10])
plt.legend()
plt.grid(True)
plt.title("ChoiceTriggered")
plt.xlabel('Time (seconds)')
plt.ylabel('GCaMP dF/F (%)')


#%%
plt.subplot(1,3,2)
PSTHplot(PSTH_RewardedL, "b", "darkblue", "RewardedTrials")
PSTHplot(PSTH_UnRewardedL, "r", "darkred", "UnRewardedTrials")
ymax = np.max([np.max(np.mean(PSTH_Rewarded,axis=0))+1,5]) 
plt.ylim([-1,ymax])
plt.xlim([-5,10])
plt.legend()
plt.grid(True)
plt.title("L-ChoiceTriggered")
plt.xlabel('Time (seconds)')
plt.ylabel('GCaMP dF/F (%)')

#%%
plt.subplot(1,3,3)
PSTHplot(PSTH_RewardedR, "b", "darkblue", "RewardedTrials")
PSTHplot(PSTH_UnRewardedR, "r", "darkred", "UnRewardedTrials")
ymax = np.max([np.max(np.mean(PSTH_Rewarded,axis=0))+1,5]) 
plt.ylim([-1,ymax])
plt.xlim([-5,10])
plt.legend()
plt.grid(True)
plt.title("R-ChoiceTriggered")
plt.xlabel('Time (seconds)')
plt.ylabel('GCaMP dF/F (%)')

#%%
plt.figure()
plt.imshow(PSTH_Rewarded,vmin=0,vmax=ymax)
plt.xticks(ticks=[0, 50, 100,150,200,250,300], labels=['-5', '-2.5','0','2.5','5','7.5','10'])
plt.xlabel('sec from choice')
plt.ylabel('trials')
plt.colorbar(shrink=0.5)

plt.figure()
plt.imshow(PSTH_UnRewarded,vmin=0,vmax=ymax)
plt.xticks(ticks=[0, 50, 100,150,200,250,300], labels=['-5', '-2.5','0','2.5','5','7.5','10'])
plt.xlabel('sec from choice')
plt.ylabel('trials')
plt.colorbar(shrink=0.5)


    
    
    