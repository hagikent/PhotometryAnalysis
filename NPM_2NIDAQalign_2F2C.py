# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:29:29 2022

Run after "NPM_1preprocessing"
To align photometry data and TTLs generated with BPOD, logged with NI+Bonsai 

BPOD TTLs (designed by HH)
'barcode': start with 0.020,then 0.010 * 1-20
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


#NMs
#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220409/KH_FB7"
#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220409/KH_FB8"
#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220409/KH_FB9"

#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220410/KH_FB7"
#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220410/KH_FB8"
#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220410/KH_FB9"

#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220411/KH_FB7"
#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220411/KH_FB8"
#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220411/KH_FB9"
#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220420/KH_FB10_1"
#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220421/KH_FB10"
#AnalDir = "/Users/kenta/Library/CloudStorage/OneDrive-AllenInstitute/Data/220422/KH_FB10"

#Win
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220323\KH_FB1"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220421\KH_FB10"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220502\KH_FB8"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220504\KH_FB8"

AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220506\KH_FB15"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220506\KH_FB8"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220505\KH_FB8"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220509\KH_FB16_halted"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220509\KH_FB15"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220512\KH_FB16"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220516\KH_FB13"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220516\KH_FB14"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220517\KH_FB8"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220518\KH_FB14"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220518\KH_FB8"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220523\KH_FB14"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220524\KH_FB8"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220524\KH_FB14"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220603\SS_FB1_1"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220603\SS_FB1_2"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220603\KH_FB19"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220616\KH_FB8"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220614\KH_FB8"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220615\KH_FB13"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220616\KH_FB14"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220609\KH_FB17"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220610\KH_FB17"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220617\KH_FB20"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220601\KH_FB14"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220527\KH_FB14"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220613\KH_FB22"
#AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220621\KH_FB17"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220622\KH_FB14"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220622\KH_FB17"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220623\KH_FB26"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220623\KH_FB27"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220624\KH_FB26"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220624\KH_FB14"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220625\KH_FB26"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220625\KH_FB27"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220626\KH_FB26"
AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\220626\KH_FB27"

FlagNoRawLick = 0

#GCaMP_dF_F = np.load(glob.glob(AnalDir + os.sep + "GCaMP_dF_F.npy")[0])
GCaMP_dF_F = np.load(glob.glob(AnalDir + os.sep + "R2_dF_F.npy")[0])
R_dF_F = np.load(glob.glob(AnalDir + os.sep + "G2_dF_F.npy")[0])

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
    
elif (len(TTLsignal)/1000) / len(TTLts) == 2:  #this shouldn't happen, though...
    TTLsignal2 = TTLsignal[1::2]
    TTLsignal = TTLsignal[0::2]
    plt.figure()
    plt.plot(TTLsignal)
    plt.plot(TTLsignal2)
    print("Num Analog Channel: 2")
        
elif (len(TTLsignal)/1000) / len(TTLts) >= 3:
    TTLsignal1 = TTLsignal[0::3]
    plt.figure()
    plt.plot(TTLsignal1,label='Events')
    
    if FlagNoRawLick == 0: 
        TTLsignal2 = TTLsignal[1::3]
        TTLsignal3 = TTLsignal[2::3]
        plt.plot(TTLsignal2,label='LickL')
        plt.plot(TTLsignal3,label='LickR')    
    
    plt.legend()
    print("Num Analog Channel: 3")
else:
    print("Something is wrong with TimeStamps or Analog Recording...")
    
#%% analoginputs binalize
TTLsignal = TTLsignal1
TTLsignal[TTLsignal < 3] = 0
TTLsignal[TTLsignal >= 3] = 1
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
    if FlagNoRawLick == 1:
        TTLsignal2[:] = 0
        
    TTLsignal2[TTLsignal2 < 0.5] = 0
    TTLsignal2[TTLsignal2 >= 0.5] = 1
    TTLsignal2_shift = np.roll(TTLsignal2, 1)
    diff2 = TTLsignal2 - TTLsignal2_shift
    
    TTL2_p = []
    for ii in range(len(TTLsignal2)):
        if diff2[ii] == 1:
            TTL2_p = np.append(TTL2_p, ii) 

            
if 'TTLsignal3' in locals():
    if FlagNoRawLick == 1:
        TTLsignal3[:] = 0
        
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
UnRewardedL = []
RewardedR = []
UnRewardedR = []

for ii in range(len(TTL_l_align)-1):
    if TTL_l_align[ii] == 2 and TTL_l_align[ii+1] == 30:  #30:reward, #40: ITI start
        RewardedL = np.append(RewardedL,ii)
    if TTL_l_align[ii] == 2 and TTL_l_align[ii+1] == 40:
         UnRewardedL = np.append(UnRewardedL,ii)   
    if TTL_l_align[ii] == 3 and TTL_l_align[ii+1] == 30:
        RewardedR = np.append(RewardedR,ii)
    if TTL_l_align[ii] == 3 and TTL_l_align[ii+1] == 40:
        UnRewardedR = np.append(UnRewardedR,ii)

UnRewarded = np.union1d(UnRewardedL, UnRewardedR)
Rewarded = np.union1d(RewardedL, RewardedR)

Ignored = []

for ii in range(len(TTL_l_align)-1):
    if TTL_l_align[ii] == 1 and TTL_l_align[ii+1] == 40:  #1:GoCue, #40: ITI start
        Ignored = np.append(Ignored,ii)

#%% Barcode Decode (220626 updated)
BarcodeP = TTL_p[TTL_l == 20]
BarcodeBin = np.zeros((len(BarcodeP),20))

for ii in range(len(BarcodeP)):
    for jj in range(20):
        BarcodeBin[ii,jj] = TTLsignal1[int(BarcodeP[ii])+30+20*jj+5]

#%%
##To match behavior... To Check
bitHighPositionThis=[]

for ii in range(len(BarcodeP)):
    bitHighPositionThis[ii] =  np.where(BarcodeBin[ii]==1)
        
        
        
        
        
        
        
        
        
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
plt.savefig(AnalDir + os.sep + "ReactionTime.png")

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

plt.figure(figsize=(20, 6))
plt.plot(time_seconds, GCaMP_dF_F*100, 'g')
plt.plot(time_seconds, R_dF_F*100, 'magenta')        ############3R
plt.plot(time_seconds, np.zeros(len(time_seconds)),'--k')
plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')
plt.title('dF/F')
plt.grid(True)

plt.scatter(TTL_p_align[TTL_l_align == 1]/20,np.ones(len(TTL_p_align[TTL_l_align == 1]))*30,label='go cue')
plt.scatter(TTL_p_align[TTL_l_align == 2]/20,np.ones(len(TTL_p_align[TTL_l_align == 2]))*28,label='Choice L')
plt.scatter(TTL_p_align[TTL_l_align == 3]/20,np.ones(len(TTL_p_align[TTL_l_align == 3]))*26,label='Choice R')
plt.scatter(TTL_p_align[TTL_l_align == 30]/20,np.ones(len(TTL_p_align[TTL_l_align == 30]))*24,label='Reward')



if 'TTL3_p' and 'TTL2_p' in locals():
    plt.scatter(TTL2_p_align/20,np.ones(len(TTL2_p_align))*20,label='Lick L (raw)')
    plt.scatter(TTL3_p_align/20,np.ones(len(TTL3_p_align))*18,label='Lick R (raw)')

plt.legend()
plt.savefig(AnalDir + os.sep + "CaBehavOverview.png")
    
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

PSTH_CueRewarded = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[Rewarded.astype(int)-1], 100, 200)
PSTH_CueRewardedL = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[RewardedL.astype(int)-1], 100, 200)
PSTH_CueRewardedR = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[RewardedR.astype(int)-1], 100, 200)

PSTH_CueUnRewarded = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[UnRewarded.astype(int)-1], 100, 200)
PSTH_CueUnRewardedL = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[UnRewardedL.astype(int)-1], 100, 200)
PSTH_CueUnRewardedR = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[UnRewardedR.astype(int)-1], 100, 200)

PSTH_Ignored = PSTHmaker(GCaMP_dF_F*100, TTL_p_align[Ignored.astype(int)], 100, 200) #Cue aligned
#%% 
#######################################
#Plotting

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
plt.ylabel('dF/F (%)')


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
plt.ylabel('dF/F (%)')

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
plt.ylabel('dF/F (%)')
plt.savefig(AnalDir + os.sep + "PSTH_Choice.png")

#%% Color map
plt.figure()
plt.imshow(PSTH_Rewarded,vmin=-1,vmax=ymax)
plt.xticks(ticks=[0, 50, 100,150,200,250,300], labels=['-5', '-2.5','0','2.5','5','7.5','10'])
plt.xlabel('sec from choice')
plt.ylabel('trials')
plt.colorbar(shrink=0.5)
plt.axvline(x=100,color='white',linestyle='--')
plt.title("ChoiceTriggered_Rewarded")
plt.savefig(AnalDir + os.sep + "ColorMap_ChoiceRew.png")

plt.figure()
plt.imshow(PSTH_UnRewarded,vmin=-1,vmax=ymax)
plt.xticks(ticks=[0, 50, 100,150,200,250,300], labels=['-5', '-2.5','0','2.5','5','7.5','10'])
plt.xlabel('sec from choice')
plt.ylabel('trials')
plt.colorbar(shrink=0.5)
plt.axvline(x=100,color='white',linestyle='--')
plt.title("ChoiceTriggered_UnRewarded")
plt.savefig(AnalDir + os.sep + "ColorMap_ChoiceUnRew.png")

#%%  Cue Triggered Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
PSTHplot(PSTH_CueRewarded, "b", "darkblue", "RewardedTrials")
PSTHplot(PSTH_CueUnRewarded, "r", "darkred", "UnRewardedTrials")
ymax = np.max([np.max(np.mean(PSTH_CueRewarded,axis=0))+1,5]) 
plt.ylim([-1,ymax])
plt.xlim([-5,10])
plt.legend()
plt.grid(True)
plt.title("CueTriggered")
plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')


#%%
plt.subplot(1,3,2)
PSTHplot(PSTH_CueRewardedL, "b", "darkblue", "RewardedTrials")
PSTHplot(PSTH_CueUnRewardedL, "r", "darkred", "UnRewardedTrials")
ymax = np.max([np.max(np.mean(PSTH_CueRewarded,axis=0))+1,5]) 
plt.ylim([-1,ymax])
plt.xlim([-5,10])
plt.legend()
plt.grid(True)
plt.title("L-CueTriggered")
plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')

#%%
plt.subplot(1,3,3)
PSTHplot(PSTH_CueRewardedR, "b", "darkblue", "RewardedTrials")
PSTHplot(PSTH_CueUnRewardedR, "r", "darkred", "UnRewardedTrials")
ymax = np.max([np.max(np.mean(PSTH_CueRewarded,axis=0))+1,5]) 
plt.ylim([-1,ymax])
plt.xlim([-5,10])
plt.legend()
plt.grid(True)
plt.title("R-CueTriggered")
plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')
plt.savefig(AnalDir + os.sep + "PSTH_Cue.png")

#%% Color map
plt.figure()
plt.imshow(PSTH_CueRewarded,vmin=-0.5,vmax=ymax)
plt.xticks(ticks=[0, 50, 100,150,200,250,300], labels=['-5', '-2.5','0','2.5','5','7.5','10'])
plt.xlabel('sec from Cue')
plt.ylabel('trials')
plt.colorbar(shrink=0.5)
plt.axvline(x=100,color='white',linestyle='--')
plt.title("CueTriggered_Rewarded")
plt.savefig(AnalDir + os.sep + "ColorMap_CueRew.png")


plt.figure()
plt.imshow(PSTH_CueUnRewarded,vmin=-0.5,vmax=ymax)
plt.xticks(ticks=[0, 50, 100,150,200,250,300], labels=['-5', '-2.5','0','2.5','5','7.5','10'])
plt.xlabel('sec from Cue')
plt.ylabel('trials')
plt.colorbar(shrink=0.5)
plt.axvline(x=100,color='white',linestyle='--')
plt.title("CueTriggered_UnRewarded")
plt.savefig(AnalDir + os.sep + "ColorMap_CueUnRew.png")

#%%
plt.figure()
PSTHplot(PSTH_Ignored, "k", "gray", "IgnoredTrials")
ymax = np.max([np.max(np.mean(PSTH_CueRewarded,axis=0))+1,5]) 
plt.ylim([-1,ymax])
plt.xlim([-5,10])
plt.legend()
plt.grid(True)
plt.title("CueTriggered")
plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')

plt.figure()
plt.imshow(PSTH_Ignored,vmin=-0.5,vmax=ymax)
plt.xticks(ticks=[0, 50, 100,150,200,250,300], labels=['-5', '-2.5','0','2.5','5','7.5','10'])
plt.xlabel('sec from Cue')
plt.ylabel('trials')
plt.colorbar(shrink=0.5)
plt.axvline(x=100,color='white',linestyle='--')
plt.title("CueTriggered_`Ignored")
#plt.savefig(AnalDir + os.sep + "ColorMap_CueRew.png")


    