# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 08:39:59 2022

Run after "NPM_2NIDAQalign"
To align pre-processed/trial-organised photometry data to model-fitting done in DataJoint 

Dependency:UtilFunctions_KH (_get_independent_variableKH; align_phys_to_behav_trials)

@author: Kenta M. Hagihara @SvobodaLab
"""

import os
os.chdir (r"C:\Users\kenta.hagihara\Documents\GitHub\map-ephys")

import json
json_open = open('dj_local_conf.json', 'r') 
config = json.load(json_open)

import datajoint as dj
dj.config['database.host'] = config["database.host"]
dj.config['database.user'] = config ["database.user"]
dj.config['database.password'] = config["database.password"]
dj.conn().connect()

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
plt.rcParams.update({'font.size': 12})
from pipeline import lab, get_schema_name, experiment, foraging_model, ephys, foraging_analysis, histology, ccf
from pipeline import ophys
from pipeline.plot import unit_psth
from pipeline.plot.foraging_model_plot import plot_session_model_comparison, plot_session_fitted_choice, _get_specified_model_fitting_results
from pipeline import lab, foraging_model, util
from pipeline.model.util import moving_average

from UtilFunctions_KH import _get_independent_variableKH, align_phys_to_behav_trials

#%%

#AnalDir = r'E:\Data\fpFIP_KH-FB32_2022-08-08_09-47-30'
#AnalDir = r'E:\Data\fpNPM_KH-FB8_2022-05-24_14-53-36'
#sess_key = {'subject_id': 632105, 'session': 16}
#sess_key = {'subject_id': 619199, 'session': 29}

# for 230213
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\fpNPM_KH-FB31_2022-09-01_09-32-00"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\fpFIP_KH-FB32_2022-09-09_10-07-45"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\fpFIP_KH-FB33_2022-08-25_09-46-43"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB42"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB43"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB46"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB48"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB49"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB52"
AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB53"

#sess_key = {'subject_id': 632104, 'session': 33}  #FB31
#sess_key = {'subject_id': 632105, 'session': 39}  #FB32
#sess_key = {'subject_id': 632106, 'session': 21}  #FB33
#sess_key = {'subject_id': 634704, 'session': 29}  #FB42  #DA:1 instead of 3
#sess_key = {'subject_id': 637701, 'session': 12}  #FB43
#sess_key = {'subject_id': 639872, 'session': 27}  #FB46
#sess_key = {'subject_id': 639875, 'session': 19}  #FB48 nope
#sess_key = {'subject_id': 639876, 'session': 19}  #FB49
#sess_key = {'subject_id': 641494, 'session': 8}  #FB52
sess_key = {'subject_id': 641495, 'session': 17}  #FB53

df = pd.read_pickle(AnalDir + os.sep + 'DataFrame.pkl')

traceN=3 #0-3 for usual recordings
#%% Model List (ID + details)
#foraging_model.Model()
#plot_session_model_comparison(sess_key, model_comparison_idx=1, sort='aic')

model_id = 16 #LNP Sugrue/Newsome as a default for now
#model_id = 18 #Bari2019 CK
#model_id = 20 #Hattori2019 CK
#model_id = 12 #Bari

#%% read LVs
LatVars=_get_independent_variableKH(sess_key, model_id=model_id, var_name=None)

#%% Trial Align
ophys_barcode = df['Barcode'].tolist() # This is so far manually imported from the photometry standalone
behav_trialN, behav_barcode = (experiment.TrialNote & sess_key & 'trial_note_type = "bitcode"').fetch('trial', 'trial_note', order_by='trial')

trial_aligned = align_phys_to_behav_trials(ophys_barcode, list(behav_barcode), list(behav_trialN))

mp=trial_aligned['phys_to_behav_mapping']

df_aligned=df.iloc[0:0]
LatVars_aligned=LatVars.iloc[0:0]

for ii in range(len(mp)):
    
    ophys_trial_this = mp[ii][0]-1  #note ophys is pd-row index-based
    behave_trial_this= mp[ii][1]
    
    if any(LatVars['trial'] == behave_trial_this):      #when false, ignored trial (no latent variables)
        df_aligned = df_aligned.append(df.iloc[ophys_trial_this,:])
        LatVars_aligned = LatVars_aligned.append(LatVars[LatVars['trial'] == behave_trial_this])
    

#%% RPE

temp = df_aligned['Resp_e']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

rpe=LatVars_aligned['rpe'].values

#%%
df_aligned.to_pickle(AnalDir + os.sep + 'Photometry_aligned.pkl')
LatVars_aligned.to_pickle(AnalDir + os.sep + 'LatVars_aligned.pkl')


#%% 231128 Additional Analysis
import statsmodels.api as sm
from pipeline import lab, foraging_model, util
from pipeline.model.util import moving_average


# == Fetch data ==
choice_history, reward_history, iti, p_reward, _ = foraging_model.get_session_history(sess_key, remove_ignored=True)

n_trials = np.shape(choice_history)[1]
p_reward_fraction = p_reward[1, :] / (np.sum(p_reward, axis=0))
ignored_trials = np.isnan(choice_history[0])
rewarded_trials = np.any(reward_history, axis=0)
unrewarded_trials = np.logical_not(np.logical_or(rewarded_trials, ignored_trials))


figT=plt.figure('Summary:' + str(sess_key['subject_id']), figsize=(14, 12))
gs = gridspec.GridSpec(16,12)
plt.subplot(gs[0:4, 0:12])


# Rewarded trials
plt.plot(np.nonzero(rewarded_trials)[0], 0.5 + (choice_history[0, rewarded_trials] - 0.5) * 1.4,
        '|', color='black', markersize=20, markeredgewidth=1)

# Unrewarded trials
plt.plot(np.nonzero(unrewarded_trials)[0], 0.5 + (choice_history[0, unrewarded_trials] - 0.5) * 1.4,
        '|', color='gray', markersize=12, markeredgewidth=1)

# Ignored trials
plt.plot(np.nonzero(ignored_trials)[0], [1.1] * sum(ignored_trials),
        'x', color='red', markersize=6, markeredgewidth=0.5)

# Base probability
plt.plot(np.arange(0, n_trials), p_reward_fraction, color=[0.8, 0.8, 0], label='base rew. prob.', lw=1.5)

# Smoothed choice history
smooth_factor = 5
y = moving_average(choice_history, smooth_factor)
x = np.arange(0, len(y)) + int(smooth_factor / 2)
plt.plot(x, y, linewidth=1.5, color='black', label='choice (smooth = %g)' % smooth_factor)

remove_ignored=False
results_to_plot = results = _get_specified_model_fitting_results(sess_key, model_id)

for idx, result in results_to_plot.iterrows():
    trial, right_choice_prob = (foraging_model.FittedSessionModel.TrialLatentVariable
                        & dict(result) & 'water_port="right"').fetch('trial', 'choice_prob')
    plt.plot(np.arange(0, n_trials) if remove_ignored else trial, right_choice_prob, linewidth=2,color=[0.2, 0.2, 1] ,
            label='Model Choice Probability')

#plt.legend(fontsize=10)
plt.legend(fontsize=10, bbox_to_anchor=(1, 1.3))
locs, labels = plt.yticks()  # Get the current locations and labels.
plt.yticks(np.arange(2))  # Set label locations.
plt.ylabel('P(c(t) = r)')
plt.title('SubjectID: ' + str(sess_key['subject_id']) + '  Session: ' + str(sess_key['session']))


#%%
def PSTH_baseline(PSTH, preW):
    
    for ii in range(np.shape(PSTH)[0]):
        
        Trace_this = PSTH[ii]
        Trace_this_base = Trace_this[100-preW:100]
        Trace_this_subtracted = Trace_this - np.mean(Trace_this_base)        
        
        if ii == 0:
            PSTHbase = Trace_this_subtracted
        else:
            PSTHbase = np.vstack([PSTHbase,Trace_this_subtracted])
    
    return PSTHbase

#%% RPE quantile PSTH
cmap_name = 'bwr' 
cmap = plt.get_cmap(cmap_name)

PSTH_all=df_aligned["PSTH3"].values

preW=100
sampling_rate=20
def PSTHplot(PSTH, MainColor, SubColor, LabelStr):
    plt.plot(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, np.mean(PSTH.T,axis=1),label=LabelStr,color = MainColor)
    y11 =  np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
    y22 =  np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
    plt.fill_between(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, y11, y22, facecolor=SubColor, alpha=0.5)


def PSTHplot2(PSTH, MainColor):
    plt.plot(np.arange(len(np.mean(PSTH)))/20 - 5, np.mean(PSTH),c=MainColor)
    y11 =  np.mean(PSTH.T,axis=0) + np.std(PSTH.T,axis=0)/np.sqrt(np.shape(PSTH)[0])
    y22 =  np.mean(PSTH.T,axis=0) - np.std(PSTH.T,axis=0)/np.sqrt(np.shape(PSTH)[0])
    plt.fill_between(np.arange(len(np.mean(PSTH)))/20 - 5, y11, y22, facecolor=MainColor[:3], alpha=0.5)

#edges = np.linspace(np.min(rpe) - 0.001, np.max(rpe) + 0.001, 6 + 1)

plt.subplot(gs[4:8, 6:9])
PSTHplot2(PSTH_all[rpe<-0.66], cmap(1.0))
PSTHplot2(PSTH_all[(rpe>=-0.66) & (rpe<-0.33)], cmap(0.8))
PSTHplot2(PSTH_all[(rpe>=-0.33) & (rpe<0)], cmap(0.6))
PSTHplot2(PSTH_all[(rpe>=0) & (rpe<0.33)], cmap(0.4))
PSTHplot2(PSTH_all[(rpe>=0.33) & (rpe<0.66)], cmap(0.2))
PSTHplot2(PSTH_all[rpe>=0.66], cmap(0.0))
plt.xlabel('time - cue (s)')
plt.ylabel('dF/F')

#based
PSTH_all_base = PSTH_baseline(PSTH_all, 40)
preW=100

plt.subplot(gs[4:8, 9:12])
PSTHplot(PSTH_all_base[rpe<-0.66], cmap(1.0),cmap(1.0),'')
PSTHplot(PSTH_all_base[(rpe>=-0.66) & (rpe<-0.33)], cmap(0.8),cmap(0.8),'')
PSTHplot(PSTH_all_base[(rpe>=-0.33) & (rpe<0)], cmap(0.6),cmap(0.6),'')
PSTHplot(PSTH_all_base[(rpe>=0) & (rpe<0.33)], cmap(0.4),cmap(0.4),'')
PSTHplot(PSTH_all_base[(rpe>=0.33) & (rpe<0.66)], cmap(0.2),cmap(0.2),'')
PSTHplot(PSTH_all_base[rpe>=0.66], cmap(0.0),cmap(0.0),'')
plt.xlabel('time - cue (s)')
plt.ylabel('dF/F')

#%%
temp = df_aligned['Resp_l'] #Reward
RewardV=[]
for item in temp:
    RewardV = np.append(RewardV, item[traceN])

#To exclude NaN
RewardV=RewardV[10:-1]
CurrentR = df_aligned['Reward_ID']
HistoryR_1 = CurrentR.shift(periods=1)[10:-1]
HistoryR_2 = CurrentR.shift(periods=2)[10:-1]
HistoryR_3 = CurrentR.shift(periods=3)[10:-1]
HistoryR_4 = CurrentR.shift(periods=4)[10:-1]
HistoryR_5 = CurrentR.shift(periods=5)[10:-1]
HistoryR_6 = CurrentR.shift(periods=6)[10:-1]
HistoryR_7 = CurrentR.shift(periods=7)[10:-1]
HistoryR_8 = CurrentR.shift(periods=8)[10:-1]
HistoryR_9 = CurrentR.shift(periods=9)[10:-1]
HistoryR_10 = CurrentR.shift(periods=10)[10:-1]
CurrentR = CurrentR[10:-1]

trial_data = np.array([CurrentR.values,
                       HistoryR_1.values,
                       HistoryR_2.values,
                       HistoryR_3.values,
                       HistoryR_4.values,
                       HistoryR_5.values,
                       HistoryR_6.values,
                       HistoryR_7.values,
                       HistoryR_8.values,
                       HistoryR_9.values,
                       HistoryR_10.values])

outcome_data = RewardV

trial_data = trial_data.astype(float)
outcome_data = outcome_data.astype(float)
X = np.column_stack((np.ones(len(outcome_data)), trial_data.T))

model_BGplot = sm.OLS(outcome_data, X)
result_BGplot = model_BGplot.fit()
coefficients = result_BGplot.params
confidence_interval = result_BGplot.conf_int()

plt.subplot(gs[8:12, 0:3])
plt.plot(np.arange(0, -11, -1), result_BGplot.params[1:], color='k', linewidth=2,)
plt.axhline(0, color=[0.4, 0.4, 0.4], linestyle='--', linewidth=1)
plt.xlabel('Trial')
plt.ylabel('Beta')
plt.title('Bayer-Glimcher against dF/F (95% CI)')

y11 =  confidence_interval[1:,0]
y22 =  confidence_interval[1:,1]
plt.fill_between(np.arange(0, -11, -1), y11, y22, facecolor=[0.5, 0.5, 0.5], alpha=0.5)



temp = df_aligned['Resp_l']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

plt.subplot(gs[8:12, 6:9])
plt.scatter(rpe[df_aligned['Reward_ID']==1],Y_this[df_aligned['Reward_ID']==1], color=[0, 0, 1, 0.5],label='Rewarded')
plt.scatter(rpe[df_aligned['Reward_ID']==0],Y_this[df_aligned['Reward_ID']==0], color=[1, 0, 0, 0.5],label='UnRewarded')

#plt.scatter(rpe,Y_this)
plt.plot(np.arange(-1,1,0.05), np.zeros(len(np.arange(-1,1,0.05))),'--k')   
plt.plot(np.zeros(len(np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05))),np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05),'--k')  
plt.legend()
plt.xlabel('RPE')
plt.ylabel('Reward_response')
plt.tight_layout()



#%% Total Action Value
#Reward
temp = df_aligned['Resp_l']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

sv = LatVars_aligned['total_action_value'].values

plt.subplot(gs[12:16, 3:6])
plt.scatter(sv[df_aligned['Reward_ID']==1],Y_this[df_aligned['Reward_ID']==1],color=[0, 0, 1, 0.4],label='Rewarded')
plt.scatter(sv[df_aligned['Reward_ID']==0],Y_this[df_aligned['Reward_ID']==0],color=[1, 0, 0, 0.4],label='UnRewarded')
plt.xlabel('total_action_value')
plt.ylabel('Reward_response')
plt.plot(np.arange(0,1,0.05), np.zeros(len(np.arange(0,1,0.05))),'--k')

#%% Chosen Q
Qc=[]
for ii in range(len(LatVars_aligned)):
    if LatVars_aligned['choice_lr'].values[ii]==0:
        Qc.append(LatVars_aligned['left_action_value'].values[ii])
    else:
        Qc.append(LatVars_aligned['right_action_value'].values[ii])


temp = df_aligned['Resp_l']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])
    
plt.subplot(gs[8:12, 3:6])
plt.scatter(Qc,Y_this, color=[0, 0, 0, 0.2])

#plt.scatter(rpe,Y_this)
plt.plot(np.arange(0,1,0.05), np.zeros(len(np.arange(0,1,0.05))),'--k')   
#plt.plot(np.zeros(len(np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05))),np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05),'--k')  
plt.xlabel('Qc')
plt.ylabel('Reward_response')

#%% Regression with Q and R
temp = df_aligned['Resp_l'] #Reward
RewardV=[]
for item in temp:
    RewardV = np.append(RewardV, item[traceN])

#To exclude NaN
CurrentR = df_aligned['Reward_ID']

trial_data = np.array([CurrentR.values, Qc])[:,:-1]
outcome_data = RewardV[:-1]

trial_data = trial_data.astype(float)
outcome_data = outcome_data.astype(float)
X = np.column_stack((np.ones(len(outcome_data)), trial_data.T))

model_QcR = sm.OLS(outcome_data, X)
result_QcR= model_QcR.fit()
coefficients = result_QcR.params
confidence_interval = result_QcR.conf_int()

plt.subplot(gs[12:16, 0:3])
plt.errorbar(coefficients[1], coefficients[2], yerr = (confidence_interval[2,1]-confidence_interval[2,0])/2, xerr = (confidence_interval[1,1]-confidence_interval[1,0])/2, capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
plt.xlim([-0.1, 0.1])
plt.ylim([-0.1, 0.1])
plt.axhline(0, color=[0.4, 0.4, 0.4], linestyle='--', linewidth=1)
plt.axvline(0, color=[0.4, 0.4, 0.4], linestyle='--', linewidth=1)
plt.xlabel('Reward')
plt.ylabel('Qc')
plt.title('Beta for dF/F')

#%% pStay, deltaP
numBins = 7

Choice=LatVars_aligned['choice_lr'].values
Pl=LatVars_aligned['left_choice_prob'].values
Pr=np.ones(len(LatVars_aligned)) - LatVars_aligned['left_choice_prob'].values

#pStay2=np.zeros(len(Pr))
#for ii in range(len(Pl)-1):
#    pStay2[ii] = Pl[ii]*Pl[ii+1] + Pr[ii]*Pr[ii+1]
# to use mouse's actual current choice 

Stay=np.zeros(len(Choice)) #Note if mouse stays the same choice in the "next(t+1)" trial
for ii in range(len(Choice)-1):
    if Choice[ii]==Choice[ii+1]:
        Stay[ii] = 1;

pStay=np.array(Pr[1:])
pStay=np.append(pStay, np.nan)
pStay[Choice==0] = 1 - pStay[Choice==0]

pChoicePred = Pr;
pChoicePred[Choice==0] = 1 - pChoicePred[Choice==0];

edges = np.linspace(min(rpe) - 0.001, max(rpe) + 0.001, numBins + 1)

meanChange = np.mean(Stay - pChoicePred);

deltaP_Mean = np.zeros(numBins)
deltaP_std = np.zeros(numBins)
meanPe = np.zeros(numBins)

for j in range(numBins):
    currInd = np.where((rpe >= edges[j]) & (rpe < edges[j + 1]))[0]
    currInd = np.setdiff1d(currInd,len(Stay)-1)
    
    pRcurr = pChoicePred[currInd]
    #pRnext = np.nanmean(Stay[np.intersect1d(currInd + 1, np.arange(len(Choice)))])
    pRnext = Stay[currInd + 1]  
    #pRnext = Stay[currInd]  
    
    meanPe[j] = np.nanmean(rpe[currInd])
    deltaP_Mean[j] = np.nanmean([pRnext - pRcurr]) - meanChange
    #deltaP_Mean[j] = pRnext - pRcurr
    deltaP_std[j] = np.nanstd([pRnext - pRcurr])/np.sqrt(len(pRnext))
    
plt.subplot(gs[4:8, 3:6])
plt.plot(meanPe, deltaP_Mean, color='k', )
y11 =  deltaP_Mean + deltaP_std
y22 =  deltaP_Mean - deltaP_std
plt.fill_between(meanPe, y11, y22, facecolor='k', alpha=0.2)
plt.axhline(0, color=[0.4, 0.4, 0.4], linestyle='--', linewidth=1)
plt.xlim([-1, 1])
plt.ylabel('Delta P')
plt.xlabel('RPE')

#%% Additional Plots

plt.subplot(gs[4:8, 0:3])
plt.scatter(rpe[:-1], pStay[:-1], color=[0, 0, 0, 0.1],label='Rewarded')
plt.axvline(0, color=[0.4, 0.4, 0.4], linestyle='--', linewidth=1)
plt.ylim([0,1])
plt.ylabel('pStay')
plt.xlabel('RPE')

plt.subplot(gs[12:16, 6:9])
plt.scatter(Y_this[:-1], pStay[:-1], color=[0, 0, 0, 0.1],label='Rewarded')
plt.axvline(0, color=[0.4, 0.4, 0.4], linestyle='--', linewidth=1)
plt.ylim([0,1])
plt.ylabel('pStay')
plt.xlabel('DA dF/F')

plt.subplot(gs[8:12, 9:12])
colors = np.arange(len(rpe))
plt.scatter(rpe,Y_this, c=colors, cmap='turbo')
plt.xlim([-1,1])
plt.xlabel('RPE')
plt.ylabel('Reward_response')
plt.colorbar(label="Trial")
plt.axhline(0, color=[0.4, 0.4, 0.4], linestyle='--', linewidth=1)
plt.axvline(0, color=[0.4, 0.4, 0.4], linestyle='--', linewidth=1)

plt.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92, wspace=10,hspace=15)

#%%
plt.figure(figsize=(10, 8))

plt.subplot(2,2,1)
PSTHplot2(PSTH_all[rpe<-0.66], cmap(1.0))
PSTHplot2(PSTH_all[(rpe>=-0.66) & (rpe<-0.33)], cmap(0.8))
PSTHplot2(PSTH_all[(rpe>=-0.33) & (rpe<0)], cmap(0.6))
PSTHplot2(PSTH_all[(rpe>=0) & (rpe<0.33)], cmap(0.4))
PSTHplot2(PSTH_all[(rpe>=0.33) & (rpe<0.66)], cmap(0.2))
PSTHplot2(PSTH_all[rpe>=0.66], cmap(0.0))
plt.xlabel('time - cue (s)')
plt.ylabel('dF/F')

#based
PSTH_all_base = PSTH_baseline(PSTH_all, 40)
preW=100

plt.subplot(2,2,2)
PSTHplot(PSTH_all_base[rpe<-0.66], cmap(1.0),cmap(1.0),'')
PSTHplot(PSTH_all_base[(rpe>=-0.66) & (rpe<-0.33)], cmap(0.8),cmap(0.8),'')
PSTHplot(PSTH_all_base[(rpe>=-0.33) & (rpe<0)], cmap(0.6),cmap(0.6),'')
PSTHplot(PSTH_all_base[(rpe>=0) & (rpe<0.33)], cmap(0.4),cmap(0.4),'')
PSTHplot(PSTH_all_base[(rpe>=0.33) & (rpe<0.66)], cmap(0.2),cmap(0.2),'')
PSTHplot(PSTH_all_base[rpe>=0.66], cmap(0.0),cmap(0.0),'')
plt.xlabel('time - cue (s)')
plt.ylabel('dF/F')


##
temp = df_aligned['Resp_l']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

plt.subplot(2,2,3)
plt.scatter(rpe[df_aligned['Reward_ID']==1],Y_this[df_aligned['Reward_ID']==1], color=[0, 0, 1, 0.5],label='Rewarded')
plt.scatter(rpe[df_aligned['Reward_ID']==0],Y_this[df_aligned['Reward_ID']==0], color=[1, 0, 0, 0.5],label='UnRewarded')

#plt.scatter(rpe,Y_this)
plt.plot(np.arange(-1,1,0.05), np.zeros(len(np.arange(-1,1,0.05))),'--k')   
plt.plot(np.zeros(len(np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05))),np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05),'--k')  
plt.legend()
plt.xlabel('RPE')
plt.ylabel('Reward_response')
plt.tight_layout()


temp = df_aligned['Resp_l'] - df_aligned['Resp_base']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

plt.subplot(2,2,4)
plt.scatter(rpe[df_aligned['Reward_ID']==1],Y_this[df_aligned['Reward_ID']==1], color=[0, 0, 1, 0.5],label='Rewarded')
plt.scatter(rpe[df_aligned['Reward_ID']==0],Y_this[df_aligned['Reward_ID']==0], color=[1, 0, 0, 0.5],label='UnRewarded')

#plt.scatter(rpe,Y_this)
plt.plot(np.arange(-1,1,0.05), np.zeros(len(np.arange(-1,1,0.05))),'--k')   
plt.plot(np.zeros(len(np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05))),np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05),'--k')  
plt.legend()
plt.xlabel('RPE')
plt.ylabel('Reward_response')
plt.tight_layout()










#%% old plots dump
'''
plt.figure(figsize=(12, 9))
plt.subplot(6,1,2)
plt.plot(LatVars_aligned['left_action_value'],color='blue',label='QL')
plt.plot(LatVars_aligned['right_action_value'],color='orange', label='QR')
#plt.title('L(blue) / R(orange) action values')
plt.xticks([0],[""])
plt.legend()

plt.subplot(6,1,3)
plt.plot(LatVars_aligned['relative_action_value_lr'],color='black')
plt.title('relative_action_value: L - R')
plt.xticks([0],[""])

plt.subplot(6,1,4)
plt.plot(LatVars_aligned['total_action_value'],color='black')
plt.title('total_action_value')
plt.xticks([0],[""])

plt.subplot(6,1,5)
plt.plot(LatVars_aligned['rpe'],color='black')
plt.title('RewardPredictionError')
plt.xticks([0],[""])

temp = df_aligned['Resp_l']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

plt.subplot(6,1,6)
plt.plot(Y_this,color='maroon')
plt.title('Reward_responses')

ax = plt.subplot(6,1,1)
plot_session_fitted_choice(sess_key, specified_model_ids=16, ax=ax, remove_ignored=False)
plt.xticks([0],[""])
'''

'''
plt.scatter(np.roll(rpe[df_aligned['Reward_ID']==1],0),Y_this[df_aligned['Reward_ID']==1], color=[0, 0, 1, 0.5],label='Rewarded')
plt.scatter(np.roll(rpe[df_aligned['Reward_ID']==0],0),Y_this[df_aligned['Reward_ID']==0], color=[1, 0, 0, 0.5],label='UnRewarded')
#plt.scatter(rpe,Y_this)


plt.plot(np.arange(-1,1,0.05), np.zeros(len(np.arange(-1,1,0.05))),'--k')   
plt.plot(np.zeros(len(np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05))),np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05),'--k')  
plt.legend()
plt.xlabel('RPE')
plt.ylabel('Cue_response')


plt.subplot(1,2,2)
temp = df_aligned['Resp_l']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

plt.scatter(rpe[df_aligned['Reward_ID']==1],Y_this[df_aligned['Reward_ID']==1], color=[0, 0, 1, 0.5],label='Rewarded')
plt.scatter(rpe[df_aligned['Reward_ID']==0],Y_this[df_aligned['Reward_ID']==0], color=[1, 0, 0, 0.5],label='UnRewarded')
#plt.scatter(rpe,Y_this)
plt.plot(np.arange(-1,1,0.05), np.zeros(len(np.arange(-1,1,0.05))),'--k')   
plt.plot(np.zeros(len(np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05))),np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05),'--k')  
plt.legend()
plt.xlabel('RPE')
plt.ylabel('Reward_response')
'''
#%%
'''
plt.figure(figsize=(16, 8))

plt.subplot(1,2,1)
temp = df_aligned['Resp_e']- df_aligned['Resp_base']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

rpe=LatVars_aligned['rpe'].values

plt.scatter(rpe[df_aligned['Reward_ID']==1],Y_this[df_aligned['Reward_ID']==1], color=[0, 0, 1, 0.5],label='Rewarded')
plt.scatter(rpe[df_aligned['Reward_ID']==0],Y_this[df_aligned['Reward_ID']==0], color=[1, 0, 0, 0.5],label='UnRewarded')
plt.plot(np.arange(-1,1,0.05), np.zeros(len(np.arange(-1,1,0.05))),'--k')   
plt.plot(np.zeros(len(np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05))),np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05),'--k')  
plt.legend()
plt.xlabel('RPE')
plt.ylabel('Cue_response, based-subtracted')


plt.subplot(1,2,2)
temp = df_aligned['Resp_l'] - df_aligned['Resp_base']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

plt.scatter(rpe[df_aligned['Reward_ID']==1],Y_this[df_aligned['Reward_ID']==1], color=[0, 0, 1, 0.5],label='Rewarded')
plt.scatter(rpe[df_aligned['Reward_ID']==0],Y_this[df_aligned['Reward_ID']==0], color=[1, 0, 0, 0.5],label='UnRewarded')
plt.plot(np.arange(-1,1,0.05), np.zeros(len(np.arange(-1,1,0.05))),'--k')   
plt.plot(np.zeros(len(np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05))),np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05),'--k')  
plt.legend()
plt.xlabel('RPE')
plt.ylabel('Reward_response, based-subtracted')
'''

#%%
'''
#Cue
plt.figure(figsize=(20, 6))

temp = df_aligned['Resp_e']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

sv = LatVars_aligned['total_action_value'].values

plt.subplot(1,3,1)
plt.scatter(sv[df_aligned['Reward_ID']==1],Y_this[df_aligned['Reward_ID']==1],color=[0, 0, 1, 0.4],label='Rewarded')
plt.scatter(sv[df_aligned['Reward_ID']==0],Y_this[df_aligned['Reward_ID']==0],color=[1, 0, 0, 0.4],label='UnRewarded')
plt.xlabel('total_action_value')
plt.ylabel('Cue_response')
plt.plot(np.arange(0,1,0.05), np.zeros(len(np.arange(0,1,0.05))),'--k')  

#Reward
temp = df_aligned['Resp_l']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

sv = LatVars_aligned['total_action_value'].values

plt.subplot(1,3,2)
plt.scatter(sv[df_aligned['Reward_ID']==1],Y_this[df_aligned['Reward_ID']==1],color=[0, 0, 1, 0.4],label='Rewarded')
plt.scatter(sv[df_aligned['Reward_ID']==0],Y_this[df_aligned['Reward_ID']==0],color=[1, 0, 0, 0.4],label='UnRewarded')
plt.xlabel('total_action_value')
plt.ylabel('Reward_response')
plt.plot(np.arange(0,1,0.05), np.zeros(len(np.arange(0,1,0.05))),'--k')

#ITI
temp = df_aligned['Resp_t']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

sv = LatVars_aligned['total_action_value'].values

plt.subplot(1,3,3)
plt.scatter(sv[df_aligned['Reward_ID']==1],Y_this[df_aligned['Reward_ID']==1],color=[0, 0, 1, 0.4],label='Rewarded')
plt.scatter(sv[df_aligned['Reward_ID']==0],Y_this[df_aligned['Reward_ID']==0],color=[1, 0, 0, 0.4],label='UnRewarded')
plt.xlabel('total_action_value')
plt.ylabel('ITI_response')
plt.plot(np.arange(0,1,0.05), np.zeros(len(np.arange(0,1,0.05))),'--k')  

#DataJoint Querry Exmples
#foraging_analysis.SessionStats & (lab.WaterRestriction & 'water_restriction_number = "KH_FB8"') & 'session=6'
#(foraging_model.FittedSessionModel.TrialLatentVariable & 'model_id = 16') & (lab.WaterRestriction & 'water_restriction_number = "KH_FB8"') & 'session=29'
'''



'''
temp = df_aligned['Resp_l'] - df_aligned['Resp_base']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

plt.subplot(gs[8:12, 6:9])
plt.scatter(rpe[df_aligned['Reward_ID']==1],Y_this[df_aligned['Reward_ID']==1], color=[0, 0, 1, 0.5],label='Rewarded')
plt.scatter(rpe[df_aligned['Reward_ID']==0],Y_this[df_aligned['Reward_ID']==0], color=[1, 0, 0, 0.5],label='UnRewarded')

#plt.scatter(rpe,Y_this)
plt.plot(np.arange(-1,1,0.05), np.zeros(len(np.arange(-1,1,0.05))),'--k')   
plt.plot(np.zeros(len(np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05))),np.arange(min(Y_this)*1.2, max(Y_this)*1.2, 0.05),'--k')  
plt.legend()
plt.xlabel('RPE')
plt.ylabel('Reward_response_BaseSubtracted')
plt.tight_layout()
'''




