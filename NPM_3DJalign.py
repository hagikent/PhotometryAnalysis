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
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 12})
from pipeline import lab, get_schema_name, experiment, foraging_model, ephys, foraging_analysis, histology, ccf
from pipeline import ophys
from pipeline.plot import unit_psth
from pipeline.plot.foraging_model_plot import plot_session_model_comparison, plot_session_fitted_choice

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
AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB46"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB48"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB49"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB52"
#AnalDir = r"C:\Users\kenta.hagihara\Dropbox\Public\Testdata221026_Stefano\KH_FB53"

#sess_key = {'subject_id': 632104, 'session': 33}  #FB31
#sess_key = {'subject_id': 632105, 'session': 39}  #FB32
#sess_key = {'subject_id': 632106, 'session': 21}  #FB33
#sess_key = {'subject_id': 634704, 'session': 29}  #FB42  #DA:1 instead of 3
#sess_key = {'subject_id': 637701, 'session': 12}  #FB43
sess_key = {'subject_id': 639872, 'session': 27}  #FB46
#sess_key = {'subject_id': 639875, 'session': 19}  #FB48 nope
#sess_key = {'subject_id': 639876, 'session': 19}  #FB49
#sess_key = {'subject_id': 641494, 'session': 8}  #FB52
#sess_key = {'subject_id': 641495, 'session': 17}  #FB53

df = pd.read_pickle(AnalDir + os.sep + 'DataFrame.pkl')

#%% Model List (ID + details)
foraging_model.Model()
plot_session_model_comparison(sess_key, model_comparison_idx=1, sort='aic')

model_id = 16 #LNP Sugrue/Newsome as a default for now
#model_id = 18 #Hattori2019 CK

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
    

#%%
traceN=3 #0-3 for usual recordings

plt.figure(figsize=(16, 12))
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

#%% RPE
plt.figure(figsize=(16, 8))

plt.subplot(1,2,1)
temp = df_aligned['Resp_e']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[traceN])

rpe=LatVars_aligned['rpe'].values

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

#%%

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


#%%
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

#%%
df_aligned.to_pickle(AnalDir + os.sep + 'Photometry_aligned.pkl')
LatVars_aligned.to_pickle(AnalDir + os.sep + 'LatVars_aligned.pkl')


