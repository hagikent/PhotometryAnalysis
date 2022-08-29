# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 08:39:59 2022

Run after "NPM_2NIDAQalign"
To align pre-processed/trial-organised photometry data to model-fitting done in DataJoint 

Dependency:UtilFunctions_KH (_get_independent_variableKH; align_phys_to_behav_trials)

@author: Kenta M. Hagihara @SvobodaLab
"""


import json
json_open = open('dj_local_conf.json', 'r') 
config = json.load(json_open)

import datajoint as dj
dj.config['database.host'] = config["database.host"]
dj.config['database.user'] = config ["database.user"]
dj.config['database.password'] = config["database.password"]
dj.conn().connect()

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 10})
from pipeline import lab, get_schema_name, experiment, foraging_model, ephys, foraging_analysis, histology, ccf
from pipeline import ophys
from pipeline.plot import unit_psth
from pipeline.plot.foraging_model_plot import plot_session_model_comparison, plot_session_fitted_choice

from UtilFunctions_KH import _get_independent_variableKH, align_phys_to_behav_trials



AnalDir = r"C:\Users\kenta.hagihara\OneDrive - Allen Institute\Data\FIP\220808\KH_FB32"
sess_key = {'subject_id': 632105, 'session': 16}
df = pd.read_pickle(AnalDir + os.sep + 'DataFrame.pkl')

#%% Model List (ID + details)
foraging_model.Model()
plot_session_model_comparison(sess_key, model_comparison_idx=1, sort='aic')

model_id = 16 #LNP Sugrue/Newsome as a default for now

#%% read LVs
LatVars=_get_independent_variableKH(sess_key, model_id=model_id, var_name=None)

#%%
plt.subplot(6,1,2)
plt.plot(LatVars['left_action_value'],color='blue')
plt.plot(LatVars['right_action_value'],color='orange')
plt.title('L(blue) / R(orange) action values')

plt.subplot(6,1,3)
plt.plot(LatVars['relative_action_value_lr'],color='black')
plt.title('relative_action_value: L - R')

plt.subplot(6,1,4)
plt.plot(LatVars['total_action_value'],color='black')
plt.title('total_action_value')

plt.subplot(6,1,5)
plt.plot(LatVars['rpe'],color='black')
plt.title('RewardPredictionError')

ax = plt.subplot(6,1,1)
plot_session_fitted_choice(sess_key, specified_model_ids=16, ax=ax, remove_ignored=False)


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
plt.figure()
plt.subplot(6,1,2)
plt.plot(LatVars_aligned['left_action_value'],color='blue')
plt.plot(LatVars_aligned['right_action_value'],color='orange')
plt.title('L(blue) / R(orange) action values')

plt.subplot(6,1,3)
plt.plot(LatVars_aligned['relative_action_value_lr'],color='black')
plt.title('relative_action_value: L - R')

plt.subplot(6,1,4)
plt.plot(LatVars_aligned['total_action_value'],color='black')
plt.title('total_action_value')

plt.subplot(6,1,5)
plt.plot(LatVars_aligned['rpe'],color='black')
plt.title('RewardPredictionError')

temp = df_aligned['Resp_l']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[3])

plt.subplot(6,1,6)
plt.plot(Y_this,color='maroon')
plt.title('FP_Late_response')

ax = plt.subplot(6,1,1)
plot_session_fitted_choice(sess_key, specified_model_ids=16, ax=ax, remove_ignored=False)


#%%
temp = df_aligned['Resp_l']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[3])

plt.figure()
plt.scatter(LatVars_aligned['rpe'],Y_this)
plt.xlabel('RPE')
plt.ylabel('Late_response')


#%%
plt.figure()

temp = df_aligned['Resp_l']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[3])

plt.scatter(LatVars_aligned['rpe'],Y_this)
plt.xlabel('RPE')
plt.ylabel('Reward_response')


#%%

plt.figure()

temp = df_aligned['Resp_e']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[3])

plt.subplot(1,3,1)
plt.scatter(LatVars_aligned['total_action_value'],Y_this)
plt.xlabel('total_action_value')
plt.ylabel('Cue_response')


temp = df_aligned['Resp_l']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[3])

plt.subplot(1,3,2)
plt.scatter(LatVars_aligned['total_action_value'],Y_this)
plt.xlabel('total_action_value')
plt.ylabel('Reward_response')

#%
temp = df_aligned['Resp_t']
Y_this=[]
for item in temp:
    Y_this = np.append(Y_this, item[3])

plt.subplot(1,3,3)
plt.scatter(LatVars_aligned['total_action_value'],Y_this)
plt.xlabel('total_action_value')
plt.ylabel('ITI_response')


#DataJoint Querry Exmples
#foraging_analysis.SessionStats & (lab.WaterRestriction & 'water_restriction_number = "KH_FB8"') & 'session=6'
#(foraging_model.FittedSessionModel.TrialLatentVariable & 'model_id = 16') & (lab.WaterRestriction & 'water_restriction_number = "KH_FB8"') & 'session=29'



