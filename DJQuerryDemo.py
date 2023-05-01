# -*- coding: utf-8 -*-
"""
DataJoint Basic Querry adopted from Han's notebook

Created on Tue Nov  8 15:02:24 2022

@author: kenta.hagihara
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

import pandas as pd
from pipeline import experiment, lab, foraging_analysis

#%%

session_with_h2o_id = experiment.Session * lab.WaterRestriction.proj('water_restriction_number')

all_foraging_sessions = session_with_h2o_id & (foraging_analysis.SessionTaskProtocol & 'session_task_protocol = 100')
allen_foraging_sessions =  all_foraging_sessions & 'rig LIKE "AIND%"'


sessions_with_kh_mice = session_with_h2o_id & 'water_restriction_number LIKE "KH_%"'

#%%
summary_kh = dj.U('water_restriction_number').aggr(sessions_with_kh_mice, total_sessions='COUNT(*)', first_date='MIN(session_date)', last_date='MAX(session_date)')