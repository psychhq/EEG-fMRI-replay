# -*- coding: utf-8 -*-
"""
# Created on Wed Jun 15 23:03:12 2022
# SCRIPT: CREATE EVENT.TSV FILES FROM THE BEHAVIORAL DATA FOR BIDS
# PROJECT: FMRIREPLAY
# WRITTEN BY QI HUANG
"""
import os
import glob
import warnings
from os.path import join as opj
import pandas as pd
import numpy as np
from collections import Counter as count
from pandas.core.common import SettingWithCopyWarning
# import libraries for bids interaction:
from bids.layout import BIDSLayout
# filter warning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# define paths depending on the operating system (OS) platform:
project = 'fMRIreplay'
# initialize empty paths:
path_root = None
sub_list = None
# create the path
project_name = 'fmrireplay-behavior'
path_root = opj(os.getcwd().split(project)[0] ,'fMRIreplay_hq')
path_bids = opj(path_root ,'fmrireplay-bids','BIDS')
path_fmriprep = opj(path_bids ,'derivatives','fmrireplay-fmriprep')
path_behavior = opj(path_bids, 'sourcedata', project_name)

def cvfold(trials,run):
    if trials >= 0 and trials <= 35 and run == 1:
        return 1
    elif trials > 35 and trials <= 71 and run == 1:
        return 2
    elif trials >= 0 and trials <= 35 and run == 2:
        return 3
    elif trials > 35 and trials <= 71 and run == 2:
        return 4
    elif trials >= 0 and trials <= 35 and run == 3:
        return 5
    elif trials > 35 and trials <= 71  and run == 3:
        return 6
    elif trials >= 0 and trials <= 35  and run == 4:
        return 7
    elif trials > 35 and trials <= 71  and run == 4:
        return 8
    
def cvfold_rep(trials,run):
    if trials >= 0 and trials < 7 and run == 1:  #7
        return 1
    elif trials >= 7 and trials < 15 and run == 1: #8
        return 2
    elif trials >= 15 and trials < 22 and run == 1:
        return 3
    elif trials >= 22 and trials < 30 and run == 1:
        return 4
    elif trials >= 0 and trials < 7 and run == 2:
        return 5
    elif trials >= 7 and trials < 15 and run == 2:
        return 6
    elif trials >= 15 and trials < 22 and run == 2:
        return 7
    elif trials >= 22 and trials < 30 and run == 2:
        return 8
    
# grab the list of subjects from the bids data set:
layout = BIDSLayout(path_bids)
# get all subject ids:
subject_list = sorted(layout.get_subjects())
# sub_list=['04']
# create a template to add the "sub-" prefix to the ids
sub_template = ['sub-'] * len(subject_list)

# add the prefix to all ids:
sub_list = ["%s%s" % t for t in zip(sub_template, subject_list)]
# if user defined to run specific subject
# sub_list = sub_list[int(sys.argv[1]):int(sys.argv[2])]


# In[]:
'''
======================================
VISUAL FUNCTIONAL LOCALIZER
======================================
'''
for subject in subject_list:
    subject = subject_list[0]
    sub = ["sub-%s" %subject]
    path_behavior_vfl = opj(path_behavior,'VFL','*_%s_*.csv' %subject) 
    path_behavior_learning = opj(path_behavior,'learning','*_%s_*.csv' %subject) 
    path_behavior_re = opj(path_behavior,'replay','*_%s_*.csv' %subject) 
    # create the path to the data file
    path_vfl_event = sorted(glob.glob(path_behavior_vfl), key=lambda f: os.path.basename(f))
    path_learning_event = sorted(glob.glob(path_behavior_learning), key=lambda f: os.path.basename(f))
    path_re_event = sorted(glob.glob(path_behavior_re), key=lambda f: os.path.basename(f))
    
    # read the visual functional localizer trial data
    df_vfl_event = [pd.read_csv(f, sep=',') for f in path_vfl_event]
    #select useful columns
    df_vfl_event = [df_vfl_event[f][['cons','stimMarker','textMarker','trials.thisRepN','trials.thisTrialN','text_6.started','image_2.started','key_resp.corr',
                     'subject_id']] for f in range(len(path_vfl_event))]
    # delete unrelated rows
    df_vfl_event=[df_vfl_event[f].dropna(axis=0, how='any').reset_index(drop=True) for f in range(len(path_vfl_event))]
    if subject == 'sub-08':    
        df_vfl_event[0].drop(df_vfl_event[0].tail(2).index,inplace=True)
    elif subject == 'sub-09':
        df_vfl_event[1], df_vfl_event[0] = df_vfl_event[0], df_vfl_event[1] 
    
    # # judge the number of trials
    # for f in range(len(path_vfl_event)):
    #     if len(df_vfl_event[f]) != 104 :
    #         print ('the number of trials in run %s is not right!' % (f+1))
    # rename the data file
    if subject == 'sub-08':    
        rename_vfl = {'trials.thisTrialN' : 'trials',
                      'text_6.started' : 'start',
                      'image_2.started' : 'onset',
                      'key_resp.corr' : 'accuracy',
                      'trials.thisRepN' : 'run'}
        df_vfl_event = [df_vfl_event[f].rename(columns=rename_vfl) for f in range(len(path_vfl_event))]
    elif subject == 'sub-09':
        rename_vfl = {'trials.thisTrialN' : 'trials',
                      'text_6.started' : 'start',
                      'image_2.started' : 'onset',
                      'key_resp.corr' : 'accuracy',
                      'trials.thisRepN' : 'run',
                      'subject_id' : 'participant'}
        df_vfl_event = [df_vfl_event[f].rename(columns=rename_vfl) for f in range(len(path_vfl_event))]
        for f in range(len(path_vfl_event)):
            df_vfl_event[f]['participant'] = 9
    df_vfl_event[0]['session'] = 1
    df_vfl_event[0]['run'] = df_vfl_event[0]['run'] + 1
    df_vfl_event[1]['session'] = 2
    df_vfl_event[1]['run'] = df_vfl_event[1]['run'] + 2
    
    for f in range(len(path_vfl_event)):
        if f == 0:
            df_vfl_event[f]['onset'] = df_vfl_event[f]['onset'].sub(
                float(df_vfl_event[f][(df_vfl_event[f]['run']==1) 
                                      & (df_vfl_event[f]['trials']==0)]['start']))
        elif f == 1:
            onset = [df_vfl_event[f][(df_vfl_event[f]['run']==i)]['onset'].sub(
                float(df_vfl_event[f][(df_vfl_event[f]['run']==i) 
                                      & (df_vfl_event[f]['trials']==0)]['start'])) for i in np.arange(2,5)]
            onset = pd.concat([onset[0],onset[1],onset[2]],axis=0)
            df_vfl_event[f]['onset'] = onset

    accuracy_vfl = [(count(df_vfl_event[f]['accuracy'])[1])/(len(df_vfl_event[f]['accuracy'] == 1)) for f in range(len(path_vfl_event))]

    for f in range(len(path_vfl_event)):
        df_vfl_event[f]['duration']=1
        df_vfl_event[f]['task'] = 'VFL'
        df_vfl_event[f]['fold'] = df_vfl_event[f].apply(lambda x: cvfold(x['trials'],x['run']),axis=1)
        
    classfication = df_vfl_event[1]['run'].unique()
   
    df_vfl_event_temp = [df_vfl_event[1][df_vfl_event[1]['run']==i] for i in np.arange(2,5)]
    df_vfl_event = df_vfl_event + df_vfl_event_temp
    del df_vfl_event[1]
    
    for f in range(len(df_vfl_event)):
        file_name = '%s_task-vfl_run-%02.0f_events.tsv' %(subject,df_vfl_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_vfl_event[f].to_csv(write_path, sep='\t')



# In[]:
'''
======================================
CUE REACTIVATION
======================================
'''

for subject in sub_list:
    path_behavior_vfl = opj(path_behavior,subject,'VFL','*csv') 
    path_behavior_learning = opj(path_behavior,subject,'learning','*csv') 
    path_behavior_re = opj(path_behavior,subject,'reactivation and replay','*csv') 
    # create the path to the data file
    path_vfl_event = sorted(glob.glob(path_behavior_vfl), key=lambda f: os.path.basename(f))
    path_learning_event = sorted(glob.glob(path_behavior_learning), key=lambda f: os.path.basename(f))
    path_re_event = sorted(glob.glob(path_behavior_re), key=lambda f: os.path.basename(f))    

    # read the single item decoding trial data
    df_rea_event = [pd.read_csv(f, sep=',') for f in path_re_event]
    #select useful columns
    if subject == 'sub-09':
        df_rea_event = [df_rea_event[f][['run.thisRepN','trials_reactivation.thisN','text.started','text_cue.text',
                                                 'reactivation_cue.Triggers','text_cue_reactivation.started',
                                                 'key_QA_animate.corr','key_resp_rating.keys',
                                                 'subject_id','session']] for f in range(len(path_re_event))]
    
        # delete unrelated rows
        df_rea_event=[df_rea_event[0].dropna(axis=0, how='any',subset=['run.thisRepN']).reset_index(drop=True) for f in range(len(path_re_event))]
    
        # rename the data file
        rename_rea = {'run.thisRepN' : 'run',
                      'trials_reactivation.thisN' : 'trials',
                      'text.started' : 'start',
                      'text_cue.text' : 'cue',
                      'reactivation_cue.Triggers' : 'Marker',
                      'text_cue_reactivation.started' : 'onset',
                      'key_QA_animate.corr' : 'accuracy',
                      'key_resp_rating.keys' : 'vividness',
                      'subject_id' : 'participant'}
        df_rea_event = df_rea_event[0].rename(columns=rename_rea) 
        accuracy_rea = (count(df_rea_event['accuracy'])[1])/(len(df_rea_event['accuracy'] == 1))
    elif subject == 'sub-08':
        df_rea_event = [df_rea_event[f][['run.thisRepN','trials_reactivation.thisN','text.started','text_cue.text',
                                                 'reactivation_cue.Triggers','text_cue_reactivation.started',
                                                 'key_QA_animate.corr','key_resp_rating.keys',
                                                 'participant','session']] for f in range(len(path_re_event))]
    
        # delete unrelated rows
        df_rea_event=[df_rea_event[0].dropna(axis=0, how='any',subset=['run.thisRepN']).reset_index(drop=True) for f in range(len(path_re_event))]
    
        # rename the data file
        rename_rea = {'run.thisRepN' : 'run',
                      'trials_reactivation.thisN' : 'trials',
                      'text.started' : 'start',
                      'text_cue.text' : 'cue',
                      'reactivation_cue.Triggers' : 'Marker',
                      'text_cue_reactivation.started' : 'onset',
                      'key_QA_animate.corr' : 'accuracy',
                      'key_resp_rating.keys' : 'vividness'
                      }
        df_rea_event = df_rea_event[0].rename(columns=rename_rea) 
        accuracy_rea = (count(df_rea_event['accuracy'])[1])/(len(df_rea_event['accuracy'] == 1))

    df_rea_event['duration'] = 4
    df_rea_event['run'] = df_rea_event['run'] + 1
    df_rea_event['task'] = 'REA'
    df_rea_event['fold'] = divmod(df_rea_event.index.values,15)[0]+1
    
    onset = [df_rea_event[(df_rea_event['run']==i)]['onset'].sub(
        float(df_rea_event[(df_rea_event['run']==i) 
                              & (df_rea_event['trials']==0)]['start'])) for i in np.arange(1,4)]
    onset = pd.concat([onset[0],onset[1],onset[2]],axis=0)
    df_rea_event['onset'] = onset
    del df_rea_event['start']
    
    classfication = df_rea_event['run'].unique()
   
    df_rea_event_list = []
    df_rea_event_list = [df_rea_event[df_rea_event['run']==i] for i in np.arange(1,4)]
    
    for f in range(len(df_rea_event_list)):
        file_name = '%s_task-rea_run-%02.0f_events.tsv' %(subject,df_rea_event_list[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_rea_event_list[f].to_csv(write_path, sep='\t')
        


# In[]:
'''
======================================
CUE REPLAY
======================================
'''

for subject in sub_list:
    path_behavior_vfl = opj(path_behavior,subject,'VFL','*csv') 
    path_behavior_learning = opj(path_behavior,subject,'learning','*csv') 
    path_behavior_re = opj(path_behavior,subject,'reactivation and replay','*csv') 
    # create the path to the data file
    path_vfl_event = sorted(glob.glob(path_behavior_vfl), key=lambda f: os.path.basename(f))
    path_learning_event = sorted(glob.glob(path_behavior_learning), key=lambda f: os.path.basename(f))
    path_re_event = sorted(glob.glob(path_behavior_re), key=lambda f: os.path.basename(f))  
    

    # read the arbitrary sequence decoding with cue trial data
    df_rep_event = [pd.read_csv(f, sep=',') for f in path_re_event]
    #select useful columns
    if subject == 'sub-09':
        df_rep_event = [df_rep_event[f][['runs_replay.thisRepN','trials_replay.thisN','text.started','text_cue_replay.text',
                                                 'replay_cue.Trigger','text_cue_replay.started','key_resp_replay_judge.corr',
                                                 'subject_id','session']] for f in range(len(path_re_event))]
    
        # delete unrelated rows
        df_rep_event=[df_rep_event[0].dropna(axis=0, how='any',subset=['text_cue_replay.text']).reset_index(drop=True) for f in range(len(path_re_event))]
    
        # rename the data file
        rename_rep = {'runs_replay.thisRepN' : 'run',
                      'trials_replay.thisN' : 'trials',
                      'text.started' : 'start',
                      'text_cue_replay.text' : 'cue',
                      'replay_cue.Trigger' : 'Marker',
                      'text_cue_replay.started' : 'onset',
                      'key_resp_replay_judge.corr' : 'accuracy',
                      'subject_id' : 'participant'}
        df_rep_event = df_rep_event[0].rename(columns=rename_rep) 
        accuracy_rep = (count(df_rep_event['accuracy'])[1])/(len(df_rep_event['accuracy'] == 1))
    elif subject == 'sub-08':
        df_rep_event = [df_rep_event[f][['runs_replay.thisRepN','trials_replay.thisN','text.started','text_cue_replay.text',
                                                 'replay_cue.Trigger','text_cue_replay.started','key_resp_replay_judge.corr',
                                                 'participant','session']] for f in range(len(path_re_event))]
    
        # delete unrelated rows
        df_rep_event=[df_rep_event[0].dropna(axis=0, how='any',subset=['text_cue_replay.text']).reset_index(drop=True) for f in range(len(path_re_event))]
    
        # rename the data file
        rename_rep = {'runs_replay.thisRepN' : 'run',
                      'trials_replay.thisN' : 'trials',
                      'text.started' : 'start',
                      'text_cue_replay.text' : 'cue',
                      'replay_cue.Trigger' : 'Marker',
                      'text_cue_replay.started' : 'onset',
                      'key_resp_replay_judge.corr' : 'accuracy'}
        df_rep_event = df_rep_event[0].rename(columns=rename_rep) 
        accuracy_rep = (count(df_rep_event['accuracy'])[1])/(len(df_rep_event['accuracy'] == 1))

    df_rep_event['duration'] = 10
    df_rep_event['run'] = df_rep_event['run'] + 1
    df_rep_event['task'] = 'REP'
    # df_rep_event['fold'] = divmod(df_rep_event.index.values,15)[0]+1
    
    onset = [df_rep_event[(df_rep_event['run']==i)]['onset'].sub(
        float(df_rep_event[(df_rep_event['run']==i) 
                              & (df_rep_event['trials']==0)]['start'])) for i in np.arange(1,3)]
    onset = pd.concat([onset[0],onset[1]],axis=0)
    df_rep_event['onset'] = onset
    del df_rep_event['start']
    df_rep_event['fold'] = df_rep_event.apply(lambda x: cvfold_rep(x['trials'],x['run']),axis=1)

    classfication = df_rep_event['run'].unique()
   
    df_rep_event_list = []
    df_rep_event_list = [df_rep_event[df_rep_event['run']==i] for i in np.arange(1,3)]
    
    
    for f in range(len(df_rep_event_list)):
        file_name = '%s_task-rep_run-%02.0f_events.tsv' %(subject,df_rep_event_list[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_rep_event_list[f].to_csv(write_path, sep='\t')