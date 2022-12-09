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
path_behavior = opj(path_bids, 'derivatives', project_name)

def cvfold(trials,run):
    if trials >= 0 and trials <= 25 and run == 1:
        return 1
    elif trials > 25 and trials <= 51 and run == 1:
        return 2
    elif trials > 51 and trials <= 77 and run == 2:
        return 3
    elif trials > 77 and trials <= 103 and run == 2:
        return 4
    elif trials >= 0 and trials <= 25 and run == 3:
        return 5
    elif trials > 25 and trials <= 51  and run == 3:
        return 6
    elif trials > 51 and trials <= 77  and run == 4:
        return 7
    elif trials > 77 and trials <= 103  and run == 4:
        return 8
# grab the list of subjects from the bids data set:
layout = BIDSLayout(path_bids)
# get all subject ids:
sub_list = sorted(layout.get_subjects())
# sub_list=['04']
# create a template to add the "sub-" prefix to the ids
sub_template = ['sub-'] * len(sub_list)
# add the prefix to all ids:
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]
# if user defined to run specific subject
# sub_list = sub_list[int(sys.argv[1]):int(sys.argv[2])]

for subject in sub_list:
    path_behavior_vfl = opj(path_behavior,subject,'VFL','*csv') 
    path_behavior_sid = opj(path_behavior,subject,'SID','*csv') 
    path_behavior_asd = opj(path_behavior,subject,'ASD','*csv') 
    path_behavior_msd = opj(path_behavior,subject,'MSD','*csv')
    # create the path to the data file
    path_vfl_event = sorted(glob.glob(path_behavior_vfl), key=lambda f: os.path.basename(f))
    path_sid_event = sorted(glob.glob(path_behavior_sid), key=lambda f: os.path.basename(f))
    path_asd_event = sorted(glob.glob(path_behavior_asd), key=lambda f: os.path.basename(f))
    path_msd_event = sorted(glob.glob(path_behavior_msd), key=lambda f: os.path.basename(f))
    '''
    ======================================
    VISUAL FUNCTIONAL LOCALIZER
    ======================================
    '''
    # read the visual functional localizer trial data
    df_vfl_event = [pd.read_csv(f, sep=',') for f in path_vfl_event]
    start_onset =  [df_vfl_event[f]['text_6.started'].dropna(axis=0, how='any') for f in range(len(path_vfl_event))]
    #select useful columns
    df_vfl_event = [df_vfl_event[f][['Ori','Marker','trials.thisTrialN','image_2.started','key_resp.corr',
                     'participant','session']] for f in range(len(path_vfl_event))]

    # delete unrelated rows
    df_vfl_event=[df_vfl_event[f].dropna(axis=0, how='any') for f in range(len(path_vfl_event))]
    # judge the number of trials
    for f in range(len(path_vfl_event)):
        if len(df_vfl_event[f]) != 104 :
            print ('the number of trials in run %s is not right!' % (f+1))
    # rename the data file
    rename_vfl = {'trials.thisTrialN' : 'trials',
                  'image_2.started' : 'onset',
                  'key_resp.corr' : 'accuracy',
                  'session' : 'run'}
    df_vfl_event = [df_vfl_event[f].rename(columns=rename_vfl) for f in range(len(path_vfl_event))]
    for f in range(len(path_vfl_event)):
        df_vfl_event[f]['onset'] = df_vfl_event[f]['onset'] - start_onset[f].iloc[0]
    accuracy_vfl = [(count(df_vfl_event[f]['accuracy'])[1])/(len(df_vfl_event[f]['accuracy'] == 1)) for f in range(len(path_vfl_event))]
    vfl_event_spec = {
        'correct_rejection': {'Ori': 0, 'accuracy': 1},
        'hit': {'Ori': 180, 'accuracy': 1},
        'false_alarm': {'Ori': 0, 'accuracy': 0},
        'miss': {'Ori': 180, 'accuracy': 0},
    }
    for f in range(len(path_vfl_event)):
        df_vfl_event[f]['duration']=2
        df_vfl_event[f]['task'] = 'VFL'
        df_vfl_event[f]['session'] = 1
        df_vfl_event[f]['fold'] = df_vfl_event[f].apply(lambda x: cvfold(x['trials'],x['run']),axis=1)
    for f in range(len(path_vfl_event)):
        file_name = '%s_task-vfl_run-%02.0f_events.tsv' %(subject,df_vfl_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name) #sub_list[0]
        df_vfl_event[f].to_csv(write_path, sep='\t')
    '''
    ======================================
    SINGLE ITEM DECODING
    ======================================
    '''
    # read the single item decoding trial data
    df_sid_event = [pd.read_csv(f, sep=',') for f in path_sid_event]
    start_onset =  [df_sid_event[f]['text_6.started'].dropna(axis=0, how='any') for f in range(len(path_sid_event))]

    #select useful columns
    df_sid_event = [df_sid_event[f][['Ori','Marker2','trials.thisTrialN','key_resp.keys','Correct',
                                     'text_8.started','participant','sess']] for f in range(len(path_sid_event))]
    # delete unrelated rows
    df_sid_event=[df_sid_event[f].dropna(axis=0, how='any') for f in range(len(path_sid_event))]
    # judge the number of trials
    for f in range(len(path_sid_event)):
        if len(df_sid_event[f]) != 104 :
            print ('the number of trials in run %s is not right!' % (f+1))
    # rename the data file
    rename_sid = {'Ori' : 'Catch',
                  'Marker2':'Marker',
                  'trials.thisTrialN' : 'trials',
                  'text_8.started' : 'onset',
                  'key_resp.keys' : 'response',
                  'Correct' : 'accuracy',
                  'sess' : 'run'}
    df_sid_event = [df_sid_event[f].rename(columns=rename_sid) for f in range(len(path_sid_event))]
    accuracy_sid = [(count(df_sid_event[f]['accuracy'])[1])/count(df_sid_event[f]['Catch'] == 2)[1] for f in range(len(path_sid_event))]
    sid_event_spec = {
        'correct_rejection': {'Catch': 1, 'accuracy': 1},
        'hit': {'Catch': 2, 'accuracy': 1},
        'false_alarm': {'Catch': 1, 'accuracy': 0},
        'miss': {'Catch': 2, 'accuracy': 0},
    }
    for f in range(len(path_sid_event)):
        df_sid_event[f]['duration']=2
        df_sid_event[f]['task'] = 'SID'
        df_sid_event[f]['session'] = 2
        df_sid_event[f]['fold'] = df_sid_event[f].apply(lambda x: cvfold(x['trials'],x['run']),axis=1)
        file_name = '%s_task-sid_run-%02.0f_events_cat.tsv' %(subject,df_sid_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_sid_event[f].to_csv(write_path, sep='\t')
    for f in range(len(path_sid_event)):
        df_sid_event[f] = df_sid_event[f][df_sid_event[f]['onset'] != 'None']
        df_sid_event[f]['onset'] = df_sid_event[f]['onset'].astype('float64') - start_onset[f].iloc[0]
        file_name = '%s_task-sid_run-%02.0f_events.tsv' %(subject,df_sid_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_sid_event[f].to_csv(write_path, sep='\t')
    
    '''
    ======================================
    ARBITARY SEQUENCE DECODING WITH CUE
    ======================================
    '''
    # read the arbitrary sequence decoding with cue trial data
    df_asd_cue_event = [pd.read_csv(f, sep=',') for f in path_asd_event]
    start_onset =  [df_asd_cue_event[f]['text_6.started'].dropna(axis=0, how='any') for f in range(len(path_asd_event))]
    #select useful columns
    df_asd_cue_event = [df_asd_cue_event[f][['marker','length','startfrom','isi','Sign','Sign2','M','M2','M3','M4','trials.thisTrialN',
                                         'text_30.started','text_33.started','key_resp_12.keys','text_35.started',
                                         'key_resp_13.keys','text_37.started','key_resp_14.keys','participant','sess']] for f in range(len(path_asd_event))]
    # delete unrelated rows
    df_asd_cue_event=[df_asd_cue_event[f].dropna(axis=0, how='any') for f in range(len(path_asd_event))]
    # judge the number of trials
    for f in range(len(df_asd_cue_event)):
        if len(df_asd_cue_event[f]) != 10 :
            print ('the number of trials in run %s is not right!' % (f+1))
    # rename the data file
    rename_asd = {'trials.thisTrialN' : 'trial',
                  'text_30.started' : 'startcue',
                  'text_33.started':'cue_1',
                  'key_resp_12.keys' : 'catch_1',
                  'text_35.started' : 'cue_2',
                  'key_resp_13.keys' : 'catch_2',
                  'text_37.started' : 'cue_3',
                  'key_resp_14.keys' : 'catch_3',
                  'sess' : 'run'}
    df_asd_cue_event = [df_asd_cue_event[f].rename(columns=rename_asd) for f in range(len(path_asd_event))]
    # accuracy_asd = [(count(df_asd_cue_event[f]['accuracy'])[1])/count(df_asd_cue_event[f]['Catch'] == 2)[1] for f in range(len(path_asd_event))]
    for f in range(len(path_asd_event)):
        df_asd_cue_event[f]['catch_0'] = 'None'
        x = pd.melt(df_asd_cue_event[f], id_vars =['trial','marker','length','startfrom','isi','Sign','Sign2','participant','run'], 
                    value_vars =['M','M2','M3','M4'],var_name='marker_type', value_name='marker_value') 
        y = pd.melt(df_asd_cue_event[f], id_vars =['trial','marker','length','startfrom','isi','Sign','Sign2','participant','run'], 
                    value_vars =['catch_0','catch_1','catch_2','catch_3'],var_name='catch_type', value_name='catch_value') 
        z = pd.melt(df_asd_cue_event[f], id_vars =['trial','marker','length','startfrom','isi','Sign','Sign2','participant','run'], 
                    value_vars =['startcue','cue_1','cue_2','cue_3'],var_name='onset_type', value_name='onset_value') 
        df_asd_cue_event[f]=pd.concat([x,y.iloc[:,-2:],z.iloc[:,-2:]],axis=1)
        df_asd_cue_event[f]['task'] = 'asd-cue'
        df_asd_cue_event[f]['session'] = 3
        df_asd_cue_event[f]['duration'] = 2
        file_name = '%s_task-asd-cue_run-%02.0f_events_cat.tsv'  %(subject,df_asd_cue_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_asd_cue_event[f].to_csv(write_path, sep='\t')
    
    for f in range(len(path_asd_event)):
        df_asd_cue_event[f] = df_asd_cue_event[f][df_asd_cue_event[f]['onset_value'] != 'None']
        df_asd_cue_event[f]['onset_value'] = df_asd_cue_event[f]['onset_value'].astype('float64') - start_onset[f].iloc[0]
        file_name = '%s_task-asd-cue_run-%02.0f_events.tsv'  %(subject,df_asd_cue_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_asd_cue_event[f].to_csv(write_path, sep='\t')
    '''
    ======================================
    ARBITARY SEQUENCE DECODING WITHOUT CUE
    ======================================
    '''
    # read the arbitrary sequence decoding without cue trial data
    df_asd_nocue_event = [pd.read_csv(f, sep=',') for f in path_asd_event]
    start_onset =  [df_asd_nocue_event[f]['text_6.started'].dropna(axis=0, how='any') for f in range(len(path_asd_event))]

    #select useful columns
    df_asd_nocue_event = [df_asd_nocue_event[f][['marker','length','startfrom','Sign','Sign2','Types','Mw','End',
                                                 'trials_3.thisTrialN','text_46.started','text_49.started','key_resp_26.keys','key_resp_16.keys',
                                                 'participant','sess']] for f in range(len(path_asd_event))]
    # delete unrelated rows
    df_asd_nocue_event = [df_asd_nocue_event[f].dropna(axis=0, how='any')  for f in range(len(path_asd_event))]
    # judge the number of trials
    for f in range(len(path_asd_event)):
        if len(df_asd_nocue_event[f]) != 60 :
            print ('the number of trials in run %s is not right!' % (f+1))
    # rename the data file
    rename_asd = {'Marker2':'Marker',
                  'trials_3.thisTrialN' : 'trials',
                  'text_46.started' : 'cue_onset',
                  'text_49.started' : 'imagine_onset',
                  'key_resp_26.keys' : 'catch1',
                  'key_resp_16.keys' : 'con_rating',
                  'sess' : 'run'}
    df_asd_nocue_event = [df_asd_nocue_event[f].rename(columns=rename_asd) for f in range(len(path_asd_event))]
    # accuracy_asd = [(count(df_asd_nocue_event[f]['accuracy'])[1])/count(df_asd_nocue_event[f]['Catch'] == 2)[1] for f in range(len(path_asd_event))]
    for f in range(len(path_asd_event)):
        df_asd_nocue_event[f]['duration'] = 4
        df_asd_nocue_event[f]['task'] = 'ASD-nocue'
        df_asd_nocue_event[f]['session'] = 4
        df_asd_nocue_event[f] = df_asd_nocue_event[f].reset_index(drop=True)
        for i in range(len(df_asd_nocue_event[0])):
            if df_asd_nocue_event[f]['Types'][i] == u'想五次':
                df_asd_nocue_event[f]['Types'][i] = 'five'
            elif df_asd_nocue_event[f]['Types'][i] == u'想一次':
                df_asd_nocue_event[f]['Types'][i] = 'one'
            if df_asd_nocue_event[f]['End'][i] == u'→A':
                df_asd_nocue_event[f]['End'][i] = 'A'
            elif df_asd_nocue_event[f]['End'][i] == u'→B':
                df_asd_nocue_event[f]['End'][i] = 'B'
            elif df_asd_nocue_event[f]['End'][i] == u'→C':
                df_asd_nocue_event[f]['End'][i] = 'C'
            elif df_asd_nocue_event[f]['End'][i] == u'→D':
                df_asd_nocue_event[f]['End'][i] = 'D'
    for f in range(len(path_asd_event)):
        file_name = '%s_task-asd-nocue_run-%02.0f_events_cat.tsv'  %(subject,df_asd_nocue_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_asd_nocue_event[f].to_csv(write_path, sep='\t')
    
    for f in range(len(path_asd_event)):
        df_asd_nocue_event[f] = df_asd_nocue_event[f][df_asd_nocue_event[f]['imagine_onset'] != 'None']
        df_asd_nocue_event[f]['imagine_onset'] = df_asd_nocue_event[f]['imagine_onset'].astype('float64') - start_onset[f].iloc[0]
        df_asd_nocue_event[f]['cue_onset'] = df_asd_nocue_event[f]['cue_onset'].astype('float64') - start_onset[f].iloc[0]
        file_name = '%s_task-asd-nocue_run-%02.0f_events.tsv'  %(subject,df_asd_nocue_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_asd_nocue_event[f].to_csv(write_path, sep='\t')
    '''
    ======================================
    MEANINGFUL SEQUENCE DECODING WITH CUE
    ======================================
    '''
    # read the arbitrary sequence decoding with cue trial data
    df_msd_cue_event = [pd.read_csv(f, sep=',') for f in path_msd_event]
    start_onset =  [df_msd_cue_event[f]['text_6.started'].dropna(axis=0, how='any') for f in range(len(path_msd_event))]

    #select useful columns
    df_msd_cue_event = [df_msd_cue_event[f][['marker','length','startfrom','isi','Sign','Sign2','M','M2','M3','M4','trials.thisTrialN',
                                         'text_30.started','text_33.started','key_resp_12.keys','text_35.started',
                                         'key_resp_13.keys','text_37.started','key_resp_14.keys','participant','sess']] for f in range(len(path_msd_event))]
    # delete unrelated rows
    df_msd_cue_event=[df_msd_cue_event[f].dropna(axis=0, how='any') for f in range(len(path_msd_event))]
    # judge the number of trials
    for f in range(len(df_msd_cue_event)):
        if len(df_msd_cue_event[f]) != 12 :
            print ('the number of trials in run %s is not right!' % (f+1))
        # else:
        #     print ('the number of trials in run %s is totally right!' % (f+1))
    # rename the data file
    rename_msd = {'trials.thisTrialN' : 'trial',
                  'text_30.started' : 'startcue',
                  'text_33.started':'cue_1',
                  'key_resp_12.keys' : 'catch_1',
                  'text_35.started' : 'cue_2',
                  'key_resp_13.keys' : 'catch_2',
                  'text_37.started' : 'cue_3',
                  'key_resp_14.keys' : 'catch_3',
                  'sess' : 'run'}
    df_msd_cue_event = [df_msd_cue_event[f].rename(columns=rename_msd) for f in range(len(path_msd_event))]
    # accuracy_msd = [(count(df_msd_cue_event[f]['accuracy'])[1])/count(df_msd_cue_event[f]['Catch'] == 2)[1] for f in range(len(path_msd_event))]
    
    for f in range(len(path_asd_event)):
        df_msd_cue_event[f]['catch_0'] = 'None'
        x = pd.melt(df_msd_cue_event[f], id_vars =['trial','marker','length','startfrom','isi','Sign','Sign2','participant','run'], 
                    value_vars =['M','M2','M3','M4'],var_name='marker_type', value_name='marker_value') 
        y = pd.melt(df_msd_cue_event[f], id_vars =['trial','marker','length','startfrom','isi','Sign','Sign2','participant','run'], 
                    value_vars =['catch_0','catch_1','catch_2','catch_3'],var_name='catch_type', value_name='catch_value') 
        z = pd.melt(df_msd_cue_event[f], id_vars =['trial','marker','length','startfrom','isi','Sign','Sign2','participant','run'], 
                    value_vars =['startcue','cue_1','cue_2','cue_3'],var_name='onset_type', value_name='onset_value') 
        df_msd_cue_event[f]=pd.concat([x,y.iloc[:,-2:],z.iloc[:,-2:]],axis=1)
        df_msd_cue_event[f]['task'] = 'MSD-cue'
        df_msd_cue_event[f]['session'] = 5
        df_msd_cue_event[f]['duration'] = 2
    for f in range(len(path_msd_event)):
        file_name = '%s_task-msd-cue_run-%02.0f_events_cat.tsv'  %(subject,df_msd_cue_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_msd_cue_event[f].to_csv(write_path, sep='\t')
        
    for f in range(len(path_msd_event)):
        df_msd_cue_event[f] = df_msd_cue_event[f][df_msd_cue_event[f]['onset_value'] != 'None']
        df_msd_cue_event[f]['onset_value'] = df_msd_cue_event[f]['onset_value'].astype('float64') - start_onset[f].iloc[0]
        file_name = '%s_task-asd-cue_run-%02.0f_events.tsv'  %(subject,df_msd_cue_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_msd_cue_event[f].to_csv(write_path, sep='\t')
        
    '''
    ======================================
    MEANINGFUL SEQUENCE DECODING WITHOUT CUE
    ======================================
    '''
    # read the arbitrary sequence decoding without cue trial data
    df_msd_nocue_event = [pd.read_csv(f, sep=',') for f in path_msd_event]
    start_onset =  [df_msd_nocue_event[f]['text_6.started'].dropna(axis=0, how='any') for f in range(len(path_msd_event))]
    #select useful columns
    df_msd_nocue_event = [df_msd_nocue_event[f][['marker','length','startfrom','Sign','Sign2','Types','Mw','End',
                                                 'trials_3.thisTrialN','text_46.started','text_49.started','key_resp_27.keys','key_resp_16.keys',
                                                 'participant','sess']] for f in range(len(path_msd_event))]
    # delete unrelated rows
    df_msd_nocue_event = [df_msd_nocue_event[f].dropna(axis=0, how='any')  for f in range(len(path_msd_event))]
    # judge the number of trials
    for f in range(len(path_msd_event)):
        if len(df_msd_nocue_event[f]) != 60 :
            print ('the number of trials in run %s is not right!' % (f+1))
    # rename the data file
    rename_msd = {'Marker2':'Marker',
                  'trials_3.thisTrialN' : 'trials',
                  'text_46.started' : 'cue_onset',
                  'text_49.started' : 'imagine_onset',
                  'key_resp_26.keys' : 'catch1',
                  'key_resp_16.keys' : 'con_rating',
                  'sess' : 'run'}
    df_msd_nocue_event = [df_msd_nocue_event[f].rename(columns=rename_msd) for f in range(len(path_msd_event))]
    # accuracy_msd = [(count(df_msd_nocue_event[f]['accuracy'])[1])/count(df_msd_nocue_event[f]['Catch'] == 2)[1] for f in range(len(path_msd_event))]
    for f in range(len(path_msd_event)):
        df_msd_nocue_event[f]['duration'] = 4
        df_msd_nocue_event[f]['task'] = 'ASD-nocue'
        df_msd_nocue_event[f]['session'] = 6
        df_msd_nocue_event[f] = df_msd_nocue_event[f].reset_index(drop=True)
        for i in range(len(df_msd_nocue_event[0])):
            if df_msd_nocue_event[f]['Types'][i] == u'想五次':
                df_msd_nocue_event[f]['Types'][i] = 'five'
            elif df_msd_nocue_event[f]['Types'][i] == u'想一次':
                df_msd_nocue_event[f]['Types'][i] = 'one'
            if df_msd_nocue_event[f]['End'][i] == u'→A':
                df_msd_nocue_event[f]['End'][i] = 'A'
            elif df_msd_nocue_event[f]['End'][i] == u'→B':
                df_msd_nocue_event[f]['End'][i] = 'B'
            elif df_msd_nocue_event[f]['End'][i] == u'→C':
                df_msd_nocue_event[f]['End'][i] = 'C'
            elif df_msd_nocue_event[f]['End'][i] == u'→D':
                df_msd_nocue_event[f]['End'][i] = 'D'
    for f in range(len(path_msd_event)):
        file_name = '%s_task-msd-nocue_run-%02.0f_events_cat.tsv'  %(subject,df_msd_nocue_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_msd_nocue_event[f].to_csv(write_path, sep='\t')

    for f in range(len(path_msd_event)):
        df_msd_nocue_event[f] = df_msd_nocue_event[f][df_msd_nocue_event[f]['imagine_onset'] != 'None']
        df_msd_nocue_event[f]['imagine_onset'] = df_msd_nocue_event[f]['imagine_onset'].astype('float64') - start_onset[f].iloc[0]
        df_msd_nocue_event[f]['cue_onset'] = df_msd_nocue_event[f]['cue_onset'].astype('float64') - start_onset[f].iloc[0]
        file_name = '%s_task-msd-nocue_run-%02.0f_events.tsv'  %(subject,df_msd_nocue_event[f]['run'].iloc[0])
        write_path = opj(path_bids,subject,'func',file_name)
        df_msd_nocue_event[f].to_csv(write_path, sep='\t')











