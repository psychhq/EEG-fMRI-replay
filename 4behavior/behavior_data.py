# -*- coding: utf-8 -*-
"""
# Created on Wed Jun 15 23:03:12 2022
# SCRIPT: CREATE EVENT.TSV FILES FROM THE BEHAVIORAL DATA FOR BIDS
# PROJECT: FMRIREPLAY
# WRITTEN BY QI HUANG
"""
# In[1]: packages, functions and data
import os
import glob
import warnings
from os.path import join as opj
import pandas as pd
import numpy as np
import operator
from collections import Counter as count
from pandas.core.common import SettingWithCopyWarning
# import libraries for bids interaction:
from bids.layout import BIDSLayout
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

# filter warning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# initialize empty paths:
path_root = None
sub_list = None
# create the path
path_bids = opj(os.getcwd().split('fmrireplay')[0], 'fmrireplay')
path_behavior_input = opj(path_bids, 'sourcedata', 'behavior-rawdata')
path_behavior_output = opj(path_bids, 'derivatives', 'behavior')

# grab the list of subjects from the bids data set:
layout = BIDSLayout(path_bids)
# get all subject ids:
sub_list = sorted(layout.get_subjects())

# create the dataframe for visualization of the accuracy
task_accuracy = pd.DataFrame(sub_list, columns=['subjectid'])
task_accuracy.insert(1, 'VFL accuracy', '')
task_accuracy.insert(2, 'learning accuracy', '')
task_accuracy.insert(3, 'last run"s learning accuracy', '')
task_accuracy.insert(4, 'replay accuracy', '')
task_accuracy.insert(5, 'vividness rating', '')
# create the dataframe for visualization of the number of trials
task_ntrials = pd.DataFrame(sub_list, columns=['subjectid'])
task_ntrials.insert(1, 'VFL', '')
task_ntrials.insert(2, 'learning', '')
task_ntrials.insert(3, 'replay', '')
# create the dataframe for cue replay task accuracy
cue_replay_accuracy = pd.DataFrame(sub_list, columns=['subjectid'])
cue_replay_accuracy.insert(1, 'cue_replay', '')

# CREATE OUTPUT DIRECTORIES IF THE DO NOT EXIST YET:
for subject in sub_list:
    sub = "sub-%s" % subject
    path_beh_sub = opj(path_behavior_output, 'sub_level', sub)
    if not os.path.exists(path_beh_sub):
        os.makedirs(path_beh_sub)

# In[2]: VISUAL FUNCTIONAL LOCALIZER

for subject in sub_list:
    sub = "sub-%s" % subject
    # get the path of VFL files
    path_behavior_vfl = opj(path_behavior_input, 'VFL', 'VFL_%s_*.csv' % subject)
    # create the path to the data file
    path_vfl_event = sorted(glob.glob(path_behavior_vfl), key=lambda f: os.path.basename(f))
    # read the visual functional localizer trial data
    df_vfl_event = [pd.read_csv(f, sep=',') for f in path_vfl_event]
    # select useful columns
    df_vfl_event = [df_vfl_event[f][['cons', 'stimMarker', 'textMarker', 'trials.thisRepN',
                                     'trials.thisTrialN', 'text_6.started', 'image_2.started', 'text_10.started',
                                     'key_resp.rt', 'key_resp.corr', 'subject_id']] for f in range(len(path_vfl_event))]
    # delete unrelated rows by NaN value
    df_vfl_event = [df_vfl_event[f].dropna(axis=0, how='any', subset=['cons']).reset_index(drop=True) for f in
                    range(len(path_vfl_event))]
    # rename the columns(variables)
    rename_vfl = {'trials.thisTrialN': 'trials',
                  'text_6.started': 'start',
                  'image_2.started': 'onset',
                  'text_10.started': 'onset_semantic',
                  'key_resp.rt': 'resp_rt',
                  'key_resp.corr': 'accuracy',
                  'trials.thisRepN': 'run',
                  'subject_id': 'participant'}
    df_vfl_event = [df_vfl_event[f].rename(columns=rename_vfl) for f in range(len(path_vfl_event))]

    # if one file from one subject
    if len(path_vfl_event) == 1:
        df_vfl_event = df_vfl_event[0]
        # set some new values for all first session
        df_vfl_event['session'] = 1
        df_vfl_event['run'] = df_vfl_event['run'] + 1
        # calculate onset time of visual stimuli
        onset = [df_vfl_event[(df_vfl_event['run'] == i)]['onset'].sub(
            float(df_vfl_event[(df_vfl_event['run'] == i) & (df_vfl_event['trials'] == 0)]['start'])) for i in
            np.arange(1, 5)]
        # to the dataframe
        onset = pd.concat([onset[0], onset[1], onset[2], onset[3]], axis=0)
        df_vfl_event['onset'] = onset

        # calculate onset time of semantic stimuli
        onset_semantic = [df_vfl_event[(df_vfl_event['run'] == i)]['onset_semantic'].sub(
            float(df_vfl_event[(df_vfl_event['run'] == i)
                               & (df_vfl_event['trials'] == 0)]['start'])) for i in np.arange(1, 5)]
        # to the dataframe
        onset_semantic = pd.concat([onset_semantic[0], onset_semantic[1],
                                    onset_semantic[2], onset_semantic[3]], axis=0)
        df_vfl_event['onset_semantic'] = onset_semantic

        # to the dataframe
        df_vfl_event.loc[df_vfl_event['resp_rt'].isna(), 'resp_rt'] = 1
        onset_resp = [
            df_vfl_event[(df_vfl_event['run'] == i)]['onset_semantic'] + df_vfl_event[(df_vfl_event['run'] == i)][
                'resp_rt']
            for i in np.arange(1, 5)]
        onset_resp = pd.concat([onset_resp[0], onset_resp[1],
                                onset_resp[2], onset_resp[3]], axis=0)
        df_vfl_event['onset_resp'] = onset_resp
    # if there are two files from one subject
    elif len(path_vfl_event) == 2:
        print(sub)
        # set some new values for the second session
        df_vfl_event[0]['session'] = 1
        df_vfl_event[0]['run'] = df_vfl_event[0]['run'] + 1
        df_vfl_event[1]['session'] = 2
        df_vfl_event[1]['run'] = df_vfl_event[1]['run'] + 3
        # calculate onset time for each run
        for f in range(len(path_vfl_event)):
            if f == 0:
                onset1 = [df_vfl_event[f][(df_vfl_event[f]['run'] == i)]['onset'].sub(
                    float(df_vfl_event[f][(df_vfl_event[f]['run'] == i)
                                          & (df_vfl_event[f]['trials'] == 0)]['start'])) for i in np.arange(1, 3)]
                onset = pd.concat([onset1[0], onset1[1]], axis=0)
                df_vfl_event[f]['onset'] = onset

                onset1_semantic = [df_vfl_event[f][(df_vfl_event[f]['run'] == i)]['onset_semantic'].sub(
                    float(df_vfl_event[f][(df_vfl_event[f]['run'] == i)
                                          & (df_vfl_event[f]['trials'] == 0)]['start'])) for i in np.arange(1, 3)]
                onset = pd.concat([onset1_semantic[0], onset1_semantic[1]], axis=0)
                df_vfl_event[f]['onset_semantic'] = onset

                df_vfl_event[f].loc[df_vfl_event[f]['resp_rt'].isna(), 'resp_rt'] = 1
                onset_resp = [df_vfl_event[f][(df_vfl_event[f]['run'] == i)]['onset_semantic'] +
                              df_vfl_event[f][(df_vfl_event[f]['run'] == i)]['resp_rt']
                              for i in np.arange(1, 3)]
                onset_resp = pd.concat([onset_resp[0], onset_resp[1]], axis=0)
                df_vfl_event[f]['onset_resp'] = onset_resp

            elif f == 1:
                onset2 = [df_vfl_event[f][(df_vfl_event[f]['run'] == i)]['onset'].sub(
                    float(df_vfl_event[f][(df_vfl_event[f]['run'] == i)
                                          & (df_vfl_event[f]['trials'] == 0)]['start'])) for i in np.arange(3, 5)]
                onset = pd.concat([onset2[0], onset2[1]], axis=0)
                df_vfl_event[f]['onset'] = onset

                onset2_semantic = [df_vfl_event[f][(df_vfl_event[f]['run'] == i)]['onset_semantic'].sub(
                    float(df_vfl_event[f][(df_vfl_event[f]['run'] == i)
                                          & (df_vfl_event[f]['trials'] == 0)]['start'])) for i in np.arange(3, 5)]
                onset = pd.concat([onset2_semantic[0], onset2_semantic[1]], axis=0)
                df_vfl_event[f]['onset_semantic'] = onset

                df_vfl_event[f].loc[df_vfl_event[f]['resp_rt'].isna(), 'resp_rt'] = 1
                onset_resp = [df_vfl_event[f][(df_vfl_event[f]['run'] == i)]['onset_semantic'] +
                              df_vfl_event[f][(df_vfl_event[f]['run'] == i)]['resp_rt']
                              for i in np.arange(3, 5)]
                onset_resp = pd.concat([onset_resp[0], onset_resp[1]], axis=0)
                df_vfl_event[f]['onset_resp'] = onset_resp
        temp = pd.concat([df_vfl_event[0], df_vfl_event[1]])
        df_vfl_event = temp

    # set some new values
    df_vfl_event['duration'] = 1
    df_vfl_event['duration_resp'] = 0
    df_vfl_event['task'] = 'VFL'
    # calculate the accuracy for VFL task
    task_accuracy['VFL accuracy'].loc[int(subject) - 1] = (count(df_vfl_event['accuracy'])[1]) / (
        len(df_vfl_event['accuracy'] == 1))
    # calculate the trials for VFL task
    task_ntrials['VFL'].loc[int(subject) - 1] = (len(df_vfl_event['accuracy'] == 1))
    # save the dataframe to the csv file
    classfication = df_vfl_event['run'].unique()
    df_vfl_event_temp = [df_vfl_event[df_vfl_event['run'] == i] for i in np.arange(1, 5)]
    for f in range(len(df_vfl_event_temp)):
        file_name = '%s_task-vfl_run-%02.0f_events.tsv' % (subject, df_vfl_event_temp[f]['run'].iloc[0])
        write_path = opj(path_behavior_output, 'sub_level', sub, file_name)
        df_vfl_event_temp[f].to_csv(write_path, sep='\t')

# In[3]:LEARNING SESSION

for subject in sub_list:
    # subject = sub_list[31]
    # subject = sub_list[0]
    sub = "sub-%s" % subject
    path_behavior_learning = opj(path_behavior_input, 'learning', 'Sequence_learning_%s_*.csv' % subject)
    # create the path to the data file
    path_learning_event = sorted(glob.glob(path_behavior_learning), key=lambda f: os.path.basename(f))
    # read the learning and test trial data
    df_learning_event = [pd.read_csv(f, sep=',') for f in path_learning_event]
    # select useful columns
    df_learning_event = [
        df_learning_event[f][['block.thisRepN', 'text_6.started',
                              'seq1_1_1.started', 'seq1_2.started', 'image_2.started',
                              'miniblock.thisN', 'trial.thisRepN', 'subject_id'
                              ]] for f in range(len(path_learning_event))]
    # rename the columns(variables)
    rename_learning = {'block.thisRepN': 'run',
                       'text_6.started': 'start',
                       'seq1_1_1.started': 'onset_a',
                       'seq1_2.started': 'onset_b',
                       'image_2.started': 'end',
                       'miniblock.thisN': 'mini_block',
                       'trial.thisRepN': 'block_trial',
                       'subject_id': 'participant'}
    df_learning_event = [df_learning_event[f].rename(columns=rename_learning) for f in range(len(path_learning_event))]

    # if one file from one subject
    if len(path_learning_event) == 1:
        df_learning_event = df_learning_event[0]
        df_learning_event['session'] = 1
        df_learning_event['run'] = df_learning_event['run'] + 1
        end_time = df_learning_event.loc[pd.Series(map(operator.not_, df_learning_event['end'].isnull())), 'end']
        df_learning_event = df_learning_event.loc[pd.Series(map(operator.not_, df_learning_event['onset_a'].isnull())),
                            :]

        # calculate onset time
        for i in np.arange(1, 4):
            for onset in ['onset_a', 'onset_b']:
                df_learning_event.loc[df_learning_event['run'] == i, onset] = df_learning_event.loc[
                    (df_learning_event['run'] == i), onset].sub(
                    float(df_learning_event[(df_learning_event['run'] == i)
                                            & (df_learning_event['mini_block'] == 0)
                                            & (df_learning_event['block_trial'] == 0)]['start']))
            # calculate the end time
            df_learning_event.loc[df_learning_event['run'] == i, 'end'] = end_time.iloc[(i - 1)] - float(
                df_learning_event[(df_learning_event['run'] == i)
                                  & (df_learning_event['mini_block'] == 0)
                                  & (df_learning_event['block_trial'] == 0)]['start'])

    # if there are two files from one subject
    elif len(path_learning_event) == 2:
        # set some new values for the second session
        df_learning_event[0]['session'] = 1
        df_learning_event[0]['run'] = df_learning_event[0]['run'] + 1
        df_learning_event[1]['session'] = 2
        df_learning_event[1]['run'] = df_learning_event[1]['run'] + 2
        # calculate onset time for each run
        for f in range(len(path_learning_event)):
            if f == 0:
                end_time = df_learning_event[f].loc[
                    pd.Series(map(operator.not_, df_learning_event[f]['end'].isnull())), 'end']
                df_learning_event[f] = df_learning_event[f].loc[
                                       pd.Series(map(operator.not_, df_learning_event[f]['onset_a'].isnull())),
                                       :]
                for onset in ['onset_a', 'onset_b']:
                    # calculate onset time
                    df_learning_event[f][onset] = df_learning_event[f][(df_learning_event[f]['run'] == 1)][onset].sub(
                        float(df_learning_event[f][(df_learning_event[f]['run'] == 1)
                                                   & (df_learning_event[f]['mini_block'] == 0)
                                                   & (df_learning_event[f]['block_trial'] == 0)]['start']))

                df_learning_event[f].loc[df_learning_event[f]['run'] == 1, 'end'] = end_time.iloc[0] - float(
                    df_learning_event[f][(df_learning_event[f]['run'] == 1)
                                         & (df_learning_event[f]['mini_block'] == 0)
                                         & (df_learning_event[f]['block_trial'] == 0)]['start'])

            elif f == 1:
                end_time = df_learning_event[f].loc[
                    pd.Series(map(operator.not_, df_learning_event[f]['end'].isnull())), 'end']
                df_learning_event[f] = df_learning_event[f].loc[
                                       pd.Series(map(operator.not_, df_learning_event[f]['onset_a'].isnull())),
                                       :]
                for i in np.arange(2, 4):
                    for onset in ['onset_a', 'onset_b']:
                        # calculate onset time
                        df_learning_event[f].loc[(df_learning_event[f]['run'] == i), onset] = \
                            df_learning_event[f].loc[(df_learning_event[f]['run'] == i), onset].sub(
                                float(df_learning_event[f][(df_learning_event[f]['run'] == i)
                                                           & (df_learning_event[f]['mini_block'] == 0)
                                                           & (df_learning_event[f]['block_trial'] == 0)]['start']))
                    df_learning_event[f].loc[df_learning_event[f]['run'] == i, 'end'] = end_time.iloc[i - 2] - float(
                        df_learning_event[f][(df_learning_event[f]['run'] == i)
                                             & (df_learning_event[f]['mini_block'] == 0)
                                             & (df_learning_event[f]['block_trial'] == 0)]['start'])
        temp = pd.concat([df_learning_event[0], df_learning_event[1]])
        df_learning_event = temp

    # delete unrelated rows
    df_learning_event = df_learning_event.dropna(axis=0, how='any', subset=['onset_a']).reset_index(drop=True)
    # set some new values
    df_learning_event['duration'] = 1.5
    df_learning_event['task'] = 'learning'
    # delete unrelated variables
    del df_learning_event['start']
    del df_learning_event['mini_block']
    del df_learning_event['block_trial']
    df_learning_event.reset_index(drop=False, inplace=True)
    df_learning_event = df_learning_event.rename(columns={'index': 'trial'})

    # save the dataframe to the csv file
    classfication = df_learning_event['run'].unique()
    df_learning_event_temp = [df_learning_event[df_learning_event['run'] == i] for i in np.arange(1, 4)]
    for f in range(len(df_learning_event_temp)):
        file_name = '%s_task-learning_run-%02.0f_events.tsv' % (subject, df_learning_event_temp[f]['run'].iloc[0])
        write_path = opj(path_behavior_output, 'sub_level', sub, file_name)
        df_learning_event_temp[f].to_csv(write_path, sep='\t')

# In[4]: TEST SESSION

for subject in sub_list:
    # subject = sub_list[31]
    sub = "sub-%s" % subject
    print(sub)
    path_behavior_learning = opj(path_behavior_input, 'learning', 'Sequence_learning_%s_*.csv' % subject)
    # create the path to the data file
    path_learning_event = sorted(glob.glob(path_behavior_learning), key=lambda f: os.path.basename(f))
    # read the learning and test trial data
    df_learning_event = [pd.read_csv(f, sep=',') for f in path_learning_event]
    # select useful columns
    df_testing_event = [
        df_learning_event[f][['block.thisRepN', 'test_trial.thisRepN', 'trial.thisRepN', 'miniblock.thisN',
                              'text_6.started', 'probe_marker', 'probe_target.started', 'text_25.started',
                              'text_23.started', 'test_pic_.started', 'test_marker', 'correct_label', 'test_resp.keys',
                              'test_resp.rt', 'test_resp.corr', 'subject_id']] for f in range(len(path_learning_event))]

    # rename the columns(variables)
    rename_learning = {'block.thisRepN': 'run',
                       'test_trial.thisRepN': 'test_trials',
                       'trial.thisRepN': 'block_trial',
                       'miniblock.thisN': 'mini_block',
                       'text_6.started': 'start',
                       'probe_marker': 'Marker',
                       'probe_target.started': 'probe_onset',
                       'text_25.started': 'start_anchor',
                       'text_23.started': 'end',
                       'test_pic_.started': 'test_onset',
                       'test_marker': 'TestMarker',
                       'correct_label': 'reference_answer',
                       'test_resp.keys': 'response',
                       'test_resp.rt': 'response_time',
                       'test_resp.corr': 'accuracy',
                       'subject_id': 'participant'}
    df_testing_event = [df_testing_event[f].rename(columns=rename_learning) for f in range(len(path_learning_event))]

    # if one file from one subject
    if len(path_learning_event) == 1:
        df_testing_event = df_testing_event[0]
        df_testing_event['session'] = 1
        df_testing_event['run'] = df_testing_event['run'] + 1
        start = df_testing_event.loc[
            ((df_testing_event['block_trial'] == 0) & (df_testing_event['mini_block'] == 0)), 'start']
        start_anchor = df_testing_event.loc[
            pd.Series(map(operator.not_, df_testing_event['start_anchor'].isnull())), 'start_anchor']
        end_time = df_testing_event.loc[pd.Series(map(operator.not_, df_testing_event['end'].isnull())), 'end']
        df_testing_event = df_testing_event.loc[pd.Series(map(operator.not_, df_testing_event['probe_onset'].isnull())),
                           :]
        # calculate onset time
        for i in np.arange(1, 4):
            for onset in ['probe_onset', 'test_onset']:
                df_testing_event.loc[df_testing_event['run'] == i, onset] = df_testing_event.loc[
                                                                                (df_testing_event['run'] == i), onset] - \
                                                                            start.iloc[i - 1]
            # start anchor time
            df_testing_event.loc[df_testing_event['run'] == i, 'start_anchor'] = start_anchor.iloc[(i - 1)] - \
                                                                                 start.iloc[i - 1]
            # end time
            df_testing_event.loc[df_testing_event['run'] == i, 'end'] = end_time.iloc[(i - 1)] - start.iloc[i - 1]
    # if there are two files from one subject
    elif len(path_learning_event) == 2:
        # set some new values for the second session
        df_testing_event[0]['session'] = 1
        df_testing_event[0]['run'] = df_testing_event[0]['run'] + 1
        df_testing_event[1]['session'] = 2
        df_testing_event[1]['run'] = df_testing_event[1]['run'] + 2
        # calculate onset time for each run
        for f in range(len(path_learning_event)):
            if f == 0:
                start = df_testing_event[f].loc[
                    ((df_testing_event[f]['block_trial'] == 0) & (df_testing_event[f]['mini_block'] == 0)), 'start']
                start_anchor = df_testing_event[f].loc[
                    pd.Series(map(operator.not_, df_testing_event[f]['start_anchor'].isnull())), 'start_anchor']
                end_time = df_testing_event[f].loc[
                    pd.Series(map(operator.not_, df_testing_event[f]['end'].isnull())), 'end']
                df_testing_event[f] = df_testing_event[f].loc[
                                      pd.Series(map(operator.not_, df_testing_event[f]['probe_onset'].isnull())), :]
                for onset in ['probe_onset', 'test_onset']:
                    # calculate onset time
                    df_testing_event[f][onset] = df_testing_event[f][(df_testing_event[f]['run'] == 1)][onset] - \
                                                 start.iloc[0]

                df_testing_event[f].loc[df_testing_event[f]['run'] == 1, 'start_anchor'] = start_anchor.iloc[0] - \
                                                                                           start.iloc[0]

                df_testing_event[f].loc[df_testing_event[f]['run'] == 1, 'end'] = end_time.iloc[0] - start.iloc[0]

            elif f == 1:
                start = df_testing_event[f].loc[
                    ((df_testing_event[f]['block_trial'] == 0) & (df_testing_event[f]['mini_block'] == 0)), 'start']
                start_anchor = df_testing_event[f].loc[
                    pd.Series(map(operator.not_, df_testing_event[f]['start_anchor'].isnull())), 'start_anchor']
                end_time = df_testing_event[f].loc[
                    pd.Series(map(operator.not_, df_testing_event[f]['end'].isnull())), 'end']
                df_testing_event[f] = df_testing_event[f].loc[
                                      pd.Series(map(operator.not_, df_testing_event[f]['probe_onset'].isnull())), :]
                for i in np.arange(2, 4):
                    for onset in ['probe_onset', 'test_onset']:
                        # calculate onset time
                        df_testing_event[f].loc[(df_testing_event[f]['run'] == i), onset] = \
                            df_testing_event[f].loc[(df_testing_event[f]['run'] == i), onset] - start.iloc[i - 2]

                    df_testing_event[f].loc[df_testing_event[f]['run'] == i, 'start_anchor'] = \
                        start_anchor.iloc[i - 2] - start.iloc[i - 2]

                    df_testing_event[f].loc[df_testing_event[f]['run'] == i, 'end'] = end_time.iloc[i - 2] - start.iloc[
                        i - 2]
        temp = pd.concat([df_testing_event[0], df_testing_event[1]])
        df_testing_event = temp
    # delete unrelated rows
    df_testing_event['rt_onset'] = df_testing_event['test_onset'] + df_testing_event['response_time']
    df_testing_event = df_testing_event.dropna(axis=0, how='any', subset=['Marker']).reset_index(drop=True)
    # set some new values
    df_testing_event['probe_duration'] = 4
    df_testing_event['task'] = 'test'
    # delete unrelated variables
    del df_testing_event['start']
    del df_testing_event['mini_block']
    del df_testing_event['block_trial']
    # calculate the accuracy for VFL task
    task_accuracy['learning accuracy'].loc[int(subject) - 1] = (count(df_testing_event['accuracy'])[1]) / (
        len(df_testing_event['accuracy'] == 1))
    task_accuracy['last run"s learning accuracy'].loc[int(subject) - 1] = (count(
        df_testing_event[df_testing_event['run'] == 3]['accuracy'])[1]) / (len(
        df_testing_event[df_testing_event['run'] == 3]['accuracy'] == 1))
    # calculate the trials for VFL task
    task_ntrials['learning'].loc[int(subject) - 1] = (len(df_testing_event['accuracy'] == 1))
    df_testing_event = df_testing_event.reset_index(drop=True)
    # save the dataframe to the csv file
    classfication = df_testing_event['run'].unique()
    df_testing_event_temp = [df_testing_event[df_testing_event['run'] == i] for i in np.arange(1, 4)]
    for f in range(len(df_testing_event_temp)):
        file_name = '%s_task-testing_run-%02.0f_events.tsv' % (subject, df_testing_event_temp[f]['run'].iloc[0])
        write_path = opj(path_behavior_output, 'sub_level', sub, file_name)
        df_testing_event_temp[f].to_csv(write_path, sep='\t')

# In[4]:CUE REPLAY

accuracy_list = []
for subject in subj_list:
    sub = "sub-%s" % subject
    # get the path of VFL files
    path_behavior_re = opj(path_behavior_input, 'replay', 'cue_replay_%s_*.csv' % subject)
    # create the path to the data file
    path_re_event = sorted(glob.glob(path_behavior_re), key=lambda f: os.path.basename(f))
    # read the visual functional localizer trial data
    df_rep_event = [pd.read_csv(f, sep=',') for f in path_re_event]
    for f in range(len(path_re_event)):
        df_rep_event[f].loc[:, 'key_resp_10.stopped'] = df_rep_event[f].loc[:, 'key_resp_10.started'] + df_rep_event[
                                                                                                            f].loc[:,
                                                                                                        'key_resp_10.rt']
    # select useful columns
    df_rep_event = [
        df_rep_event[f][['runs_replay.thisRepN', 'trials_replay.thisN', 'key_resp_10.stopped', 'text_cue_replay.text',
                         'replay_cue.Trigger', 'text_cue_replay.started',
                         'replay_judge_cRESP', 'key_resp_replay_judge.keys',
                         'key_resp_rating.keys',
                         'key_resp_replay_judge.corr', 'subject_id', 'session'
                         ]] for f in range(len(path_re_event))]
    # delete unrelated rows by NaN value
    df_rep_event = [df_rep_event[f].dropna(axis=0, how='any', subset=['text_cue_replay.started']).reset_index(drop=True)
                    for f in range(len(path_re_event))]
    # rename the columns(variables)
    rename_rep = {'runs_replay.thisRepN': 'run',
                  'trials_replay.thisN': 'trials',
                  'key_resp_10.stopped': 'start',
                  'text_cue_replay.text': 'cue',
                  'replay_cue.Trigger': 'Marker',
                  'text_cue_replay.started': 'onset',
                  'replay_judge_cRESP': 'reference_answer',
                  'key_resp_replay_judge.keys': 'response',
                  'key_resp_replay_judge.corr': 'accuracy',
                  'key_resp_rating.keys': 'rating',
                  'subject_id': 'participant'}
    df_rep_event = [df_rep_event[f].rename(columns=rename_rep) for f in range(len(path_re_event))]
    # set some new values for all first session
    df_rep_event[0]['session'] = 1
    df_rep_event[0]['run'] = df_rep_event[0]['run'] + 1
    # if one file from one subject   
    if len(path_re_event) == 1:
        df_rep_event = df_rep_event[0]
        # calculate onset time
        onset = [df_rep_event[(df_rep_event['run'] == i)]['onset'].sub(
            float(df_rep_event[(df_rep_event['run'] == i)
                               & (df_rep_event['trials'] == 0)]['start'])) for i in np.arange(1, 4)]
        onset = pd.concat([onset[0], onset[1], onset[2]], axis=0)
        df_rep_event['onset'] = onset
    # if there are two files from one subject
    elif len(path_re_event) == 2:
        # set some new values for the second session
        df_rep_event[1]['session'] = 2
        df_rep_event[1]['run'] = df_rep_event[1]['run'] + df_rep_event[0]['run'].iloc[-1] + 1
        # calculate onset time for each run
        if sub == 'sub-07' or sub == 'sub-32':
            for f in range(len(path_re_event)):
                if f == 0:
                    onset1 = [df_rep_event[f][(df_rep_event[f]['run'] == i)]['onset'].sub(
                        float(df_rep_event[f][(df_rep_event[f]['run'] == i)
                                              & (df_rep_event[f]['trials'] == 0)]['start'])) for i in np.arange(1, 3)]
                    onset = pd.concat([onset1[0], onset1[1]], axis=0)
                    df_rep_event[f]['onset'] = onset
                elif f == 1:
                    onset2 = df_rep_event[f][(df_rep_event[f]['run'] == 3)]['onset'].sub(
                        float(df_rep_event[f][(df_rep_event[f]['run'] == 3)
                                              & (df_rep_event[f]['trials'] == 0)]['start']))
                    df_rep_event[f]['onset'] = onset2
        elif sub == 'sub-40':
            for f in range(len(path_re_event)):
                if f == 0:
                    onset1 = [df_rep_event[f][(df_rep_event[f]['run'] == i)]['onset'].sub(
                        float(df_rep_event[f][(df_rep_event[f]['run'] == i)
                                              & (df_rep_event[f]['trials'] == 0)]['start'])) for i in np.arange(1, 2)]
                    df_rep_event[f]['onset'] = onset1[0]
                elif f == 1:
                    onset2 = [df_rep_event[f][(df_rep_event[f]['run'] == i)]['onset'].sub(
                        float(df_rep_event[f][(df_rep_event[f]['run'] == i)
                                              & (df_rep_event[f]['trials'] == 0)]['start'])) for i in np.arange(2, 4)]
                    onset = pd.concat([onset2[0], onset2[1]], axis=0)
                    df_rep_event[f]['onset'] = onset
        temp = pd.concat([df_rep_event[0], df_rep_event[1]])
        df_rep_event = temp
    # set some new values
    df_rep_event['duration'] = 10
    df_rep_event['task'] = 'REP'
    # calculate the accuracy for VFL task    
    if subject == '01' or subject == '05' or subject == '07' or subject == '23' or subject == '30' or subject == '42' or subject == '43' or subject == '49':
        df_rep_event['reference_answer'] = 3 - df_rep_event['reference_answer']


        def pred_acc(reference_answer, response):
            if (response == reference_answer):
                return 1
            else:
                return 0


        df_rep_event['response'] = df_rep_event['response'].replace({'None': '0'})
        df_rep_event['response'] = pd.to_numeric(df_rep_event['response'])
        df_rep_event['accuracy'] = df_rep_event.apply(lambda x: pred_acc(x['reference_answer'], float(x['response'])),
                                                      axis=1)
    # calculate the accuracy for cue replay task
    task_accuracy['replay accuracy'].loc[int(subject) - 1] = (count(df_rep_event['accuracy'])[1]) / (
        len(df_rep_event['accuracy'] == 1))
    # calculate the trials for cue replay task
    task_ntrials['replay'].loc[int(subject) - 1] = (len(df_rep_event['accuracy'] == 1))
    df_rep_event = df_rep_event.reset_index(drop=True)
    accuracy = df_rep_event['accuracy']
    accuracy_list.append(accuracy)


    def str_to_float(s):
        if s == 'None':
            return np.nan
        elif s != 'None':
            return float(s)


    ratings = np.nanmean(np.array(list(map(str_to_float, df_rep_event['rating']))))
    task_accuracy['vividness rating'].loc[int(subject) - 1] = ratings
    # save the dataframe to the csv file
    classfication = df_rep_event['run'].unique()
    df_rep_event_temp = [df_rep_event[df_rep_event['run'] == i] for i in np.arange(1, 4)]
    for f in range(len(df_rep_event_temp)):
        file_name = '%s_task-rep_run-%02.0f_events.tsv' % (subject, df_rep_event_temp[f]['run'].iloc[0])
        write_path = opj(path_behavior_output, 'sub_level', sub, file_name)
        df_rep_event_temp[f].to_csv(write_path, sep='\t')

# In[5]: BEHAVIORAL ACCURACY

# set the theme
sns.set_theme(style="ticks", color_codes=True)
# visualize the included subejcts
loweeg = [45, 38, 36, 34, 28, 21, 16, 15, 11, 5, 4, 2]
incomplete = [42, 22, 4]
extremeheadmotion = [38, 28, 16, 15, 13, 9, 5, 0]
delete_list = np.unique(np.hstack(
    [loweeg, incomplete, extremeheadmotion]))[::-1]
task_accuracy_1 = task_accuracy.drop(index=delete_list)
subject_num = len(task_accuracy_1)

# visualize the data
plt.figure(dpi=400, figsize=(12, 5))
sns.barplot(data=task_accuracy_1.iloc[:, 0:5])
sns.swarmplot(data=task_accuracy_1.iloc[:, 0:5], palette='dark:0', alpha=0.35)
plt.xticks(fontsize=10)
plt.title('%1.0f subjects behavior accuracy of all tasks' % subject_num)
plt.xlabel("Task")
plt.ylabel("Accuracy")
plt.savefig(opj(path_behavior_output, '%1.0f subjects behavior accuracy of all tasks.svg' % subject_num),
            format='svg')
plt.show()

file_name = 'behavior_accuracy_final.csv'
write_path = opj(path_behavior_output, file_name)
task_accuracy_1.to_csv(write_path, sep=',')

# output the statistical results
variable_name = ['VFL accuracy', 'learning accuracy',
                 'last run"s learning accuracy',
                 'replay accuracy', 'vividness rating']
stats_df = pd.DataFrame(index=variable_name, columns=['mean', 'sd', 't', 'dof', 'p'])
for i in variable_name:
    stats_df.loc[i, 'mean'] = np.mean(task_accuracy_1[i])
    stats_df.loc[i, 'sd'] = np.std(task_accuracy_1[i])
    if i != 'vividness rating':
        ttest = pg.ttest(x=list(task_accuracy_1[i]), y=0.5, paired=False, alternative='greater').round(5)
        stats_df.loc[i, 't'] = ttest['T'][0]
        stats_df.loc[i, 'dof'] = ttest['dof'][0]
        stats_df.loc[i, 'p'] = ttest['p-val'][0]
        stats_df.loc[i, 'cohensd'] = ttest['cohen-d'][0]

