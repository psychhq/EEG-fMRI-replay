#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[1]: packages and path settings
# ======================================================================
# PACKAGES AND PATH SETTING
# ======================================================================

import os
import glob
import warnings
from os.path import join as opj
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import numpy as np
import copy
from bids.layout import BIDSLayout

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# name of the current project:
sub_list = None
subject = None
# path to the project root:
project_name = 'replay-onset'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_onset = opj(path_bids, 'sourcedata', 'onsets', 'merge')
path_behavior_input = opj(path_bids, 'sourcedata', 'fmrireplay-rawbehavior')
path_onset_output = opj(path_bids, 'derivatives', project_name)
# define the subject id
layout = BIDSLayout(path_bids)
sub_list = sorted(layout.get_subjects())
# delete the subjects
loweeg = [45, 38, 36, 34, 28, 21, 15, 11, 5, 4]
incomplete = [42, 22, 4]
extremeheadmotion = [38, 28, 16, 15, 13, 9, 5, 0]
delete_list = np.unique(
    np.hstack([loweeg, incomplete, extremeheadmotion]))[::-1]
[sub_list.pop(i) for i in delete_list]


def behavior_start(path_behavior_input, subject):
    # load the behavioral data file
    behavior_path = opj(path_behavior_input, 'replay', 'cue_replay_%s_*.csv' % subject)
    # create the path to the data file
    behavior_path = sorted(glob.glob(behavior_path), key=lambda f: os.path.basename(f))
    # read the visual functional localizer trial data
    behavior_file = [pd.read_csv(f, sep=',') for f in behavior_path]
    # select useful columns
    behavior_file = [
        behavior_file[f][['runs_replay.thisRepN', 'trials_replay.thisN', 'text.started',
                          'text_cue_replay.started']] for f in range(len(behavior_path))]
    # delete unrelated rows by NaN value
    behavior_file = [
        behavior_file[f].dropna(axis=0, how='any', subset=['text_cue_replay.started']).reset_index(drop=True)
        for f in range(len(behavior_path))]
    # rename the columns(variables)
    rename_beh = {'runs_replay.thisRepN': 'run',
                  'trials_replay.thisN': 'trials',
                  'text.started': 'start',
                  'text_cue_replay.started': 'onset'}
    behavior_file = [behavior_file[f].rename(columns=rename_beh) for f in range(len(behavior_path))]
    # set some new values for all first session
    behavior_file[0]['session'] = 1
    behavior_file[0]['run'] = behavior_file[0]['run'] + 1
    # if one file from one subject
    if len(behavior_path) == 1:
        behavior_file = behavior_file[0]
        # calculate onset time
        onset = [behavior_file[(behavior_file['run'] == i)]['onset'].sub(
            float(behavior_file[(behavior_file['run'] == i)
                                & (behavior_file['trials'] == 0)]['start'])) for i in np.arange(1, 4)]
        onset = pd.concat([onset[0], onset[1], onset[2]], axis=0)
        behavior_file['onset'] = onset
    # if there are two files from one subject
    elif len(behavior_path) == 2:
        # set some new values for the second session
        behavior_file[1]['session'] = 2
        behavior_file[1]['run'] = behavior_file[1]['run'] + behavior_file[0]['run'].iloc[-1] + 1
        # calculate onset time for each run
        if sub == 'sub-07' or sub == 'sub-32':
            for f in range(len(behavior_path)):
                if f == 0:
                    onset1 = [behavior_file[f][(behavior_file[f]['run'] == i)]['onset'].sub(
                        float(behavior_file[f][(behavior_file[f]['run'] == i)
                                               & (behavior_file[f]['trials'] == 0)]['start'])) for i in np.arange(1, 3)]
                    onset = pd.concat([onset1[0], onset1[1]], axis=0)
                    behavior_file[f]['onset'] = onset
                elif f == 1:
                    onset2 = behavior_file[f][(behavior_file[f]['run'] == 3)]['onset'].sub(
                        float(behavior_file[f][(behavior_file[f]['run'] == 3)
                                               & (behavior_file[f]['trials'] == 0)]['start']))
                    behavior_file[f]['onset'] = onset2
        elif sub == 'sub-40':
            for f in range(len(behavior_path)):
                if f == 0:
                    onset1 = [behavior_file[f][(behavior_file[f]['run'] == i)]['onset'].sub(
                        float(behavior_file[f][(behavior_file[f]['run'] == i)
                                               & (behavior_file[f]['trials'] == 0)]['start'])) for i in np.arange(1, 2)]
                    behavior_file[f]['onset'] = onset1[0]
                elif f == 1:
                    onset2 = [behavior_file[f][(behavior_file[f]['run'] == i)]['onset'].sub(
                        float(behavior_file[f][(behavior_file[f]['run'] == i)
                                               & (behavior_file[f]['trials'] == 0)]['start'])) for i in np.arange(2, 4)]
                    onset = pd.concat([onset2[0], onset2[1]], axis=0)
                    behavior_file[f]['onset'] = onset
        temp = pd.concat([behavior_file[0], behavior_file[1]])
        behavior_file = temp
    # devide behavior event into different files by runs
    class_run_beh = behavior_file['run'].unique()
    behavior_event = [behavior_file[behavior_file['run'] == i].reset_index(drop=True) for i in class_run_beh]
    # get the absolute time onset for each run
    abs_start = np.array([behavior_event[f].loc[0, ['onset']] for f in range(len(behavior_event))])

    return abs_start


# In[2]: onset data analysis
# ======================================================================
# MAXIMUM PEAK PROBABILITY ONSET DATA ANALYSIS (2 conditions)
# ======================================================================

for subject in sub_list:
    # subject = sub_list[7]
    # set the subject string
    sub = 'sub-%s' % subject
    # load the onset data file
    onset_path = opj(path_onset, 'sub_%s_replayonsets_trialMax.txt' % subject)
    onset_file = pd.read_csv(onset_path, sep=',')

    # get the probable replay onset event
    onset_data = onset_file.loc[(onset_file.prob_lag30 > 0), :]

    # set the relative replay event onset
    onset_data.loc[:, ['onset']] = np.nan
    onset_data['onset'] = onset_data.loc[:, ['points_time']] / 1000
    # set the duration of replay event
    onset_data.loc[:, ['duration']] = 0

    # set the probability of replay event
    onset_data = onset_data.rename(columns={'prob_lag30': 'probability'})

    # devide replay event into different files by runs
    class_run = onset_data['points_runNum'].unique()
    onset_run_data = [onset_data[onset_data['points_runNum'] == i] for i in class_run]

    # rearrange the replay onset event
    onset_run_event = [
        onset_run_data[i].loc[:, ['onset', 'duration', 'points_TrialNum', 'points_runNum',
                                  'isfwdTrial', 'probability']] for i in range(len(onset_run_data))]

    # get start point of each subject's and each run
    abs_start = behavior_start(path_behavior_input, subject)
    for i in range(len(onset_run_event)):
        onset_run_event[i]['onsets'] = onset_run_event[i]['onset'] + abs_start[i]

    # save as the onset run files
    for f in range(len(onset_run_event)):
        file_name = '%s_cue_replay_onset_run-%02.0f_events.tsv' % (subject, onset_run_event[f]['points_runNum'].iloc[0])
        write_path = opj(path_onset_output, 'Max_onset', sub)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        onset_run_event[f].to_csv(opj(write_path, file_name), sep='\t', index=0)

# In[4]: onset data analysis
# ======================================================================
# ALL PERMUTATION PROBABILITY ONSET DATA ANALYSIS (4 conditions)
# ======================================================================

for subject in sub_list:
    # subject = sub_list[1]
    # set the subject string
    sub = 'sub-%s' % subject
    # load the onset data file
    onset_path = opj(path_onset, 'sub_%s_replayonsets_withThresh.txt' % subject)
    onset_file = pd.read_csv(onset_path, sep=',')

    # get the probable replay onset event
    onset_data = onset_file[((onset_file.prob_fwd_lag30 > 0) | (onset_file.prob_bwd_lag30 > 0))]

    # take simultaneously fwd and bwd replay as two time point
    temp_1 = onset_data.loc[((onset_file.prob_fwd_lag30 > 0) & (onset_file.prob_bwd_lag30 > 0)), :]
    temp_2 = onset_data.loc[((onset_file.prob_fwd_lag30 > 0) & (onset_file.prob_bwd_lag30 > 0)), :]
    temp_1.loc[:, 'isfwdReplay'] = 1
    temp_2.loc[:, 'isfwdReplay'] = 0
    temp_all = pd.concat([temp_1, temp_2])

    # sort out other trials
    temp_others = onset_data.loc[~((onset_file.prob_fwd_lag30 > 0) & (onset_file.prob_bwd_lag30 > 0)), :]
    # probable forward replay event
    temp_others.loc[(temp_others.prob_fwd_lag30 > 0), ['isfwdReplay']] = 1.
    # probable backward replay event
    temp_others.loc[(temp_others.prob_bwd_lag30 > 0), ['isfwdReplay']] = 0.

    onset_data = pd.concat([temp_all, temp_others]).sort_values(by=["points_runNum", "points_TrialNum"])

    # set the relative replay event onset
    onset_data.loc[:, ['onset']] = np.nan
    onset_data['onset'] = onset_data.loc[:, ['points_time']] / 1000
    # set the duration of replay event
    onset_data.loc[:, ['duration']] = 0

    # set the probability of replay event
    onset_data.loc[:, ['probability']] = np.nan
    onset_data.loc[(onset_data.isfwdReplay == 1), 'probability'] = onset_data.loc[
        (onset_data.isfwdReplay == 1), 'prob_fwd_lag30']
    onset_data.loc[(onset_data.isfwdReplay == 0), 'probability'] = onset_data.loc[
        (onset_data.isfwdReplay == 0), 'prob_bwd_lag30']

    # devide replay event into different files by runs
    class_run = onset_data['points_runNum'].unique()
    onset_run_data = [onset_data[onset_data['points_runNum'] == i] for i in class_run]

    # rearrange the replay onset event
    onset_run_event = [
        onset_run_data[i].loc[:, ['onset', 'duration', 'points_TrialNum', 'points_runNum',
                                  'isfwdTrial', 'isfwdReplay', 'probability']] for i in range(len(onset_run_data))]

    # get start point of each subject's and each run
    abs_start = behavior_start(path_behavior_input, subject)
    for i in range(len(onset_run_event)):
        onset_run_event[i]['onsets'] = onset_run_event[i]['onset'] + abs_start[i]

    # save as the onset run files
    for f in range(len(onset_run_event)):
        file_name = '%s_cue_replay_onset_run-%02.0f_events.tsv' % (subject, onset_run_event[f]['points_runNum'].iloc[0])
        write_path = opj(path_onset_output, 'Perm_onset_4con', sub)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        onset_run_event[f].to_csv(opj(write_path, file_name), sep='\t', index=0)

# In[5]: onset data analysis
# ======================================================================
# ALL PERMUTATION PROBABILITY ONSET DATA ANALYSIS (2 conditions)
# ======================================================================

for subject in sub_list:
    # subject = sub_list[1]
    # set the subject string
    sub = 'sub-%s' % subject
    # load the onset data file
    onset_path = opj(path_onset, 'sub_%s_replayonsets_withThresh.txt' % subject)
    onset_file = pd.read_csv(onset_path, sep=',')

    # get the probable replay onset event
    onset_data_fwd = onset_file[((onset_file.prob_fwd_lag30 > 0) & (onset_file.isfwdTrial == 1))]
    onset_data_bwd = onset_file[((onset_file.prob_bwd_lag30 > 0) & (onset_file.isfwdTrial == 0))]
    onset_data = pd.concat((onset_data_fwd, onset_data_bwd)).sort_values(by=["points_runNum", "points_TrialNum"])

    # set the relative replay event onset
    onset_data.loc[:, ['onset']] = np.nan
    onset_data['onset'] = onset_data.loc[:, ['points_time']] / 1000

    # set the duration of replay event
    onset_data.loc[:, ['duration']] = 0

    # set the probability of replay event
    onset_data.loc[:, ['probability']] = np.nan
    onset_data.loc[(onset_file.isfwdTrial == 1), 'probability'] = onset_data.loc[
        (onset_file.isfwdTrial == 1), 'prob_fwd_lag30']
    onset_data.loc[(onset_file.isfwdTrial == 0), 'probability'] = onset_data.loc[
        (onset_file.isfwdTrial == 0), 'prob_bwd_lag30']

    # devide replay event into different files by runs
    class_run = onset_data['points_runNum'].unique()
    onset_run_data = [onset_data[onset_data['points_runNum'] == i] for i in class_run]

    # rearrange the replay onset event
    onset_run_event = [
        onset_run_data[i].loc[:, ['onset', 'duration', 'points_TrialNum', 'points_runNum', 'isfwdTrial', 'probability']]
        for
        i in range(len(onset_run_data))]
    # get start point of each subject's and each run
    abs_start = behavior_start(path_behavior_input, subject)
    for i in range(len(onset_run_event)):
        onset_run_event[i]['onsets'] = onset_run_event[i]['onset'] + abs_start[i]

    # save as the onset run files
    for f in range(len(onset_run_event)):
        file_name = '%s_cue_replay_onset_run-%02.0f_events.tsv' % (subject, onset_run_event[f]['points_runNum'].iloc[0])
        write_path = opj(path_onset_output, 'Perm_onset_2con', sub)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        onset_run_event[f].to_csv(opj(write_path, file_name), sep='\t', index=0)

# In[5]: onset data analysis

# ======================================================================
# ALL RAW ONSET DATA ANALYSIS (2 CONDITIONS)
# ======================================================================
import copy

for subject in sub_list:
    # subject = sub_list[1]
    # set the subject string
    sub = 'sub-%s' % subject
    # load the onset data file
    onset_path = opj(path_onset, 'sub_%s_replayonsets_raw.txt' % subject)
    onset_file = pd.read_csv(onset_path, sep=',')
    # set the probability of replay event
    onset_file.loc[:, ['probability']] = np.nan
    onset_file.loc[(onset_file.isfwdTrial == 1), 'probability'] = onset_file.loc[
        (onset_file.isfwdTrial == 1), 'prob_fwd_lag30']
    onset_file.loc[(onset_file.isfwdTrial == 0), 'probability'] = onset_file.loc[
        (onset_file.isfwdTrial == 0), 'prob_bwd_lag30']

    # set the relative replay event onset
    onset_file.loc[:, ['onset']] = np.nan
    onset_file['onset'] = onset_file.loc[:, ['points_time']] / 1000

    # devide replay event into different files by runs
    class_run = onset_file['points_runNum'].unique()
    onset_run_data = [onset_file[onset_file['points_runNum'] == i] for i in class_run]

    # rearrange the replay onset event
    onset_run_event = [
        onset_run_data[i].loc[:, ['onset', 'points_TrialNum', 'points_runNum', 'probability']]
        for i in range(len(onset_run_data))]

    # get start point of each subject's and each run
    abs_start = behavior_start(path_behavior_input, subject)
    for i in range(len(onset_run_event)):
        onset_run_event[i]['onsets'] = onset_run_event[i]['onset'] + abs_start[i]
        onset_run_event[i]['TRscan'] = np.ceil(onset_run_event[i]['onsets'] / 1.3)

    for i in range(len(onset_run_event)):
        df_mean = onset_run_event[i].groupby('TRscan')['probability'].mean()
        # for idx in df_mean.index:
        #     print(idx)
        #     onset_run_event[i].loc[onset_run_event[i]['TRscan']==idx,'probability'] = df_mean[idx]
        prob_mean = pd.DataFrame(df_mean)
        prob_mean = prob_mean.reset_index()
        # save as the onset run files
        file_name = '%s_cue_replay_onset_run-%02.0f_prob.tsv' % (subject, i + 1)
        write_path = opj(path_onset_output, 'raw_onset', sub)

        if not os.path.exists(write_path):
            os.makedirs(write_path)
        prob_mean.to_csv(opj(write_path, file_name), sep='\t', index=0)


# In[3]: own sample onset to the TR
import math

test = copy.deepcopy(onset_run_event[0]).reset_index(drop=True)
test.loc[:, ['onsets(TR)']] = pd.Series(map(lambda x: math.ceil(x), np.array(test.loc[:, ['onsets']]) / 1.3))
x = pd.Series(map(lambda x: math.ceil(x), np.array(test.loc[:, ['onsets']]) / 1.3))
num_tr = np.unique(x)
y = np.array(test.loc[:, ['onsets']]) / 1.3

# whether both forward and backward replay at the same timepoint
test2 = onset_file[((onset_file.prob_fwd_lag30 > 0) & (onset_file.prob_bwd_lag30 > 0))]
