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
from collections import Counter as count
from pandas.core.common import SettingWithCopyWarning
# import libraries for bids interaction:
from bids.layout import BIDSLayout
import seaborn as sns
import matplotlib.pyplot as plt

# filter warning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# initialize empty paths:
path_root = None
sub_list = None
# create the path
project_name = 'behavior'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_behavior_input = opj(path_bids, 'sourcedata', 'fmrireplay-rawbehavior')
path_behavior_output = opj(path_bids, 'derivatives', project_name)


# path_decoding = opj(path_bids, "derivatives", 'decoding')
# path_out_all = opj(path_decoding, 'all_subject_results')

# define some functions
def cvfold(trials, run):
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
    elif trials > 35 and trials <= 71 and run == 3:
        return 6
    elif trials >= 0 and trials <= 35 and run == 4:
        return 7
    elif trials > 35 and trials <= 71 and run == 4:
        return 8


def cvfold_learning(trials, run):
    if trials >= 0 and trials < 9 and run == 1:  # 9
        return 1
    elif trials >= 9 and trials < 12 and run == 1:  # 3
        return 2
    elif trials >= 0 and trials < 6 and run == 2:  # 6
        return 2
    elif trials >= 6 and trials < 12 and run == 2:  # 6
        return 3
    elif trials >= 0 and trials < 3 and run == 3:  # 3
        return 3
    elif trials >= 3 and trials < 12 and run == 3:  # 9
        return 4


### define some parameters

# grab the list of subjects from the bids data set:
layout = BIDSLayout(path_bids)
# get all subject ids:
subject_list = sorted(layout.get_subjects())
# if user defined to run specific subject
# subject_list = subject_list[40:49]
# create a template to add the "sub-" prefix to the ids
sub_template = ['sub-'] * len(subject_list)
# add the prefix to all ids:
sub_list = ["%s%s" % t for t in zip(sub_template, subject_list)]

# create the dataframe for visualization of the accuracy
task_accuracy = pd.DataFrame(subject_list, columns=['subjectid'])
task_accuracy.insert(1, 'VFL accuracy', '')
task_accuracy.insert(2, 'learning accuracy', '')
task_accuracy.insert(3, 'last run"s learning accuracy', '')
task_accuracy.insert(4, 'replay accuracy', '')
# create the dataframe for visualization of the number of trials
task_ntrials = pd.DataFrame(subject_list, columns=['subjectid'])
task_ntrials.insert(1, 'VFL', '')
task_ntrials.insert(2, 'learning', '')
task_ntrials.insert(3, 'replay', '')
# create the dataframe for cue replay task accuracy
cue_replay_accuracy = pd.DataFrame(subject_list, columns=['subjectid'])
cue_replay_accuracy.insert(1, 'cue_replay', '')

# CREATE OUTPUT DIRECTORIES IF THE DO NOT EXIST YET:


# In[2]: VISUAL FUNCTIONAL LOCALIZER
'''
======================================
VISUAL FUNCTIONAL LOCALIZER
======================================
'''
for subject in subject_list:
    # test for the specific subject
    # subject = subject_list[6]
    # get the sub-*
    sub = "sub-%s" % subject
    # get the path of VFL files
    path_behavior_vfl = opj(path_behavior_input, 'VFL', 'VFL_%s_*.csv' % subject)
    # create the path to the data file
    path_vfl_event = sorted(glob.glob(path_behavior_vfl), key=lambda f: os.path.basename(f))
    # read the visual functional localizer trial data
    df_vfl_event = [pd.read_csv(f, sep=',') for f in path_vfl_event]
    # select useful columns
    df_vfl_event = [df_vfl_event[f][['cons', 'stimMarker', 'textMarker', 'trials.thisRepN',
                                     'trials.thisTrialN', 'text_6.started', 'image_2.started',
                                     'key_resp.corr', 'subject_id']] for f in range(len(path_vfl_event))]
    # delete unrelated rows by NaN value
    df_vfl_event = [df_vfl_event[f].dropna(axis=0, how='any').reset_index(drop=True) for f in
                    range(len(path_vfl_event))]
    # rename the columns(variables)
    rename_vfl = {'trials.thisTrialN': 'trials',
                  'text_6.started': 'start',
                  'image_2.started': 'onset',
                  'key_resp.corr': 'accuracy',
                  'trials.thisRepN': 'run',
                  'subject_id': 'participant'}
    df_vfl_event = [df_vfl_event[f].rename(columns=rename_vfl) for f in range(len(path_vfl_event))]
    # set some new values for all first session
    df_vfl_event[0]['session'] = 1
    df_vfl_event[0]['run'] = df_vfl_event[0]['run'] + 1
    # if one file from one subject
    if len(path_vfl_event) == 1:
        df_vfl_event = df_vfl_event[0]
        # calculate onset time
        onset = [df_vfl_event[(df_vfl_event['run'] == i)]['onset'].sub(
            float(df_vfl_event[(df_vfl_event['run'] == i)
                               & (df_vfl_event['trials'] == 0)]['start'])) for i in np.arange(1, 5)]
        # to the dataframe
        onset = pd.concat([onset[0], onset[1], onset[2], onset[3]], axis=0)
        df_vfl_event['onset'] = onset
    # if there are two files from one subject
    elif len(path_vfl_event) == 2:
        # set some new values for the second session
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
            elif f == 1:
                onset2 = [df_vfl_event[f][(df_vfl_event[f]['run'] == i)]['onset'].sub(
                    float(df_vfl_event[f][(df_vfl_event[f]['run'] == i)
                                          & (df_vfl_event[f]['trials'] == 0)]['start'])) for i in np.arange(3, 5)]
                onset = pd.concat([onset2[0], onset2[1]], axis=0)
                df_vfl_event[f]['onset'] = onset
        temp = pd.concat([df_vfl_event[0], df_vfl_event[1]])
        df_vfl_event = temp
    # set some new values
    df_vfl_event['duration'] = 1
    df_vfl_event['task'] = 'VFL'
    # the fold for cross validation during decoding
    df_vfl_event['fold'] = df_vfl_event.apply(lambda x: cvfold(x['trials'], x['run']), axis=1)
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
        write_path = opj(path_behavior_output, sub, file_name)
        df_vfl_event_temp[f].to_csv(write_path, sep='\t')

# In[3]:LEARNING AND TEST
'''
======================================
LEARNING AND TEST
======================================
'''
for subject in subject_list:
    # test for the specific subject
    # subject = subject_list[31]
    # get the sub-*
    sub = "sub-%s" % subject
    path_behavior_learning = opj(path_behavior_input, 'learning', 'Sequence_learning_%s_*.csv' % subject)
    # create the path to the data file
    path_learning_event = sorted(glob.glob(path_behavior_learning), key=lambda f: os.path.basename(f))
    # read the learning and test trial data
    df_learning_event = [pd.read_csv(f, sep=',') for f in path_learning_event]
    # select useful columns
    df_learning_event = [
        df_learning_event[f][['test_trial.thisRepN', 'block.thisRepN', 'trial.thisRepN', 'miniblock.thisN',
                              'probe_target.started', 'probe_marker', 'test_marker',
                              'correct_label', 'test_resp.keys', 'test_resp.corr',
                              'text_6.started', 'subject_id'
                              ]] for f in range(len(path_learning_event))]
    # rename the columns(variables)
    rename_learning = {'block.thisRepN': 'run',
                       'test_trial.thisRepN': 'trials',
                       'trial.thisRepN': 'trial_start',
                       'miniblock.thisN': 'trial_start2',
                       'text_6.started': 'start',
                       'probe_marker': 'Marker',
                       'probe_target.started': 'onset',
                       'test_marker': 'TestMarker',
                       'correct_label': 'reference_answer',
                       'test_resp.keys': 'response',
                       'test_resp.corr': 'accuracy',
                       'subject_id': 'participant'}
    df_learning_event = [df_learning_event[f].rename(columns=rename_learning) for f in range(len(path_learning_event))]
    # set some new values for all first session
    df_learning_event[0]['session'] = 1
    df_learning_event[0]['run'] = df_learning_event[0]['run'] + 1
    # if one file from one subject
    if len(path_learning_event) == 1:
        df_learning_event = df_learning_event[0]
        # calculate onset time
        onset = [df_learning_event[(df_learning_event['run'] == i)]['onset'].sub(
            float(df_learning_event[(df_learning_event['run'] == i)
                                    & (df_learning_event['trial_start'] == 0)
                                    & (df_learning_event['trial_start2'] == 0)]['start'])) for i in np.arange(1, 4)]
        # to the dataframe
        onset = pd.concat([onset[0], onset[1], onset[2]], axis=0)
        df_learning_event['onset'] = onset
    # if there are two files from one subject
    elif len(path_learning_event) == 2:
        # set some new values for the second session
        df_learning_event[1]['session'] = 2
        df_learning_event[1]['run'] = df_learning_event[1]['run'] + 2
        # calculate onset time for each run
        for f in range(len(path_learning_event)):
            if f == 0:
                onset1 = [df_learning_event[f][(df_learning_event[f]['run'] == i)]['onset'].sub(
                    float(df_learning_event[f][(df_learning_event[f]['run'] == i)
                                               & (df_learning_event[f]['trial_start'] == 0)
                                               & (df_learning_event[f]['trial_start2'] == 0)]['start'])) for i in
                    np.arange(1, 2)]
                # onset = pd.concat([onset1[0],onset1[1]],axis=0)
                df_learning_event[f]['onset'] = onset1[0]
            elif f == 1:
                onset2 = [df_learning_event[f][(df_learning_event[f]['run'] == i)]['onset'].sub(
                    float(df_learning_event[f][(df_learning_event[f]['run'] == i)
                                               & (df_learning_event[f]['trial_start'] == 0)
                                               & (df_learning_event[f]['trial_start2'] == 0)]['start'])) for i in
                    np.arange(2, 4)]
                onset = pd.concat([onset2[0], onset2[1]], axis=0)
                df_learning_event[f]['onset'] = onset
        temp = pd.concat([df_learning_event[0], df_learning_event[1]])
        df_learning_event = temp
    # delete unrelated rows
    df_learning_event = df_learning_event.dropna(axis=0, how='any', subset=['Marker']).reset_index(drop=True)
    # set some new values
    df_learning_event['duration'] = 4
    df_learning_event['task'] = 'learning'
    # delete unrelated variables
    del df_learning_event['start']
    del df_learning_event['trial_start']
    del df_learning_event['trial_start2']
    # calculate the accuracy for VFL task
    task_accuracy['learning accuracy'].loc[int(subject) - 1] = (count(df_learning_event['accuracy'])[1]) / (
        len(df_learning_event['accuracy'] == 1))
    task_accuracy['last run"s learning accuracy'].loc[int(subject) - 1] = (count(
        df_learning_event[df_learning_event['run'] == 3]['accuracy'])[1]) / (len(
        df_learning_event[df_learning_event['run'] == 3]['accuracy'] == 1))
    # calculate the trials for VFL task
    task_ntrials['learning'].loc[int(subject) - 1] = (len(df_learning_event['accuracy'] == 1))
    df_learning_event = df_learning_event.reset_index(drop=True)
    # save the dataframe to the csv file
    classfication = df_learning_event['run'].unique()
    df_learning_event_temp = [df_learning_event[df_learning_event['run'] == i] for i in np.arange(1, 4)]
    for f in range(len(df_learning_event_temp)):
        file_name = '%s_task-learning_run-%02.0f_events.tsv' % (subject, df_learning_event_temp[f]['run'].iloc[0])
        write_path = opj(path_behavior_output, sub, file_name)
        df_learning_event_temp[f].to_csv(write_path, sep='\t')

# In[4]:CUE REPLAY
'''
======================================
CUE REPLAY
======================================
'''
accuracy_list = []
for subject in subject_list:
    # test for the specific subject
    # subject = subject_list[1]
    # get the sub-*
    sub = "sub-%s" % subject
    # get the path of VFL files
    path_behavior_re = opj(path_behavior_input, 'replay', 'cue_replay_%s_*.csv' % subject)
    # create the path to the data file
    path_re_event = sorted(glob.glob(path_behavior_re), key=lambda f: os.path.basename(f))
    # read the visual functional localizer trial data
    df_rep_event = [pd.read_csv(f, sep=',') for f in path_re_event]
    for f in range(len(path_re_event)):
        df_rep_event[f].loc[:, 'key_resp_10.stopped'] = df_rep_event[f].loc[:, 'key_resp_10.started'] + df_rep_event[f].loc[:, 'key_resp_10.rt']
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
    # the fold for cross validation during decoding
    df_rep_event['fold'] = divmod(df_rep_event.index.values, 12)[0] + 1
    # save the dataframe to the csv file
    classfication = df_rep_event['run'].unique()
    df_rep_event_temp = [df_rep_event[df_rep_event['run'] == i] for i in np.arange(1, 4)]
    for f in range(len(df_rep_event_temp)):
        file_name = '%s_task-rep_run-%02.0f_events.tsv' % (subject, df_rep_event_temp[f]['run'].iloc[0])
        write_path = opj(path_behavior_output, sub, file_name)
        df_rep_event_temp[f].to_csv(write_path, sep='\t')

# In[5]: BEHAVIORAL ACCURACY
'''
======================================
BEHAVIORAL ACCURACY
======================================
'''
# set the theme
sns.set_theme(style="ticks", color_codes=True)
# visualize the useful subejcts
# lowfmri = [37, 16, 15, 13, 11, 0]
loweeg = [45, 38, 36, 34, 28, 21, 15, 11, 5, 4]
incomplete = [42, 22, 4]
extremeheadmotion = [38, 28, 16, 15, 13, 9, 5, 0]
delete_list = np.unique(np.hstack(
    [loweeg, incomplete, extremeheadmotion]))[::-1]
task_accuracy_1 = task_accuracy.drop(index=delete_list)
subject_num = len(task_accuracy_1)

# visualize the data
plt.figure(dpi=400, figsize=(8, 6))
sns.barplot(data=task_accuracy)
sns.swarmplot(data=task_accuracy, palette='dark:0', alpha=0.35)
plt.xticks(fontsize=10)
plt.title('The behavioral accuracy of VFL and Replay %1.0f subjects' % subject_num)
plt.xlabel("Task")
plt.ylabel("Accuracy")
plt.savefig(opj(path_behavior_output, 'The behavioral accuracy of VFL and Replay %1.0f subjects.png' % subject_num),
            dpi=400, bbox_inches='tight')
plt.show()

file_name = 'behavior accuracy.csv'
write_path = opj(path_behavior_output, file_name)
task_accuracy.to_csv(write_path, sep=',')

# In[6]: CHECK THE CORRESPONDENCE BETWEEN EEG AND FMRI BEHAVIORAL DATA IN CUE REPLAY TASK

# cue_replay_accuracy = pd.DataFrame(accuracy_list)
# cue_replay_accuracy = np.transpose(cue_replay_accuracy)
#
# path_accuracy = '/home/huangqi/Data/BIDS/sourcedata/CUE_replay_ACC.txt'
# zhibing_accuracy = pd.read_csv(path_accuracy, sep=',', header=None)
# judge = np.full([96, 50], np.nan)
#
# for j in range(50):
#     for i in range(96):
#         judge[i, j] = (zhibing_accuracy.iloc[i, j] == cue_replay_accuracy.iloc[i, j])
