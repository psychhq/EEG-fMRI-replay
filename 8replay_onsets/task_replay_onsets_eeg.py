#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: IDENTIFY REPLAY ONSETS IN TASK
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITIVE NEUROSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

# In[IMPORT RELEVANT PACKAGES]:
import os
import warnings
import numpy as np
import pandas as pd
from os.path import join as opj
from bids.layout import BIDSLayout
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from pandas.core.common import SettingWithCopyWarning
from replay_onsets_functions import (confound_tr_task, behavior_start, spm_hrf)

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# In[SETUP NECESSARY PATHS ETC]:
# path to the project root:
path_bids = opj(os.getcwd().split('fmrireplay')[0], 'fmrireplay')
path_onset = opj(path_bids, 'sourcedata', 'replay_onsets', 'cue_replay_onsets')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_behavior = opj(path_bids, 'sourcedata', 'behavior-rawdata')
path_output = opj(path_bids, 'derivatives', 'replay_onsets')

# define the subject id
layout = BIDSLayout(path_bids)
sub_list = sorted(layout.get_subjects())
# delete the subjects
loweeg = [45, 38, 36, 34, 28, 21, 15, 11, 5, 4, 2]
incomplete = [42, 22, 4]
extremeheadmotion = [38, 28, 16, 15, 13, 9, 5, 0]
delete_list = np.unique(
    np.hstack([loweeg, incomplete, extremeheadmotion]))[::-1]
[sub_list.pop(i) for i in delete_list]
# time of repetition
tr = 1.3


# In[convolve the replay onsets with HRF]:
def replay_eeg(subject):
    print('=====sub-%s start!=====' % subject)
    # set the subject string
    sub = 'sub-%s' % subject
    # get the absolute time start point
    abs_start = behavior_start(path_behavior, subject)
    # get the confound file to calculate the TR scans
    scan_tr = confound_tr_task(path_fmriprep, subject)
    # load the onset data file
    onset_path = opj(path_onset, 'sub_%s_replayonsets_raw.txt' % subject)
    onset_file = pd.read_csv(onset_path, sep=',')
    onset_file.loc[:, ['probability']] = np.nan
    onset_file['probability'] = onset_file['prob_fwd_lag30'] + onset_file['prob_bwd_lag30']
    # set the relative replay event onset
    onset_file.loc[:, ['onset']] = np.nan
    onset_file['onset'] = onset_file.loc[:, ['points_time']] / 1000
    onset_file['onset'] = onset_file['onset'].apply(
        lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
    # devide replay event into different files by runs
    class_run = onset_file['points_runNum'].unique()
    class_run.sort()
    onset_run_data = [onset_file[onset_file['points_runNum'] == i] for i in class_run]
    # rearrange the replay onset event
    onset_run_event = [onset_run_data[i].loc[:, ['onset', 'probability']] for i in range(len(onset_run_data))]
    for i in range(len(onset_run_event)):
        onset_run_event[i]['onset'] = onset_run_event[i]['onset'] + abs_start[i]
        onset_run_event[i]['onset'] = onset_run_event[i]['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        # get the maximum time point in a run
        max_time = int(scan_tr[i]) * tr
        # create the time frame list
        scan_list = pd.DataFrame(np.linspace(0, max_time, int(max_time * 100) + 1), columns=['onset'])
        scan_list['onset'] = scan_list['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        scan_list['probability'] = 0
        # concatenate scan list and onset_run_event
        temp = pd.concat((onset_run_event[i], scan_list), axis=0)
        tempa = temp.drop_duplicates(subset='onset').sort_values(by='onset').reset_index(drop=True)
        # spm HRF kernel
        hkernel = spm_hrf(tr=1.3, oversampling=100)
        # convolve the HRF kernel and probability event
        tempa['conv_reg'] = np.array(np.convolve(tempa['probability'], hkernel)[:tempa['probability'].size])
        # interpolation
        f = interp1d(tempa['onset'], tempa['conv_reg'])
        # get the specific time point BOLD signal
        slice_time_ref = 0.5
        start_time = slice_time_ref * tr
        end_time = (scan_tr[i] - 1 + slice_time_ref) * tr
        frame_times = np.linspace(start_time, end_time, scan_tr[i])
        # resample BOLD signal
        resample_onset = pd.DataFrame(f(frame_times).T, columns=['HRF_onset_TR'])
        # save the data
        file_name = '%s_cue_replay_onsets_run-%02.0f_events.tsv' % (sub, (i + 1))
        write_path = opj(path_output, 'sub_level', sub)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        resample_onset.to_csv(opj(write_path, file_name), sep='\t', index=0)


Parallel(n_jobs=64)(delayed(replay_eeg)(subject) for subject in sub_list)
