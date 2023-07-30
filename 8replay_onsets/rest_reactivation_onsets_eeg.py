#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: IDENTIFY EEG-BASED REACTIVATION ONSETS IN REST
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITIVE NEUROSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

# In[IMPORT RELEVANT PACKAGES]:
import os
import warnings
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
from os.path import join as opj
from bids.layout import BIDSLayout
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from pandas.core.common import SettingWithCopyWarning
from replay_onsets_functions import (confound_tr_rest, pre_rest_start, post_rest_start, spm_hrf)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# In[SETUP NECESSARY PATHS ETC]:
path_bids = opj(os.getcwd().split('fmrireplay')[0], 'fmrireplay')
path_onset_pre = opj(path_bids, 'sourcedata', 'replay_onsets', 'pre_rest_replay_onsets')
path_onset_post = opj(path_bids, 'sourcedata', 'replay_onsets', 'post_rest_replay_onsets')
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
subject = None
tr = 1.3


# In[RESTING STATE REACTIVATION PROBABILITY ONSETS]:
def conv(subject):
    print('=====sub-%s start!=====' % subject)
    # set the subject string
    sub = 'sub-%s' % subject
    # get the absolute time start point for pre resting and post resting state
    abs_start_pre = pre_rest_start(path_behavior, subject)
    abs_start_post = post_rest_start(path_behavior, subject)
    # get the confound file to calculate the TR scans
    scan_tr_pre = confound_tr_rest(path_fmriprep, subject)[0]
    scan_tr_post = confound_tr_rest(path_fmriprep, subject)[1]
    # load the onset data file
    pre_react = pd.read_csv(opj(path_onset_pre, 'sub_%s_preRest_2_prob.txt' % subject), sep=',')
    post_react = pd.read_csv(opj(path_onset_post, 'sub_%s_postRest_2_prob.txt' % subject), sep=',')

    # pack the start time point and scan numbers
    abs_start = [abs_start_pre, abs_start_post]
    scan_tr = [scan_tr_pre, scan_tr_post]

    # summarize all the states' probability
    pre_react['probability'] = np.sum((pre_react['A'], pre_react['B'],
                                       pre_react['C'], pre_react['D']), axis=0)
    post_react['probability'] = np.sum((post_react['A'], post_react['B'],
                                        post_react['C'], post_react['D']), axis=0)

    # EEG-based reactivation probability onsets in rest
    consname = {0: {'cons': 'pre'}, 1: {'cons': 'post'}, }
    for i, file in enumerate([pre_react, post_react]):
        # set the relative replay event onset
        file['onset'] = file.loc[:, ['points_time']] / 1000 + abs_start[i]
        file['onset'] = file['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        file = file.loc[:, ['onset', 'probability']]
        # get the maximum time point in a run
        max_time = int(scan_tr[i]) * tr
        # create the time frame list
        scan_list = pd.DataFrame(np.linspace(0, max_time, int(max_time * 100) + 1), columns=['onset'])
        scan_list['onset'] = scan_list['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        scan_list['probability'] = 0
        # concatenate scan list and onset_run_event
        temp = pd.concat((file, scan_list), axis=0)
        tempa = temp.drop_duplicates(subset='onset').sort_values(by='onset').reset_index(drop=True)
        # spm HRF kernel
        hrfkernel = spm_hrf(tr=1.3, oversampling=100)
        # convolve the HRF kernel and reactivation probability event
        tempa['conv_reg'] = np.array(np.convolve(tempa['probability'], hrfkernel)[:tempa['probability'].size])
        # interpolation
        f = interp1d(tempa['onset'], tempa['conv_reg'])
        # get the specific time point BOLD signal (middle time point)
        slice_time_ref = 0.5
        start_time = slice_time_ref * tr
        end_time = (scan_tr[i] - 1 + slice_time_ref) * tr
        frame_times = np.linspace(start_time, end_time, scan_tr[i])
        # resample BOLD signal
        resample_onset = pd.DataFrame(f(frame_times).T, columns=['HRF_onset_TR'])
        # save the data
        file_name = '%s_%s_rest_reactivation_events.tsv' % (sub, consname[i]['cons'])
        write_path = opj(path_output, 'sub_level', sub)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        resample_onset.to_csv(opj(write_path, file_name), sep='\t', index=0)

    # average the probability within whole sessions for paired t-test
    pre_react_mean = np.mean(pre_react['probability'], axis=0)
    post_react_mean = np.mean(post_react['probability'], axis=0)

    return pre_react_mean, post_react_mean


react_prob = Parallel(n_jobs=64)(delayed(conv)(subject) for subject in sub_list)

# In[THE DIFFERENCE OF PRE- AND POST-REST REACTIVATION PROBABILITY]:
react_prob = pd.DataFrame(react_prob)
# calculate the difference of reactivation probability between pre rest and post rest
react_prob['difference'] = react_prob.loc[:, 1] - react_prob.loc[:, 0]
# calculate the standardization score for difference values
react_prob['z-difference'] = stats.zscore(react_prob['difference'])
# save the paired t-test between post and pre rest excluded outlier
t_test_results = pg.ttest(x=react_prob.loc[1:34, 1], y=react_prob.loc[1:34, 0], paired=True, alternative='greater')
# save the reactivation probability
react_prob.to_csv(opj(path_output, 'reactivation_prob.csv'), sep=',', index=0)
