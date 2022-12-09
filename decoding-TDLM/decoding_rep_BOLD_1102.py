#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: FMRI-DECODING
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITVIE NERUOSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

# In[IMPORT RELEVANT PACKAGES]:
import glob
import os
import logging
import time
from os.path import join as opj
import copy
import numpy as np
from nilearn import image, masking
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import warnings
from scipy.stats import pearsonr
from bids.layout import BIDSLayout
from joblib import Parallel, delayed

# import seaborn as sns
# import collections
# DEAL WITH ERRORS, WARNING ETC
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


# In[DEFINITION OF ALL FUNCTIONS]:

# set the classlabel based on marker
def classlabel(Marker):
    if Marker == 11 or Marker == 21 or Marker == 41 or Marker == 51:
        return 'girl'
    elif Marker == 12 or Marker == 22 or Marker == 42 or Marker == 52:
        return 'scissors'
    elif Marker == 13 or Marker == 23 or Marker == 43:
        return 'zebra'
    elif Marker == 14 or Marker == 24 or Marker == 44:
        return 'banana'


class TaskData:
    def __init__(
            self,
            events,
            confounds,
            task,
            name,
            num_vol_run_1,
            num_vol_run_2,
            num_vol_run_3,
            num_vol_run_4,
            # trial_type,
            bold_delay,
            interval,
    ):
        # define name of the task data subset:
        self.name = name
        # define the task condition the task data subset if from:
        self.task = task
        # define the delay (in seconds) by which onsets are moved:
        self.bold_delay = bold_delay
        # define the number of TRs from event onset that should be selected:
        self.interval = interval
        # define the repetition time (TR) of the mri data acquisition:
        self.t_tr = 1.3

        # select events: upright stimulus, correct answer trials only:
        if task == "VFL":
            self.events = events.loc[
                          (events["task"] == 'VFL') &
                          (events["accuracy"] == 1),
                          :,
                          ]
            # define the number of volumes per task run:
            self.num_vol_run = [[], [], [], []]
            self.num_vol_run[0] = num_vol_run_1
            self.num_vol_run[1] = num_vol_run_2
            self.num_vol_run[2] = num_vol_run_3
            self.num_vol_run[3] = num_vol_run_4

        if task == 'REP':
            self.events = events.loc[
                          (events["task"] == 'REP')
                          & (events["accuracy"] == 1),
                          :,
                          ]
            # define the number of volumes per task run:
            self.num_vol_run = [[], [], []]
            self.num_vol_run[0] = num_vol_run_1
            self.num_vol_run[1] = num_vol_run_2
            self.num_vol_run[2] = num_vol_run_3

        self.confounds = confounds
        # reset the indices of the data frame:
        self.events.reset_index(drop=True, inplace=True)
        # sort all values by session and run:
        self.events.sort_values(by=["run"])
        # call further function upon initialization:
        self.define_trs_pre()
        self.rmheadmotion()
        self.define_trs_post()
        self.get_stats()

    def define_trs_pre(self):
        # import relevant functions:
        import numpy as np
        # select all events onsets:
        self.event_onsets = self.events["onset"]
        # add the selected delay to account for the peak of the hrf
        self.bold_peaks = (self.event_onsets + (self.bold_delay * self.t_tr))  # delay is the time, not the TR
        # divide the expected time-point of bold peaks by the repetition time:
        self.peak_trs = self.bold_peaks / self.t_tr
        if self.task == 'VFL' or self.task == 'learning' or self.task == 'REP':
            # add the number of run volumes to the tr indices:
            run_volumes = [sum(self.num_vol_run[0:int(self.events["run"].iloc[row] - 1)])
                           for row in range(len(self.events["run"]))]
            run_volumes = pd.Series(run_volumes)
        # add the number of volumes of each run:
        trs = round(self.peak_trs + run_volumes)  # (run-1)*run_trs + this_run_peak_trs
        # copy the relevant trs as often as specified by the interval:
        a = np.transpose(np.tile(trs, (self.interval, 1)))
        # create same-sized matrix with trs to be added:
        b = np.full((len(trs), self.interval), np.arange(self.interval))
        # assign the final list of trs:
        self.trs = np.array(np.add(a, b).flatten(), dtype=int)
        # save the TRs of the stimulus presentations
        self.stim_trs = round(self.event_onsets / self.t_tr + run_volumes)

    def rmheadmotion(self):
        self.fd = self.confounds.loc[:, 'framewise_displacement'].fillna(0).reset_index(drop=True)
        self.extras = np.array(self.fd[self.fd.values > 0.2].index)
        self.rmheadmotions = np.zeros(len(self.stim_trs))
        for i in range(len(self.stim_trs)):
            if any(extra in np.arange(self.stim_trs[i], (self.stim_trs[i] + self.interval)) for extra in self.extras):
                self.rmheadmotions[i] = 1
            else:
                self.rmheadmotions[i] = 0
        self.rmheadmotions = self.rmheadmotions.astype(bool)
        self.events_rm = self.events.loc[~self.rmheadmotions, :]
        self.events_rm.reset_index(drop=True, inplace=True)
        # sort all values by session and run:
        self.events_rm.sort_values(by=["run"])

    def define_trs_post(self):
        # import relevant functions:
        import numpy as np
        # select all events onsets:
        self.event_onsets = self.events_rm["onset"]
        # add the selected delay to account for the peak of the hrf
        self.bold_peaks = (self.event_onsets + (self.bold_delay * self.t_tr))  # delay is the time, not the TR
        # divide the expected time-point of bold peaks by the repetition time:
        self.peak_trs = self.bold_peaks / self.t_tr
        if self.task == 'VFL' or self.task == 'learning' or self.task == 'REP':
            # add the number of run volumes to the tr indices:
            run_volumes = [sum(self.num_vol_run[0:int(self.events_rm["run"].iloc[row] - 1)])
                           for row in range(len(self.events_rm["run"]))]
            self.run_volumes = pd.Series(run_volumes)
        # add the number of volumes of each run:
        self.trs_round = round(self.peak_trs + run_volumes)  # (run-1)*run_trs + this_run_peak_trs
        # copy the relevant trs as often as specified by the interval:
        self.a = np.transpose(np.tile(self.trs_round, (self.interval, 1)))
        # create same-sized matrix with trs to be added:
        self.b = np.full((len(self.trs_round), self.interval), np.arange(self.interval))
        # assign the final list of trs:
        self.trs = np.array(np.add(self.a, self.b).flatten(), dtype=int)
        # save the TRs of the stimulus presentations
        self.stim_trs = round(self.event_onsets / self.t_tr + run_volumes)

    def get_stats(self):
        import numpy as np
        self.num_trials = len(self.events_rm)
        self.runs = np.repeat(
            np.array(self.events_rm["run"], dtype=int), self.interval)
        self.trials = np.repeat(
            np.array(self.events_rm["trials"], dtype=int), self.interval)
        self.sess = np.repeat(
            np.array(self.events_rm["session"], dtype=int), self.interval)
        self.stim = np.repeat(
            np.array(self.events_rm["stim_label"], dtype=object), self.interval)
        self.fold = np.repeat(
            np.array(self.events_rm["fold"], dtype=object), self.interval)
        if self.task == 'REP':
            self.marker = np.repeat(
                np.array(self.events_rm["Marker"], dtype=object), self.interval)
        self.stim_trs = np.repeat(np.array(self.stim_trs, dtype=int), self.interval)


def detrend(data):
    from nilearn.signal import clean
    data_detrend = clean(signals=data, t_r=1.3, detrend=True, standardize=False)
    return data_detrend


def standardize(data):
    from nilearn.signal import clean
    data_standardize = clean(signals=data, t_r=1.3, detrend=False, standardize=True)
    return data_standardize


def TransM_cross(x):
    # create an empty matrix
    transition_matrix = np.zeros([2, 2])
    # transition
    for a in range(1):
        transition_matrix[x[a] - 1][x[a + 1] - 1] = 1
    return (transition_matrix)


# detect the autocorrelation between hippocampus and visual cortex
# probability_matrix = np.transpose(bold_cross)
def TDLM_cross(probability_matrix):
    rp = [1, 2]
    # get the transition matrix
    T1 = TransM_cross(rp)
    T2 = np.transpose(T1)
    # probability data
    X1 = np.array(probability_matrix)

    # detect for each timelag
    def Crosscorr(X1, T, l):
        orig = np.dot(X1[0:-(2 * l), :], T)
        proj = X1[l:-l, :]
        ## Scale variance
        corrtemp = np.full(np.size(proj, axis=1), np.nan)
        for iseq in range(np.size(proj, axis=1)):
            if (np.nansum(orig[:, iseq]) != 0) & (np.nansum(proj[:, iseq]) != 0):
                corrtemp[iseq] = pearsonr(orig[:, iseq], proj[:, iseq]).statistic
        sf = np.nanmean(corrtemp)
        return sf

    for l in range(1, maxLag + 1):
        cross[0, l - 1] = Crosscorr(X1=X1, T=T1, l=l)
        cross[1, l - 1] = Crosscorr(X1=X1, T=T2, l=l)


# In[SETUP NECESSARY PATHS ETC]:
# name of the current project:
sub_list = None
subject = None
# path to the project root:
project_name = 'fmrireplay-BOLD'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_code = opj(path_bids, 'code', 'decoding-TDLM')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_masks = opj(path_bids, 'derivatives', 'masks')
path_decoding = opj(path_bids, 'derivatives', 'decoding')
path_behavior = opj(path_bids, 'derivatives', 'behavior')
path_out_cross = opj(path_decoding, 'all_subject_results', 'crosscorrelation')

# define the subject id
layout = BIDSLayout(path_bids)
sub_list = sorted(layout.get_subjects())
# choose the specific subjects
# sub_list = sub_list[0:40]
# lowfmri = [37, 16, 15, 13, 11, 0]
loweeg = [45, 38, 36, 34, 28, 21, 15, 11, 5, 4]
incomplete = [42, 22, 4]
extremeheadmotion = [38, 28, 16, 15, 13, 9, 5, 0]
delete_list = np.unique(
    np.hstack([loweeg, incomplete, extremeheadmotion]))[::-1]
[sub_list.pop(i) for i in delete_list]

# create a new subject 'sub-*' list 
subnum_list = copy.deepcopy(sub_list)
sub_template = ["sub-"] * len(sub_list)
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]
subnum_list = list(map(int, subnum_list))

# In[paramters]
# some parameters
long_interval = 8
# the number of timelags
maxLag = long_interval
# just for plot
nbins = maxLag + 1
# the number of subjects
nSubj = len(sub_list)
# the nan matrix for forward design matrix (2 directions * 8 timelags conditions)
cross = np.full([2, maxLag], np.nan)

# In[for parallel function]:
def decoding(subject):
    # for subject in sub_list:
    # print(subject)
    # In[CREATE PATHS TO OUTPUT DIRECTORIES]:
    path_out = opj(path_decoding, subject)
    path_out_figs = opj(path_out, "plots")
    path_out_data = opj(path_out, "data")
    path_out_logs = opj(path_out, "logs")
    path_out_masks = opj(path_out, "masks")
    path_out_cross = opj(path_decoding, 'all_subject_results', 'crosscorrelation')
    # CREATE OUTPUT DIRECTORIES IF THE DO NOT EXIST YET:
    for path in [path_out_figs, path_out_data, path_out_logs, path_out_masks, path_out_cross]:
        if not os.path.exists(path):
            os.makedirs(path)

    # In[LOAD BEHAVIORAL (THE EVENTS.TSV FILES OF THE SUBJECT) and MRI DATA]:
    # paths to all events files of the current subject:
    path_events_rep = opj(path_behavior, subject, "*rep*events.tsv")
    path_events_rep = sorted(glob.glob(path_events_rep), key=lambda f: os.path.basename(f))
    # import events file and save data in dataframe:
    df_events_rep = pd.concat(
        (pd.read_csv(f, sep="\t") for f in path_events_rep), ignore_index=True
    )
    df_events_rep['stim_label'] = df_events_rep.apply(lambda x: classlabel(x.Marker), axis=1)
    del df_events_rep['Unnamed: 0']
    del df_events_rep['start']

    # comfound data from fMRIprep
    path_confounds_rep = glob.glob(opj(path_fmriprep, subject, 'func', '*replay*confounds_timeseries.tsv'))
    df_confound_rep_list = [pd.read_csv(i, sep="\t") for i in path_confounds_rep]
    df_confound_rep = pd.concat([df_confound_rep_list[0],
                                 df_confound_rep_list[1],
                                 df_confound_rep_list[2]])

    # load the cue replay mri task files:
    path_func_task_rep = opj(path_masks, "smooth", subject, "*", "*task-replay*nii.gz")
    path_func_task_rep = sorted(glob.glob(path_func_task_rep), key=lambda f: os.path.basename(f))
    # load smoothed functional mri data for all task runs:
    data_task_rep = [image.load_img(i) for i in path_func_task_rep]

    # In[DEFINE THE TASK CONDITION]:
    test_rep_long = TaskData(
        events=df_events_rep,
        confounds=df_confound_rep,
        task='REP',
        bold_delay=0,
        interval=long_interval,
        name='test-rep_long',
        num_vol_run_1=np.size(data_task_rep[0].dataobj, axis=3),
        num_vol_run_2=np.size(data_task_rep[1].dataobj, axis=3),
        num_vol_run_3=np.size(data_task_rep[2].dataobj, axis=3),
        num_vol_run_4=0,
    )

    # In[VISUAL CORTEX AND HIPPOCAMPUS BOLD SIGNAL AUTOCORRELATION]:
    tr_onset = np.unique(test_rep_long.stim_trs)

    # load the hippocampus/MTL/entorhinal and visual task mask in cue replay task
    def ROI_BOLD(mask, data_rep, onset, region):
        # load the path of masks:
        path_mask = sorted(glob.glob(
            opj(path_masks, mask, subject, "*", "*task-replay*.nii.gz")
        ), key=lambda f: os.path.basename(f))
        mask_dy = [image.load_img(i)
                   for i in copy.deepcopy(path_mask[0:3])]
        masked_data = [masking.apply_mask(data, mask_dy[i])
                       for (i, data) in enumerate(data_rep)]
        # detrend the BOLD signal in each run
        data_detrend_test = [detrend(i)
                             for i in masked_data]
        # average the BOLD signal in voxel level
        data_detrend_test = [data_detrend_test[i].mean(axis=1)
                             for i in np.arange(len(data_detrend_test))]
        # concat three runs' BOLD signal
        data_detrend_test = np.array(np.hstack(data_detrend_test))
        # data_detrend_test_stand = preprocessing.scale(data_detrend_test)#standardize(data_detrend_test)
        # get each trials' 10 TR
        data_trial = [data_detrend_test[onset[i]:onset[i] + 10, ]
                      for i in range(len(onset))]  # the least duration in a single trial
        data_trial = np.array(data_trial)
        # the relative signal strength
        data = (data_trial / np.max(data_trial))
        # visual_data_max = visual_data/visual_data.max()
        BOLDsignal = np.reshape(data_trial, np.size(data))

        return BOLDsignal

    # load the BOLD data
    MTL_bold = ROI_BOLD('mask_hippocampus', data_task_rep, tr_onset, 'hippocampus')
    visual_bold = ROI_BOLD('mask_visual', data_task_rep, tr_onset, 'visual cortex')
    bold_cross = np.vstack([MTL_bold, visual_bold])

    # calculate the autocorrelation
    TDLM_cross(np.transpose(bold_cross))
    return cross


# In[]:
cross_TDLM = Parallel(n_jobs=64)(delayed(decoding)(subject) for subject in sub_list)

# In[]
# load the output
path_out_all = opj(path_decoding, 'all_subject_results')
cross_TDLM_seq_list = np.array(cross_TDLM)

# calculate the mean and standard deviation
s1mean = np.insert(np.nan, 1, (cross_TDLM_seq_list[:, 0, :] - cross_TDLM_seq_list[:, 1, :]).mean(axis=0))
s1sem = np.insert(np.nan, 1, (cross_TDLM_seq_list[:, 0, :] - cross_TDLM_seq_list[:, 1, :]).std(axis=0) / np.sqrt(nSubj))

# plot the cross-correlation sequenceness
x = np.arange(0, maxLag + 1, 1)
plt.figure(dpi=400, figsize=(10, 5))
plt.xticks(np.arange(len(x)), x, fontsize=10)
plt.plot(x, s1mean, color='darkred', linestyle='-', linewidth=1)
plt.fill_between(x, s1mean - s1sem, s1mean + s1sem,
                 color='lightcoral', alpha=0.5, linewidth=1)
plt.axhline(y=0, color='silver', linestyle='--')
plt.title('Correlation: fwd-bkw: Hippocampus to Visual Cortex')
plt.xlabel('lag (TRs)')
plt.ylabel('fwd minus bkw sequenceness')
plt.savefig(opj(path_out_cross, 'Hippocampus to Visual Cortex.png'),
            dpi=400, bbox_inches='tight')
plt.show()
