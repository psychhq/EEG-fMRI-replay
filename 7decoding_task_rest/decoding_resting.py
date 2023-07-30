# -*- coding: utf-8 -*-
# # SCRIPT: FMRI-DECODING IN RESTING
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # REFERENCED TO LENNART WITTKUHN, SCHUCK NICOLAS, 2021
# # CONTACT: STATE KEY LABORATORY OF COGNITIVE NEUROSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

# In[IMPORT RELEVANT PACKAGES]:

import os
import sys
import copy
import glob
import scipy
import itertools
import warnings
import numpy as np
import pandas as pd
from nilearn.signal import clean
from nilearn import image, masking
from bids.layout import BIDSLayout
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from os.path import join as opj
from scipy.stats import pearsonr, ttest_1samp
from sklearn.linear_model import LogisticRegression
from pandas.core.common import SettingWithCopyWarning
from statsmodels.stats.multitest import multipletests
# DEAL WITH ERRORS, WARNING ETC
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

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


# fMRI event function
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
                          :, ]
            # define the number of volumes per task run: 4 runs for VFL
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
        if self.task == 'VFL' or self.task == 'REP':
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
        if self.task == 'VFL' or self.task == 'REP':
            # add the number of run volumes to the tr indices:
            run_volumes = [sum(self.num_vol_run[0:int(self.events_rm["run"].iloc[row] - 1)])
                           for row in range(len(self.events_rm["run"]))]
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

    def get_stats(self):
        import numpy as np
        self.num_trials = len(self.events_rm)
        self.runs = np.repeat(
            np.array(self.events_rm["run"], dtype=int), self.interval)
        self.trials = np.repeat(
            np.array(self.events_rm["trials"], dtype=int), self.interval)
        self.sess = np.repeat(
            np.array(self.events_rm["session"], dtype=int), self.interval)
        if self.task == 'REP':
            self.marker = np.repeat(np.array(self.events_rm["Marker"], dtype=object), self.interval)
        self.stim = np.repeat(
            np.array(self.events_rm["stim_label"], dtype=object), self.interval)
        self.stim_trs = np.repeat(np.array(self.stim_trs, dtype=int), self.interval)

    def zscore(self, signals, run_list):
        from nilearn.signal import clean
        import numpy as np
        # get boolean indices for all run indices in the run list:
        run_indices = np.isin(self.runs, list(run_list))
        # standardize data all runs in the run list:
        self.data_zscored = clean(
            signals=signals[self.trs[run_indices]],
            t_r=1.3,
            detrend=False,
            standardize=True)

    def predict(self, clf, run_list):
        # import packages:
        import pandas as pd
        # get classifier class predictions:
        pred_class = clf.predict(self.data_zscored)
        # get classifier probabilistic predictions:
        pred_proba = clf.predict_proba(self.data_zscored)
        # get the classes of the classifier:
        classes_names = clf.classes_
        # get boolean indices for all run indices in the run list:
        run_indices = np.isin(self.runs, list(run_list))
        # create a dataframe with the probabilities of each class:
        df = pd.DataFrame(pred_proba, columns=classes_names)
        # get the number of predictions made:
        num_pred = len(df)
        # get the number of trials in the test set:
        num_trials = int(num_pred / self.interval)
        # add the predicted class label to the dataframe:
        df["pred_label"] = pred_class
        # add the condition marker to the dataframe:
        df['marker'] = self.marker[run_indices]
        # add the true stimulus label to the dataframe:
        df["stim"] = self.stim[run_indices]
        # add the volume number (TR) to the dataframe:
        df["tr"] = self.trs[run_indices]
        # add the sequential TR to the dataframe:
        df["seq_tr"] = np.tile(np.arange(1, self.interval + 1), num_trials)
        # add the counter of trs on which the stimulus was presented
        df["stim_tr"] = self.stim_trs[run_indices]
        # add the trial number to the dataframe:
        df["trials"] = self.trials[run_indices]
        # add the run number to the dataframe:
        df["run"] = self.runs[run_indices]
        # add the session number to the dataframe:
        df["session"] = self.sess[run_indices]
        # add the participant id to the dataframe:
        df["participant"] = np.repeat(self.events_rm["participant"].unique(), num_pred)
        # add the name of the classifier to the dataframe:
        df["test_set"] = np.repeat(self.name, num_pred)
        # return the dataframe:
        return df


def detrend(data):
    from nilearn.signal import clean
    data_detrend = clean(signals=data, t_r=1.3,
                         detrend=True, standardize=False)
    return data_detrend


def standardize(data):
    from nilearn.signal import clean
    data_standardize = clean(signals=data, t_r=1.3,
                             detrend=False, standardize=True)
    return data_standardize


def melt_df(df, melt_columns):
    # save the column names of the dataframe in a list:
    column_names = df.columns.tolist()
    # remove the stimulus classes from the column names;
    id_vars = [x for x in column_names if x not in melt_columns]
    # melt the dataframe creating one column with value_name and var_name:
    df_melt = pd.melt(df, value_name="probability",
                      var_name="class", id_vars=id_vars)
    # return the melted dataframe:
    return df_melt


# combine all masks from the feature selection by intersection:
def multimultiply(arrays):
    import copy
    # start with the first array:
    array_union = copy.deepcopy(arrays[0].astype(np.int32))
    # loop through all arrays
    for i in range(len(arrays)):
        # multiply every array with all previous array
        array_union = np.multiply(
            array_union, copy.deepcopy(arrays[i].astype(np.int32)))
    # return the union of all arrays:
    return array_union


def stim_index(x):
    if x == 'girl':
        return 1
    elif x == 'scissors':
        return 2
    elif x == 'zebra':
        return 3
    elif x == 'banana':
        return 4


def TransM(x):
    # create an empty matrix
    transition_matrix = np.zeros([4, 4])
    # transition
    for a in range(3):
        transition_matrix[x[a] - 1][x[a + 1] - 1] = 1
    return (transition_matrix)


def TransM_cross(x):
    # create an empty matrix
    transition_matrix = np.zeros([2, 2])
    # transition
    for a in range(1):
        transition_matrix[x[a] - 1][x[a + 1] - 1] = 1
    return (transition_matrix)


def matlab_percentile(x, p):
    p = np.asarray(p, dtype=float)
    n = len(x)
    p = (p - 50) * n / (n - 1) + 50
    p = np.clip(p, 0, 100)
    return np.percentile(x, p)


# detect the autocorrelation between hippocampus and visual cortex
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

    cross = np.full([2, maxLag_cross], np.nan)
    for l in range(1, maxLag_cross + 1):
        cross[0, l - 1] = Crosscorr(X1=X1, T=T1, l=l)
        cross[1, l - 1] = Crosscorr(X1=X1, T=T2, l=l)

    return cross


# temp_pred_prob is a function to get the specific condition's dynamic probability
def temp_pred_prob_rest(data, condition, picList, path_data, subject):
    # filter the data
    pred_filter = data[(data["class"] != 'other') &
                       (data['run_study'] == condition)]
    pred_filter['class_index'] = pred_filter['class'].map(
        lambda a: stim_index(a))
    pred_filter_sort = pred_filter.sort_values(by=['class_index', 'tr'])
    pred_filter_sort.to_csv(opj(path_data, '%s fMRI %s decoding probability in rest.csv' % (subject, condition)),
                            index=False)
    pred_array = np.array(pred_filter_sort.loc[:, 'probability'])
    # get the probability array
    pred_matrix = np.reshape(pred_array, (int(len(pred_array) / 4), 4), order='F')
    # get the probability matrix (Time * States)
    pred_matrix_seq = [pd.Series(pred_matrix[:, picList[0] - 1]),
                       pd.Series(pred_matrix[:, picList[1] - 1]),
                       pd.Series(pred_matrix[:, picList[2] - 1]),
                       pd.Series(pred_matrix[:, picList[3] - 1])]
    pred_matrix_seq = pd.concat(pred_matrix_seq, axis=1)
    return pred_matrix_seq


# TDLM for cue replay detecting within subject
def TDLM(prob_matrix, all_sequence, condition):
    for i in range(len(all_sequence)):
        # real sequence
        rp = all_sequence[i, :]
        # get the transition matrix
        T1 = TransM(rp)
        T2 = np.transpose(T1)
        # no lag and several lags = nbins
        X1 = np.array(prob_matrix)
        # timelag matrix
        dm = scipy.linalg.toeplitz(X1[:, 0], [np.zeros((nbins, 1))])
        dm = dm[:, 1:]
        # 3 loops for another 3 states
        for k in range(1, 4):
            temp = scipy.linalg.toeplitz(X1[:, k], [np.zeros((nbins, 1))])
            temp = temp[:, 1:]
            dm = np.hstack((dm, temp))
        # the next time point needed to be predicted
        Y = X1
        # build a new framework for first GLM betas
        betas_1GLM = np.full([nstates * maxLag, nstates], np.nan)
        # detect for each timelag
        for l in range(maxLag):
            temp_zinds = np.array(range(0, nstates * maxLag, maxLag)) + l
            # First GLM
            design_mat_1 = np.hstack(
                (dm[:, temp_zinds], np.ones((len(dm[:, temp_zinds]), 1))))
            temp = np.dot(np.linalg.pinv(design_mat_1), Y)
            betas_1GLM[temp_zinds, :] = temp[:-1, :]
        betasnbins64 = np.reshape(
            betas_1GLM, [maxLag, np.square(nstates)], order="F")
        # Second GLM
        design_mat_2 = np.transpose(np.vstack((np.reshape(T1, 16, order="F"),
                                               np.reshape(T2, 16, order="F"),
                                               np.reshape(np.eye(nstates), 16, order="F"),
                                               np.reshape(np.ones([nstates, nstates]), 16, order="F"))))
        betas_2GLM = np.dot(np.linalg.pinv(design_mat_2),
                            np.transpose(betasnbins64))
        # four different design matrices for regressor multiple to the temporal data
        # linear regression for the backward and forward replay
        if condition == 'pre_rest':
            sequenceness[0, 0, i, 1:] = betas_2GLM[0, :]  # fwd condition-fwd replay
            sequenceness[0, 1, i, 1:] = betas_2GLM[1, :]  # fwd condition-bwd replay
        elif condition == 'post_rest':
            sequenceness[1, 0, i, 1:] = betas_2GLM[0, :]  # bwd condition-fwd replay
            sequenceness[1, 1, i, 1:] = betas_2GLM[1, :]  # bwd condition-bwd replay


def TDLM_fig_each(condition, direction, data, permutation, path_data, path_fig, subject):
    if direction == 'fwd':
        plotcolor = 'darkred'
        title = '%s fwd replay in  %s rest ' % (subject, condition)
        filename = '%s fwd replay in %s rest.svg' % (subject, condition)
        datafile = '%s fwd replay in %s rest.csv' % (subject, condition)
    elif direction == 'bwd':
        plotcolor = 'dodgerblue'
        title = '%s bwd replay in  %s rest' % (subject, condition)
        filename = '%s bwd replay in %s rest.svg' % (subject, condition)
        datafile = '%s bwd replay in %s rest.csv' % (subject, condition)

    # export the data files
    data_file = pd.DataFrame({'sequence': data,
                              'permutation': permutation})
    data_file.to_csv(opj(path_data, datafile), index=False)

    # plot the sequenceness
    x = np.arange(0, nbins, 1)
    plt.figure(dpi=400, figsize=(7, 5))
    plt.xticks(range(len(x)), x, fontsize=10)
    plt.plot(x, data, color=plotcolor, linestyle='-', linewidth=1)
    plt.axhline(y=permutation, color='silver', linestyle='--')
    plt.title(title)
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path_fig, filename), format='svg')
    plt.show()


# plot the all averaged subjects' sequenceness
def TDLM_fig(direction, data, permutation, condition, path):
    if direction == 'fwd':
        plotcolor = 'darkred'
        fillcolor = 'lightcoral'
        title = 'fwd replay in  %s rest' % condition
        filename = 'subject mean fwd replay in %s rest.svg' % condition
        datafile = 'fwd replay in %s rest.csv' % condition

    elif direction == 'bwd':
        plotcolor = 'dodgerblue'
        fillcolor = 'lightskyblue'
        title = 'bwd replay in sequential %s rest' % condition
        filename = 'subject mean bwd replay in  %s rest.svg' % condition
        datafile = 'bwd replay in %s rest.csv' % condition

    elif direction == 'fwd-bwd':
        plotcolor = 'dodgerblue'
        fillcolor = 'lightskyblue'
        title = 'fwd-bwd replay in sequential %s rest' % condition
        filename = 'subject mean fwd-bwd replay in  %s rest.svg' % condition
        datafile = 'fwd-bwd replay in %s rest.csv' % condition

    # export the data files
    s1mean = data.mean(axis=0)
    s1std = data.std(axis=0)
    s1sem = data.std(axis=0) / np.sqrt(len(data))
    data_file = pd.DataFrame({'mean_sequence': s1mean,
                              'std_sequence': s1std,
                              'permutation': permutation})
    data_file.to_csv(opj(path, datafile), index=False)

    # plot the sequenceness
    x = np.arange(0, nbins, 1)
    plt.figure(dpi=400, figsize=(7, 5))
    plt.xticks(range(len(x)), x, fontsize=10)
    plt.plot(x, s1mean, color=plotcolor, linestyle='-', linewidth=1)
    plt.fill_between(x, s1mean - s1sem, s1mean + s1sem,
                     color=fillcolor, alpha=0.5, linewidth=1)
    plt.axhline(y=permutation, color='silver', linestyle='--')
    plt.title(title)
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path, filename),
                format='svg')
    plt.show()


# In[SETUP NECESSARY PATHS ETC]:

# path to the project root:
path_bids = opj(os.getcwd().split('fmrireplay')[0], 'fmrireplay')
path_code = opj(path_bids, 'code', 'decoding_task_rest')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_masks = opj(path_bids, 'derivatives', 'masks')
path_level1_vfl = opj(path_bids, "derivatives", "glm-vfl", "l1pipeline")
path_decoding = opj(path_bids, 'derivatives', 'decoding', 'sub_level')
path_behavior = opj(path_bids, 'derivatives', 'behavior', 'sub_level')
path_out_rest = opj(path_bids, 'derivatives', 'decoding', 'rest_decoding')
path_out_cross = opj(path_bids, 'derivatives', 'decoding', 'cross_correlation')
for path in [path_decoding, path_behavior, path_out_rest]:
    if not os.path.exists(path):
        os.makedirs(path)

# load the learning sequence
learning_sequence = pd.read_csv(opj(path_bids, 'sourcedata', 'sequence.csv'), sep=',')

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

# create a new subject 'sub-*' list
subnum_list = copy.deepcopy(sub_list)
sub_template = ["sub-"] * len(sub_list)
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]
subnum_list = list(map(int, subnum_list))
subject = None

# In[LOAD TDLM PARAMETERS]

# the number of VFL runs
n_run = 4
# parameters of TDLM
# the list of all the sequences
uniquePerms = list(itertools.permutations([1, 2, 3, 4], 4))
# the number of sequences
nShuf = len(uniquePerms)
# all possible sequences
all_sequence = np.array(uniquePerms)
# the number of time lags
maxLag = 15
nbins = maxLag + 1
maxLag_cross = 8
# the number of states in decoding models
nstates = 4
# the number of subjects
nSubj = len(sub_list)
# predefine GLM data frame
betas_1GLM = None
betas_2GLM = None

# the nan matrix for forward design matrix (2 directions * 8 timelags conditions)
cross_pre = np.full([2, 8], np.nan)
cross_post = np.full([2, 8], np.nan)
# the nan matrix for pre- and post-resting design matrix
# (2 experimental conditions * 2 replay directions * 24 shuffles * 9 bins)
sequenceness = np.full([2, 2, len(all_sequence), nbins], np.nan)


# In[for parallel function]:
def decoding(subject):
    # In[CREATE PATHS TO OUTPUT DIRECTORIES]:
    path_out = opj(path_decoding, subject)
    path_out_figs = opj(path_out, "plots", 'rest')
    path_out_data = opj(path_out, "data")
    path_out_masks = opj(path_out, "masks")
    # CREATE OUTPUT DIRECTORIES IF THE DO NOT EXIST YET:
    for path in [path_out_figs, path_out_data, path_out_masks]:
        if not os.path.exists(path):
            os.makedirs(path)

    # In[LOAD BEHAVIORAL DATA (THE EVENTS.TSV FILES OF THE SUBJECT)]:

    # load VFL events files
    path_events_vfl = opj(path_behavior, subject, "*vfl*events.tsv")
    path_events_vfl = sorted(glob.glob(path_events_vfl), key=lambda f: os.path.basename(f))
    df_events_vfl = [pd.read_csv(path_events_vfl[f], sep="\t") for f in range(len(path_events_vfl))]
    df_events_vfl = pd.concat([df_events_vfl[0], df_events_vfl[1], df_events_vfl[2], df_events_vfl[3]])
    df_events_vfl['stim_label'] = df_events_vfl.apply(lambda x: classlabel(x.stimMarker), axis=1)
    del df_events_vfl['Unnamed: 0']
    del df_events_vfl['start']

    # load VFL confound files
    path_confounds_vfl = glob.glob(opj(path_fmriprep, subject, 'func', '*vfl*confounds_timeseries.tsv'))
    df_confound_vfl_list = [pd.read_csv(i, sep="\t") for i in path_confounds_vfl]
    df_confound_vfl = pd.concat(
        [df_confound_vfl_list[0], df_confound_vfl_list[1], df_confound_vfl_list[2], df_confound_vfl_list[3]])

    # In[CREATE PATHS TO THE MRI DATA]:

    # load mask files:
    path_mask_task = opj(path_masks, 'mask_vis_mtl', subject, "*", "*task*.nii.gz")
    path_mask_task = sorted(glob.glob(path_mask_task), key=lambda f: os.path.basename(f))
    mask_vis_mtl = [np.asanyarray(image.load_img(i).dataobj) for i in copy.deepcopy(path_mask_task[-4:])]

    # load the visual functional localizer mri task files:
    path_func_task_vfl = opj(path_masks, "smooth", subject, "*", "*task-vfl*nii.gz")
    path_func_task_vfl = sorted(glob.glob(path_func_task_vfl), key=lambda f: os.path.basename(f))
    data_task_vfl = [image.load_img(i) for i in path_func_task_vfl]

    # load the t-maps of the first-level glm:
    path_tmap_vfl = opj(path_level1_vfl, "contrasts", subject, "*", "spmT*.nii")
    path_tmap_vfl = sorted(glob.glob(path_tmap_vfl), key=lambda f: os.path.dirname(f))
    tmaps_vfl = [np.asanyarray(image.load_img(i).dataobj) for i in copy.deepcopy(path_tmap_vfl)]

    # load rest data
    path_rest = opj(path_masks, 'smooth', subject, '*', '*task-rest*.nii.gz')
    path_rest = sorted(glob.glob(path_rest),
                       key=lambda f: os.path.dirname(f))
    data_rest = [image.load_img(i) for i in path_rest]

    # In[FEATURE SELECTION FOR VISUAL FUNCTIONAL LOCALIZER TASK]:

    # FEATURE SELECTION: MASKS THE T-MAPS WITH THE ANATOMICAL MASKS
    # check if any value in the supposedly binary mask is bigger than 1:
    for i in mask_vis_mtl:
        if np.any(i > 1):
            sys.exit("Values > 1 in the anatomical ROI!")
    # get combination of anatomical mask and t-map
    tmaps_masked = [np.multiply(i, j) for (i, j) in zip(copy.deepcopy(mask_vis_mtl), copy.deepcopy(tmaps_vfl))]
    # masked tmap into image like object:
    tmaps_masked_img = [image.new_img_like(i, j) for (i, j) in zip(path_tmap_vfl, copy.deepcopy(tmaps_masked))]

    # FEATURE SELECTION: THRESHOLD THE MASKED T-MAPS
    # set the threshold:
    threshold = 3
    # threshold the masked tmap image:
    tmaps_masked_thresh_img = [image.threshold_img(i, threshold) for i in copy.deepcopy(tmaps_masked_img)]
    # extract data from the thresholded images
    tmaps_masked_thresh = [np.asanyarray(image.load_img(i).dataobj) for i in tmaps_masked_thresh_img]

    # FEATURE SELECTION: BINARIZE THE THRESHOLDED MASKED T-MAPS
    # replace all NaNs with 0:
    tmaps_masked_thresh_bin = [np.where(np.isnan(i), 0, i) for i in copy.deepcopy(tmaps_masked_thresh)]
    # replace all other values with 1:
    tmaps_masked_thresh_bin = [np.where(i > 0, 1, i) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
    # turn the 3D-array into booleans:
    tmaps_masked_thresh_bin = [i.astype(bool) for i in copy.deepcopy(tmaps_masked_thresh_bin)]

    # SAVE THE UNION MASK (get interesction mask from the four-task-run t-tmap threshold masks )
    mask_label = 'union'
    masks_union = multimultiply(
        tmaps_masked_thresh_bin).astype(int).astype(bool)
    masks_union_nii = image.new_img_like(path_func_task_vfl[0], masks_union)
    path_save = opj(
        path_out_masks, '{}_cue_replay_mask-{}.nii.gz'.format(subject, mask_label))
    masks_union_nii.to_filename(path_save)

    # In[DEFINE THE CLASSIFIERS]:

    class_labels = ["girl", "scissors", "zebra", "banana"]
    # create a dictionary with all values as independent instances:
    # see here: https://bit.ly/2J1DvZm
    clf_set = {
        key: LogisticRegression(
            C=1.0,  # Inverse of regularization strength
            penalty="l2",
            multi_class="ovr",
            solver="lbfgs",
            # Algorithm to use in the optimization problem, Stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno
            max_iter=4000,
            class_weight="balanced",
            random_state=42,  # to shuffle the data
        )
        for key in class_labels
    }

    # In[DEFINE THE TASK CONDITION]:
    # for task_bold_delay in [3,4,5,6]:
    train_VFL_peak = TaskData(
        events=df_events_vfl,
        confounds=df_confound_vfl,
        task="VFL",
        bold_delay=3,
        interval=1,
        name="train-VFL_peak",
        num_vol_run_1=np.size(data_task_vfl[0].dataobj, axis=3),
        num_vol_run_2=np.size(data_task_vfl[1].dataobj, axis=3),
        num_vol_run_3=np.size(data_task_vfl[2].dataobj, axis=3),
        num_vol_run_4=np.size(data_task_vfl[3].dataobj, axis=3),
    )

    # In[VISUAL CORTEX AND HIPPOCAMPUS BOLD SIGNAL AUTOCORRELATION]:

    # load the hippocampus and visual task mask in cue replay task
    def ROI_BOLD(mask, subject=subject,
                 data_rep=data_rest):
        # load the path of masks:
        path_mask = sorted(glob.glob(
            opj(path_masks, mask, subject, "*", "*task-rest*.nii.gz")
        ), key=lambda f: os.path.basename(f))
        mask_dy = [image.load_img(i)
                   for i in copy.deepcopy(path_mask[0:2])]
        masked_data = [masking.apply_mask(data, mask_dy[i])
                       for (i, data) in enumerate(data_rep)]
        # detrend the BOLD signal in each run
        data_detrend_test = [detrend(i) for i in masked_data]
        # average the BOLD signal in voxel level
        data_detrend_test = [data_detrend_test[i].mean(axis=1) for i in np.arange(len(data_detrend_test))]
        # the relative signal strength
        BOLDsignal = [(i / np.max(i)) for i in data_detrend_test]

        return BOLDsignal

    # load the BOLD data
    MTL_bold = ROI_BOLD('mask_hippocampus')
    visual_bold = ROI_BOLD('mask_visual')
    bold_cross_pre = np.vstack([MTL_bold[0], visual_bold[0]])
    bold_cross_post = np.vstack([MTL_bold[1], visual_bold[1]])

    # calculate the autocorrelation
    cross_pre = TDLM_cross(np.transpose(bold_cross_pre))
    cross_post = TDLM_cross(np.transpose(bold_cross_post))

    # In[SEPARATE RUN-WISE STANDARDIZATION (Z-SCORING) OF VISUAL FUNCTIONAL LOCALIZER]:
    data_list = []
    rest_interval = 1
    # DETREND AND Z-STANDARDIZED THE RESTING FMRI DATA
    # mask all VFL task runs separately:
    data_task_vfl_masked = [masking.apply_mask(i, masks_union_nii) for i in data_task_vfl]
    # mask all cue replay runs with the averaged feature selection masks:
    data_rest_masked = [masking.apply_mask(i, masks_union_nii) for i in data_rest]
    # detrend each VFL task run separately:
    data_task_vfl_masked_detrend = [clean(i, detrend=True, standardize=False) for i in data_task_vfl_masked]
    # detrend each cue replay run separately:
    data_rest_masked_detrend = [clean(i, detrend=True, standardize=True) for i in data_rest_masked]
    # combine the detrended data of all runs:
    data_task_vfl_masked_detrend = np.vstack(data_task_vfl_masked_detrend)
    # standardized for the all trials in VFL
    train_VFL_peak.zscore(signals=data_task_vfl_masked_detrend, run_list=[1, 2, 3, 4])
    # write session and run labels:
    run_labels = [i.split(subject + "_")[1].split("_space")[0] for i in path_rest]
    # loop through all classifiers in the classifier set:
    for clf_name, clf in clf_set.items():
        # get the example labels for all functional localizer trials:
        train_stim = copy.deepcopy(train_VFL_peak.stim)
        # replace labels for single-label classifiers:
        if clf_name in class_labels:
            # replace all other labels with other
            train_stim = ["other" if x != clf_name else x for x in train_stim]
            # turn into a numpy array
            train_stim = np.array(train_stim, dtype=object)
        # train the classifier
        clf.fit(train_VFL_peak.data_zscored, train_stim)
        # classifier prediction: predict on test data and save the data:
        pred_rest_class = [clf.predict(i) for i in data_rest_masked_detrend]
        pred_rest_proba = [clf.predict_proba(i) for i in data_rest_masked_detrend]
        # get the class names of the classifier:
        classes_names = clf.classes_
        # save classifier predictions on resting state scans
        for t, name in enumerate(pred_rest_proba):
            # create a dataframe with the probabilities of each class:
            df_rest_pred = pd.DataFrame(
                pred_rest_proba[t], columns=classes_names)
            # get the number of predictions made:
            num_pred = len(df_rest_pred)
            # get the number of trials in the test set:
            num_trials = int(num_pred / rest_interval)
            # add the predicted class label to the dataframe:
            df_rest_pred['pred_label'] = pred_rest_class[t]
            # add the true stimulus label to the dataframe:
            df_rest_pred['stim'] = np.full(num_pred, np.nan)
            # add the volume number (TR) to the dataframe:
            df_rest_pred['tr'] = np.arange(1, num_pred + 1)
            # add the sequential TR to the dataframe:
            df_rest_pred['seq_tr'] = np.arange(1, num_pred + 1)
            # add the trial number to the dataframe:
            df_rest_pred['trial'] = np.tile(np.arange(1, rest_interval + 1), num_trials)
            # add the run number to the dataframe:
            df_rest_pred['run_study'] = np.repeat(run_labels[t], num_pred)
            # add the inter trial interval to the dataframe:
            df_rest_pred['tITI'] = np.tile('rest', num_pred)
            # add the participant id to the dataframe:
            df_rest_pred['id'] = np.repeat(df_events_vfl['participant'].unique(), num_pred)
            # add the name of the classifier to the dataframe:
            df_rest_pred['test_set'] = np.repeat('rest', num_pred)
            # add the name of the classifier to the dataframe;
            df_rest_pred['classifier'] = np.repeat(clf_name, num_pred)
            # add a label that indicates the mask / training regime:
            df_rest_pred['mask'] = np.repeat(mask_label, len(df_rest_pred))
            # melt the data frame:
            df_pred_melt = melt_df(df=df_rest_pred, melt_columns=train_stim)
            # append dataframe to list of dataframe results:
            data_list.append(df_pred_melt)

    # In[TDLM FOR EACH SUBJECT]:
    # load the sequence list
    x = learning_sequence[learning_sequence['subject'] == subject].iloc[0]
    picList = list(map(int, list((x[1:5]))))
    # temporal prediction probability
    pred_list = copy.deepcopy(data_list)
    pred_append = pd.concat([pred_list[i] for i in range(len(pred_list))], axis=0)

    pred_pre = temp_pred_prob_rest(pred_append, 'task-rest_run-1', picList, path_out_data, subject)
    pred_post = temp_pred_prob_rest(pred_append, 'task-rest_run-2', picList, path_out_data, subject)
    # detect the sequenceness for two resting state sessions
    TDLM(prob_matrix=pred_pre, all_sequence=all_sequence, condition='pre_rest')
    TDLM(prob_matrix=pred_post, all_sequence=all_sequence, condition='post_rest')

    # In[PLOT the FIGURE]
    # (2 experimental conditions * 2 replay directions * 24 shuffles * 9 bins)
    # pre resting
    pre_fwd_sequence = sequenceness[0, 0, 0, :]
    pre_fwd_permutation = matlab_percentile(np.amax(abs(sequenceness[0, 0, 1:, 1:]), axis=1), 95)

    pre_bwd_sequence = sequenceness[0, 1, 0, :]
    pre_bwd_permutation = matlab_percentile(np.amax(abs(sequenceness[0, 1, 1:, 1:]), axis=1), 95)

    TDLM_fig_each('pre_rest', 'fwd', pre_fwd_sequence, pre_fwd_permutation, path_out_data, path_out_figs, subject)
    TDLM_fig_each('pre_rest', 'bwd', pre_bwd_sequence, pre_bwd_permutation, path_out_data, path_out_figs, subject)

    # post resting
    post_fwd_sequence = sequenceness[1, 0, 0, :]
    post_fwd_permutation = matlab_percentile(np.amax(abs(sequenceness[1, 0, 1:, 1:]), axis=1), 95)

    post_bwd_sequence = sequenceness[1, 1, 0, :]
    post_bwd_permutation = matlab_percentile(np.amax(abs(sequenceness[1, 1, 1:, 1:]), axis=1), 95)

    TDLM_fig_each('post_rest', 'fwd', post_fwd_sequence, post_fwd_permutation, path_out_data, path_out_figs, subject)
    TDLM_fig_each('post_rest', 'bwd', post_bwd_sequence, post_bwd_permutation, path_out_data, path_out_figs, subject)

    # post-pre resting
    post_pre_fwd_sequence = (sequenceness[1, 0, 0, :] - sequenceness[0, 0, 0, :])
    post_pre_fwd_permutation = matlab_percentile(
        np.amax(abs(sequenceness[1, 0, 1:, 1:] - sequenceness[0, 0, 1:, 1:]), axis=1), 95)

    post_pre_bwd_sequence = (sequenceness[1, 1, 0, :] - sequenceness[0, 1, 0, :])
    post_pre_bwd_permutation = matlab_percentile(
        np.amax(abs(sequenceness[1, 1, 1:, 1:] - sequenceness[0, 1, 1:, 1:]), axis=1), 95)

    # plot the sequenceness
    TDLM_fig_each('post_pre', 'fwd', post_pre_fwd_sequence, post_pre_fwd_permutation, path_out_data, path_out_figs,
                  subject)
    TDLM_fig_each('post_pre', 'bwd', post_pre_bwd_sequence, post_pre_bwd_permutation, path_out_data, path_out_figs,
                  subject)

    return cross_pre, cross_post, sequenceness


# In[PARALLEL FUNCTION]:

sequenceness_lists = Parallel(n_jobs=64)(delayed(decoding)(subject) for subject in sub_list)

# In[CROSS CORRELATION]:
cross_pre_list = np.array([sequenceness_lists[i][0] for i in range(len(sub_list))])
cross_post_list = np.array([sequenceness_lists[i][1] for i in range(len(sub_list))])

for (con, cross_bold) in zip(['pre', 'post'], [cross_pre_list, cross_post_list]):
    cc_df = pd.DataFrame((cross_bold[:, 0, :] - cross_bold[:, 1, :]))
    # perform one-sample t-test for each column
    p_values = []
    for col in cc_df.columns:
        t_stat, p_val = ttest_1samp(cc_df[col], 0)
        p_values.append(p_val)
    # perform multiple comparisons correction using the Benjamini-Hochberg procedure
    reject, p_values_corrected, alpha_sidak, alpha_bonf = multipletests(p_values, method='fdr_bh')
    # create a new dataframe to store the results
    results = pd.DataFrame({
        'Column': cc_df.columns,
        'P-value': p_values,
        'P-value (corrected)': p_values_corrected,
        'Reject null hypothesis': reject})
    results.to_csv(opj(path_out_cross, '%s rest_cross_correlation_t_test.csv' % con), index=False)

    # calculate the mean and standard deviation
    cc_mean = np.insert(np.nan, 1, (cross_bold[:, 0, :] - cross_bold[:, 1, :]).mean(axis=0))
    cc_std = np.insert(np.nan, 1, (cross_bold[:, 0, :] - cross_bold[:, 1, :]).std(axis=0))
    cc_sem = np.insert(np.nan, 1, (cross_bold[:, 0, :] - cross_bold[:, 1, :]).std(axis=0) / np.sqrt(nSubj))

    # plot the cross-correlation sequenceness
    x = np.arange(0, maxLag_cross+1, 1)
    plt.figure(dpi=400, figsize=(8, 5))
    plt.xticks(np.arange(len(x)), x, fontsize=10)
    plt.plot(x, cc_mean, color='darkred', linestyle='-', linewidth=1)
    plt.fill_between(x, cc_mean - cc_sem, cc_mean + cc_sem,
                     color='lightcoral', alpha=0.5, linewidth=1)
    plt.axhline(y=0, color='silver', linestyle='--')
    plt.title('Cross Correlation in %s rest: Hippocampus to Visual Cortex' % con)
    plt.xlabel('lag (TRs)')
    plt.ylabel('Cross correlation')
    plt.savefig(opj(path_out_cross, 'Cross Correlation in %s rest: Hippocampus to Visual Cortex.svg' % con))
    plt.show()

    # export the cross correlation file
    cross_correlation_file = pd.DataFrame({'cross_correlation_mean': cc_mean,
                                           'cross_correlation_STD': cc_std,
                                           'cross_correlation_SEM': cc_sem})

    cross_correlation_file.to_csv(opj(path_out_cross, '%s rest_cross_correlation.csv' % con), index=False)


# In[TDLM AND PLOT THE FIGURE]:

sequenceness_list = np.array([sequenceness_lists[i][2] for i in range(len(sub_list))])
# pre resting
# forward replay
pre_fwd_sequence_all = sequenceness_list[:, 0, 0, 0, :]
pre_fwd_permutation_all = matlab_percentile(np.amax(abs(sequenceness_list[:, 0, 0, 1:, 1:].mean(axis=0)), axis=1), 95)
# backward replay
pre_bwd_sequence_all = sequenceness_list[:, 0, 1, 0, :]
pre_bwd_permutation_all = matlab_percentile(np.amax(abs(sequenceness_list[:, 0, 1, 1:, 1:].mean(axis=0)), axis=1), 95)
# plot the sequenceness
TDLM_fig('fwd', pre_fwd_sequence_all, pre_fwd_permutation_all, 'pre_rest', path_out_rest)
TDLM_fig('bwd', pre_bwd_sequence_all, pre_bwd_permutation_all, 'pre_rest', path_out_rest)

# post resting
# forward replay
post_fwd_sequence_all = sequenceness_list[:, 1, 0, 0, :]
post_fwd_permutation_all = matlab_percentile(np.amax(abs(sequenceness_list[:, 1, 0, 1:, 1:].mean(axis=0)), axis=1), 95)
# backward replay
post_bwd_sequence_all = sequenceness_list[:, 1, 1, 0, :]
post_bwd_permutation_all = matlab_percentile(np.amax(abs(sequenceness_list[:, 1, 1, 1:, 1:].mean(axis=0)), axis=1), 95)
# plot the sequenceness
TDLM_fig('fwd', post_fwd_sequence_all, post_fwd_permutation_all, 'post_rest', path_out_rest)
TDLM_fig('bwd', post_bwd_sequence_all, post_bwd_permutation_all, 'post_rest', path_out_rest)

# post-pre resting
# forward replay
post_pre_fwd_sequence_all = sequenceness_list[:, 1, 0, 0, :] - sequenceness_list[:, 0, 0, 0, :]
post_pre_fwd_permutation_all = matlab_percentile(
    np.amax(abs((sequenceness_list[:, 1, 0, 1:, 1:] - sequenceness_list[:, 0, 0, 1:, 1:]).mean(axis=0)), axis=1), 95)
# backward replay
post_pre_bwd_sequence_all = sequenceness_list[:, 1, 1, 0, :] - sequenceness_list[:, 0, 1, 0, :]
post_pre_bwd_permutation_all = matlab_percentile(
    np.amax(abs((sequenceness_list[:, 1, 1, 1:, 1:] - sequenceness_list[:, 0, 1, 1:, 1:]).mean(axis=0)), axis=1), 95)
# plot the sequenceness
TDLM_fig('fwd', post_pre_fwd_sequence_all, post_pre_fwd_permutation_all, 'post_pre', path_out_rest)
TDLM_fig('bwd', post_pre_bwd_sequence_all, post_pre_bwd_permutation_all, 'post_pre', path_out_rest)
