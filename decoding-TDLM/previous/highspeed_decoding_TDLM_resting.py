# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:25:21 2022

Detecting replay from fMRI resting state data by TDLM

@author: Qi Huang
"""
# In[IMPORT RELEVANT PACKAGES]:
'''
========================================================================
IMPORT RELEVANT PACKAGES:
========================================================================
'''
import glob
import os
# import yaml
import logging
import time
from os.path import join as opj
import sys
import copy
from pprint import pformat
import numpy as np
# from nilearn.input_data import NiftiMasker
from nilearn import plotting, image, masking
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import math
from matplotlib import pyplot as plt
import itertools
import warnings
from nilearn.signal import clean
import scipy
from scipy.stats import pearsonr
# import scipy.io as scio
# from sklearn.svm import SVC
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import RidgeClassifier
# from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn.preprocessing import StandardScaler
# from sklearn import preprocessing
from bids.layout import BIDSLayout
from joblib import Parallel, delayed
import seaborn as sns

# import collections
# DEAL WITH ERRORS, WARNING ETC
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

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
        # delay is the time, not the TR
        self.bold_peaks = (self.event_onsets + (self.bold_delay * self.t_tr))
        # divide the expected time-point of bold peaks by the repetition time:
        self.peak_trs = self.bold_peaks / self.t_tr
        if self.task == 'VFL' or self.task == 'learning' or self.task == 'REP':
            # add the number of run volumes to the tr indices:
            run_volumes = [sum(self.num_vol_run[0:int(self.events["run"].iloc[row] - 1)])
                           for row in range(len(self.events["run"]))]
            run_volumes = pd.Series(run_volumes)
        # add the number of volumes of each run:
        # (run-1)*run_trs + this_run_peak_trs
        trs = round(self.peak_trs + run_volumes)
        # copy the relevant trs as often as specified by the interval:
        a = np.transpose(np.tile(trs, (self.interval, 1)))
        # create same-sized matrix with trs to be added:
        b = np.full((len(trs), self.interval), np.arange(self.interval))
        # assign the final list of trs:
        self.trs = np.array(np.add(a, b).flatten(), dtype=int)
        # save the TRs of the stimulus presentations
        self.stim_trs = round(self.event_onsets / self.t_tr + run_volumes)

    def rmheadmotion(self):
        self.fd = self.confounds.loc[:, 'framewise_displacement'].fillna(
            0).reset_index(drop=True)
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
        # delay is the time, not the TR
        self.bold_peaks = (self.event_onsets + (self.bold_delay * self.t_tr))
        # divide the expected time-point of bold peaks by the repetition time:
        self.peak_trs = self.bold_peaks / self.t_tr
        if self.task == 'VFL' or self.task == 'learning' or self.task == 'REP':
            # add the number of run volumes to the tr indices:
            run_volumes = [sum(self.num_vol_run[0:int(self.events_rm["run"].iloc[row] - 1)])
                           for row in range(len(self.events_rm["run"]))]
            self.run_volumes = pd.Series(run_volumes)
        # add the number of volumes of each run:
        # (run-1)*run_trs + this_run_peak_trs
        self.trs_round = round(self.peak_trs + run_volumes)
        # copy the relevant trs as often as specified by the interval:
        self.a = np.transpose(np.tile(self.trs_round, (self.interval, 1)))
        # create same-sized matrix with trs to be added:
        self.b = np.full((len(self.trs_round), self.interval),
                         np.arange(self.interval))
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
        self.stim_trs = np.repeat(
            np.array(self.stim_trs, dtype=int), self.interval)

    def zscore(self, signals, fold_list):
        from nilearn.signal import clean
        import numpy as np
        # get boolean indices for all run indices in the run list:
        fold_indices = np.isin(self.fold, list(fold_list))
        # standardize data all runs in the run list:
        self.data_zscored = clean(
            signals=signals[self.trs[fold_indices]],
            # sessions=self.runs[fold_indices],
            t_r=1.3,
            detrend=False,
            standardize=True)

    def predict(self, clf, fold_list):
        # import packages:
        import pandas as pd
        # get classifier class predictions:
        pred_class = clf.predict(self.data_zscored)
        # get classifier probabilistic predictions:
        pred_proba = clf.predict_proba(self.data_zscored)
        # get the classes of the classifier:
        classes_names = clf.classes_
        # get boolean indices for all run indices in the run list:
        fold_indices = np.isin(self.fold, list(fold_list))
        # create a dataframe with the probabilities of each class:
        df = pd.DataFrame(pred_proba, columns=classes_names)
        # get the number of predictions made:
        num_pred = len(df)
        # get the number of trials in the test set:
        num_trials = int(num_pred / self.interval)
        # add the predicted class label to the dataframe:
        df["pred_label"] = pred_class
        df['marker'] = self.marker[fold_indices]
        # add the true stimulus label to the dataframe:
        df["stim"] = self.stim[fold_indices]
        # add the volume number (TR) to the dataframe:
        df["tr"] = self.trs[fold_indices]
        # add the sequential TR to the dataframe:
        df["seq_tr"] = np.tile(np.arange(1, self.interval + 1), num_trials)
        # add the counter of trs on which the stimulus was presented
        df["stim_tr"] = self.stim_trs[fold_indices]
        # add the trial number to the dataframe:
        df["trials"] = self.trials[fold_indices]
        # add the run number to the dataframe:
        df["run"] = self.runs[fold_indices]
        # add the session number to the dataframe:
        df["session"] = self.sess[fold_indices]
        df["fold"] = self.fold[fold_indices]
        # add the inter trial interval to the dataframe:
        # df["tITI"] = self.itis[fold_indices]
        # add the participant id to the dataframe:
        df["participant"] = np.repeat(
            self.events["participant"].unique(), num_pred)
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


def show_weights(array):
    # https://stackoverflow.com/a/50154388
    import numpy as np
    # import seaborn as sns
    n_samples = array.shape[0]
    classes, bins = np.unique(array, return_counts=True)
    n_classes = len(classes)
    weights = n_samples / (n_classes * bins)
    sns.barplot(classes, weights)
    plt.xlabel("class label")
    plt.ylabel("weight")
    plt.show()


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


def temp_pred_prob(data, condition, picList):
    # data = pre_list_append
    # condition  = 51.
    pre_list_filter = data[(data["class"] != 'other') &
                           (data["classifier"] != 'log_reg') &
                           (data["test_set"] == 'test-rep_long') &
                           (data['marker'] == condition)]
    pre_list_filter['class_index'] = pre_list_filter['class'].map(
        lambda a: stim_index(a))
    pre_list_filter_sort = pre_list_filter.sort_values(
        by=['class_index', 'run', 'trials'])
    pred_array = np.array(pre_list_filter_sort.loc[:, 'probability'])
    pred_matrix = np.reshape(
        pred_array, (int(len(pred_array) / 4), 4), order='F')
    pred_matrix_seq = [pd.Series(pred_matrix[:, picList[0] - 1]),
                       pd.Series(pred_matrix[:, picList[1] - 1]),
                       pd.Series(pred_matrix[:, picList[2] - 1]),
                       pd.Series(pred_matrix[:, picList[3] - 1])]
    pred_matrix_seq = pd.concat(pred_matrix_seq, axis=1)
    return pred_matrix_seq


def Temporal_probability_each(condition, pred_matrix, long_interval, subject, path):
    pre_matrix_re = np.array([np.array(pred_matrix[n]).reshape(int(len(pred_matrix[n]) / long_interval), long_interval)
                              for n in range(np.size(pred_matrix, axis=1))])
    s1mean = pre_matrix_re[0, :, :].mean(axis=0)
    s2mean = pre_matrix_re[1, :, :].mean(axis=0)
    s3mean = pre_matrix_re[2, :, :].mean(axis=0)
    s4mean = pre_matrix_re[3, :, :].mean(axis=0)
    s1sem = pre_matrix_re[0, :, :].std(axis=0) / np.sqrt(np.size(pre_matrix_re, axis=1))
    s2sem = pre_matrix_re[1, :, :].std(axis=0) / np.sqrt(np.size(pre_matrix_re, axis=1))
    s3sem = pre_matrix_re[2, :, :].std(axis=0) / np.sqrt(np.size(pre_matrix_re, axis=1))
    s4sem = pre_matrix_re[3, :, :].std(axis=0) / np.sqrt(np.size(pre_matrix_re, axis=1))

    x = np.arange(1, long_interval + 1)
    plt.figure(dpi=400)
    plt.xticks(fontsize=10)
    plt.plot(x, s1mean, color='darkred', linestyle='-',
             linewidth=1, label='Serial event 1')
    plt.fill_between(x, s1mean - s1sem, s1mean + s1sem,
                     color='lightcoral', alpha=0.3, linewidth=1)
    plt.plot(x, s2mean, color='orangered', linestyle='-',
             linewidth=1, label='Serial event 2')
    plt.fill_between(x, s2mean - s2sem, s2mean + s2sem,
                     color='lightsalmon', alpha=0.3, linewidth=1)
    plt.plot(x, s3mean, color='gold', linestyle='-',
             linewidth=1, label='Serial event 3')
    plt.fill_between(x, (s3mean - s3sem), (s3mean + s3sem),
                     color='khaki', alpha=0.3, linewidth=1)
    plt.plot(x, s4mean, color='darkgreen', linestyle='-',
             linewidth=1, label='Serial event 4')
    plt.fill_between(x, (s4mean - s4sem), (s4mean + s4sem),
                     color='palegreen', alpha=0.3, linewidth=1)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title('%s condition %s sequential probability' % (subject, condition))
    plt.xlabel('time(TR)')
    plt.ylabel('probability')
    plt.ylim(0, 0.6)
    plt.savefig(opj(path, '%s condition %s sequential probability.png' % (subject, condition)),
                dpi=400, bbox_inches='tight')
    plt.show()


def Temporal_probability(condition, pred_matrix, long_interval, nSubj, path):
    pre_matrix_re = [np.array(pred_matrix[n]).reshape(int(len(pred_matrix[n]) / long_interval), long_interval, 4)
                     for n in range(len(pred_matrix))]
    pre_matrix_re_mean = [pre_matrix_re[i].mean(
        axis=0) for i in range(len(pre_matrix_re))]
    pre_matrix_re_mean = np.array(pre_matrix_re_mean)

    s1mean = pre_matrix_re_mean[:, :, 0].mean(axis=0)
    s2mean = pre_matrix_re_mean[:, :, 1].mean(axis=0)
    s3mean = pre_matrix_re_mean[:, :, 2].mean(axis=0)
    s4mean = pre_matrix_re_mean[:, :, 3].mean(axis=0)
    s1sem = pre_matrix_re_mean[:, :, 0].std(axis=0) / np.sqrt(nSubj)
    s2sem = pre_matrix_re_mean[:, :, 1].std(axis=0) / np.sqrt(nSubj)
    s3sem = pre_matrix_re_mean[:, :, 2].std(axis=0) / np.sqrt(nSubj)
    s4sem = pre_matrix_re_mean[:, :, 3].std(axis=0) / np.sqrt(nSubj)

    x = np.arange(1, long_interval + 1)
    plt.figure(dpi=400)
    plt.xticks(fontsize=10)
    plt.plot(x, s1mean, color='darkred', linestyle='-',
             linewidth=1, label='Serial event 1')
    plt.fill_between(x, s1mean - s1sem, s1mean + s1sem,
                     color='lightcoral', alpha=0.3, linewidth=1)
    plt.plot(x, s2mean, color='orangered', linestyle='-',
             linewidth=1, label='Serial event 2')
    plt.fill_between(x, s2mean - s2sem, s2mean + s2sem,
                     color='lightsalmon', alpha=0.3, linewidth=1)
    plt.plot(x, s3mean, color='gold', linestyle='-',
             linewidth=1, label='Serial event 3')
    plt.fill_between(x, (s3mean - s3sem), (s3mean + s3sem),
                     color='khaki', alpha=0.3, linewidth=1)
    plt.plot(x, s4mean, color='darkgreen', linestyle='-',
             linewidth=1, label='Serial event 4')
    plt.fill_between(x, (s4mean - s4sem), (s4mean + s4sem),
                     color='palegreen', alpha=0.3, linewidth=1)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.title('condition %s sequential probability' % condition)
    plt.xlabel('time(TR)')
    plt.ylabel('probability')
    plt.ylim(0, 0.4)
    plt.savefig(opj(path, 'condition %s sequential probability.png' % condition),
                dpi=400, bbox_inches='tight')
    plt.show()


# TDLM for cue replay detecting within subject
def TDLM(probability_matrix, all_sequence, condition):
    rp = []
    # print(isub,sub)
    # detect for each 2048 ms trial
    # loops for real sequence and permutation trials
    for i in range(len(all_sequence)):
        # real sequence
        rp = all_sequence[i, :]
        # get the transition matrix
        T1 = TransM(rp)
        T2 = np.transpose(T1)
        # no lag and several lags = nbins
        nbins = maxLag + 1
        # X1 = seq_pred_matrix[13*i:13*(i+1),:] #####
        X1 = np.array(probability_matrix)
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
        # four different design matrix for regressor multiple to the temporal data
        # linear regression for the backward and forward replay
        if condition == 'sf':
            sf[0, i, 1:, 0] = betas_2GLM[0, :]  # 51 condition forward
            sf[0, i, 1:, 1] = betas_2GLM[1, :]  # 51 condition backward
        elif condition == 'sb':
            sf[1, i, 1:, 0] = betas_2GLM[0, :]  # 52 condition forward
            sf[1, i, 1:, 1] = betas_2GLM[1, :]  # 52 condition backward


def TDLM_fig_each(direction, data, permutation, condition, path, task_bold_delay, mask_name):
    if direction == 'forward':
        plotcolor = 'darkred'
        title = '%s Forward replay in sequential %s condition (%1.0f TR delay) %s.png' % (subject,
                                                                                          condition, task_bold_delay,
                                                                                          mask_name)
        filename = '%s Forward replay in sequential %s condition (%1.0f TR delay) %s.png' % (subject,
                                                                                             condition, task_bold_delay,
                                                                                             mask_name)
    elif direction == 'backward':
        plotcolor = 'dodgerblue'
        title = '%s Backward replay in sequential %s condition (%1.0f TR delay) %s.png' % (subject,
                                                                                           condition, task_bold_delay,
                                                                                           mask_name)
        filename = '%s Backward replay in sequential %s condition (%1.0f TR delay) %s.png' % (subject,
                                                                                              condition,
                                                                                              task_bold_delay,
                                                                                              mask_name)
    elif direction == 'for-back':
        plotcolor = 'darkgreen'
        title = '%s Forward-Backward replay in sequential %s condition (%1.0f TR delay) %s.png' % (subject,
                                                                                                   condition,
                                                                                                   task_bold_delay,
                                                                                                   mask_name)
        filename = '%s Forward-Backward replay in sequential %s condition (%1.0f TR delay) %s.png' % (subject,
                                                                                                      condition,
                                                                                                      task_bold_delay,
                                                                                                      mask_name)

    x = np.arange(0, nbins, 1)
    plt.figure(dpi=400, figsize=(10, 5))
    plt.xticks(range(len(x)), x, fontsize=10)
    plt.plot(x, data, color=plotcolor, linestyle='-', linewidth=1)
    plt.axhline(y=permutation, color='silver', linestyle='--')
    plt.axhline(y=(-permutation), color='silver', linestyle='--')
    plt.title(title)
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path, filename),
                dpi=400, bbox_inches='tight')
    plt.show()


# plot the all averaged subjects' sequenceness
# probability_matrix = np.transpose(bold_cross)
def TDLM_fig(direction, data, permutation, condition, path, task_bold_delay, mask_name):
    if direction == 'forward':
        plotcolor = 'darkred'
        fillcolor = 'lightcoral'
        title = 'Forward replay in sequential %s condition (%1.0f th TR peak) %s.png' % (
            condition, task_bold_delay, mask_name)
        filename = 'subject mean Forward replay in sequential %s condition (%1.0f th TR peak) %s.png' % (
            condition, task_bold_delay, mask_name)
    elif direction == 'backward':
        plotcolor = 'dodgerblue'
        fillcolor = 'lightskyblue'
        title = 'Backward replay in sequential %s condition (%1.0f th TR peak) %s.png' % (
            condition, task_bold_delay, mask_name)
        filename = 'subject mean Backward replay in sequential %s condition (%1.0f th TR peak) %s.png' % (
            condition, task_bold_delay, mask_name)
    elif direction == 'for-back':
        plotcolor = 'darkgreen'
        fillcolor = 'lightgreen'
        title = 'Forward-Backward replay in sequential %s condition (%1.0f th TR peak) %s.png' % (
            condition, task_bold_delay, mask_name)
        filename = 'subject mean Forward-Backward replay in sequential %s condition (%1.0f th TR peak) %s.png' % (
            condition, task_bold_delay, mask_name)

    s1mean = data.mean(axis=0)
    s1sem = data.std(axis=0) / np.sqrt(len(data))
    x = np.arange(0, nbins, 1)
    plt.figure(dpi=400, figsize=(10, 5))
    plt.xticks(range(len(x)), x, fontsize=10)
    plt.plot(x, s1mean, color=plotcolor, linestyle='-', linewidth=1)
    plt.fill_between(x, s1mean - s1sem, s1mean + s1sem,
                     color=fillcolor, alpha=0.5, linewidth=1)
    plt.axhline(y=permutation, color='silver', linestyle='--')
    plt.axhline(y=(-permutation), color='silver', linestyle='--')
    plt.title(title)
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path, 'mean replay', filename),
                dpi=400, bbox_inches='tight')
    plt.show()


# In[SETUP NECESSARY PATHS ETC]:
'''
========================================================================
SETUP NECESSARY PATHS ETC:
========================================================================
'''
# name of the current project:
project = "fmrireplay"
path_root = None
sub_list = None
subject = None
# path to the project root:
project_name = "fmrireplay-decoding"
path_root = opj(os.getcwd().split(project)[0], project)
path_bids = opj(path_root, "fmrireplay-bids", "BIDS")
path_bids_nf = opj(path_root, "BIDS-nofieldmap")
path_code = opj(path_bids_nf, "code", "decoding-TDLM")
path_fmriprep = opj(path_bids_nf, "derivatives", "fmrireplay-fmriprep")
path_masks = opj(path_bids_nf, "derivatives", "fmrireplay-masks")
path_glm_vfl = opj(path_bids_nf, "derivatives", "fmrireplay-glm-vfl")
path_level1_vfl = opj(path_glm_vfl, "l1pipeline")
path_glm_rep = opj(path_bids_nf, "derivatives", "fmrireplay-glm-replay")
path_level1_rep = opj(path_glm_rep, "l1pipeline")
path_decoding = opj(path_bids_nf, "derivatives", project_name)
path_behavior = opj(path_bids_nf, 'derivatives', 'fmrireplay-behavior')
path_out_all = opj(path_decoding, 'all_subject_results')
# load the learning sequence
learning_sequence = pd.read_csv(opj(path_behavior, 'sequence.csv'), sep=',')

# define the subject id
layout = BIDSLayout(path_bids_nf)
sub_list = sorted(layout.get_subjects())
# delete the subjects
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



'''
========================================================================
DEFINITION OF ALL FUNCTIONS
========================================================================
'''
class TaskData:

    def __init__(
            self, events, condition, name, trial_type, bold_delay=0,
            interval=1, t_tr=1.25, num_vol_run=530):  #?
        import pandas as pd
        # define name of the task data subset:
        self.name = name
        # define the task condition the task data subset if from:
        self.condition = condition
        # define the delay (in seconds) by which onsets are moved:
        self.bold_delay = bold_delay
        # define the number of TRs from event onset that should be selected:
        self.interval = interval
        # define the repetition time (TR) of the mri data acquisition:
        self.t_tr = t_tr
        # define the number of volumes per task run:
        self.num_vol_run = num_vol_run
        # select events: upright stimulus, correct answer trials only:
        if trial_type == 'stimulus':
            self.events = events.loc[
                          (events['condition'] == condition) &
                          (events['trial_type'] == trial_type) &
                          (events['stim_orient'] == 0) &
                          (events['serial_position'] == 1) &
                          (events['accuracy'] != 0),
                          :]
        elif trial_type == 'cue':
            self.events = events.loc[
                          (events['condition'] == condition) &
                          (events['trial_type'] == trial_type),
                          :]
        # reset the indices of the data frame:
        self.events.reset_index()
        # sort all values by session and run:
        self.events.sort_values(by=['session', 'run_session'])
        # call further function upon initialization:
        self.define_trs()
        self.get_stats()
        if condition == 'sequence':
            self.sequence = events.loc[
                             (events['condition'] == condition) &
                             (events['trial_type'] == trial_type) &
                             (events['stim_orient'] == 0) &
                             (events['serial_position'] >= 0) &
                             (events['accuracy'] != 0),
                             :]
            self.sequences = np.array(self.sequence)[:,12]
            self.real_trials = np.unique(self.trials)
            self.real_trial = len(self.real_trials)
            self.sequences = self.sequences.reshape(self.real_trial, 5)



    def define_trs(self):
        # import relevant functions:
        import numpy as np
        # select all events onsets:
        self.event_onsets = self.events['onset']
        # add the selected delay to account for the peak of the hrf
        self.bold_peaks = self.events['onset'] + self.bold_delay  #delay is the time, not the TR
        # divide the expected time-point of bold peaks by the repetition time:
        self.peak_trs = self.bold_peaks / self.t_tr
        # add the number of run volumes to the tr indices:
        run_volumes = (self.events['run_study']-1) * self.num_vol_run
        # add the number of volumes of each run:
        trs = round(self.peak_trs + run_volumes) # (run-1)*run_trs + this_run_peak_trs
        # copy the relevant trs as often as specified by the interval:
        a = np.transpose(np.tile(trs, (self.interval, 1)))
        # create same-sized matrix with trs to be added:
        b = np.full((len(trs), self.interval), np.arange(self.interval))
        # assign the final list of trs:
        self.trs = np.array(np.add(a, b).flatten(), dtype=int)
        # save the TRs of the stimulus presentations
        self.stim_trs = round(self.event_onsets / self.t_tr + run_volumes) # continuous trs for counting

    def get_stats(self):
        import numpy as np
        self.num_trials = len(self.events)
        self.runs = np.repeat(np.array(self.events['run_study'], dtype=int), self.interval)
        self.trials = np.repeat(np.array(self.events['trial'], dtype=int), self.interval)
        self.sess = np.repeat(np.array(self.events['session'], dtype=int), self.interval)
        self.stim = np.repeat(np.array(self.events['stim_label'], dtype=object), self.interval)
        self.itis = np.repeat(np.array(self.events['interval_time'], dtype=float), self.interval)
        self.stim_trs = np.repeat(np.array(self.stim_trs, dtype=int), self.interval)

    def zscore(self, signals, run_list, t_tr=1.25):
        from nilearn.signal import clean
        import numpy as np
        # get boolean indices for all run indices in the run list:
        run_indices = np.isin(self.runs, list(run_list))
        # standardize data all runs in the run list:
        self.data_zscored = clean(
            signals=signals[self.trs[run_indices]],
            sessions=self.runs[run_indices],
            t_r=t_tr,
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
        df['pred_label'] = pred_class
        # add the true stimulus label to the dataframe:
        df['stim'] = self.stim[run_indices]
        # add the volume number (TR) to the dataframe:
        df['tr'] = self.trs[run_indices]
        # add the sequential TR to the dataframe:
        df['seq_tr'] = np.tile(np.arange(1, self.interval + 1), num_trials)
        # add the counter of trs on which the stimulus was presented
        df['stim_tr'] = self.stim_trs[run_indices]
        # add the trial number to the dataframe:
        df['trial'] = self.trials[run_indices]
        # add the run number to the dataframe:
        df['run_study'] = self.runs[run_indices]
        # add the session number to the dataframe:
        df['session'] = self.sess[run_indices]
        # add the inter trial interval to the dataframe:
        df['tITI'] = self.itis[run_indices]
        # add the participant id to the dataframe:
        df['id'] = np.repeat(self.events['subject'].unique(), num_pred)
        # add the name of the classifier to the dataframe:
        df['test_set'] = np.repeat(self.name, num_pred)
        # return the dataframe:
        return df


def detrend(data, t_tr=1.25):
    from nilearn.signal import clean
    data_detrend = clean(
        signals=data, t_r=t_tr, detrend=True, standardize=False)
    return data_detrend


def show_weights(array):
    # https://stackoverflow.com/a/50154388
    import numpy as np
    import seaborn as sns
    n_samples = array.shape[0]
    classes, bins = np.unique(array, return_counts=True)
    n_classes = len(classes)
    weights = n_samples / (n_classes * bins)
    sns.barplot(classes, weights)
    plt.xlabel('class label')
    plt.ylabel('weight')
    plt.show()


def melt_df(df, melt_columns):
    # save the column names of the dataframe in a list:
    column_names = df.columns.tolist()
    # remove the stimulus classes from the column names;
    id_vars = [x for x in column_names if x not in melt_columns]
    # melt the dataframe creating one column with value_name and var_name:
    df_melt = pd.melt(
            df, value_name='probability', var_name='class', id_vars=id_vars)
    # return the melted dataframe:
    return df_melt

# combine all masks from the feature selection by intersection:
def multimultiply(arrays):
    import copy
    # start with the first array:
    array_union = copy.deepcopy(arrays[0].astype(np.int))
    # loop through all arrays
    for i in range(len(arrays)):
        # multiply every array with all previous array
        array_union = np.multiply(array_union, copy.deepcopy(arrays[i].astype(np.int)))
    # return the union of all arrays:
    return(array_union)

# function of producing a sequence transition matrix
def TransM(x):
    # create an empty matrix
    transition_matrix = np.zeros([5,5])
    # transition
    for a in range(4):
       transition_matrix[x[a]-1][x[a+1]-1] = 1
    return(transition_matrix)

# help to order the classifiers
def stim_index(x):
    if x =='face':
     return 1
    elif x =='house':
     return 2
    elif x =='cat':
     return 3
    elif x =='shoe':
     return 4
    elif x =='chair':
     return 5

'''
========================================================================
LOAD TDLM PARAMETERS:
========================================================================
'''
# DEFINE DECODING SPECIFIC PARAMETERS:
# define the mask to be used:
mask = 'visual'  # visual or whole
# applied time-shift to account for the BOLD delay, in seconds:
bold_delay = 4  # 4, 5 or 6 secs
# define the degree of smoothing of the functional data
smooth = 4


# DEFINE RELEVANT VARIABLES:
# time of repetition (TR), in seconds:
t_tr = params['mri']['tr']
# number of volumes (TRs) for each functional task run:
n_tr_run = 530
# acquisition time window of one sequence trial, in seconds:
t_win = 16
# number of measurements that are considered per sequence time window:
n_tr_win = round(t_win / t_tr)
# number of oddball trials in the experiment:
n_tr_odd = 600
# number of sequence trials in the experiment:
n_tr_seq = 75
# number of repetition trials in the experiment:
n_tr_rep = 45
# number of scanner triggers before the experiment starts:
n_tr_wait = params['mri']['num_trigger']
# number of functional task runs in total:
n_run = params['mri']['num_runs']
# number of experimental sessions in total:
n_ses = params['mri']['num_sessions']
# number of functional task runs per session:
n_run_ses = int(n_run / n_ses)


# the number of subjects
nSubj = 32
# the list of all the sequences
uniquePerms = list(itertools.permutations([1,2,3,4,5],5))
# the number of sequences
nShuf = len(uniquePerms)
# the number of timelags, including 32, 64, 128, 512and 2048ms
maxLag_rest = 20
# the number of states (decoding models)
nstates = 5 #len(betas_de_re[0,])
# pre-resting and post-resting states
resting = 2



'''
========================================================================
DECODING, PREDICTION AND TDLM
========================================================================
'''
# CREATE PATHS TO OUTPUT DIRECTORIES:
# for n,sub in zip(range(len(suball)),suball):
sub=suball[20]

# output path
path_decoding = opj(path_tardis, 'TDLM')
path_out = opj(path_decoding, sub)
path_out_figs = opj(path_out, 'plots')
path_out_data = opj(path_out, 'data')
path_out_logs = opj(path_out, 'logs')
path_out_masks = opj(path_out, 'masks')


# CREATE PATHS TO OUTPUT DIRECTORIES:
for path in [path_out_figs, path_out_data, path_out_logs, path_out_masks]:
    if not os.path.exists(path):
        os.makedirs(path)


# SETUP LOGGING:
# remove all handlers associated with the root logger object:
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# get current data and time as a string:
timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
# create path for the logging file
log = opj(path_out_logs, '%s-%s.log' % (timestr, sub))
# start logging:
logging.basicConfig(
    filename=log, level=logging.DEBUG, format='%(asctime)s %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S')


 # ADD BASIC SCRIPT INFORMATION TO THE LOGGER:
logging.info('Running decoding script')
logging.info('operating system: %s' % sys.platform)
logging.info('project name: %s' % project)
logging.info('participant: %s' % sub)
logging.info('mask: %s' % mask)
logging.info('bold delay: %d secs' % bold_delay)
logging.info('smoothing kernel: %d mm' % smooth)



# LOAD BEHAVIORAL DATA (THE EVENTS.TSV FILES OF THE SUBJECT):
# paths to all events files of the current subject:
path_events = opj(cwd1,'highspeed-bids','forBIDS', sub, 'ses-*', 'func', '*tsv')
#dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
path_events = sorted(glob.glob(path_events), key=lambda f: os.path.basename(f))
logging.info('found %d event files' % len(path_events))
logging.info('paths to events files (sorted):\n%s' % pformat(path_events))
# import events file and save data in dataframe:
df_events = pd.concat((pd.read_csv(f, sep='\t') for f in path_events),
                      ignore_index=True)


# CREATE PATHS TO THE MRI DATA:
# define path to input directories:
path_fmriprep = opj(cwd1, 'highspeed-fmriprep', 'fmriprep', sub)
path_level1 = opj(cwd1, 'highspeed-glm', 'l1pipeline')
path_masks = opj(cwd1, 'highspeed-masks', 'masks')

logging.info('path to fmriprep files: %s' % path_fmriprep)
logging.info('path to level 1 files: %s' % path_level1)
logging.info('path to mask files: %s' % path_masks)
paths = {
    'fmriprep': opj(cwd1, 'highspeed-fmriprep', 'fmriprep', sub),
    'level1': opj(cwd1, 'highspeed-glm', 'l1pipeline'),
    'masks': opj(cwd1, 'highspeed-masks', 'masks')
}

# load the visual mask task files:
path_mask_vis_task = opj(path_masks, 'mask_visual', sub, '*', '*task-highspeed*.nii.gz')
path_mask_vis_task = sorted(glob.glob(path_mask_vis_task), key=lambda f: os.path.basename(f))
logging.info('found %d visual mask task files' % len(path_mask_vis_task))
logging.info('paths to visual mask task files:\n%s' % pformat(path_mask_vis_task))

# load the hippocampus mask task files:
path_mask_hpc_task = opj(path_masks, 'mask_hippocampus', sub, '*', '*task-highspeed*.nii.gz')
path_mask_hpc_task = sorted(glob.glob(path_mask_hpc_task), key=lambda f: os.path.basename(f))
logging.info('found %d hpc mask files' % len(path_mask_hpc_task))
logging.info('paths to hpc mask task files:\n%s' % pformat(path_mask_hpc_task))

# load the whole brain mask files:
path_mask_whole_task = opj(path_fmriprep, '*', 'func', '*task-highspeed*T1w*brain_mask.nii.gz')
path_mask_whole_task = sorted(glob.glob(path_mask_whole_task), key=lambda f: os.path.basename(f))
logging.info('found %d whole-brain masks' % len(path_mask_whole_task))
logging.info('paths to whole-brain mask files:\n%s' % pformat(path_mask_whole_task))

# load the functional mri task files:
path_func_task = opj(path_level1, 'smooth', sub, '*', '*task-highspeed*nii.gz')
path_func_task = sorted(glob.glob(path_func_task), key=lambda f: os.path.basename(f))
logging.info('found %d functional mri task files' % len(path_func_task))
logging.info('paths to functional mri task files:\n%s' % pformat(path_func_task))

# define path to the functional resting state runs:
path_rest = opj(path_masks, 'smooth', sub, '*', '*task-rest*nii.gz')
path_rest = sorted(glob.glob(path_rest), key=lambda f: os.path.basename(f))
logging.info('found %d functional mri rest files' % len(path_rest))
logging.info('paths to functional mri rest files:\n%s' % pformat(path_rest))

# load the anatomical mri file:
path_anat = opj(path_fmriprep, 'anat', '%s_desc-preproc_T1w.nii.gz' % sub)
path_anat = sorted(glob.glob(path_anat), key=lambda f: os.path.basename(f))
logging.info('found %d anatomical mri file' % len(path_anat))
logging.info('paths to anatoimical mri files:\n%s' % pformat(path_anat))

# load the confounds files:
path_confs_task = opj(path_fmriprep, '*', 'func', '*task-highspeed*confounds_regressors.tsv')
path_confs_task = sorted(glob.glob(path_confs_task), key=lambda f: os.path.basename(f))
logging.info('found %d confounds files' % len(path_confs_task))
logging.info('found %d confounds files' % len(path_confs_task))
logging.info('paths to confounds files:\n%s' % pformat(path_confs_task))

# load the spm.mat files:
path_spm_mat = opj(path_level1, 'contrasts', sub, '*', 'SPM.mat')
path_spm_mat = sorted(glob.glob(path_spm_mat), key=lambda f: os.path.dirname(f))
logging.info('found %d spm.mat files' % len(path_spm_mat))
logging.info('paths to spm.mat files:\n%s' % pformat(path_spm_mat))

# load the t-maps of the first-level glm:
path_tmap = opj(path_level1, 'contrasts', sub, '*', 'spmT*.nii')
path_tmap = sorted(glob.glob(path_tmap), key=lambda f: os.path.dirname(f))
logging.info('found %d t-maps' % len(path_tmap))
logging.info('paths to t-maps files:\n%s' % pformat(path_tmap))


# LOAD THE MRI DATA:
anat = image.load_img(path_anat[0])
logging.info('successfully loaded %s' % path_anat[0])
# load visual mask:
mask_vis = image.load_img(path_mask_vis_task[0]).get_data().astype(int)
logging.info('successfully loaded one visual mask file!')
# load tmap data:
tmaps = [image.load_img(i).get_data().astype(float) for i in copy.deepcopy(path_tmap)]
logging.info('successfully loaded the tmaps!')
# load hippocampus mask:
mask_hpc = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_hpc_task)]
logging.info('successfully loaded one visual mask file!')


# FEATURE SELECTION: MASKS THE T-MAPS WITH THE ANATOMICAL MASKS
# check if any value in the supposedly binary mask is bigger than 1:
if np.any(mask_vis > 1):
    logging.info('WARNING: detected values > 1 in the anatomical ROI!')
    sys.exit("Values > 1 in the anatomical ROI!")
# get combination of anatomical mask and t-map
tmaps_masked = [np.multiply(mask_vis, i) for i in copy.deepcopy(tmaps)]
# masked tmap into image like object:
tmaps_masked_img = [image.new_img_like(i, j) for (i, j) in zip(path_tmap, copy.deepcopy(tmaps_masked))]

# for i, path in enumerate(tmaps_masked_img):
#     path_save = opj(path_out_masks, '%s_run-%02d_tmap_masked.nii.gz' % (sub, i + 1))
#     path.to_filename(path_save)

# plot masked t-maps
# logging.info('plotting masked tmaps with anatomical as background:')
# for i, path in enumerate(tmaps_masked_img):
#     logging.info('plotting masked tmap %d of %d' % (i+1, len(tmaps_masked_img)))
#     path_save = opj(path_out_figs, '%s_run-%02d_tmap_masked.png' % (sub, i+1))
#     plotting.plot_roi(path, anat, title=os.path.basename(path_save),
#                       output_file=path_save, colorbar=True)


# FEATURE SELECTION: THRESHOLD THE MASKED T-MAPS

# set the threshold:
threshold = params['mri']['thresh']
logging.info('thresholding t-maps with a threshold of %s' % str(threshold))
# threshold the masked tmap image:
tmaps_masked_thresh_img = [image.threshold_img(i, threshold) for i in copy.deepcopy(tmaps_masked_img)]

# logging.info('plotting thresholded tmaps with anatomical as background:')
# for i, path in enumerate(tmaps_masked_thresh_img):
#     path_save = opj(path_out_figs, '%s_run-%02d_tmap_masked_thresh.png' % (sub, i+1))
#     logging.info('plotting masked tmap %s (%d of %d)'
#                  % (path_save, i + 1, len(tmaps_masked_thresh_img)))
#     plotting.plot_roi(path, anat, title=os.path.basename(path_save),
#                       output_file=path_save, colorbar=True)

# extract data from the thresholded images
tmaps_masked_thresh = [image.load_img(i).get_data().astype(float) for i in tmaps_masked_thresh_img]

# calculate the number of tmap voxels:
# all the voxels in the t-maps of brain
num_tmap_voxel = [np.size(i) for i in copy.deepcopy(tmaps_masked_thresh)]
logging.info('number of feature selected voxels: %s' % pformat(num_tmap_voxel))

# selected voxels in the t-maps
num_above_voxel = [np.count_nonzero(i) for i in copy.deepcopy(tmaps_masked_thresh)]
logging.info('number of voxels above threshold: %s' % pformat(num_above_voxel))

# rest voxels
num_below_voxel = [np.count_nonzero(i == 0) for i in copy.deepcopy(tmaps_masked_thresh)]
logging.info('number of voxels below threshold: %s' % pformat(num_below_voxel))

# plot the distribution of t-values:
# for i, run_mask in enumerate(tmaps_masked_thresh):
#     masked_tmap_flat = run_mask.flatten()
#     masked_tmap_flat = masked_tmap_flat[~np.isnan(masked_tmap_flat)]
#     masked_tmap_flat = masked_tmap_flat[~np.isnan(masked_tmap_flat) & ~(masked_tmap_flat == 0)]
#     path_save = opj(path_out_figs, '%s_run-%02d_tvalue_distribution.png' % (sub, i+1))
#     logging.info('plotting thresholded t-value distribution %s (%d of %d)'
#                  % (path_save, i+1, len(tmaps_masked_thresh)))
#     fig = plt.figure()
#     plt.hist(masked_tmap_flat, bins='auto')
#     plt.xlabel('t-values')
#     plt.ylabel('number')
#     plt.title('t-value distribution (%s, run-%02d)' % (sub, i+1))
#     plt.savefig(path_save)

# create a dataframe with the number of voxels
df_thresh = pd.DataFrame({
    'id': [sub] * n_run,
    'run': np.arange(1,n_run+1),
    'n_total': num_tmap_voxel,
    'n_above': num_above_voxel,
    'n_below': num_below_voxel
})
file_name = opj(path_out_data, '%s_thresholding.csv' % (sub))
df_thresh.to_csv(file_name, sep=',', index=False)


# FEATURE SELECTION: BINARIZE THE THRESHOLDED MASKED T-MAPS
# replace all NaNs with 0:
tmaps_masked_thresh_bin = [np.where(np.isnan(i), 0, i) for i in copy.deepcopy(tmaps_masked_thresh)]
# replace all other values with 1:
tmaps_masked_thresh_bin = [np.where(i > 0, 1, i) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
# turn the 3D-array into booleans:
tmaps_masked_thresh_bin = [i.astype(bool) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
# create image like object:
masks_final = [image.new_img_like(path_func_task[0], i.astype(np.int)) for i in copy.deepcopy(tmaps_masked_thresh_bin)]

# logging.info('plotting final masks with anatomical as background:')
# for i, path in enumerate(masks_final):
#     filename = '%s_run-%02d_tmap_masked_thresh.nii.gz'\
#                % (sub, i + 1)
#     path_save = opj(path_out_masks, filename)
#     logging.info('saving final mask %s (%d of %d)'
#                  % (path_save, i+1, len(masks_final)))
#     path.to_filename(path_save)
#     path_save = opj(path_out_figs, '%s_run-%02d_visual_final_mask.png'
#                     % (sub, i + 1))
#     logging.info('plotting final mask %s (%d of %d)'
#                  % (path_save, i + 1, len(masks_final)))
#     plotting.plot_roi(path, anat, title=os.path.basename(path_save),
#                       output_file=path_save, colorbar=True)


# LOAD SMOOTHED FMRI DATA FOR ALL FUNCTIONAL TASK RUNS:
# load smoothed functional mri data for all eight task runs:
logging.info('loading %d functional task runs ...' % len(path_func_task))
data_task = [image.load_img(i) for i in path_func_task]
logging.info('loading successful!')

# DEFINE THE FUNCTIONAL FMRI TASK DATA
# training decoding model
train_odd_peak = TaskData(
        events=df_events, condition='oddball', trial_type='stimulus',
        bold_delay=4, interval=1, name='train-odd_peak')

# test decoding model in 4th TR
test_odd_peak = TaskData(
        events=df_events, condition='oddball', trial_type='stimulus',
        bold_delay=4, interval=1, name='test-odd_peak')

# test decoding model in all 7 TRs
test_odd_long = TaskData(
        events=df_events, condition='oddball', trial_type='stimulus',
        bold_delay=0, interval=7, name='test-odd_long')

# test sequence condition in all 13 TRs
test_seq_long = TaskData(
        events=df_events, condition='sequence', trial_type='stimulus',
        bold_delay=0, interval=13, name='test-seq_long')

# test repetition condition in all 13 TRs
test_rep_long = TaskData(
        events=df_events, condition='repetition', trial_type='stimulus',
        bold_delay=0, interval=13, name='test-rep_long')

# test sequence condition
# test_seq_cue = TaskData(
#         events=df_events, condition='sequence', trial_type='cue',
#         bold_delay=0, interval=5, name='test-seq_cue')
# test_rep_cue = TaskData(
#         events=df_events, condition='repetition', trial_type='cue',
#         bold_delay=0, interval=5, name='test-rep_cue')
test_sets = [
        test_odd_peak, test_odd_long, test_seq_long, test_rep_long
        # ,test_seq_cue, test_rep_cue
        ]




# DEFINE THE CLASSIFIERS
class_labels = ['cat', 'chair', 'face', 'house', 'shoe']
# create a dictionary with all values as independent instances:
# see here: https://bit.ly/2J1DvZm
clf_set = {key: LogisticRegression(
    C=1., # Inverse of regularization strength
    penalty='l2', multi_class='ovr', solver='lbfgs', # Algorithm to use in the optimization problem, Stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno
    max_iter=4000, class_weight='balanced', random_state=42  # to shuffle the data
    ) for key in class_labels}

classifiers = {
    'log_reg': LogisticRegression(
        C=1., penalty='l2', multi_class='multinomial', solver='lbfgs',
        max_iter=4000, class_weight='balanced', random_state=42)}
clf_set.update(classifiers)


# DECODING ON RESTING STATE DATA BY TDLM:

data_list = []
runs = list(range(1, n_run+1))
#mask_label = 'cv'

# LOAD THE FMRI DATA
logging.info('Loading fMRI data of %d resting state runs ...' % len(path_rest))
data_rest = [image.load_img(i) for i in path_rest]
logging.info('loading successful!')

# SAVE THE UNION MASK
mask_label = 'union'
masks_union = multimultiply(tmaps_masked_thresh_bin).astype(int).astype(bool)
masks_union_nii = image.new_img_like(path_func_task[0], masks_union)
path_save = opj(path_out_masks, '{}_task-rest_mask-{}.nii.gz'.format(sub, mask_label))
masks_union_nii.to_filename(path_save)

# APPLIED THE MASK TO THE RESTING DATA
# mask all resting state runs with the averaged feature selection masks:
data_rest_masked = [masking.apply_mask(i, masks_union_nii) for i in data_rest]
# detrend and standardize each resting state run separately:
data_rest_final = [clean(i, detrend=True, standardize=True) for i in data_rest_masked]

# DETREND AND Z-STANDARDIZED THE RESTING FMRI DATA
# mask all functional task runs separately:
data_task_masked = [masking.apply_mask(i, masks_union_nii) for i in data_task]
# detrend each task run separately:
data_task_masked_detrend = [clean(i, detrend=True, standardize=False) for i in data_task_masked]
# combine the detrended data of all runs:
data_task_masked_detrend = np.vstack(data_task_masked_detrend)
# select only oddball data and standardize:
train_odd_peak.zscore(signals=data_task_masked_detrend, run_list=runs)
# write session and run labels:
ses_labels = [i.split(sub + "_")[1].split("_task")[0] for i in path_rest]
run_labels = [i.split("prenorm_")[1].split("_space")[0] for i in path_rest]
file_names = ['_'.join([a, b]) for (a, b) in zip(ses_labels, run_labels)]
rest_interval = 1
# save the voxel patterns:
num_voxels = len(train_odd_peak.data_zscored[0])
voxel_labels = ['v' + str(x) for x in range(num_voxels)]
df_patterns = pd.DataFrame(train_odd_peak.data_zscored, columns=voxel_labels)
# add the stimulus labels to the dataframe:
df_patterns['label'] = copy.deepcopy(train_odd_peak.stim)
# add the participant id to the dataframe:
df_patterns['id'] = np.repeat(df_events['subject'].unique(), len(train_odd_peak.stim))
# add the mask label:
df_patterns['mask'] = np.repeat(mask_label, len(train_odd_peak.stim))
# split pattern dataframe by label:
df_pattern_list = [df_patterns[df_patterns['label'] == i] for i in df_patterns['label'].unique()]
# create file path to save the dataframe:
file_paths = [opj(path_out_data, '{}_voxel_patterns_{}_{}'.format(sub, mask_label, i)) for i in df_patterns['label'].unique()]
# save the final dataframe to a .csv-file:
[i.to_csv(j + '.csv', sep=',', index=False) for (i, j) in zip(df_pattern_list, file_paths)]
# save only the voxel patterns as niimg-like objects
[masking.unmask(X=i.loc[:, voxel_labels].to_numpy(), mask_img=masks_union_nii).to_filename(j + '.nii.gz') for (i, j) in zip(df_pattern_list, file_paths)]
#[image.new_img_like(path_func_task[0], i.loc[:, voxel_labels].to_numpy()).to_filename(j + '.nii.gz') for (i, j) in zip(df_pattern_list, file_paths)]


# DECODING RESTING STATE DATA:
for clf_name, clf in clf_set.items():
    # print classifier name:
    logging.info('classifier: %s' % clf_name)
    # get the example labels for all slow trials:
    train_stim = copy.deepcopy(train_odd_peak.stim)
    # replace labels for single-label classifiers:
    if clf_name in class_labels:
        # replace all other labels with other
        train_stim = ['other' if x != clf_name else x for x in train_stim]
        # turn into a numpy array
        train_stim = np.array(train_stim, dtype=object)
    # train the classifier
    clf.fit(train_odd_peak.data_zscored, train_stim)

    # APPLIED THE CLASSIFIERS TO THE RESTING STATE DATA
    # classifier prediction: predict on test data and save the data:
    pred_rest_class = [clf.predict(i) for i in data_rest_final]
    pred_rest_proba = [clf.predict_proba(i) for i in data_rest_final]

    # SAVE THE PREDITION OF RESTING STATE DATA
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
        df_rest_pred['trial'] = np.tile(
                np.arange(1, rest_interval + 1), num_trials)
        # add the run number to the dataframe:
        df_rest_pred['run_study'] = np.repeat(run_labels[t], num_pred)
        # add the session number to the dataframe:
        df_rest_pred['session'] = np.repeat(ses_labels[t], len(df_rest_pred))
        # add the inter trial interval to the dataframe:
        df_rest_pred['tITI'] = np.tile('rest', num_pred)
        # add the participant id to the dataframe:
        df_rest_pred['id'] = np.repeat(df_events['subject'].unique(), num_pred)
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



# 2048ms and specific classifier's prediction for 15 trials with different sequences with 13 TRs




# multinomial decoding
pre_list=copy.deepcopy(data_list)
select_list_multi = np.arange(20,24)
pre_list_append_m = []
pre_list_append_m_sort = []
pred_array = []
pred_matrix = []
pre_list_append_m = [pre_list[i] for i in select_list_multi]
for i in range(4):
    pre_list_append_m[i]['class_index'] = pre_list_append_m[i]['class'].map(lambda a : stim_index(a))

pre_list_append_m_sort= [pre_list_append_m[i].sort_values(by=['class_index','trial']) for i in range(4)]
# get the prediction matrix
pred_array = [np.array(pre_list_append_m_sort[i].loc[:,'probability']) for i in range(4)]
pred_matrix = [np.reshape(pred_array[i],(233,5),order='F') for i in range(4)]


# ovr decoding
pre_list_append_ovr = []
pre_list_append_ovr = pd.concat([pre_list[i] for i in range(len(pre_list))], axis=0)
pre_list_filter_ovr = pre_list_append_ovr[(pre_list_append_ovr["tITI"] == 2.048)&(pre_list_append_ovr["class"] != 'other')]
pre_list_filter_ovr['class_index'] = pre_list_filter_ovr['class'].map(lambda a : stim_index(a))
pre_list_filter_ovr_sort = pre_list_filter_ovr.sort_values(by=['class_index','trial'])
# get the prediction matrix
pred_array = np.array(pre_list_filter_ovr_sort.loc[:,'probability'])
pred_matrix = np.reshape(pred_array,(195,5),order='F')


# # parameters for TDLM
# # beta parameters of 8826 voexls for 5 decoding models
# betas_de_re = clf.coef_.T
# # intercept parameters for 5 decoding models
# intercepts_de_re = clf.intercept_.T

# get all the sequences the participants will be presented in the sequential condition
rp = []
for i in test_seq_long.real_trials-1:
    # real sequence
    temp=test_seq_long.sequences[i,]
    # get the integer of sequence
    rp.append(temp.astype(np.int32))
# all the sequences for the specific subject
real_sequences = np.unique(rp,axis=0)
# all possible sequences
all_sequence = np.array(uniquePerms)
# rest of sequences for the specific subject
set_y = set(map(tuple, real_sequences))
idx = [tuple(point) not in set_y for point in all_sequence]
rest_sequences = all_sequence[idx]
# GLM predefine
betas_1GLM = None
betas_2GLM = None

# the nan matrix for forward design matrix (40 subjects * 4 resting states * 120 shuffles * 2100 timelags)
# sf_rest = np.full([nSubj,resting,nShuf,maxLag_rest+1],np.nan)
sf_rest_less = np.full([resting, len(rest_sequences),maxLag_rest+1],np.nan)
sf_rest_more = np.full([resting, len(real_sequences),maxLag_rest+1],np.nan)
sb_rest_less = np.full([resting, len(rest_sequences),maxLag_rest+1],np.nan)
sb_rest_more = np.full([resting, len(real_sequences),maxLag_rest+1],np.nan)

logging.info('predict the resting data')

# the probability of prediction of training model to sequence data
# classes_names = clf.classes_
# pred_resting_decoding_prob = [1/(1+np.exp(-(np.dot(i,betas_de_re)
#                                     + repmat(intercepts_de_re, len(i),1)))) for i in data_rest_final]

# pred_resting_decoding_clas = [[''.join(clf.classes_[np.where(np.array(pred_resting_decoding_prob[i][j,:])
#                                                               == np.max(np.array(pred_resting_decoding_prob[i][j,:])))])
#                                 for j in range(len(pred_resting_decoding_prob[0]))]
#                               for i in range(len(data_rest_final))]

# plot the decoded state space
plt.figure(dpi=400)
plt.xticks(fontsize=10)
plt.imshow(pred_matrix[3], aspect='auto', interpolation='none',
           origin='lower',cmap='hot')
plt.title('Decoded State Space')
plt.xlabel('States')
plt.ylabel('time points (TR)')
plt.xticks(range(5),('face', 'house', 'cat', 'shoe', 'chair'))
plt.colorbar()
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Resting\\decoded state space-%s.png' % sub),
            dpi = 400,bbox_inches = 'tight')
plt.show()

#less frequent sequences
for i in range(resting):
    print(i)
    X = pred_matrix[i]
    for j,jseq in zip(range(len(rest_sequences)),rest_sequences):
        # real sequence
        seq = jseq
        # get the integer of sequence
        seq = np.array(seq, dtype=int)
        # get the transition matrix (forward)
        T1 = TransM(seq)
        # get the transition matrix (backward)
        T2 = np.transpose(T1)
        # bins
        nbins = maxLag_rest+1
        # timelag matrix
        dm = scipy.linalg.toeplitz(X[:,0],[np.zeros((nbins,1))])#####
        dm = dm[:,1:]

        # 4 loops for another 4 states
        for k in range(1,nstates):
            temp = scipy.linalg.toeplitz(X[:,k],[np.zeros((nbins,1))]);
            temp = temp[:,1:]
            dm = np.hstack((dm,temp))

        # the next time point needed to be predicted
        Y=X
        # build a new framework for first GLM betas
        betas_1GLM = np.full([nstates*maxLag_rest, nstates],np.nan)

        #detect for each timelag
        for l in range(maxLag_rest):
            temp_zinds = np.array(range(0,nstates*maxLag_rest,maxLag_rest)) + l
            # Multinomial regression in First GLM
            design_mat_1 = np.hstack((dm[:,temp_zinds], np.ones((len(dm[:,temp_zinds]),1))))
            temp = np.dot(np.linalg.pinv(design_mat_1), Y)
            betas_1GLM[temp_zinds,:] = temp[:-1,:]
            # Single regression in First GLM
            # design_mat_1=[]
            # design_mat_1 = [np.hstack((dm[:,temp_zinds[m:m+1]],np.ones((len(dm[:,temp_zinds]),1)))) for m in range(5)]
            # temp = [np.dot(np.linalg.pinv(design_mat_1[m]),Y) for m in range(5)]
            # for m in range(5):
            #     betas_1GLM[temp_zinds[m:m+1],:] = temp[m][:-1,:]####?
        betasnbins64 = np.reshape(betas_1GLM,[maxLag_rest,np.square(nstates)],order = "F")#####
        # Second GLM
        design_mat_2 = np.transpose(np.vstack((np.reshape(T1,25,order = "F"),
                                               np.reshape(T2,25,order = "F"),
                                               np.reshape(np.eye(nstates),25,order = "F"),
                                               np.reshape(np.ones([nstates,nstates]),25,order = "F"))))
        betas_2GLM = np.dot(np.linalg.pinv(design_mat_2),np.transpose(betasnbins64))
        # four different design matrix for regressor multiple to the temporal data
        # linear regression for the backward and forward replay

        sf_rest_less[i,j,1:] = betas_2GLM[0,:]
        sb_rest_less[i,j,1:] = betas_2GLM[1,:]
        # post-resting: i = 0 and 2
        # pre-resting: i = 1 and 3


    for j,jseq in zip(range(len(real_sequences)),real_sequences):
        # real sequence
        seq = jseq
        # get the integer of sequence
        seq = np.array(seq, dtype=int)
        # get the transition matrix (forward)
        T1 = TransM(seq)
        # get the transition matrix (backward)
        T2 = np.transpose(T1)
        # bins
        nbins = maxLag_rest+1
        # timelag matrix
        dm = scipy.linalg.toeplitz(X[:,0],[np.zeros((nbins,1))])#####
        dm = dm[:,1:]

        # 4 loops for another 4 states
        for k in range(1,nstates):
            temp = scipy.linalg.toeplitz(X[:,k],[np.zeros((nbins,1))]);
            temp = temp[:,1:]
            dm = np.hstack((dm,temp))

        # the next time point needed to be predicted
        Y=X
        # build a new framework for first GLM betas
        betas_1GLM = np.full([nstates*maxLag_rest, nstates],np.nan)

        #detect for each timelag
        for l in range(maxLag_rest):
            temp_zinds = np.array(range(0,nstates*maxLag_rest,maxLag_rest)) + l
            # Multinomial regression in First GLM
            design_mat_1 = np.hstack((dm[:,temp_zinds], np.ones((len(dm[:,temp_zinds]),1))))
            temp = np.dot(np.linalg.pinv(design_mat_1), Y)
            betas_1GLM[temp_zinds,:] = temp[:-1,:]
            # Single regression in First GLM
            # design_mat_1=[]
            # design_mat_1 = [np.hstack((dm[:,temp_zinds[m:m+1]],np.ones((len(dm[:,temp_zinds]),1)))) for m in range(5)]
            # temp = [np.dot(np.linalg.pinv(design_mat_1[m]),Y) for m in range(5)]
            # for m in range(5):
            #     betas_1GLM[temp_zinds[m:m+1],:] = temp[m][:-1,:]####?
        betasnbins64 = np.reshape(betas_1GLM,[maxLag_rest,np.square(nstates)],order = "F")#####
        # Second GLM
        design_mat_2 = np.transpose(np.vstack((np.reshape(T1,25,order = "F"),
                                               np.reshape(T2,25,order = "F"),
                                               np.reshape(np.eye(nstates),25,order = "F"),
                                               np.reshape(np.ones([nstates,nstates]),25,order = "F"))))
        betas_2GLM = np.dot(np.linalg.pinv(design_mat_2),np.transpose(betasnbins64))
        # four different design matrix for regressor multiple to the temporal data
        # linear regression for the backward and forward replay

        sf_rest_more[i,j,1:] = betas_2GLM[0,:]
        sb_rest_more[i,j,1:] = betas_2GLM[1,:]

        # post-resting: i = 0 and 2
        # pre-resting: i = 1 and 3

# mean the all the more frequent sequence in forward replay
sf_rest_more_meanseq = sf_rest_more.mean(axis=1)
sf_rest_more_post = sf_rest_more[0,:,:]
sf_rest_more_pre = sf_rest_more[1,:,:]

sf_rest_more_post_m = sf_rest_more_post.mean(axis=0)
sf_rest_more_pre_m = sf_rest_more_pre.mean(axis=0)
# sf_rest_more_meanseq_s2post = sf_rest_more_meanseq[2,:]
# sf_rest_more_meanseq_s2pre = sf_rest_more_meanseq[3,:]
sf_rest_more_meanseq_dif = sf_rest_more_meanseq_post - sf_rest_more_meanseq_pre
# mean the all the less frequent sequence in forward replay
sf_rest_less_meanseq = sf_rest_less.mean(axis=1)
sf_rest_less_meanseq_post = sf_rest_less_meanseq[0,:]
sf_rest_less_meanseq_pre = sf_rest_less_meanseq[1,:]
# sf_rest_less_meanseq_s2post = sf_rest_less_meanseq[2,:]
# sf_rest_less_meanseq_s2pre = sf_rest_less_meanseq[3,:]
sf_rest_less_meanseq_dif = sf_rest_less_meanseq_post - sf_rest_less_meanseq_pre
# mean the all the more frequent sequence in backward replay
sb_rest_more_meanseq = sb_rest_more.mean(axis=1)
sb_rest_more_meanseq_post = sb_rest_more_meanseq[0,:]
sb_rest_more_meanseq_pre = sb_rest_more_meanseq[1,:]
# sb_rest_more_meanseq_s2post = sb_rest_more_meanseq[2,:]
# sb_rest_more_meanseq_s2pre = sb_rest_more_meanseq[3,:]
sb_rest_more_meanseq_dif = sb_rest_more_meanseq_post - sb_rest_more_meanseq_pre
# mean the all the less frequent sequence in backward replay
sb_rest_less_meanseq = sb_rest_less.mean(axis=1)
sb_rest_less_meanseq_post = sb_rest_less_meanseq[0,:]
sb_rest_less_meanseq_pre = sb_rest_less_meanseq[1,:]
# sb_rest_less_meanseq_s2pre = sb_rest_less_meanseq[3,:]
# sb_rest_less_meanseq_s2post = sb_rest_less_meanseq[2,:]
sb_rest_less_meanseq_dif = sb_rest_less_meanseq_post - sb_rest_less_meanseq_pre



# PLOT THE SEQUENCENESS
plt.figure(dpi=400)
plt.xticks(fontsize=10)
x=np.arange(0,nbins)
l1=plt.plot(x,sf_rest_less_meanseq_pre,color='c', linestyle='--',label='less-pre-rest')
l2=plt.plot(x,sf_rest_less_meanseq_post,color='lightblue', linestyle='-',label='less-post-rest')
l3=plt.plot(x,sf_rest_more_pre_m,color='lightcoral', linestyle=':',label='more-pre-rest')
l4=plt.plot(x,sf_rest_more_post_m,color='rosybrown', linestyle='solid',label='more-post-rest')
# l5=plt.plot(x,sf_rest_less_meanseq_s2pre,color='dodgerblue', linestyle='--',label='s2-less-pre-rest')
# l6=plt.plot(x,sf_rest_less_meanseq_s2post,color='steelblue', linestyle='-',label='s2-less-post-rest')
# l7=plt.plot(x,sf_rest_more_meanseq_s2pre,color='peru', linestyle=':',label='s2-more-pre-rest')
# l8=plt.plot(x,sf_rest_more_meanseq_s2post,color='orangered', linestyle='solid',label='s2-more-post-rest')
# plt.plot(x,sf_rest_less_meanseq_pre,color='c', linestyle='-')
# plt.plot(x,sf_rest_less_meanseq_post,color='lightblue', linestyle='--')
# plt.plot(x,sf_rest_more_meanseq_pre,color='lightcoral', linestyle=':')
# plt.plot(x,sf_rest_more_meanseq_post,color='rosybrown', linestyle='solid')
# plt.plot(x,sf_rest_less_meanseq_s2pre,color='dodgerblue', linestyle='--')
# plt.plot(x,sf_rest_less_meanseq_s2post,color='steelblue', linestyle='-')
# plt.plot(x,sf_rest_more_meanseq_s2pre,color='peru', linestyle=':')
# plt.plot(x,sf_rest_more_meanseq_s2post,color='orangered', linestyle='solid')
plt.title('Forward Replay')
plt.xlabel('Timelag')
plt.ylabel('Sequenceness')
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Resting\\Forward Replay-%s.png' % sub),
            dpi = 400,bbox_inches = 'tight')
plt.show()


plt.figure(dpi=400)
plt.xticks(fontsize=10)
x=np.arange(0,nbins)
l1=plt.plot(x,sb_rest_less_meanseq_pre,color='c', linestyle='--',label='less-pre-rest')
l2=plt.plot(x,sb_rest_less_meanseq_post,color='lightblue', linestyle='-',label='less-post-rest')
l3=plt.plot(x,sb_rest_more_meanseq_pre,color='lightcoral', linestyle=':',label='more-pre-rest')
l4=plt.plot(x,sb_rest_more_meanseq_post,color='rosybrown', linestyle='solid',label='more-post-rest')
# l5=plt.plot(x,sb_rest_less_meanseq_s2pre,color='dodgerblue', linestyle='--',label='s2-less-pre-rest')
# l6=plt.plot(x,sb_rest_less_meanseq_s2post,color='steelblue', linestyle='-',label='s2-less-post-rest')
# l7=plt.plot(x,sb_rest_more_meanseq_s2pre,color='peru', linestyle=':',label='s2-more-pre-rest')
# l8=plt.plot(x,sb_rest_more_meanseq_s2post,color='orangered', linestyle='solid',label='s2-more-post-rest')
# plt.plot(x,sb_rest_less_meanseq_s1pre,color='c', linestyle='-')
# plt.plot(x,sb_rest_less_meanseq_s1post,color='lightblue', linestyle='--')
# plt.plot(x,sb_rest_more_meanseq_s1pre,color='lightcoral', linestyle=':')
# plt.plot(x,sb_rest_more_meanseq_s1post,color='rosybrown', linestyle='solid')
# plt.plot(x,sb_rest_less_meanseq_s2pre,color='dodgerblue', linestyle='--')
# plt.plot(x,sb_rest_less_meanseq_s2post,color='steelblue', linestyle='-')
# plt.plot(x,sb_rest_more_meanseq_s2pre,color='peru', linestyle=':')
# plt.plot(x,sb_rest_more_meanseq_s2post,color='orangered', linestyle='solid')
plt.title('Backward Replay')
plt.xlabel('Timelag')
plt.ylabel('Sequenceness')
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Resting\\Backward Replay-%s.png' % sub),
            dpi = 400,bbox_inches = 'tight')
plt.show()


plt.figure(dpi=400)
plt.xticks(fontsize=10)
x=np.arange(0,nbins)
l1=plt.plot(x,sf_rest_more_meanseq_dif,color='c', linestyle='--',label='Post-Pre More frequent')
l2=plt.plot(x,sf_rest_less_meanseq_dif,color='lightblue', linestyle='-',label='Post-Pre Less frequent')
plt.title('Forward Replay')
plt.xlabel('Timelag')
plt.ylabel('Forward Sequenceness')
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Resting\\Forward Replay-difference-%s.png' % sub),
            dpi = 400,bbox_inches = 'tight')
plt.show()


plt.figure(dpi=400)
plt.xticks(fontsize=10)
x=np.arange(0,nbins)
l1=plt.plot(x,sb_rest_more_meanseq_dif,color='c', linestyle='--',label='Post-Pre More frequent')
l2=plt.plot(x,sb_rest_less_meanseq_dif,color='lightblue', linestyle='-',label='Post-Pre Less frequent')
plt.title('Backward Replay')
plt.xlabel('Timelag')
plt.ylabel('Backward Sequenceness')
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Resting\\Backward Replay-difference-%s.png' % sub),
            dpi = 400,bbox_inches = 'tight')
plt.show()