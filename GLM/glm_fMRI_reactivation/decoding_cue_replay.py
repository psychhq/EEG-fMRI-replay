#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: FMRI-DECODING
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITVIE NERUOSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

# In[IMPORT RELEVANT PACKAGES]:
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

        if task == 'REP':
            self.events = events.loc[
                          (events["task"] == 'REP'),
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
            self.define_trs_pre()
            self.rmheadmotion_rep()
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

    def rmheadmotion_rep(self):
        self.fd = self.confounds.loc[:, 'framewise_displacement'].fillna(
            0).reset_index(drop=True)
        self.extras = np.array(self.fd[self.fd.values > 0.2].index)
        self.rmheadmotions = np.zeros(len(self.stim_trs))
        for i in range(len(self.stim_trs)):
            if any(extra in np.arange(self.stim_trs[i], (self.stim_trs[i] + self.interval)) for extra in self.extras):
                self.rmheadmotions[i] = 1
            else:
                self.rmheadmotions[i] = 0
        self.events['rmheadmotions'] = self.rmheadmotions
        self.events_rm = self.events

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


def multiunion(arrays):
    import copy
    # start with the first array:
    array_union = copy.deepcopy(arrays[0].astype(np.int32))
    # loop through all arrays
    for i in range(1, len(arrays)):
        # multiply every array with all previous array
        array_union = array_union + copy.deepcopy(arrays[i].astype(np.int32))
    array_union = np.array(array_union, dtype=bool)
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
    plt.savefig(opj(path, filename),
                dpi=400, bbox_inches='tight')
    plt.show()

# In[SETUP NECESSARY PATHS ETC]:
# name of the current project:
sub_list = None
subject = None
# path to the project root:
project_name = 'fmrireplay-replay-TDLM'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_code = opj(path_bids, 'code', 'decoding-TDLM')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_masks = opj(path_bids, 'derivatives', 'masks')
path_level1_vfl = opj(path_bids, "derivatives", "glm-vfl", "l1pipeline")
path_decoding = opj(path_bids, 'derivatives', 'decoding')
path_behavior = opj(path_bids, 'derivatives', 'behavior')
path_out_replay = opj(path_decoding, 'all_subject_results', 'replay_decoding')
path_out_meanreplay = opj(path_decoding, 'all_subject_results', 'mean_replay')
# CREATE OUTPUT DIRECTORIES IF THE DO NOT EXIST YET:
for path in [path_out_replay, path_out_meanreplay]:
    if not os.path.exists(path):
        os.makedirs(path)

# load the learning sequence
learning_sequence = pd.read_csv(opj(path_behavior, 'sequence.csv'), sep=',')

# define the subject id
layout = BIDSLayout(path_bids)
sub_list = sorted(layout.get_subjects())
# delete the subjects
# lowfmri = [37, 16, 15, 13, 11, 0]
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

# In[paramters]
# the number of VFL runs
n_run = 4
# some parameters
data_list_all = []
pred_acc_mean_sub_all = []
# the trial duration (tr)
long_interval = 8
# parameters for TDLM
# the list of all the sequences
uniquePerms = list(itertools.permutations(
    [1, 2, 3, 4], 4))  # all possibilities
# except the last one, because backward direction equal to the correct sequence
# uniquePerms = uniquePerms[0:-1]
# the number of sequences
nShuf = len(uniquePerms)
# all possible sequences
all_sequence = np.array(uniquePerms)
# the number of timelags
maxLag = 8
nbins = maxLag + 1
# the number of states (decoding models)
nstates = 4
# the number of subjects
nSubj = len(sub_list)
# the nan matrix for forward design matrix (2 directions * 23 shuffles * 10 timelags * 2 experimental conditions)
sf = np.full([2, len(all_sequence), nbins, 2], np.nan)
# predefine GLM data frame
betas_1GLM = None
betas_2GLM = None

# for test the code
# sub_list = sub_list[2:4]
# create the mask list and bold delay list
mask_name_list = ['mask_visual', 'mask_vis_mtl', 'mask_temporal',
                  'mask_prefrontal', 'mask_hippocampus', 'mask_entorhinal', 'mask_mtl']
mask_index_list = [0, 1, 2, 3]
task_bold_delay_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# the parameter for test
# subject = sub_list[0]
mask_index = mask_index_list[1]
task_bold_delay = task_bold_delay_list[2]
mask_name = mask_name_list[mask_index]
task_peak_tr = task_bold_delay


# In[for parallel function]:
def decoding(subject):
    # for subject in sub_list:
        # subject = sub_list[0]
    # print(subject)
    # In[CREATE PATHS TO OUTPUT DIRECTORIES]:
    path_out = opj(path_decoding, subject)
    path_out_figs = opj(path_out, "plots")
    path_out_data = opj(path_out, "data")
    path_out_logs = opj(path_out, "logs")
    path_out_masks = opj(path_out, "masks")
    # CREATE OUTPUT DIRECTORIES IF THE DO NOT EXIST YET:
    for path in [path_out_figs, path_out_data, path_out_logs, path_out_masks]:
        if not os.path.exists(path):
            os.makedirs(path)

    # In[SETUP LOGGING]:
    # remove all handlers associated with the root logger object:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # get current data and time as a string:
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    # create path for the logging file
    log = opj(path_out_logs, "%s-%s.log" % (timestr, subject))
    # start logging:
    logging.basicConfig(
        filename=log,
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    # In[LOAD BEHAVIORAL DATA (THE EVENTS.TSV FILES OF THE SUBJECT)]:
    # paths to all events files of the current subject:
    path_events = opj(path_behavior, subject, "*tsv")
    # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
    path_events = sorted(glob.glob(path_events),
                         key=lambda f: os.path.basename(f))

    # paths to all events files of the current subject:
    path_events_vfl = opj(path_behavior, subject, "*vfl*events.tsv")
    # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
    path_events_vfl = sorted(glob.glob(path_events_vfl),
                             key=lambda f: os.path.basename(f))
    # import events file and save data in dataframe:
    df_events_vfl = [pd.read_csv(path_events_vfl[f], sep="\t")
                     for f in range(len(path_events_vfl))]
    df_events_vfl = pd.concat(
        [df_events_vfl[0], df_events_vfl[1], df_events_vfl[2], df_events_vfl[3]])
    df_events_vfl['stim_label'] = df_events_vfl.apply(
        lambda x: classlabel(x.stimMarker), axis=1)
    del df_events_vfl['Unnamed: 0']
    del df_events_vfl['start']

    # paths to all events files of the current subject:
    path_events_rep = opj(path_behavior, subject, "*rep*events.tsv")
    # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
    path_events_rep = sorted(glob.glob(path_events_rep),
                             key=lambda f: os.path.basename(f))
    # import events file and save data in dataframe:
    # df_events_rep = pd.read_csv(path_events_rep, sep="\t")
    df_events_rep = pd.concat(
        (pd.read_csv(f, sep="\t") for f in path_events_rep), ignore_index=True
    )
    df_events_rep['stim_label'] = df_events_rep.apply(
        lambda x: classlabel(x.Marker), axis=1)
    del df_events_rep['Unnamed: 0']
    del df_events_rep['start']

    # load the confound file for VFL
    path_confounds_vfl = glob.glob(
        opj(path_fmriprep, subject, 'func', '*vfl*confounds_timeseries.tsv'))
    df_confound_vfl_list = [pd.read_csv(
        i, sep="\t") for i in path_confounds_vfl]
    df_confound_vfl = pd.concat([df_confound_vfl_list[0],
                                 df_confound_vfl_list[1],
                                 df_confound_vfl_list[2],
                                 df_confound_vfl_list[3]])

    # load the confound file for cue replay task
    path_confounds_rep = glob.glob(
        opj(path_fmriprep, subject, 'func', '*replay*confounds_timeseries.tsv'))
    df_confound_rep_list = [pd.read_csv(i, sep="\t")
                            for i in path_confounds_rep]
    df_confound_rep = pd.concat([df_confound_rep_list[0],
                                 df_confound_rep_list[1],
                                 df_confound_rep_list[2]])

    # In[CREATE PATHS TO THE MRI DATA]:
    # define path to input directories:
    mask_name_list = ['mask_visual', 'mask_vis_mtl', 'mask_temporal',
                      'mask_prefrontal', 'mask_hippocampus', 'mask_entorhinal', 'mask_mtl']

    def load_mask_file(mask_name):
        from os.path import join as opj
        import glob
        from nilearn import image
        import copy
        path_mask_task = opj(path_masks, mask_name,
                             subject, "*", "*task*.nii.gz")
        path_mask_task = sorted(glob.glob(path_mask_task),
                                key=lambda f: os.path.basename(f))
        mask = [np.asanyarray(image.load_img(i).dataobj)
                for i in copy.deepcopy(path_mask_task[-4:])]
        return mask

    # load mask files:
    mask_visual = load_mask_file(mask_name_list[0])
    mask_vis_mtl = load_mask_file(mask_name_list[1])
    mask_temporal = load_mask_file(mask_name_list[2])
    mask_prefrontal = load_mask_file(mask_name_list[3])
    mask_hippocampus = load_mask_file(mask_name_list[4])
    mask_entorhinal = load_mask_file(mask_name_list[5])
    mask_mtl = load_mask_file(mask_name_list[6])

    # load the visual functional localizer mri task files:
    path_func_task_vfl = opj(path_level1_vfl, "smooth",
                             subject, "*", "*task*nii.gz")
    path_func_task_vfl = sorted(
        glob.glob(path_func_task_vfl), key=lambda f: os.path.basename(f))
    # load smoothed functional mri data for all task runs:
    data_task_vfl = [image.load_img(i) for i in path_func_task_vfl]

    # load the cue replay mri task files:
    path_func_task_rep = opj(path_masks, "smooth",
                             subject, "*", "*task-replay*nii.gz")
    path_func_task_rep = sorted(
        glob.glob(path_func_task_rep), key=lambda f: os.path.basename(f))
    # load smoothed functional mri data for all task runs:
    data_task_rep = [image.load_img(i) for i in path_func_task_rep]

    # load the anatomical mri file:
    path_anat = opj(path_fmriprep, subject, "anat",
                    "%s_desc-preproc_T1w.nii.gz" % subject)
    path_anat = sorted(glob.glob(path_anat), key=lambda f: os.path.basename(f))
    anat = image.load_img(path_anat[0])

    # load the t-maps of the first-level glm:
    path_tmap_vfl = opj(path_level1_vfl, "contrasts",
                        subject, "*", "spmT*.nii")
    path_tmap_vfl = sorted(glob.glob(path_tmap_vfl),
                           key=lambda f: os.path.dirname(f))
    tmaps_vfl = [np.asanyarray(image.load_img(i).dataobj)
                 for i in copy.deepcopy(path_tmap_vfl)]

    # In[FEATURE SELECTION FOR VISUAL FUNCTIONAL LOCALIZER TASK]:
    # load the mask for feature selection
    # , mask_hippocampus, mask_entorhinal, mask_mtl]
    mask_list = [mask_visual, mask_vis_mtl, mask_temporal, mask_prefrontal]
    mask_name_list = ['mask_visual', 'mask_vis_mtl', 'mask_temporal',
                      'mask_prefrontal', 'mask_hippocampus', 'mask_entorhinal', 'mask_mtl']
    mask = mask_list[mask_index]
    mask_name = mask_name_list[mask_index]

    # FEATURE SELECTION: MASKS THE T-MAPS WITH THE ANATOMICAL MASKS
    # plot raw-tmaps on an anatomical background:
    for i, path in enumerate(path_tmap_vfl):
        path_save = opj(path_out_figs, "%s_run-%02d_tmap_raw.png" %
                        (subject, i + 1))
        plotting.plot_roi(
            path,
            anat,
            title=os.path.basename(path_save),
            output_file=path_save,
            colorbar=True, )

    # FEATURE SELECTION: MASKS THE T-MAPS WITH THE ANATOMICAL MASKS
    # check if any value in the supposedly binary mask is bigger than 1:
    for i in mask:
        if np.any(i > 1):
            logging.info("WARNING: detected values > 1 in the anatomical ROI!")
            sys.exit("Values > 1 in the anatomical ROI!")
    # get combination of anatomical mask and t-map
    tmaps_masked = [np.multiply(i, j) for (i, j) in zip(
        copy.deepcopy(mask), copy.deepcopy(tmaps_vfl))]
    # masked tmap into image like object:
    tmaps_masked_img = [image.new_img_like(i, j) for (
        i, j) in zip(path_tmap_vfl, copy.deepcopy(tmaps_masked))]
    for i, path in enumerate(tmaps_masked_img):
        path_save = opj(path_out_masks, "%s_run-%02d_tmap_masked_%s.nii.gz" %
                        (subject, i + 1, mask_name))
        path.to_filename(path_save)
    # plot masked t-maps
    for i, path in enumerate(tmaps_masked_img):
        logging.info('plotting masked tmap %d of %d' % (i + 1, len(tmaps_masked_img)))
        path_save = opj(path_out_figs, '%s_run-%02d_tmap_masked_%s.png' %
                        (subject, i + 1, mask_name))
        plotting.plot_roi(path, anat, title=os.path.basename(path_save),
                          output_file=path_save, colorbar=True)

    # FEATURE SELECTION: THRESHOLD THE MASKED T-MAPS
    # set the threshold:
    threshold = 3
    # threshold the masked tmap image:
    tmaps_masked_thresh_img = [image.threshold_img(
        i, threshold) for i in copy.deepcopy(tmaps_masked_img)]
    # plot threshold mask tmap:
    for i, path in enumerate(tmaps_masked_thresh_img):
        path_save = opj(path_out_figs, '%s_run-%02d_tmap_masked_thresh_%s.png' %
                        (subject, i + 1, mask_name))
        plotting.plot_roi(path, anat, title=os.path.basename(path_save),
                          output_file=path_save, colorbar=True)

    # extract data from the thresholded images
    tmaps_masked_thresh = [np.asanyarray(image.load_img(
        i).dataobj) for i in tmaps_masked_thresh_img]

    # calculate the number of tmap voxels:
    num_tmap_voxel = [np.size(i) for i in copy.deepcopy(tmaps_masked_thresh)]
    num_above_voxel = [np.count_nonzero(i)
                       for i in copy.deepcopy(tmaps_masked_thresh)]
    num_below_voxel = [np.count_nonzero(i == 0)
                       for i in copy.deepcopy(tmaps_masked_thresh)]

    # plot the distribution of t-values:
    for i, run_mask in enumerate(tmaps_masked_thresh):
        masked_tmap_flat = run_mask.flatten()
        masked_tmap_flat = masked_tmap_flat[~np.isnan(masked_tmap_flat)]
        masked_tmap_flat = masked_tmap_flat[~np.isnan(
            masked_tmap_flat) & ~(masked_tmap_flat == 0)]
        path_save = opj(path_out_figs, '%s_run-%02d_tvalue_distribution_%s.png' %
                        (subject, i + 1, mask_name))
        plt.figure()
        plt.hist(masked_tmap_flat, bins='auto')
        plt.xlabel('t-values')
        plt.ylabel('number')
        plt.title('t-value distribution (_%s, run-%02d)_%s' %
                  (subject, i + 1, mask_name))
        plt.savefig(path_save)

    # create a dataframe with the number of voxels
    df_thresh = pd.DataFrame(
        {"id": [subject] * n_run,
         "run": np.arange(1, n_run + 1),
         "n_total": num_tmap_voxel,
         "n_above": num_above_voxel,
         "n_below": num_below_voxel,
         }
    )
    file_name = opj(path_out_data, "%s_%s_thresholding.csv" %
                    (subject, mask_name))
    df_thresh.to_csv(file_name, sep=",", index=False)

    # FEATURE SELECTION: BINARIZE THE THRESHOLDED MASKED T-MAPS
    # replace all NaNs with 0:
    tmaps_masked_thresh_bin = [
        np.where(np.isnan(i), 0, i) for i in copy.deepcopy(tmaps_masked_thresh)]
    # replace all other values with 1:
    tmaps_masked_thresh_bin = [
        np.where(i > 0, 1, i) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
    # turn the 3D-array into booleans:
    tmaps_masked_thresh_bin = [
        i.astype(bool) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
    # create image like object:
    masks_final_vfl = [
        image.new_img_like(path_func_task_vfl[0], i.astype(np.int32))
        for i in copy.deepcopy(tmaps_masked_thresh_bin)]

    for i, path in enumerate(masks_final_vfl):
        filename = "%s_run-%02d_tmap_masked_thresh_%s.nii.gz" % (
            subject, i + 1, mask_name)
        path_save = opj(path_out_masks, filename)
        path.to_filename(path_save)
        path_save = opj(path_out_figs, '%s_run-%02d_visual_final_mask_%s.png'
                        % (subject, i + 1, mask_name))
        plotting.plot_roi(path, anat, title=os.path.basename(path_save),
                          output_file=path_save, colorbar=True)

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
    # clf_set = {
    #     key: LogisticRegression(
    #         C=1.0,  # Inverse of regularization strength
    #         penalty="l1",
    #         multi_class="ovr",
    #         solver="liblinear",  # Algorithm to use in the optimization problem, Stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno
    #         max_iter=4000,
    #         class_weight="balanced",
    #         random_state=42,  # to shuffle the data
    #     )
    #     for key in class_labels
    # }
    # classifiers = {
    #     "log_reg": LogisticRegression(
    #         C=1.0,
    #         penalty="l1",
    #         multi_class="multinomial",
    #         solver="saga",
    #         max_iter=4000,
    #         class_weight="balanced",
    #         random_state=42,
    #     )
    # }
    # clf_set.update(classifiers)

    # clf_set = {
    #     key: RidgeClassifier(
    #         alpha=1.0,
    #         normalize=False,
    #         max_iter=4000,
    #         class_weight='balanced',
    #         solver='cholesky',
    #         positive=False,
    #         random_state=42
    #     )
    #     for key in class_labels
    # }
    # RidgeClassifier(alpha=1.0, *, fit_intercept=True, normalize='deprecated', copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', positive=False, random_state=None)

    # In[DEFINE THE TASK CONDITION]:

    # 1. SPLIT THE EVENTS DATAFRAME FOR EACH TASK CONDITION
    # 2. RESET THE INDICES OF THE DATAFRAMES
    # 3. SORT THE ROWS OF ALL DATAFRAMES IN CHRONOLOGICAL ORDER
    # 4. PRINT THE NUMBER OF TRIALS OF EACH TASK CONDITION

    # for task_bold_delay in [3,4,5,6]:
    train_VFL_peak = TaskData(
        events=df_events_vfl,
        confounds=df_confound_vfl,
        task="VFL",
        bold_delay=task_bold_delay,
        interval=1,
        name="train-VFL_peak",
        num_vol_run_1=np.size(data_task_vfl[0].dataobj, axis=3),
        num_vol_run_2=np.size(data_task_vfl[1].dataobj, axis=3),
        num_vol_run_3=np.size(data_task_vfl[2].dataobj, axis=3),
        num_vol_run_4=np.size(data_task_vfl[3].dataobj, axis=3),
    )

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

    # In[SEPARATE RUN-WISE STANDARDIZATION (Z-SCORING) OF VISUAL FUNCTIONAL LOCALIZER]:
    data_list = []
    fold_list = [1, 2, 3, 4, 5, 6, 7, 8]
    # SAVE THE UNION MASK (get interesction mask from the four-task-run t-tmap threshold masks )
    mask_label = 'union'
    masks_union = multimultiply(
        tmaps_masked_thresh_bin).astype(int).astype(bool)
    masks_union_nii = image.new_img_like(path_func_task_vfl[0], masks_union)
    path_save = opj(
        path_out_masks, '{}_cue_replay_mask-{}.nii.gz'.format(subject, mask_label))
    masks_union_nii.to_filename(path_save)
    # DETREND AND Z-STANDARDIZED THE RESTING FMRI DATA
    # mask all VFL task runs separately:
    data_task_vfl_masked = [masking.apply_mask(
        i, masks_union_nii) for i in data_task_vfl]
    # mask all cue replay runs with the averaged feature seleasction masks:
    data_task_rep_masked = [masking.apply_mask(
        i, masks_union_nii) for i in data_task_rep]
    # detrend each VFL task run separately:
    data_task_vfl_masked_detrend = [
        clean(i, detrend=True, standardize=False) for i in data_task_vfl_masked]
    # detrend each cue replay run separately:
    data_task_rep_masked_detrend = [
        clean(i, detrend=True, standardize=False) for i in data_task_rep_masked]
    # combine the detrended data of all runs:
    data_task_vfl_masked_detrend = np.vstack(data_task_vfl_masked_detrend)
    data_task_rep_masked_detrend = np.vstack(data_task_rep_masked_detrend)
    # loop through all classifiers in the classifier set:
    for clf_name, clf in clf_set.items():
        # print classifier:
        logging.info("classifier: %s" % clf_name)
        # fit the classifier to the training data:
        train_VFL_peak.zscore(
            signals=data_task_vfl_masked_detrend, fold_list=fold_list)
        train_stim = copy.deepcopy(train_VFL_peak.stim)
        # replace labels for single-label classifiers:
        if clf_name in class_labels:
            # replace all other labels with other
            train_stim = ["other" if x != clf_name else x for x in train_stim]
            # turn into a numpy array
            train_stim = np.array(train_stim, dtype=object)
        # check weights:
        # show_weights(array=train_stim)
        # train the classifier
        clf.fit(train_VFL_peak.data_zscored, train_stim)
        # classifier prediction: predict on test data and save the data:
        test_rep_long.zscore(signals=data_task_rep_masked_detrend,
                             fold_list=fold_list)
        if test_rep_long.data_zscored.size < 0:
            continue
        # create dataframe containing classifier predictions:
        df_pred = test_rep_long.predict(clf=clf, fold_list=fold_list)
        # add the current classifier as a new column:
        df_pred["classifier"] = np.repeat(clf_name, len(df_pred))
        # add a label that indicates the mask / training regime:
        df_pred["mask"] = np.repeat(mask_label, len(df_pred))
        # melt the data frame:
        df_pred_melt = melt_df(df=df_pred, melt_columns=train_stim)
        # append dataframe to list of dataframe results:
        data_list.append(df_pred_melt)

    # In[PREDICTION PROBABILITY AND ACCURACY FOR PEAK]:
    # temporal prediction probability
    pre_list = copy.deepcopy(data_list)
    pre_list_append = pd.concat([pre_list[i]
                                 for i in range(len(pre_list))], axis=0)

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


    # In[TEMPORAL DYNAMIC PROBABILITY FOR FORWARD AND BACKWARD MENTAL SIMULATION]:
    # Temporal_probability_each is a function to plot the dynamic probability
    # Temporal_probability_each('1-2-3-4', pred_matrix_51_seq, long_interval, subject, path_out_figs)
    # Temporal_probability_each('4-3-2-1', pred_matrix_52_seq, long_interval, subject, path_out_figs)

    # return [data_list, pred_matrix_51_seq, pred_matrix_52_seq, sf]


# In[PARALLEL FUNCTION]

# sf_list =  Parallel(n_jobs=60)(delayed(decoding)(subject) for subject in sub_list)
# pred_matrix =  Parallel(n_jobs=60)(delayed(decoding)(subject) for subject in sub_list)
# data_activation_list =  Parallel(n_jobs=60)(delayed(decoding)(subject) for subject in sub_list)
# cross_TDLM = Parallel(n_jobs=60)(delayed(decoding)(subject) for subject in sub_list)
all_results_list = Parallel(n_jobs=60)(
    delayed(decoding)(subject) for subject in sub_list)
