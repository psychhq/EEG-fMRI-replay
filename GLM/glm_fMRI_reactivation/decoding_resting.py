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
from scipy import stats
from scipy.interpolate import interp1d
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
from scipy.stats import gamma

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

def multiunion(arrays):
    import copy
    # start with the first array:
    array_union = copy.deepcopy(arrays[0].astype(np.int32))
    # loop through all arrays
    for i in range(1, len(arrays)):
        # multiply every array with all previous array
        array_union = array_union + copy.deepcopy(arrays[i].astype(np.int32))
    array_union = np.array(array_union, dtype = bool)
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


def matlab_percentile(x, p):
    p = np.asarray(p, dtype=float)
    n = len(x)
    p = (p - 50) * n / (n - 1) + 50
    p = np.clip(p, 0, 100)
    return np.percentile(x, p)


def _sample_condition(exp_condition, frame_times, oversampling=50,
                      min_onset=-24):
    """Make a possibly oversampled event regressor from condition information.

    Parameters
    ----------
    exp_condition : arraylike of shape (3, n_events)
        yields description of events for this condition as a
        (onsets, durations, amplitudes) triplet

    frame_times : array of shape(n_scans)
        Sample time points.

    oversampling : int, optional
        Factor for oversampling event regressor. Default=50.

    min_onset : float, optional
        Minimal onset relative to frame_times[0] (in seconds)
        events that start before frame_times[0] + min_onset are not considered.
        Default=-24.

    Returns
    -------
    regressor : array of shape(over_sampling * n_scans)
        Possibly oversampled event regressor.

    hr_frame_times : array of shape(over_sampling * n_scans)
        Time points used for regressor sampling.

    """
    # Find the high-resolution frame_times
    n = frame_times.size
    min_onset = float(min_onset)
    n_hr = ((n - 1) * 1. / (frame_times.max() - frame_times.min())
            * (frame_times.max() * (1 + 1. / (n - 1)) - frame_times.min()
               - min_onset) * oversampling) + 1

    hr_frame_times = np.linspace(frame_times.min() + min_onset,
                                 frame_times.max() * (1 + 1. / (n - 1)),
                                 np.rint(n_hr).astype(int))

    # Get the condition information
    onsets, durations, values = tuple(map(np.asanyarray, exp_condition))
    if (onsets < frame_times[0] + min_onset).any():
        warnings.warn(('Some stimulus onsets are earlier than %s in the'
                       ' experiment and are thus not considered in the model'
                       % (frame_times[0] + min_onset)), UserWarning)

    # Set up the regressor timecourse
    tmax = len(hr_frame_times)
    regressor = np.zeros_like(hr_frame_times).astype(np.float64)
    t_onset = np.minimum(np.searchsorted(hr_frame_times, onsets), tmax - 1)
    for t, v in zip(t_onset, values):
        regressor[t] += v
    t_offset = np.minimum(np.searchsorted(hr_frame_times, onsets + durations),
                          tmax - 1)

    # Handle the case where duration is 0 by offsetting at t + 1
    for i, t in enumerate(t_offset):
        if t < (tmax - 1) and t == t_onset[i]:
            t_offset[i] += 1

    for t, v in zip(t_offset, values):
        regressor[t] -= v
    regressor = np.cumsum(regressor)

    return regressor, hr_frame_times


def _gamma_difference_hrf(tr, oversampling=100, time_length=32., onset=0.,
                          delay=6, undershoot=16., dispersion=1.,
                          u_dispersion=1., ratio=0.167):
    dt = tr / oversampling
    time_stamps = np.linspace(0, time_length,
                              np.rint(float(time_length) / dt).astype(int))
    time_stamps -= onset

    # define peak and undershoot gamma functions
    peak_gamma = gamma.pdf(
        time_stamps,
        delay / dispersion,
        loc=dt,
        scale=dispersion)
    undershoot_gamma = gamma.pdf(
        time_stamps,
        undershoot / u_dispersion,
        loc=dt,
        scale=u_dispersion)

    # calculate the hrf
    hrf = peak_gamma - ratio * undershoot_gamma
    hrf /= hrf.sum()
    return hrf


def spm_hrf(tr, oversampling=100, time_length=32., onset=0.):
    return _gamma_difference_hrf(tr, oversampling, time_length, onset)


# In[SETUP NECESSARY PATHS ETC]:
'''
========================================================================
SETUP NECESSARY PATHS ETC:
========================================================================
'''
# name of the current project:
project = "fmrireplay"
sub_list = None
subject = None
# path to the project root:
project_name = 'fmrireplay-resting-TDLM'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_code = opj(path_bids, 'code', 'decoding-TDLM')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_masks = opj(path_bids, 'derivatives', 'masks')
path_level1_vfl = opj(path_bids, "derivatives", "glm-vfl", "l1pipeline")
path_decoding = opj(path_bids, 'derivatives', 'decoding')
path_behavior = opj(path_bids, 'derivatives', 'behavior')
path_out = opj(path_decoding, 'all_subject_results')
path_out_rest = opj(path_decoding, 'all_subject_results', 'resting')
path_onset_output = opj(path_bids, 'derivatives', 'resting-replay-onset', 'raw_replay_fmri')
path_rest_decode = opj(path_bids, 'derivatives', 'decoding_correlation', 'resting', 'fMRI')


for path in [path_out, path_out_rest]:
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
'''
========================================================================
LOAD TDLM PARAMETERS:
========================================================================
'''
# DEFINE RELEVANT VARIABLES:
n_run = 4 # VFL decoding
n_resting = 2 # resting periods
data_list_all = []
nSubj = len(sub_list)
repetition_time = 1.3

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
    log = opj(path_out_logs, '%s-%s.log' % (timestr, subject))
    # start logging:
    logging.basicConfig(
        filename=log, level=logging.DEBUG, format='%(asctime)s %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')

    # In[LOAD BEHAVIORAL DATA (THE EVENTS.TSV FILES OF THE SUBJECT)]:

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

    # load the confound file for VFL
    path_confounds_vfl = glob.glob(
        opj(path_fmriprep, subject, 'func', '*vfl*confounds_timeseries.tsv'))
    df_confound_vfl_list = [pd.read_csv(
        i, sep="\t") for i in path_confounds_vfl]
    df_confound_vfl = pd.concat([df_confound_vfl_list[0],
                                 df_confound_vfl_list[1],
                                 df_confound_vfl_list[2],
                                 df_confound_vfl_list[3]])

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

    # load the visual functional localizer mri task files:
    path_func_task_vfl = opj(path_level1_vfl, "smooth",
                             subject, "*", "*task*nii.gz")
    path_func_task_vfl = sorted(
        glob.glob(path_func_task_vfl), key=lambda f: os.path.basename(f))
    # load smoothed functional mri data for all task runs:
    data_task_vfl = [image.load_img(i) for i in path_func_task_vfl]

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

    # LOAD THE FMRI DATA
    path_rest = opj(path_masks, 'smooth', subject, '*', '*task-rest*.nii.gz')
    path_rest = sorted(glob.glob(path_rest),
                       key=lambda f: os.path.dirname(f))
    data_rest = [image.load_img(i) for i in path_rest]

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
        logging.info('plotting masked tmap %d of %d' %
                     (i + 1, len(tmaps_masked_img)))
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

    # In[SEPARATE RUN-WISE STANDARDIZATION (Z-SCORING) OF VISUAL FUNCTIONAL LOCALIZER]:
    data_list = []
    fold_list = [1, 2, 3, 4, 5, 6, 7, 8]
    rest_interval = 1
    # SAVE THE UNION MASK (get interesction mask from the four-task-run t-tmap threshold masks )
    mask_label = 'union'
    masks_union = multimultiply(
        tmaps_masked_thresh_bin).astype(int).astype(bool)
    masks_union_nii = image.new_img_like(path_func_task_vfl[0], masks_union)
    path_save = opj(
        path_out_masks, '{}_task_rest_mask-{}.nii.gz'.format(subject, mask_label))
    masks_union_nii.to_filename(path_save)
    # DETREND AND Z-STANDARDIZED THE RESTING FMRI DATA
    # mask all VFL task runs separately:
    data_task_vfl_masked = [masking.apply_mask(
        i, masks_union_nii) for i in data_task_vfl]
    # mask all cue replay runs with the averaged feature selection masks:
    data_rest_masked = [masking.apply_mask(
        i, masks_union_nii) for i in data_rest]
    # detrend each VFL task run separately:
    data_task_vfl_masked_detrend = [
        clean(i, detrend=True, standardize=False) for i in data_task_vfl_masked]
    # detrend each cue replay run separately:
    data_rest_masked_detrend = [
        clean(i, detrend=True, standardize=True) for i in data_rest_masked]
    # combine the detrended data of all runs:
    data_task_vfl_masked_detrend = np.vstack(data_task_vfl_masked_detrend)
    # standardized for the all trials in VFL
    train_VFL_peak.zscore(
        signals=data_task_vfl_masked_detrend, fold_list=fold_list)
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
            df_rest_pred['trial'] = np.tile(
                np.arange(1, rest_interval + 1), num_trials)
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

    # In[]:
    # temporal prediction probability
    pre_list = copy.deepcopy(data_list)
    pre_list_append = pd.concat([pre_list[i]
                                 for i in range(len(pre_list))], axis=0)

    pred_list_pre = pre_list_append.loc[(pre_list_append['run_study'] == 'task-rest_run-1') &
                                       (pre_list_append['class'] != 'other'),
                                       ['tr', 'class', 'probability']]

    pred_list_post = pre_list_append.loc[(pre_list_append['run_study'] == 'task-rest_run-2') &
                                        (pre_list_append['class'] != 'other'),
                                        ['tr', 'class', 'probability']]

    pre_rest_prob = pred_list_pre.pivot(index='tr', columns='class', values='probability')
    post_rest_prob = pred_list_post.pivot(index='tr', columns='class', values='probability')

    rename_dic = {'girl': 'A', 'scissors': 'B', 'zebra': 'C', 'banana': 'D'}
    column_name_list = ['A', 'B', 'C', 'D']

    pre_rest_prob = pre_rest_prob.rename(columns=rename_dic).reindex(columns=column_name_list).reset_index()
    pre_rest_prob['probability'] = np.mean((pre_rest_prob['A'], pre_rest_prob['B'],
                                            pre_rest_prob['C'], pre_rest_prob['D']),axis=0)
    pre_rest_prob['onset'] = (pre_rest_prob['tr'] - 1) * repetition_time
    pre_rest_prob['duration'] = repetition_time

    post_rest_prob = post_rest_prob.rename(columns=rename_dic).reindex(columns=column_name_list).reset_index()
    post_rest_prob['probability'] = np.mean((post_rest_prob['A'], post_rest_prob['B'],
                                             post_rest_prob['C'], post_rest_prob['D']),axis=0)
    post_rest_prob['onset'] = (post_rest_prob['tr'] - 1) * repetition_time
    post_rest_prob['duration'] = repetition_time

    pre_rest_prob.to_csv(
        opj(path_rest_decode, '%s_%s_resting_fmri_decoding.tsv' % (subject, 'pre')), sep='\t',
        index=False)

    post_rest_prob.to_csv(
        opj(path_rest_decode, '%s_%s_resting_fmri_decoding.tsv' % (subject, 'post')), sep='\t',
        index=False)

    # stats.pearsonr(pre_rest_prob.loc[:,'probability'], post_rest_prob.iloc[:-1]['probability'])
    # stats.pearsonr(pre_rest_prob.loc[:,'A'], post_rest_prob.iloc[:-1]['A'])
    # stats.pearsonr(pre_rest_prob.loc[:,'B'], post_rest_prob.iloc[:-1]['B'])
    # stats.pearsonr(pre_rest_prob.loc[:,'C'], post_rest_prob.iloc[:-1]['C'])
    # stats.pearsonr(pre_rest_prob.loc[:,'D'], post_rest_prob.iloc[:-1]['D'])
    #
    # stats.pearsonr(post_rest_prob.loc[:,'A'], post_rest_prob.loc[:,'D'])
    # stats.pearsonr(post_rest_prob.loc[:,'B'], post_rest_prob.loc[:,'C'])
    # stats.pearsonr(post_rest_prob.loc[:,'C'], post_rest_prob.loc[:,'D'])
    # stats.pearsonr(post_rest_prob.loc[:,'D'], post_rest_prob.loc[:,'A'])

    scan_tr = [np.size(data_rest[0].dataobj, axis=3), np.size(data_rest[1].dataobj, axis=3)]
    # print(stats.pearsonr(save_list[2].iloc[:,0], save_list[3].iloc[:,0]))
    for i, file in enumerate([pre_rest_prob, post_rest_prob]):
        # i = 1
        # file = post_rest_prob
        file['onset'] = file['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        file = file.loc[:, ['onset', 'probability','duration']]
        # onset_run_event[i]['probability'] = mlab.detrend_mean(onset_run_event[i]['probability'], axis=0)
        # get the maximum time point in a run
        max_time = int(scan_tr[i]) * repetition_time

        exp_condition = (file['onset'],
                         file['duration'],
                         file['probability'])
        frame_times = np.arange(scan_tr[i]) * repetition_time
        hr_regressor, hr_frame_times = _sample_condition(
            exp_condition, frame_times, oversampling=100)

        # spm HRF kernel
        hkernel = spm_hrf(tr=repetition_time, oversampling=100)
        # convolve the HRF kernel and probability event
        conv_reg = np.array(np.convolve(hr_regressor, hkernel)[:hr_regressor.size])
        # conv_reg_post = np.array(np.convolve(hr_regressor, hkernel)[:hr_regressor.size])
        # stats.pearsonr(conv_reg_pre, conv_reg_post[:25447])
        # interpolation
        f = interp1d(hr_frame_times, conv_reg)
        # get the specific time point BOLD signal
        slice_time_ref = 0.5
        start_time = slice_time_ref * repetition_time
        end_time = (scan_tr[i] - 1 + slice_time_ref) * repetition_time
        frame_times = np.linspace(start_time, end_time, scan_tr[i])
        # resample BOLD signal
        resample_onset = pd.DataFrame(f(frame_times).T, columns=['reactivation_fMRI'])
        # mean BOLD signal in one TR
        # resample_onset = pd.DataFrame([tempa[(tempa['onset'] >= ts * tr) &
        #                                      (tempa['onset'] < (ts + 1) * tr)]['conv_reg'].mean()
        #                                for ts in range(scan_tr[i])], columns=['HRF_onset_TR'])
        # save the data
        consname = {0: {'cons': 'pre'}, 1: {'cons': 'post'}, }
        resample_onset.to_csv(
            opj(path_onset_output, '%s_%s_resting_fmri_event.tsv' % (subject, consname[i]['cons'])), sep='\t',
            index=False)


Parallel(n_jobs=-2)(delayed(decoding)(subject) for subject in sub_list)
