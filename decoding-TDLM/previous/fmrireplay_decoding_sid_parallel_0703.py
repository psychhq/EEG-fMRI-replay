#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: FMRI-DECODING
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITVIE NERUOSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

# In[IMPORT RELEVANT PACKAGES]:
    
import glob
import os
import yaml
import logging
import time
from os.path import join as opj
import sys
import copy
from pprint import pformat
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn import plotting, image, masking
import pandas as pd
import math
from matplotlib import pyplot as plt
import warnings
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from bids.layout import BIDSLayout
from joblib import Parallel, delayed
import seaborn as sns
# plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt.rcParams["axes.unicode_minus"] = False
# DEAL WITH ERRORS, WARNING ETC
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

# In[DEFINITION OF ALL FUNCTIONS]:
    
# set the classlabel based on marker
def classlabel(Marker):
    if Marker == 1 or Marker ==21:
        return 'girl'
    elif Marker == 2 or Marker == 22:
        return 'basketballhoop'
    elif Marker == 3 or Marker == 23:
        return 'watch'
    elif Marker == 4 or Marker == 24:
        return 'keyboard'
    elif Marker == 5 or Marker == 25:
        return 'basketball'
    elif Marker == 6 or Marker == 26:
        return 'house'
    elif Marker == 7 or Marker == 27:
        return 'shoe'
    elif Marker == 8 or Marker == 28:
        return 'clock'

class TaskData:
    def __init__(
        self,
        events,
        task,
        name,
        num_vol_run_1,
        num_vol_run_2,
        # trial_type,
        bold_delay=0,
        interval=1,
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
        # define the number of volumes per task run:
        self.num_vol_run_1 = num_vol_run_1
        self.num_vol_run_2 = num_vol_run_2
        # select events: upright stimulus, correct answer trials only:
        if task == "VFL":
            self.events = events.loc[
                (events["task"] == 'VFL')
                # & (events["trial_type"] == trial_type)
                & (events["Ori"] == 0)
                # & (events["serial_position"] == 1)
                & (events["accuracy"] != 0),
                :,
            ]
        if task == "SID":
            self.events = events.loc[
                (events["task"] == 'SID')
                # & (events["trial_type"] == trial_type)
                & (events["Catch"] == 1),
                :,
            ]
            # self.real_trials = np.unique(self.trials)
            # self.real_trial = len(self.real_trials)   
        # elif trial_type == "cue":
        #     self.events = events.loc[
        #         (events["condition"] == condition)
        #         & (events["trial_type"] == trial_type),
        #         :,
        #     ]
        # reset the indices of the data frame:
        self.events.reset_index()
        # sort all values by session and run:
        self.events.sort_values(by=["session", "run"])
        # call further function upon initialization:
        self.define_trs()
        self.get_stats()

    def define_trs(self):
        # import relevant functions:
        import numpy as np

        # select all events onsets:
        self.event_onsets = self.events["onset"]
        # add the selected delay to account for the peak of the hrf
        self.bold_peaks = (
            self.events["onset"] + self.bold_delay
        )  # delay is the time, not the TR
        # divide the expected time-point of bold peaks by the repetition time:
        self.peak_trs = self.bold_peaks / self.t_tr
        
        # add the number of run volumes to the tr indices:
        run_volumes = (.events["run"] - 1) * self.num_vol_run_1
        # add the number of volumes of each run:
        trs = round(self.peak_trs + run_volumes)  # (run-1)*run_trs + this_run_peak_trs
        # copy the relevant trs as often as specified by the interval:
        a = np.transpose(np.tile(trs, (self.interval, 1)))
        # create same-sized matrix with trs to be added:
        b = np.full((len(trs), self.interval), np.arange(self.interval))
        # assign the final list of trs:
        self.trs = np.array(np.add(a, b).flatten(), dtype=int)
        # save the TRs of the stimulus presentations
        self.stim_trs = round(
            self.event_onsets / self.t_tr + run_volumes)
        
    def get_stats(self):
        import numpy as np
        self.num_trials = len(self.events)
        self.runs = np.repeat(
            np.array(self.events["run"], dtype=int), self.interval)
        self.trials = np.repeat(
            np.array(self.events["trials"], dtype=int), self.interval)
        self.sess = np.repeat(
            np.array(self.events["session"], dtype=int), self.interval)
        self.stim = np.repeat(
            np.array(self.events["stim_label"], dtype=object), self.interval)
        self.fold = np.repeat(
            np.array(self.events["fold"], dtype=object), self.interval)
        # self.itis = np.repeat(
        #     np.array(self.events["interval_time"], dtype=float), self.interval
        # )
        self.stim_trs = np.repeat(np.array(self.stim_trs, dtype=int), self.interval)

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
        df["participant"] = np.repeat(self.events["participant"].unique(), num_pred)
        # add the name of the classifier to the dataframe:
        df["test_set"] = np.repeat(self.name, num_pred)
        # return the dataframe:
        return df

def detrend(data):
    from nilearn.signal import clean
    data_detrend = clean(signals=data, t_r=1.3, detrend=True, standardize=False)
    return data_detrend

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
    df_melt = pd.melt(df, value_name="probability", var_name="class", id_vars=id_vars)
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
    return array_union

# In[SETUP NECESSARY PATHS ETC]:

# get start time of the script:
start = time.time()
# name of the current project:
project = "fmrireplay"
project = "fMRIreplay"
path_root = None
sub_list = None
subject = None
# path to the project root:
project_name = "fmrireplay-decoding"
path_root = opj(os.getcwd().split(project)[0], "fMRIreplay_hq")
path_bids = opj(path_root, "fmrireplay-bids", "BIDS")
path_code = opj(path_bids, "code", "decoding-TDLM")
path_fmriprep = opj(path_bids, "derivatives", "fmrireplay-fmriprep")
path_glm_vfl = opj(path_bids, "derivatives", "fmrireplay-glm-vfl")
path_level1_vfl = opj(path_glm_vfl, "l1pipeline")
path_glm_sid = opj(path_bids, "derivatives", "fmrireplay-glm-sid")
path_level1_sid = opj(path_glm_sid, "l1pipeline")
path_masks = opj(path_bids, "derivatives", "fmrireplay-masks")
path_decoding = opj(path_bids, "derivatives", project_name)
# path_behavior = opj(path_bids,'derivatives','fmrireplay-behavior')

data_list_all=[] 
pred_acc_mean_sub_all = []
# define the subject id
layout = BIDSLayout(path_bids)
sub_list = sorted(layout.get_subjects())
sub_template = ["sub-"] * len(sub_list)
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]
# delete_sub = [39,36,30,23]
# [sub_list.pop(i) for i in delete_sub]
# sub-04 for test the code
# subject = sub_list[1]

# mask_list = ['mask_temporal','mask_vis','mask_hpc','mask_prefrontal']
# mask_index_list=[0,1,2,3]
bold_delay_task_list = [2,3,4,5,6,7]

# for test
subject=sub_list[0]
mask_index=mask_index_list[0]
bold_delay_task=bold_delay_task_list[0]
#         mask_run = mask_hpc[run-1]
# for subject in sub_list:
# def decoding(subject,bold_delay_task): 
    # In[LOAD PROJECT PARAMETERS]:
    # path_params = glob.glob(opj(path_code, "*parameters.yaml"))[0]
    # with open(path_params, "rb") as f:
    #     params = yaml.load(f, Loader=yaml.FullLoader)
    # f.close()
    
    # define the mask to be used:
    # applied time-shift to account for the BOLD delay, in seconds:
    # bold_delay_task = 3  # 4, 5 or 6 secs  ???
    # define the degree of smoothing of the functional data
    # smooth = 4
    # time of repetition (TR), in seconds:
    # t_tr = params["mri"]["tr"]
    # acquisition time window of one sequence trial, in seconds:
    # t_win =
    # number of measurements that are considered per sequence time window:
    # n_tr_win = round(t_win / t_tr)
    # number of functional task runs in total:
    # n_run = params['mri']['num_runs']
    n_run = 2
    
    
    # In[CREATE PATHS TO OUTPUT DIRECTORIES]:
        
    # path_fmriprep_sub = opj(path_fmriprep, subject)
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
    #ADD BASIC SCRIPT INFORMATION TO THE LOGGER]:
    # logging.info("Running decoding script")
    # logging.info("operating system: %s" % sys.platform)
    # logging.info("project name: %s" % project)
    # logging.info("participant: %s" % subject)
    # logging.info("mask: %s" % mask)
    # logging.info("bold delay: %d secs" % bold_delay_task)
    # logging.info("smoothing kernel: %d mm" % smooth)
    
    # In[LOAD BEHAVIORAL DATA (THE EVENTS.TSV FILES OF THE SUBJECT)]:
        
    # paths to all events files of the current subject:
    path_events = opj(path_bids, subject, "func", "*tsv")
    # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
    path_events = sorted(glob.glob(path_events), key=lambda f: os.path.basename(f))
    logging.info("found %d event files" % len(path_events))
    logging.info("paths to events files (sorted):\n%s" % pformat(path_events))
    # import events file and save data in dataframe:
    # df_events = pd.concat((pd.read_csv(f, sep='\t') for f in path_events),
    #                       ignore_index=True)
    
    # paths to all events files of the current subject:
    path_events_vfl = opj(path_bids, subject, "func", "*vfl*events.tsv")
    # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
    path_events_vfl = sorted(glob.glob(path_events_vfl), key=lambda f: os.path.basename(f))
    # import events file and save data in dataframe:
    df_events_vfl = pd.concat(
        (pd.read_csv(f, sep="\t") for f in path_events_vfl), ignore_index=True
    )
    df_events_vfl['stim_label'] = df_events_vfl.apply(lambda x: classlabel(x.Marker), axis = 1)
    
    # paths to all events files of the current subject:
    path_events_sid = opj(path_bids, subject, "func", "*sid*events.tsv")
    # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
    path_events_sid = sorted(glob.glob(path_events_sid), key=lambda f: os.path.basename(f))
    # import events file and save data in dataframe:
    df_events_sid = pd.concat(
        (pd.read_csv(f, sep="\t") for f in path_events_sid), ignore_index=True
    )
    df_events_sid['stim_label'] = df_events_sid.apply(lambda x: classlabel(x.Marker), axis = 1)
    # In[CREATE PATHS TO THE MRI DATA]:
    
    # define path to input directories:
    # logging.info("path to fmriprep files: %s" % path_fmriprep_sub)
    # logging.info("path to level 1 files: %s" % path_level1_vfl)
    # logging.info("path to level 1 files: %s" % path_level1_sid)
    # logging.info("path to mask files: %s" % path_masks)
    # paths = {"fmriprep": path_fmriprep_sub, "level1_vfl": path_level1_vfl,
    #          "level1_sid": path_level1_sid, "masks": path_masks}
    # load the visual mask task files:
    path_mask_vis_task = opj(path_masks, "mask_visual", subject, "*", "*task*.nii.gz")
    path_mask_vis_task = sorted(glob.glob(path_mask_vis_task), key=lambda f: os.path.basename(f))
    logging.info("found %d visual mask task files" % len(path_mask_vis_task))
    logging.info("paths to visual mask task files:\n%s" % pformat(path_mask_vis_task))
    # load the hippocampus mask task files:
    path_mask_hpc_task = opj(path_masks, "mask_hippocampus", subject, "*", "*task*.nii.gz")
    path_mask_hpc_task = sorted(glob.glob(path_mask_hpc_task), key=lambda f: os.path.basename(f))
    logging.info("found %d hpc mask files" % len(path_mask_hpc_task))
    logging.info("paths to hpc mask task files:\n%s" % pformat(path_mask_hpc_task))
    # load the temporal cortex mask task files:
    path_mask_temporal_task = opj(path_masks, "mask_temporal", subject, "*", "*task*.nii.gz")
    path_mask_temporal_task = sorted(glob.glob(path_mask_temporal_task), key=lambda f: os.path.basename(f))
    logging.info("found %d temporal mask files" % len(path_mask_temporal_task))
    logging.info("paths to temporal mask task files:\n%s" % pformat(path_mask_temporal_task))
    # load the prefrontal cortex mask task files:
    path_mask_prefrontal_task = opj(path_masks, "mask_prefrontal", subject, "*", "*task*.nii.gz")
    path_mask_prefrontal_task = sorted(glob.glob(path_mask_prefrontal_task), key=lambda f: os.path.basename(f))
    logging.info("found %d hpc mask files" % len(path_mask_prefrontal_task))
    logging.info("paths to hpc mask task files:\n%s" % pformat(path_mask_prefrontal_task))
    # load the whole brain mask files:
    path_mask_whole_task = opj(path_fmriprep, subject, "func", "*task*T1w*brain_mask.nii.gz")
    path_mask_whole_task = sorted(glob.glob(path_mask_whole_task), key=lambda f: os.path.basename(f))
    logging.info("found %d whole-brain masks" % len(path_mask_whole_task))
    logging.info("paths to whole-brain mask files:\n%s" % pformat(path_mask_whole_task))
    # load the visual functional localizer mri task files:
    path_func_task_vfl = opj(path_level1_vfl, "smooth", subject, "*", "*task*nii.gz")
    path_func_task_vfl = sorted(glob.glob(path_func_task_vfl), key=lambda f: os.path.basename(f))
    logging.info("found %d functional mri task files" % len(path_func_task_vfl))
    logging.info("paths to functional mri task files:\n%s" % pformat(path_func_task_vfl))
    # load the single item decoding mri task files:
    path_func_task_sid = opj(path_level1_sid, "smooth", subject, "*", "*task*nii.gz")
    path_func_task_sid = sorted(glob.glob(path_func_task_sid), key=lambda f: os.path.basename(f))
    logging.info("found %d functional mri task files" % len(path_func_task_sid))
    logging.info("paths to functional mri task files:\n%s" % pformat(path_func_task_sid))
    # define path to the functional resting state runs:
    # path_rest = opj(path_masks, 'smooth', subject, '*', '*task-rest*nii.gz')
    # path_rest = sorted(glob.glob(path_rest), key=lambda f: os.path.basename(f))
    # logging.info('found %d functional mri rest files' % len(path_rest))
    # logging.info('paths to functional mri rest files:\n%s' % pformat(path_rest))
    # load the anatomical mri file:
    path_anat = opj(path_fmriprep, subject, "anat", "%s_desc-preproc_T1w.nii.gz" % subject)
    path_anat = sorted(glob.glob(path_anat), key=lambda f: os.path.basename(f))
    logging.info("found %d anatomical mri file" % len(path_anat))
    logging.info("paths to anatoimical mri files:\n%s" % pformat(path_anat))
    # load the confounds files:
    # path_confs_task = opj(path_fmriprep, subject, "func", "*task*confounds_timeseries.tsv")
    # path_confs_task = sorted(glob.glob(path_confs_task), key=lambda f: os.path.basename(f))
    # logging.info("found %d confounds files" % len(path_confs_task))
    # logging.info("found %d confounds files" % len(path_confs_task))
    # logging.info("paths to confounds files:\n%s" % pformat(path_confs_task))
    # load the spm.mat files:
    path_spm_mat_vfl = opj(path_level1_vfl, "contrasts", subject, "*", "SPM.mat")
    path_spm_mat_vfl = sorted(glob.glob(path_spm_mat_vfl), key=lambda f: os.path.dirname(f))
    logging.info("found %d spm.mat files" % len(path_spm_mat_vfl))
    logging.info("paths to spm.mat files:\n%s" % pformat(path_spm_mat_vfl))
    # load the spm.mat files:
    path_spm_mat_sid = opj(path_level1_sid, "contrasts", subject, "*", "SPM.mat")
    path_spm_mat_sid = sorted(glob.glob(path_spm_mat_sid), key=lambda f: os.path.dirname(f))
    logging.info("found %d spm.mat files" % len(path_spm_mat_sid))
    logging.info("paths to spm.mat files:\n%s" % pformat(path_spm_mat_sid))
    # load the t-maps of the first-level glm:
    path_tmap_vfl = opj(path_level1_vfl, "contrasts", subject, "*", "spmT*.nii")
    path_tmap_vfl = sorted(glob.glob(path_tmap_vfl), key=lambda f: os.path.dirname(f))
    logging.info("found %d t-maps" % len(path_tmap_vfl))
    logging.info("paths to t-maps files:\n%s" % pformat(path_tmap_vfl))
    # load the t-maps of the first-level glm:
    path_tmap_sid = opj(path_level1_sid, "contrasts", subject, "*", "spmT*.nii")
    path_tmap_sid = sorted(glob.glob(path_tmap_sid), key=lambda f: os.path.dirname(f))
    logging.info("found %d t-maps" % len(path_tmap_sid))
    logging.info("paths to t-maps files:\n%s" % pformat(path_tmap_sid))
    # In[LOAD THE MRI DATA]:
    
    anat = image.load_img(path_anat[0])
    logging.info("successfully loaded %s" % path_anat[0])
    # load visual mask:
    mask_vis = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_vis_task[14:16])]
    logging.info("successfully loaded one visual mask file!")
    # load hippocampus mask:
    mask_hpc = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_hpc_task[14:16])]
    logging.info("successfully loaded one hippocampus mask file!")
    # load temporal mask:
    mask_temporal = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_temporal_task[14:16])]
    logging.info("successfully loaded one temporal mask file!")
    # load prefrontal mask:
    mask_prefrontal = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_prefrontal_task[14:16])]
    logging.info("successfully loaded one prefrontal mask file!")   
    # load wholebrain mask for SID:
    mask_brain = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_whole_task[14:16])]
    image.load_img(path_mask_whole_task[14:16]).get_data().astype(int)
    logging.info("successfully loaded one visual mask file!")
    # load tmap data:
    tmaps_vfl = [image.load_img(i).get_data().astype(float) for i in copy.deepcopy(path_tmap_vfl)]
    logging.info("successfully loaded the tmaps!")
    # load tmap data:
    tmaps_sid = [image.load_img(i).get_data().astype(float) for i in copy.deepcopy(path_tmap_sid)]
    logging.info("successfully loaded the tmaps!")    
    # LOAD SMOOTHED FMRI DATA FOR ALL FUNCTIONAL TASK RUNS:
    # load smoothed functional mri data for all eight task runs:
    logging.info("loading %d functional task runs ..." % len(path_func_task_vfl))
    data_task_vfl = [image.load_img(i) for i in path_func_task_vfl]
    logging.info("loading successful!")
    logging.info("loading %d functional task runs ..." % len(path_func_task_sid))
    data_task_sid = [image.load_img(i) for i in path_func_task_sid]
    logging.info("loading successful!")
    
    # mask_list = [mask_temporal,mask_vis,mask_hpc,mask_prefrontal]
    # mask_name_list = ['mask_temporal','mask_vis','mask_hpc','mask_prefrontal']
    # In[FEATURE SELECTION FOR VISUAL FUNCTIONAL LOCALIZER TASK]:
    taskname = 'VFL'
    ### plot raw-tmaps on an anatomical background:
    logging.info("plotting raw tmaps with anatomical as background:")
    for i, path in enumerate(path_tmap_vfl):
        logging.info("plotting raw tmap %s (%d of %d)" % (path, i + 1, len(path_tmap_vfl)))
        path_save = opj(path_out_figs, "%s_%s_run-%02d_tmap_raw.png" % (subject, taskname, i + 1))
        plotting.plot_roi(
            path,
            anat,
            title=os.path.basename(path_save),
            output_file=path_save,
            colorbar=True,)
        
    ### FEATURE SELECTION: MASKS THE T-MAPS WITH THE ANATOMICAL MASKS
    
    # v= tmaps_masked[0]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(v[:,0],v[:,1],v[:,2], zdir='z', c= 'red')
    # plt.show()
    # check if any value in the supposedly binary mask is bigger than 1:
    # if np.any(mask_vis > 1):
    #     logging.info("WARNING: detected values > 1 in the anatomical ROI!")
    #     sys.exit("Values > 1 in the anatomical ROI!")
    # get combination of anatomical mask and t-map
    tmaps_masked = [np.multiply(i, j) for (i,j) in zip(copy.deepcopy(mask_vis),copy.deepcopy(tmaps_vfl))]

    # masked tmap into image like object:
    tmaps_masked_img = [image.new_img_like(i, j) for (i, j) in zip(path_tmap_vfl, copy.deepcopy(tmaps_masked))]
    for i, path in enumerate(tmaps_masked_img):
        path_save = opj(path_out_masks, "%s_%s_run-%02d_tmap_masked.nii.gz" % (subject, taskname, i + 1))
        path.to_filename(path_save)
    # plot masked t-maps
    logging.info("plotting masked tmaps with anatomical as background:")
    for i, path in enumerate(tmaps_masked_img):
        logging.info("plotting masked tmap %d of %d" % (i + 1, len(tmaps_masked_img)))
        path_save = opj(path_out_figs, "%s_%s_run-%02d_tmap_masked.png" % (subject, taskname, i + 1))
        plotting.plot_roi(
            path,
            anat,
            title=os.path.basename(path_save),
            output_file=path_save,
            colorbar=True,)
        
    ### FEATURE SELECTION: THRESHOLD THE MASKED T-MAPS
    
    # set the threshold:
    # threshold = params["mri"]["thresh"]
    threshold = 1.96
    logging.info("thresholding t-maps with a threshold of %s" % str(threshold))
    # threshold the masked tmap image:
    tmaps_masked_thresh_img = [image.threshold_img(i, threshold) for i in copy.deepcopy(tmaps_masked_img)]
    logging.info("plotting thresholded tmaps with anatomical as background:")
    for i, path in enumerate(tmaps_masked_thresh_img):
        path_save = opj(path_out_figs, "%s_%s_run-%02d_tmap_masked_thresh.png" % (subject, taskname, i + 1))
        logging.info(
            "plotting masked tmap %s (%d of %d)"
            % (path_save, i + 1, len(tmaps_masked_thresh_img))
        )
        plotting.plot_roi(
            path,
            anat,
            title=os.path.basename(path_save),
            output_file=path_save,
            colorbar=True,
        )
    # extract data from the thresholded images
    tmaps_masked_thresh = [image.load_img(i).get_data().astype(float) for i in tmaps_masked_thresh_img]
    # calculate the number of tmap voxels:
    num_tmap_voxel = [np.size(i) for i in copy.deepcopy(tmaps_masked_thresh)]
    logging.info("number of feature selected voxels: %s" % pformat(num_tmap_voxel))
    num_above_voxel = [np.count_nonzero(i) for i in copy.deepcopy(tmaps_masked_thresh)]
    logging.info("number of voxels above threshold: %s" % pformat(num_above_voxel))
    num_below_voxel = [np.count_nonzero(i == 0) for i in copy.deepcopy(tmaps_masked_thresh)]
    logging.info("number of voxels below threshold: %s" % pformat(num_below_voxel))
    
    # plot the distribution of t-values:
    for i, run_mask in enumerate(tmaps_masked_thresh):
        masked_tmap_flat = run_mask.flatten()
        masked_tmap_flat = masked_tmap_flat[~np.isnan(masked_tmap_flat)]
        masked_tmap_flat = masked_tmap_flat[
            ~np.isnan(masked_tmap_flat) & ~(masked_tmap_flat == 0)]
        path_save = opj(path_out_figs, "%s_%s_run-%02d_tvalue_distribution.png" % (subject, taskname, i + 1))
        logging.info(
            "plotting thresholded t-value distribution %s (%d of %d)"
            % (path_save, i + 1, len(tmaps_masked_thresh)))
        fig = plt.figure()
        plt.hist(masked_tmap_flat, bins="auto")
        plt.xlabel("t-values")
        plt.ylabel("number")
        plt.title("t-value distribution (%s, run-%02d)" % (subject, i + 1))
        plt.savefig(path_save)
    # create a dataframe with the number of voxels
    df_thresh = pd.DataFrame(
        {   "id": [subject] * n_run,
            "run": np.arange(1, n_run + 1),
            "n_total": num_tmap_voxel,
            "n_above": num_above_voxel,
            "n_below": num_below_voxel,
        }
    )
    file_name = opj(path_out_data, "%s_%s_thresholding.csv" % (subject, taskname))
    df_thresh.to_csv(file_name, sep=",", index=False)
    
    ### FEATURE SELECTION: BINARIZE THE THRESHOLDED MASKED T-MAPS
    
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
        image.new_img_like(path_func_task_vfl[0], i.astype(np.int))
        for i in copy.deepcopy(tmaps_masked_thresh_bin)]
    logging.info("plotting final masks with anatomical as background:")
    for i, path in enumerate(masks_final_vfl):
        filename = "%s_run-%02d_tmap_masked_thresh.nii.gz" % (subject, i + 1)
        path_save = opj(path_out_masks, filename)
        logging.info(
            "saving final mask %s (%d of %d)" % (path_save, i + 1, len(masks_final_vfl)))
        path.to_filename(path_save)
        path_save = opj(
            path_out_figs, "%s_%s_run-%02d_visual_final_mask.png" % (subject, taskname, i + 1))
        logging.info(
            "plotting final mask %s (%d of %d)" % (path_save, i + 1, len(masks_final_vfl)))
        plotting.plot_roi(
            path,
            anat,
            title=os.path.basename(path_save),
            output_file=path_save,
            colorbar=True,)
    
    # In[FEATURE SELECTION FOR SINGLE ITEM DECODING TASK]:
    # taskname = 'SID'
    # # mask = mask_temporal
    # mask=mask_list[mask_index]
    # mask_name=mask_name_list[mask_index]
    # ### plot raw-tmaps on an anatomical background:
    # logging.info("plotting raw tmaps with anatomical as background:")
    # for i, path in enumerate(path_tmap_sid):
    #     logging.info("plotting raw tmap %s (%d of %d) %s" % (path, i + 1, len(path_tmap_sid), mask_name))
    #     path_save = opj(path_out_figs, "%s_%s_run-%02d_tmap_raw_%s.png" % (subject, taskname, i + 1, mask_name))
    #     plotting.plot_roi(
    #         path,
    #         anat,
    #         title=os.path.basename(path_save),
    #         output_file=path_save,
    #         colorbar=True,)
        
    # ### FEATURE SELECTION: MASKS THE T-MAPS WITH THE ANATOMICAL MASKS
    # # v= tmaps_masked[0]
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(v[:,0],v[:,1],v[:,2], zdir='z', c= 'red')
    # # plt.show()
    # # check if any value in the supposedly binary mask is bigger than 1:
    # # for i in range(len(mask)):
    # #     if np.any(mask[i] > 1):
    # #         logging.info("WARNING: detected values > 1 in the anatomical ROI!")
    # #         sys.exit("Values > 1 in the anatomical ROI!")
    # # get combination of anatomical mask and t-map
    # tmaps_masked = [np.multiply(i, j) for (i,j) in zip(copy.deepcopy(mask),copy.deepcopy(tmaps_sid))]
    # # masked tmap into image like object:
    # tmaps_masked_img = [image.new_img_like(i, j) for (i, j) in zip(path_tmap_sid, copy.deepcopy(tmaps_masked))]
    # for i, path in enumerate(tmaps_masked_img):
    #     path_save = opj(path_out_masks, "%s_%s_run-%02d_tmap_masked_%s.nii.gz" % (subject, taskname, i + 1, mask_name))
    #     path.to_filename(path_save)
    # # plot masked t-maps
    # logging.info("plotting masked tmaps with anatomical as background:")
    # for i, path in enumerate(tmaps_masked_img):
    #     logging.info("plotting masked tmap %d of %d %s" % (i + 1, len(tmaps_masked_img), mask_name))
    #     path_save = opj(path_out_figs, "%s_%s_run-%02d_tmap_masked_%s.png" % (subject, taskname, i + 1, mask_name))
    #     plotting.plot_roi(
    #         path,
    #         anat,
    #         title=os.path.basename(path_save),
    #         output_file=path_save,
    #         colorbar=True,)
        
    # ### FEATURE SELECTION: THRESHOLD THE MASKED T-MAPS
    
    # # set the threshold:
    # # threshold = params["mri"]["thresh"]
    # threshold = 1.96
    # logging.info("thresholding t-maps with a threshold of %s" % str(threshold))
    # # threshold the masked tmap image:
    # tmaps_masked_thresh_img = [image.threshold_img(i, threshold) for i in copy.deepcopy(tmaps_masked_img)]
    # logging.info("plotting thresholded tmaps with anatomical as background:")
    # for i, path in enumerate(tmaps_masked_thresh_img):
    #     path_save = opj(path_out_figs, "%s_%s_run-%02d_tmap_masked_thresh_%s.png" % (subject, taskname, i + 1, mask_name))
    #     logging.info(
    #         "plotting masked tmap %s (%d of %d)"
    #         % (path_save, i + 1, len(tmaps_masked_thresh_img))
    #     )
    #     plotting.plot_roi(
    #         path,
    #         anat,
    #         title=os.path.basename(path_save),
    #         output_file=path_save,
    #         colorbar=True,
    #     )
    # # extract data from the thresholded images
    # tmaps_masked_thresh = [image.load_img(i).get_data().astype(float) for i in tmaps_masked_thresh_img]
    # # calculate the number of tmap voxels:
    # num_tmap_voxel = [np.size(i) for i in copy.deepcopy(tmaps_masked_thresh)]
    # logging.info("number of feature selected voxels: %s" % pformat(num_tmap_voxel))
    # num_above_voxel = [np.count_nonzero(i) for i in copy.deepcopy(tmaps_masked_thresh)]
    # logging.info("number of voxels above threshold: %s" % pformat(num_above_voxel))
    # num_below_voxel = [np.count_nonzero(i == 0) for i in copy.deepcopy(tmaps_masked_thresh)]
    # logging.info("number of voxels below threshold: %s" % pformat(num_below_voxel))
    
    # # plot the distribution of t-values:
    # for i, run_mask in enumerate(tmaps_masked_thresh):
    #     masked_tmap_flat = run_mask.flatten()
    #     masked_tmap_flat = masked_tmap_flat[~np.isnan(masked_tmap_flat)]
    #     masked_tmap_flat = masked_tmap_flat[
    #         ~np.isnan(masked_tmap_flat) & ~(masked_tmap_flat == 0)]
    #     path_save = opj(path_out_figs, "%s_%s_run-%02d_tvalue_distribution_%s.png" % (subject, taskname, i + 1, mask_name))
    #     logging.info(
    #         "plotting thresholded t-value distribution %s (%d of %d)"
    #         % (path_save, i + 1, len(tmaps_masked_thresh)))
    #     fig = plt.figure()
    #     plt.hist(masked_tmap_flat, bins="auto")
    #     plt.xlabel("t-values")
    #     plt.ylabel("number")
    #     plt.title("t-value distribution (%s, run-%02d) %s" % (subject, i + 1, mask_name))
    #     plt.savefig(path_save)
    # # create a dataframe with the number of voxels
    # df_thresh = pd.DataFrame(
    #     {   "id": [subject] * n_run,
    #         "run": np.arange(1, n_run + 1),
    #         "n_total": num_tmap_voxel,
    #         "n_above": num_above_voxel,
    #         "n_below": num_below_voxel,
    #     }
    # )
    # file_name = opj(path_out_data, "%s_%s_thresholding_%s.csv" % (subject, taskname,mask_name))
    # df_thresh.to_csv(file_name, sep=",", index=False)
    
    # ### FEATURE SELECTION: BINARIZE THE THRESHOLDED MASKED T-MAPS
    
    # # replace all NaNs with 0:
    # tmaps_masked_thresh_bin = [
    #     np.where(np.isnan(i), 0, i) for i in copy.deepcopy(tmaps_masked_thresh)]
    # # replace all other values with 1:
    # tmaps_masked_thresh_bin = [
    #     np.where(i > 0, 1, i) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
    # # turn the 3D-array into booleans:
    # tmaps_masked_thresh_bin = [
    #     i.astype(bool) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
    # # create image like object:
    # masks_final_sid = [
    #     image.new_img_like(path_func_task_sid[0], i.astype(np.int))
    #     for i in copy.deepcopy(tmaps_masked_thresh_bin)]
    # logging.info("plotting final masks with anatomical as background:")
    # for i, path in enumerate(masks_final_sid):
    #     filename = "%s_run-%02d_tmap_masked_thresh.nii.gz" % (subject, i + 1)
    #     path_save = opj(path_out_masks, filename)
    #     logging.info(
    #         "saving final mask %s (%d of %d) %s" % (path_save, i + 1, len(masks_final_sid), mask_name))
    #     path.to_filename(path_save)
    #     path_save = opj(
    #         path_out_figs, "%s_%s_run-%02d_visual_final_mask %s.png" % (subject, taskname, i + 1, mask_name))
    #     logging.info(
    #         "plotting final mask %s (%d of %d) %s" % (path_save, i + 1, len(masks_final_sid), mask_name))
    #     plotting.plot_roi(
    #         path,
    #         anat,
    #         title=os.path.basename(path_save),
    #         output_file=path_save,
    #         colorbar=True,)
    # In[DEFINE THE CLASSIFIERS]:
    
    class_labels = ["girl", "basketballhoop", "watch", "keyboard", "basketball",'house','shoe','clock']
    # create a dictionary with all values as independent instances:
    # see here: https://bit.ly/2J1DvZm
    clf_set = {
        key: LogisticRegression(
            C=1.0,  # Inverse of regularization strength
            penalty="l2",
            multi_class="ovr",
            solver="lbfgs",  # Algorithm to use in the optimization problem, Stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno
            max_iter=4000,
            class_weight="balanced",
            random_state=42,  # to shuffle the data
        )
        for key in class_labels
    }
    classifiers = {
        "log_reg": LogisticRegression(
            C=1.0,
            penalty="l2",
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=4000,
            class_weight="balanced",
            random_state=42,
        )
    }
    clf_set.update(classifiers)
    
     
    # In[DEFINE THE TASK CONDITION]:
        
    # 1. SPLIT THE EVENTS DATAFRAME FOR EACH TASK CONDITION
    # 2. RESET THE INDICES OF THE DATAFRAMES
    # 3. SORT THE ROWS OF ALL DATAFRAMES IN CHRONOLOGICAL ORDER
    # 4. PRINT THE NUMBER OF TRIALS OF EACH TASK CONDITION

    # for bold_delay_task in [3,4,5,6]:
    train_VFL_peak = TaskData(
        events=df_events_vfl,
        task="VFL",
        # trial_type="stimulus",
        bold_delay=bold_delay_task,
        interval=1,
        name="train-VFL_peak",
        num_vol_run_1=np.size(data_task_vfl[0]._data,axis=3),
        num_vol_run_2=np.size(data_task_vfl[1]._data,axis=3),
    )
    train_SID_peak = TaskData(
        events=df_events_sid,
        task="SID",
        # trial_type="stimulus",
        bold_delay=bold_delay_task,
        interval=1,
        name="train-SID_peak",
        num_vol_run_1=np.size(data_task_sid[0]._data,axis=3),
        num_vol_run_2=np.size(data_task_sid[1]._data,axis=3),
    )
    
    
    test_VFL_peak = TaskData(
        events=df_events_vfl,
        task="VFL",
        # trial_type="stimulus",
        bold_delay=bold_delay_task,
        interval=1,
        name="test-VFL_peak",
        num_vol_run_1=np.size(data_task_vfl[0]._data,axis=3),
        num_vol_run_2=np.size(data_task_vfl[1]._data,axis=3),
    )
    test_VFL_long = TaskData(
        events=df_events_vfl,
        task="VFL",
        # trial_type="stimulus",
        bold_delay=0,
        interval=9,
        name="test-VFL_long",
        num_vol_run_1=np.size(data_task_vfl[0]._data,axis=3),
        num_vol_run_2=np.size(data_task_vfl[1]._data,axis=3),
    )
    
    
    test_SID_peak = TaskData(
        events=df_events_sid,
        task="SID",
        # trial_type="stimulus",
        bold_delay=bold_delay_task,
        interval=1,
        name="test-SID_peak",
        num_vol_run_1=np.size(data_task_sid[0]._data,axis=3),
        num_vol_run_2=np.size(data_task_sid[1]._data,axis=3),
    )
    test_SID_long = TaskData(
        events=df_events_sid,
        task="SID",
        # trial_type="stimulus",
        bold_delay=0,
        interval=9,
        name="test-SID_long",
        num_vol_run_1=np.size(data_task_sid[0]._data,axis=3),
        num_vol_run_2=np.size(data_task_sid[1]._data,axis=3),
        )
    # test_rep_long = TaskData(
    #     events=df_events,
    #     condition="repetition",
    #     trial_type="stimulus",
    #     bold_delay=0,
    #     interval=13,
    #     name="test-rep_long",
    # )
    # test_seq_cue = TaskData(
    #     events=df_events,
    #     condition="sequence",
    #     trial_type="cue",
    #     bold_delay=0,
    #     interval=5,
    #     name="test-seq_cue",
    # )
    # test_rep_cue = TaskData(
    #     events=df_events,
    #     condition="repetition",
    #     trial_type="cue",
    #     bold_delay=0,
    #     interval=5,
    #     name="test-rep_cue",
    # )
    test_sets = [
        test_VFL_peak,
        test_VFL_long,
    #     test_SID_peak,
    ]
    # In[SEPARATE RUN-WISE STANDARDIZATION (Z-SCORING) OF VISUAL FUNCTIONAL LOCALIZER]:
    
    test_VFL_long.trs[-1]=test_VFL_long.trs[-3]
    test_VFL_long.trs[-2]=test_VFL_long.trs[-3]
    
    n_fold = 8
    data_list = []
    runs = list([1,1,1,1,2,2,2,2])
    folds = list(range(1, n_fold + 1))
    mask_label = 'cv'
    logging.info("starting leave-one-run-out cross-validation")
    # for mask_label in ["cv", "cv_hpc"]:
    logging.info("testing in mask %s" % (mask_label))
    for fold, run in zip(folds, runs):
        logging.info("testing on run %d of %d ..." % (fold, len(folds)))
        # define the run indices for the training and test set:
        train_folds = [x for x in folds if x != fold]
        test_folds = [x for x in folds if x == fold]
        # get the feature selection mask of the current run:
        # if mask_label == "cv":
        mask_run = masks_final_vfl[run-1]
        # elif mask_label == "cv_hpc":
        #     mask_run = path_mask_hpc_task[runs.index(run)]
        # extract smoothed fMRI data from the mask for the cross-validation fold:
        masked_data = [masking.apply_mask(i, mask_run) for i in data_task_vfl]
        # detrend the masked fMRI data separately for each run:
        data_detrend = [detrend(i) for i in masked_data]
        # combine the detrended data of all runs:
        data_detrend = np.vstack(data_detrend)
        # loop through all classifiers in the classifier set:
        for clf_name, clf in clf_set.items():
            # print classifier:
            logging.info("classifier: %s" % clf_name)
            # fit the classifier to the training data:
            train_VFL_peak.zscore(signals=data_detrend, fold_list=train_folds)
            # get the example labels:
            train_stim = copy.deepcopy(train_VFL_peak.stim[train_VFL_peak.fold != test_folds])
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
            for test_set in test_sets:
            # test_set = test_VFL_peak
                logging.info("testing on test set %s" % test_set.name)
                test_set.zscore(signals=data_detrend, fold_list=test_folds)
                if test_set.data_zscored.size < 0:
                    continue
                # create dataframe containing classifier predictions:
                df_pred = test_set.predict(clf=clf, fold_list=test_folds)
                # add the current classifier as a new column:
                df_pred["classifier"] = np.repeat(clf_name, len(df_pred))
                # add a label that indicates the mask / training regime:
                df_pred["mask"] = np.repeat(mask_label, len(df_pred))
                # melt the data frame:
                df_pred_melt = melt_df(df=df_pred, melt_columns=train_stim)
                # append dataframe to list of dataframe results:
                data_list.append(df_pred_melt)
                
    # data_list_all[isub] = list(data_list)
    # In[PREDICTION PROBABILITY AND ACCURACY FOR PEAK]:
    
    mask_name='VFL'
    # prediction accuracy
    pred_class_mean = []
    pre_list=copy.deepcopy(data_list)
    # delete_list = [71,62,53,44,35,26,17,8]
    # [pre_list.pop(i) for i in delete_list]
    pre_list_append = pd.concat([pre_list[i] for i in range(len(pre_list))], axis=0)
    pre_list_filter = pre_list_append[(pre_list_append["class"] != 'other' ) &
                                      (pre_list_append["test_set"] == 'test-VFL_peak') &
                                      (pre_list_append["classifier"] == 'log_reg') &
                                      (pre_list_append["stim"] == pre_list_append["class"])]
    pre_list_filter['pred_acc'] = 0
    def pred_acc(stim,pred_label):
        if (stim == pred_label):
            return 1
        else: 
            return 0
    pre_list_filter['pred_acc'] = pre_list_filter.apply(lambda x: pred_acc(x['stim'],x['pred_label']), axis=1)
    pred_acc_mean = np.mean(pre_list_filter['pred_acc'])
    
    # prediction probability
    pred_acc_mean_sub = pd.DataFrame([[pred_acc_mean],[subject],[bold_delay_task]])
    # prediction probability
    pred_class_mean = []
    pre_list=copy.deepcopy(data_list)
    # delete_list = [71,62,53,44,35,26,17,8]
    # [pre_list.pop(i) for i in delete_list]
    pre_list_append = []
    pre_list_append = pd.concat([pre_list[i] for i in range(len(pre_list))], axis=0)
    pre_list_filter = pre_list_append[(pre_list_append["class"] != 'other') &
                                      (pre_list_append["classifier"] == 'log_reg') &
                                      (pre_list_append["test_set"] =='test-VFL_peak')]
    pred_class_mean = [np.mean(pre_list_filter[(pre_list_filter['stim'] == class_label) &
                                                (pre_list_filter['class'] == class_label)]['probability']) 
                        for class_label in class_labels]
    x_data = class_labels
    y_data = pred_class_mean
    # plot the prediction accuracy
    plt.figure(dpi=400,figsize=(10,5))
    plt.xticks(fontsize=10)
    for i in range(len(x_data)):
     	plt.bar(x_data[i], y_data[i])
    plt.axhline(y=0.125,color='silver',linestyle='--')
    plt.title('%s_decoding probability of visual functional localizer in %1.0f th TR peak' % (subject,train_VFL_peak.bold_delay))
    plt.xlabel("label")
    plt.ylabel("probability")
    plt.savefig(opj(path_out_figs,'%s_decoding probability of visual functional localizer in %1.0f th TR peak.png') % (subject,train_VFL_peak.bold_delay),
                dpi = 400,bbox_inches = 'tight')
    plt.show()
    
    # In[PREDICTION PROBABILITY AND ACCURACY FOR LONG TERM]:
    
    # prediction probability
    pred_class_mean = []
    pre_list=copy.deepcopy(data_list)
    # delete_list = [71,62,53,44,35,26,17,8]
    # [pre_list.pop(i) for i in delete_list]
    pre_list_append = []
    pre_list_append = pd.concat([pre_list[i] for i in range(len(pre_list))], axis=0)
    pre_list_filter = pre_list_append[(pre_list_append["class"] != 'other') &
                                      (pre_list_append["classifier"] == 'log_reg') &
                                      (pre_list_append["test_set"] == 'test-VFL_long')]
    pred_class_mean = [[np.mean(pre_list_filter[(pre_list_filter['stim'] == class_label) 
                                                & (pre_list_filter['seq_tr'] ==  seq_tr)
                                                & (pre_list_filter['class'] == class_label)]['probability']) 
                        for class_label in class_labels] for seq_tr in np.arange(1,10)]
    pred_class_mean = np.matrix(pred_class_mean)
    whole_TR = np.arange(1,10)
    
    fig, ax = plt.subplots(nrows = 2, ncols = 4,sharex=True, sharey=True,figsize=(10,4),dpi=400)
    for j, class_label in enumerate(class_labels):
        x_data = whole_TR
        y_data = pred_class_mean[:,j]
        raw = math.floor(j/4)
        col = j-4*raw 
        ax[raw][col].set_ylim(ymax=0.8)
        ax[raw][col].set_xticks(range(1,len(whole_TR)+1),labels=whole_TR)
        ax[raw][col].set_yticks([0,0.2,0.4,0.6,0.8],labels=[0,20,40,60,80])
        ax[raw][col].plot(x_data,y_data,color='black',linestyle='-')
        ax[raw][col].axhline(y=0.125,color='silver',linestyle='--')
        ax[raw][col].set_title('%s' % class_label) 
    fig.text(0.5, 0, 'Time from stimulus onset (TRs)', ha='center')
    fig.text(0, 0.5, 'Probability(%)', va='center', rotation='vertical')
    plt.tight_layout(pad=2.5, w_pad=0.5, h_pad=2)
    plt.suptitle('%s decoding probability of visual functional localizer (%1.0f th TR peak)' % (subject,train_VFL_peak.bold_delay))
    plt.savefig(opj(path_out_figs,'%s decoding probability of visual functional localizer (%1.0f th TR peak)' % (subject,train_VFL_peak.bold_delay)),
                dpi = 400,bbox_inches = 'tight')
    plt.show() 
    
    # In[SEPARATE RUN-WISE STANDARDIZATION (Z-SCORING) OF SINGLE ITEM DECODING]:
    
    # test_SID_long.trs[-1]=test_SID_long.trs[-4]
    # test_SID_long.trs[-2]=test_SID_long.trs[-4]
    # test_SID_long.trs[-3]=test_SID_long.trs[-4]
    # n_fold = 8
    # data_list = []
    # runs = list([1,1,1,1,2,2,2,2])
    # folds = list(range(1, n_fold + 1))
    # mask_label = 'cv'
    # logging.info("starting leave-one-run-out cross-validation")
    # # for mask_label in ["cv", "cv_hpc"]:
    # logging.info("testing in mask %s" % (mask_label))
    # for fold, run in zip(folds, runs):
    #     logging.info("testing on run %d of %d ..." % (fold, len(folds)))
    #     # define the run indices for the training and test set:
    #     train_folds = [x for x in folds if x != fold]
    #     test_folds = [x for x in folds if x == fold]
    #     # get the feature selection mask of the current run:
    #     # if mask_label == "cv":
    #     mask_run = masks_final_sid[run-1]
    #     # elif mask_label == "cv_hpc":
    #     #     mask_run = path_mask_hpc_task[runs.index(run)]
    #     # extract smoothed fMRI data from the mask for the cross-validation fold:
    #     masked_data = [masking.apply_mask(i, mask_run) for i in data_task_sid]
    #     # detrend the masked fMRI data separately for each run:
    #     data_detrend = [detrend(i) for i in masked_data]
    #     # combine the detrended data of all runs:
    #     data_detrend = np.vstack(data_detrend)
    #     # loop through all classifiers in the classifier set:
    #     for clf_name, clf in clf_set.items():
    #         # print classifier:
    #         logging.info("classifier: %s" % clf_name)
    #         # fit the classifier to the training data:
    #         train_SID_peak.zscore(signals=data_detrend, fold_list=train_folds)
    #         # get the example labels:
    #         train_stim = copy.deepcopy(train_SID_peak.stim[train_SID_peak.fold != test_folds])
    #         # replace labels for single-label classifiers:
    #         if clf_name in class_labels:
    #             # replace all other labels with other
    #             train_stim = ["other" if x != clf_name else x for x in train_stim]
    #             # turn into a numpy array
    #             train_stim = np.array(train_stim, dtype=object)
    #         # check weights:
    #         # show_weights(array=train_stim)
    #         # train the classifier
    #         clf.fit(train_SID_peak.data_zscored, train_stim)
    #         # classifier prediction: predict on test data and save the data:
    #         for test_set in test_sets:
    #         # test_set = test_SID_peak
    #             logging.info("testing on test set %s" % test_set.name)
    #             test_set.zscore(signals=data_detrend, fold_list=test_folds)
    #             if test_set.data_zscored.size < 0:
    #                 continue
    #             # create dataframe containing classifier predictions:
    #             df_pred = test_set.predict(clf=clf, fold_list=test_folds)
    #             # add the current classifier as a new column:
    #             df_pred["classifier"] = np.repeat(clf_name, len(df_pred))
    #             # add a label that indicates the mask / training regime:
    #             df_pred["mask"] = np.repeat(mask_label, len(df_pred))
    #             # melt the data frame:
    #             df_pred_melt = melt_df(df=df_pred, melt_columns=train_stim)
    #             # append dataframe to list of dataframe results:
    #             data_list.append(df_pred_melt)
    
    # # In[PREDICTION PROBABILITY AND ACCURACY FOR PEAK]:
    
    # # prediction accuracy
    # pred_class_mean = []
    # pre_list=copy.deepcopy(data_list)
    # # delete_list = [71,62,53,44,35,26,17,8]
    # # [pre_list.pop(i) for i in delete_list]
    # pre_list_append = pd.concat([pre_list[i] for i in range(len(pre_list))], axis=0)
    # pre_list_filter = pre_list_append[(pre_list_append["class"] != 'other' ) &
    #                                   (pre_list_append["test_set"] == 'test-SID_peak') &
    #                                   (pre_list_append["classifier"] != 'log_reg') &
    #                                   (pre_list_append["stim"] == pre_list_append["classifier"])]
    # pre_list_filter['pred_acc'] = 0
    # def pred_acc(stim,pred_label):
    #     if (stim == pred_label):
    #         return 1
    #     else: 
    #         return 0
    # pre_list_filter['pred_acc'] = pre_list_filter.apply(lambda x: pred_acc(x['stim'],x['pred_label']), axis=1)
    # pred_acc_mean = np.mean(pre_list_filter['pred_acc'])
    
    # # prediction probability
    # pred_acc_mean_sub = pd.DataFrame([[pred_acc_mean],[subject],[mask_name],[bold_delay_task]])


    # pre_list=copy.deepcopy(data_list)
    # # delete_list = [71,62,53,44,35,26,17,8]
    # # [pre_list.pop(i) for i in delete_list]
    # pre_list_append = []
    # pre_list_append = pd.concat([pre_list[i] for i in range(len(pre_list))], axis=0)
    # pre_list_filter = pre_list_append[(pre_list_append["class"] != 'other') &
    #                                   (pre_list_append["classifier"] != 'log_reg') &
    #                                   (pre_list_append["test_set"] == 'test-SID_peak')]
    # pred_class_mean = [np.mean(pre_list_filter[(pre_list_filter['stim'] == class_label) &
    #                                            (pre_list_filter['classifier'] == class_label)]['probability']) 
    #                    for class_label in class_labels]
    # x_data = class_labels
    # y_data = pred_class_mean
    # # plot the prediction accuracy
    # plt.figure(dpi=400,figsize=(10,5))
    # plt.xticks(fontsize=10)
    # for i in range(len(x_data)):
    # 	plt.bar(x_data[i], y_data[i])
    # plt.axhline(y=0.125,color='silver',linestyle='--')
    # plt.title('%s_decoding probability of SID in %1.0f th TR peak %s' % (subject,bold_delay_task,mask_name))
    # plt.xlabel("label")
    # plt.ylabel("probability")
    # plt.savefig(opj(path_out_figs,'%s_decoding probability of SID in %1.0f th TR peak %s.png') % (subject,bold_delay_task,mask_name),
    #             dpi = 400,bbox_inches = 'tight')
    # plt.show()
        
    #     # In[PREDICTION PROBABILITY AND ACCURACY FOR LONG TERM]:
        
    # # prediction probability
    # pred_class_mean = []
    # pre_list=copy.deepcopy(data_list)
    # # delete_list = [71,62,53,44,35,26,17,8]
    # # [pre_list.pop(i) for i in delete_list]
    # pre_list_append = []
    # pre_list_append = pd.concat([pre_list[i] for i in range(len(pre_list))], axis=0)
    # pre_list_filter = pre_list_append[(pre_list_append["class"] != 'other') &
    #                                   (pre_list_append["classifier"] != 'log_reg') &
    #                                   (pre_list_append["test_set"] == 'test-SID_long')]
    # pred_class_mean = [[np.mean(pre_list_filter[(pre_list_filter['stim'] == class_label) 
    #                                             & (pre_list_filter['seq_tr'] ==  seq_tr)
    #                                            & (pre_list_filter['classifier'] == class_label)]['probability']) 
    #                    for class_label in class_labels] for seq_tr in np.arange(1,10)]
    # pred_class_mean = np.matrix(pred_class_mean)
    # whole_TR = np.arange(1,10)
    
    # fig, ax = plt.subplots(nrows = 2, ncols = 4,sharex=True, sharey=True,figsize=(10,4),dpi=400)
    # for j, class_label in enumerate(class_labels):
    #     print(j)
    #     x_data = whole_TR
    #     y_data = pred_class_mean[:,j]
    #     raw = math.floor(j/4)
    #     col = j-4*raw 
    #     ax[raw][col].set_ylim(ymax=0.6)
    #     ax[raw][col].set_xticks(range(1,len(whole_TR)+1),labels=whole_TR)
    #     ax[raw][col].set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6],labels=[0,10,20,30,40,50,60])
    #     ax[raw][col].plot(x_data,y_data,color='black',linestyle='-')
    #     ax[raw][col].axhline(y=0.125,color='silver',linestyle='--')
    #     ax[raw][col].set_title('%s' % class_label) 
    # fig.text(0.5, 0, 'Time from stimulus onset (TRs)', ha='center')
    # fig.text(0, 0.5, 'Probability(%)', va='center', rotation='vertical')
    # plt.tight_layout(pad=2.5, w_pad=0.5, h_pad=2)
    # plt.suptitle('%s decoding probability of SID (%1.0f th TR peak) %s' % (subject,bold_delay_task,mask_name))
    # plt.savefig(opj(path_out_figs,'%s decoding probability of SID (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name)),
    #             dpi = 400,bbox_inches = 'tight')
    # plt.show() 
    
    return pred_acc_mean_sub


pred_acc_mean_sub_all = Parallel(n_jobs=50)(
                delayed(decoding)(subject,bold_delay_task) 
                for bold_delay_task in bold_delay_task_list for subject in sub_list)

# In[STOP LOGGING]:
# a=np.hstack(pred_class_mean_all)
end = time.time()
total_time = (end - start) / 60
logging.info("total running time: %0.2f minutes" % total_time)
logging.shutdown()
# In[analysis]

pred_acc = copy.deepcopy(pred_acc_mean_sub_all)
pred_acc = np.hstack(pred_acc)
pred_acc = np.transpose(pred_acc)
pred_acc = pd.DataFrame(pred_acc,columns=['pred','subject','bold_delay_task'])
pred_acc_mean = pred_acc.groupby('bold_delay_task')['pred'].mean()
