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
import itertools
import warnings
import scipy
import scipy.io as scio
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

# In[paramters]


# parameters for TDLM
# the list of all the sequences
# uniquePerms = list(itertools.permutations([1,2,3,4],3))
uniquePerms = list(itertools.permutations([1,2,3,4],3))

# the number of sequences
nShuf = len(uniquePerms)
# all possible sequences
all_sequence = np.array(uniquePerms)
# real_sequence = np.array(real_sequence)
# rest of sequences for the specific subject
# set_y = set(map(tuple, real_sequence))
# idx = [tuple(point) not in set_y for point in all_sequence]
# rest_sequence = all_sequence[idx]
# recombine the real sequence and rest of sequences
# new_all_sequence = np.vstack((real_sequence,rest_sequence))
# the number of timelags, 
maxLag= 7
nbins=maxLag+1
# the number of states (decoding models)
nstates = 4
# the number of subjects
# nSubj = len(suball)
nSubj = 2
# the nan matrix for forward design matrix (36 subjects * 120 shuffles * 11 timelags)
sf = np.full([nSubj,2,len(all_sequence),maxLag+1],np.nan)
sb = np.full([nSubj,2,len(all_sequence),maxLag+1],np.nan)
# predefine GLM data frame 
betas_1GLM = None
betas_2GLM = None


# In[DEFINITION OF ALL FUNCTIONS]:
    
# set the classlabel based on marker
def classlabel(Marker):
    if Marker == 11 or Marker == 21 or Marker == 41 or Marker == 51:
        return 'girl'
    elif Marker == 12 or Marker == 22 or Marker == 42  or Marker == 52:
        return 'car'
    elif Marker == 13 or Marker == 23 or Marker == 43:
        return 'zebra'
    elif Marker == 14 or Marker == 24 or Marker == 44:
        return 'balloon'
    
class TaskData:
    def __init__(
        self,
        events,
        task,
        name,
        num_vol_run_1,
        num_vol_run_2,
        num_vol_run_3,
        num_vol_run_4,
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

        # select events: upright stimulus, correct answer trials only:
        if task == "VFL":
            self.events = events.loc[
                (events["task"] == 'VFL') &
                (events["accuracy"] != 0),
                :,
            ]
            # define the number of volumes per task run:
            self.num_vol_run = [[],[],[],[]]
            self.num_vol_run[0] = num_vol_run_1
            self.num_vol_run[1] = num_vol_run_2
            self.num_vol_run[2] = num_vol_run_3
            self.num_vol_run[3] = num_vol_run_4
            
        # if task == "REA":
        #     self.events = events.loc[
        #         (events["task"] == 'REA')
        #         & (events["accuracy"] != 0),
        #         :,
        #     ]
        #     # define the number of volumes per task run:
        #     self.num_vol_run = [[],[],[]]
        #     self.num_vol_run[0] = num_vol_run_1
        #     self.num_vol_run[1] = num_vol_run_2
        #     self.num_vol_run[2] = num_vol_run_3
            
        if task == 'REP':
            self.events = events.loc[
                (events["task"] == 'REP')
                & (events["accuracy"] != 0),
                :,
            ]
            # define the number of volumes per task run:
            self.num_vol_run = [[],[]]
            self.num_vol_run[0] = num_vol_run_1
            self.num_vol_run[1] = num_vol_run_2
            
        # reset the indices of the data frame:
        self.events.reset_index(drop=True,inplace=True)
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
        if self.task == 'VFL' or self.task == 'REA' or self.task == 'REP':
            # add the number of run volumes to the tr indices:
            run_volumes = [sum(self.num_vol_run[0:int(self.events["run"].iloc[row]-1)])
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
        if self.task == 'REP':
            self.marker = np.repeat(
                np.array(self.events["Marker"], dtype=object), self.interval)
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
        df["participant"] = np.repeat(self.events["participant"].unique(), num_pred)
        # add the name of the classifier to the dataframe:
        df["test_set"] = np.repeat(self.name, num_pred)
        # return the dataframe:
        return df

def detrend(data):
    from nilearn.signal import clean
    data_detrend = clean(signals=data, t_r=1.3, detrend=True, standardize=False)
    return data_detrend

def standardize(data):
    from nilearn.signal import clean
    data_standardize = clean(signals=data, t_r=1.3, detrend=False, standardize=True)
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

def stim_index(x):
    if x =='girl': 
     return 1 
    elif x =='car':
     return 2
    elif x =='zebra':
     return 3
    elif x =='balloon':
     return 4

def TransM(x):
    # create an empty matrix
    transition_matrix = np.zeros([4,4])
    # transition
    for a in range(2):
        transition_matrix[x[a]-1][x[a+1]-1] = 1
    return(transition_matrix)

def matlab_percentile(x, p):
    p = np.asarray(p, dtype=float)
    n = len(x)
    p = (p-50)*n/(n-1) + 50
    p = np.clip(p, 0, 100)
    return np.percentile(x, p)

def TDLM(probability_matrix,all_sequence,condition,subject):
    rp = []
    # print(subi,sub)    
    #detect for each 2048 ms trial
    # loops for real sequence and permutation trials
    if subject == 'sub-08':
        subi = 0
    elif subject == 'sub-09':
        subi = 1
        
    for i in range(len(all_sequence)):
        # real sequence
        rp = all_sequence[i,]
        # get the transition matrix
        T1 = TransM(rp)
        T2 = np.transpose(T1)
        nbins = maxLag+1
        # X1 = seq_pred_matrix[13*i:13*(i+1),:] #####
        X1 = np.array(probability_matrix)
        # timelag matrix 
        dm = scipy.linalg.toeplitz(X1[:,0],[np.zeros((nbins,1))]) #####
        dm = dm[:,1:]
        # 4 loops for another 4 states
        for k in range(1,4): 
            temp = scipy.linalg.toeplitz(X1[:,k],[np.zeros((nbins,1))])
            temp = temp[:,1:]
            dm = np.hstack((dm,temp))
        # the next time point needed to be predicted 
        Y = X1
        # build a new framework for first GLM betas
        betas_1GLM = np.full([nstates*maxLag, nstates],np.nan)
        #detect for each timelag
        for l in range(maxLag): 
            temp_zinds = np.array(range(0,nstates*maxLag,maxLag)) + l
            # First GLM
            design_mat_1 = np.hstack((dm[:,temp_zinds],np.ones((len(dm[:,temp_zinds]),1))))
            temp = np.dot(np.linalg.pinv(design_mat_1),Y)
            betas_1GLM[temp_zinds,:] = temp[:-1,:]  ####?
            
        betasnbins64 = np.reshape(betas_1GLM,[maxLag,np.square(nstates)],order = "F") ######
        # Second GLM
        design_mat_2 = np.transpose(np.vstack((np.reshape(T1,16,order = "F"),
                                               np.reshape(T2,16,order = "F"),
                                               np.reshape(np.eye(nstates),16,order = "F"),
                                               np.reshape(np.ones([nstates,nstates]),16,order = "F"))))
        betas_2GLM = np.dot(np.linalg.pinv(design_mat_2) , np.transpose(betasnbins64))
        # four different design matrix for regressor multiple to the temporal data
        # linear regression for the backward and forward replay 
        if condition == 'sf':
            sf[subi,0,i,1:] = betas_2GLM[0,:]
            sb[subi,0,i,1:] = betas_2GLM[1,:]
        elif condition == 'sb':
            sf[subi,1,i,1:] = betas_2GLM[0,:]
            sb[subi,1,i,1:] = betas_2GLM[1,:]
# In[SETUP NECESSARY PATHS ETC]:

# get start time of the script:
start = time.time()
# name of the current project:

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
path_glm_rea = opj(path_bids, "derivatives", "fmrireplay-glm-rea")
path_level1_rea = opj(path_glm_rea, "l1pipeline")
path_glm_rep = opj(path_bids, "derivatives", "fmrireplay-glm-replay")
path_level1_rep = opj(path_glm_rep, "l1pipeline")
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

mask_list = ['mask_temporal','mask_visual','mask_tem_pre','mask_prefrontal','mask_mixed','mask_mtl']
mask_index_list=[0,1,2,3,4,5]
bold_delay_task_list = [1,2,3,4,5,6,7,8,9]

# for test
# subject=sub_list[0]
mask_index=mask_index_list[0]
bold_delay_task=bold_delay_task_list[0]
# #         mask_run = mask_hpc[run-1]
# In[] 
# for subject in sub_list:
def decoding(mask_index, bold_delay_task):
    # In[CREATE PATHS TO OUTPUT DIRECTORIES]:
    for subject in sub_list:
        # path_fmriprep_sub = opj(path_fmriprep, subject)
        path_out = opj(path_decoding, subject)
        path_out_figs = opj(path_out, "plots")
        path_out_data = opj(path_out, "data")
        path_out_logs = opj(path_out, "logs")
        path_out_masks = opj(path_out, "masks")
        path_out_all = opj(path_decoding, 'all_subject_results')
        # CREATE OUTPUT DIRECTORIES IF THE DO NOT EXIST YET:
        for path in [path_out_figs, path_out_data, path_out_logs, path_out_masks, path_out_all]:
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
        n_run = 4    
        # paths to all events files of the current subject:
        path_events = opj(path_bids, subject, "func", "*tsv")
        # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
        path_events = sorted(glob.glob(path_events), key=lambda f: os.path.basename(f))
        
        # paths to all events files of the current subject:
        path_events_vfl = opj(path_bids, subject, "func", "*vfl*events.tsv")
        # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
        path_events_vfl = sorted(glob.glob(path_events_vfl), key=lambda f: os.path.basename(f))
        # import events file and save data in dataframe:
        df_events_vfl = pd.concat(
            (pd.read_csv(f, sep="\t") for f in path_events_vfl), ignore_index=True
        )
        df_events_vfl['stim_label'] = df_events_vfl.apply(lambda x: classlabel(x.stimMarker), axis = 1)
        del df_events_vfl['Unnamed: 0']
        del df_events_vfl['start']
        
        # replay file
        path_events_rep = opj(path_bids, subject, "func", "*rep*events.tsv")
        # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
        path_events_rep = sorted(glob.glob(path_events_rep), key=lambda f: os.path.basename(f))
        # import events file and save data in dataframe:
        df_events_rep = pd.concat(
            (pd.read_csv(f, sep="\t") for f in path_events_rep), ignore_index=True
        )
        df_events_rep['stim_label'] = df_events_rep.apply(lambda x: classlabel(x.Marker), axis = 1)
    
        del df_events_rep['Unnamed: 0']
    
        # In[CrepTE PATHS TO THE MRI DATA]:
        
        # define path to input directories:
    
        # load the visual mask task files:
        path_mask_vis_task = opj(path_masks, "mask_visual", subject, "*", "*task*.nii.gz")
        path_mask_vis_task = sorted(glob.glob(path_mask_vis_task), key=lambda f: os.path.basename(f))
    
        # load the hippocampus mask task files:
        path_mask_mtl_task = opj(path_masks, "mask_mtl", subject, "*", "*task*.nii.gz")
        path_mask_mtl_task = sorted(glob.glob(path_mask_mtl_task), key=lambda f: os.path.basename(f))
    
        # load the temporal cortex mask task files:
        path_mask_temporal_task = opj(path_masks, "mask_temporal", subject, "*", "*task*.nii.gz")
        path_mask_temporal_task = sorted(glob.glob(path_mask_temporal_task), key=lambda f: os.path.basename(f))
    
        # load the prefrontal cortex mask task files:
        path_mask_prefrontal_task = opj(path_masks, "mask_prefrontal", subject, "*", "*task*.nii.gz")
        path_mask_prefrontal_task = sorted(glob.glob(path_mask_prefrontal_task), key=lambda f: os.path.basename(f))
    
        # load the prefrontal cortex mask task files:
        path_mask_mixed_task = opj(path_masks, "mask_mixed", subject, "*", "*task*.nii.gz")
        path_mask_mixed_task = sorted(glob.glob(path_mask_mixed_task), key=lambda f: os.path.basename(f))
    
        # load the prefrontal cortex mask task files:
        path_mask_tem_pre_task = opj(path_masks, "mask_tem_pre", subject, "*", "*task*.nii.gz")
        path_mask_tem_pre_task = sorted(glob.glob(path_mask_tem_pre_task), key=lambda f: os.path.basename(f))
    
        # load the hippocampus mask task files:
        path_mask_hippocampus_task = opj(path_masks, "mask_hippocampus", subject, "*", "*task*.nii.gz")
        path_mask_hippocampus_task = sorted(glob.glob(path_mask_hippocampus_task), key=lambda f: os.path.basename(f))
        
        # load the whole brain mask files:
        path_mask_whole_task = opj(path_fmriprep, subject, "func", "*task*T1w*brain_mask.nii.gz")
        path_mask_whole_task = sorted(glob.glob(path_mask_whole_task), key=lambda f: os.path.basename(f))
    
        # load the visual functional localizer mri task files:
        path_func_task_vfl = opj(path_level1_vfl, "smooth", subject, "*", "*task*nii.gz")
        path_func_task_vfl = sorted(glob.glob(path_func_task_vfl), key=lambda f: os.path.basename(f))
    
        # load the single item decoding mri task files:
        path_func_task_rep = opj(path_level1_rep, "smooth", subject, "*", "*task*nii.gz")
        path_func_task_rep = sorted(glob.glob(path_func_task_rep), key=lambda f: os.path.basename(f))
        
        # load the anatomical mri file:
        path_anat = opj(path_fmriprep, subject, "anat", "%s_desc-preproc_T1w.nii.gz" % subject)
        path_anat = sorted(glob.glob(path_anat), key=lambda f: os.path.basename(f))
    
        # load the spm.mat files:
        path_spm_mat_vfl = opj(path_level1_vfl, "contrasts", subject, "*", "SPM.mat")
        path_spm_mat_vfl = sorted(glob.glob(path_spm_mat_vfl), key=lambda f: os.path.dirname(f))
    
        # load the t-maps of the first-level glm:
        path_tmap_vfl = opj(path_level1_vfl, "contrasts", subject, "*", "spmT*.nii")
        path_tmap_vfl = sorted(glob.glob(path_tmap_vfl), key=lambda f: os.path.dirname(f))
    
    
    
        # In[LOAD THE MRI DATA]:
        
        anat = image.load_img(path_anat[0])
        # load visual mask:
        mask_visual = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_vis_task[10:14])]
        # load mtl mask:
        mask_mtl = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_mtl_task[10:14])]
        # load temporal mask:
        mask_temporal = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_temporal_task[10:14])]
        # load prefrontal mask:
        mask_prefrontal = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_prefrontal_task[10:14])]
        # load mixed mask:
        mask_mixed = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_mixed_task[10:14])]
        # load tem_pre mask:
        mask_tem_pre = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_tem_pre_task[10:14])]
        # load hippocampus mask:
        mask_hippocampus = [image.load_img(i) for i in copy.deepcopy(path_mask_hippocampus_task[10:14])]
        # load wholebrain mask for SID:
        mask_brain = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_whole_task[10:14])]
        image.load_img(path_mask_whole_task[10:14]).get_data().astype(int)
        
        # load tmap data:
        tmaps_vfl = [image.load_img(i).get_data().astype(float) for i in copy.deepcopy(path_tmap_vfl)]
        
        # LOAD SMOOTHED FMRI DATA FOR ALL FUNCTIONAL TASK RUNS:
        # load smoothed functional mri data for all eight task runs:
        data_task_vfl = [image.load_img(i) for i in path_func_task_vfl]
        data_task_rep = [image.load_img(i) for i in path_func_task_rep]
    
        mask_list = [mask_temporal,mask_visual,mask_tem_pre,mask_prefrontal,mask_mixed,mask_mtl,mask_hippocampus]
        mask_name_list= ['mask_temporal','mask_vis','mask_tem_pre','mask_prefrontal','mask_mixed','mask_mtl','mask_hippocampus']
        mask = mask_list[mask_index]
        mask_name = mask_name_list[mask_index]
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
        tmaps_masked = [np.multiply(i, j) for (i,j) in zip(copy.deepcopy(mask),copy.deepcopy(tmaps_vfl))]
    
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
        threshold = 3
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
        # crepte a dataframe with the number of voxels
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
    
        # In[DEFINE THE CLASSIFIERS]:
        
        class_labels = ["girl", "car", "zebra", "balloon"]
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
            num_vol_run_1 = np.size(data_task_vfl[0]._data,axis=3),
            num_vol_run_2 = np.size(data_task_vfl[1]._data,axis=3),
            num_vol_run_3 = np.size(data_task_vfl[2]._data,axis=3),
            num_vol_run_4 = np.size(data_task_vfl[3]._data,axis=3),
        )
        test_rep_long = TaskData(
            events=df_events_rep, 
            task='REP', 
            bold_delay=0, 
            interval=13,
            name='test-rep_long',
            num_vol_run_1 = np.size(data_task_rep[0]._data,axis=3),
            num_vol_run_2 = np.size(data_task_rep[1]._data,axis=3),
            num_vol_run_3 = 0,
            num_vol_run_4 = 0,
            )
        test_sets = [
            # test_VFL_peak,
            # test_VFL_long,
            # test_rea_reak,
            # test_rea_long,
            test_rep_long,
        #     test_SID_peak,
        ]
        
        # In[]
        mask_visual_dy = [image.load_img(i) for i in copy.deepcopy(path_mask_vis_task[6:8])]
        mask_hippocampus_dy = [image.load_img(i) for i in copy.deepcopy(path_mask_hippocampus_task[6:8])]

        # replay
        masked_visual_data = [masking.apply_mask(data, mask_visual_dy[i]) for (i,data) in enumerate(data_task_rep)]
        data_detrend_test = [standardize(i) for i in masked_visual_data]
        data_detrend_test = np.vstack(data_detrend_test)
        visual_data = data_detrend_test.mean(axis=1)
        tr_onset = np.unique(test_rep_long.stim_trs)
        visual_data_trial = [visual_data[tr_onset[i]:tr_onset[i]+12] for i in range(len(tr_onset))]
        visual_data_trial_mean = (np.vstack(visual_data_trial)).mean(axis=0)
        
        masked_hippocampus_data = [masking.apply_mask(data, mask_hippocampus_dy[i]) for (i,data) in enumerate(data_task_rep)]
        data_detrend_test = [standardize(i) for i in masked_hippocampus_data]
        data_detrend_test = np.vstack(data_detrend_test)
        hippocampus_data = data_detrend_test.mean(axis=1)
        tr_onset = np.unique(test_rep_long.stim_trs)
        hippocampus_data_trial = [hippocampus_data[tr_onset[i]:tr_onset[i]+12] for i in range(len(tr_onset))]
        hippocampus_data_trial_mean = (np.vstack(hippocampus_data_trial)).mean(axis=0)
        
        
        
        plt.figure(dpi=200,figsize=(10,5))
        plt.xticks(fontsize=10)
        x=np.arange(1,len(hippocampus_data_trial_mean)+1)
        l1=plt.plot(x,hippocampus_data_trial_mean,color='turquoise',linestyle='-',marker = 'o',label='hippocampus')
        l2=plt.plot(x,visual_data_trial_mean,color='red',linestyle='-',marker = 'o',label='visual cortex')
        plt.legend()
        plt.title('hippocampus and visual cortex')
        plt.xlabel('timelag (TRs)')
        plt.ylabel('BOLD signal (zscore,average all voxels in hippocampus and visual mask)')
        # plt.savefig(opj(path_out_all,'all subject all condition forward-backward replay in sequential condition (%1.0f th TR peak) %s.png' % (bold_delay_task,mask_name)),
        #             dpi = 400,bbox_inches = 'tight')
        plt.show()
        
        
        
        
        
        
        
        # In[SEPARATE RUN-WISE STANDARDIZATION (Z-SCORING) OF VISUAL FUNCTIONAL LOCALIZER]:
       
        n_fold = 8
        data_list = []
        runs = list([1,1,2,2,3,3,4,4])
        folds = list(range(1, n_fold + 1))
        mask_label = mask_name
    
        for fold, run in zip(folds, runs):
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
            masked_data_test = [masking.apply_mask(i, mask_run) for i in data_task_rep]
            # detrend the masked fMRI data separately for each run:
            data_detrend = [detrend(i) for i in masked_data]
            data_detrend_test = [detrend(i) for i in masked_data_test]
            # combine the detrended data of all runs:
            data_detrend = np.vstack(data_detrend)
            data_detrend_test = np.vstack(data_detrend_test)
            # loop through all classifiers in the classifier set:
            for clf_name, clf in clf_set.items():
                # print classifier:
                logging.info("classifier: %s" % clf_name)
                # fit the classifier to the training data:
                train_VFL_peak.zscore(signals=data_detrend, fold_list=train_folds)
                # get the example labels:
                train_stim = copy.deepcopy((train_VFL_peak.stim[train_VFL_peak.fold == train_folds[0]],
                                            train_VFL_peak.stim[train_VFL_peak.fold == train_folds[1]],
                                            train_VFL_peak.stim[train_VFL_peak.fold == train_folds[2]]))
                train_stim = pd.concat((pd.Series(train_stim[0]),pd.Series(train_stim[1]),pd.Series(train_stim[2])))
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
                    test_set.zscore(signals=data_detrend_test, fold_list=test_folds)
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
        if subject == 'sub-08':
            picList = [3,4,1,2]
            real_sequence = [(4,1,2),(1,4,3)]
            sequence51 = [(4,1,2)]
            sequence52 = [(1,4,3)]
        elif subject == 'sub-09':
            picList = [2,3,4,1]
            real_sequence = [(3,4,1),(4,3,2)]
            sequence51 = [(3,4,1)]
            sequence52 = [(4,3,2)]
        
        # prediction probability
        pred_class_mean = []
        pre_list=copy.deepcopy(data_list)
        # delete_list = [71,62,53,44,35,26,17,8]
        # [pre_list.pop(i) for i in delete_list]
        pre_list_append = []
        pre_list_append = pd.concat([pre_list[i] for i in range(len(pre_list))], axis=0)
        pre_list_filter_51 = pre_list_append[(pre_list_append["class"] != 'other') &
                                          (pre_list_append["classifier"] != 'log_reg') &
                                          (pre_list_append["test_set"] =='test-rep_long') &
                                          (pre_list_append['marker'] == 51)]
        pre_list_filter_51['class_index'] = pre_list_filter_51['class'].map(lambda a : stim_index(a))
        pre_list_filter_51_sort = pre_list_filter_51.sort_values(by=['class_index','run','trials'])
        pred_array_51 = np.array(pre_list_filter_51_sort.loc[:,'probability'])
        pred_matrix_51 = np.reshape(pred_array_51,(int(len(pred_array_51)/4),4),order='F')
        pred_matrix_51_seq = [pd.Series(pred_matrix_51[:,sequence51[0][0]-1]),
                                pd.Series(pred_matrix_51[:,sequence51[0][1]-1]),
                                pd.Series(pred_matrix_51[:,sequence51[0][2]-1]),
                                pd.Series(pred_matrix_51[:,picList[0]-1])]
        pred_matrix_51_seq = pd.concat(pred_matrix_51_seq,axis=1)
        
        # marker 52 
        pre_list_filter_52 = pre_list_append[(pre_list_append["class"] != 'other') &
                                          (pre_list_append["classifier"] != 'log_reg') &
                                          (pre_list_append["test_set"] =='test-rep_long') &
                                          (pre_list_append['marker'] == 52)]
        
        pre_list_filter_52['class_index'] = pre_list_filter_52['class'].map(lambda a : stim_index(a))
        pre_list_filter_52_sort = pre_list_filter_52.sort_values(by=['class_index','run','trials'])
        pred_array_52 = np.array(pre_list_filter_52_sort.loc[:,'probability'])
        pred_matrix_52 = np.reshape(pred_array_52,(int(len(pred_array_52)/4),4),order='F')
        pred_matrix_52_seq = [pd.Series(pred_matrix_52[:,sequence52[0][0]-1]),
                                pd.Series(pred_matrix_52[:,sequence52[0][1]-1]),
                                pd.Series(pred_matrix_52[:,sequence52[0][2]-1]),
                                pd.Series(pred_matrix_52[:,picList[3]-1])]
        pred_matrix_52_seq = pd.concat(pred_matrix_52_seq,axis=1)
        
     
        TDLM(probability_matrix=pred_matrix_51_seq,all_sequence=all_sequence,condition='sf',subject=subject)
        TDLM(probability_matrix=pred_matrix_52_seq,all_sequence=all_sequence,condition='sb',subject=subject)
    
# In[PLOT the FIGURE]
    
        if subject == 'sub-08':
            subi = 0
        elif subject == 'sub-09':
            subi = 1
        
        # sub-08 and sub-09
        
        # mean the real sequence and permutation sequences respectively
        sf_sequence_51 = sf[subi,0,0,:]
        temp_per = sf[subi,0,1:,1:]
        temp_per = abs(temp_per)
        temp_per = np.amax(temp_per,axis=1)
        sf_permutation = matlab_percentile(temp_per,95)
        
        sb_sequence_51 = sb[subi,0,0,:]
        temp_per = sb[subi,0,1:,1:]
        temp_per = abs(temp_per)
        temp_per = np.amax(temp_per,axis=1)
        sb_permutation = matlab_percentile(temp_per,95)
        
        dif = sf_sequence_51 - sb_sequence_51
        temp_per = sf[subi,0,1:,1:]-sb[subi,0,1:,1:]
        temp_per = abs(temp_per)
        temp_per = np.amax(temp_per,axis=1)
        dif_permutation = matlab_percentile(temp_per,95)
        
        # plot the results
        plt.figure(dpi=400,figsize=(10,5))
        x=np.arange(0,nbins,1)
        plt.xticks(range(len(x)),x,fontsize=10)
        l1=plt.plot(x,sf_sequence_51,color='turquoise',linestyle='-',marker = 'o')
        plt.axhline(y=sf_permutation,color='silver',linestyle='--')
        plt.axhline(y=(-sf_permutation),color='silver',linestyle='--')
        plt.title('%s Forward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name))
        plt.xlabel('timelag (TRs)')
        plt.ylabel('sequenceness')
        plt.savefig(opj(path_out_all,'%s Forward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name)),
                    dpi = 400,bbox_inches = 'tight')
        plt.show()
        
        plt.figure(dpi=400,figsize=(10,5))
        x=np.arange(0,nbins,1)
        plt.xticks(range(len(x)),x,fontsize=10)
        l1=plt.plot(x,sb_sequence_51,color='turquoise',linestyle='-',marker = 'o')
        plt.axhline(y=sb_permutation,color='silver',linestyle='--')
        plt.axhline(y=(-sb_permutation),color='silver',linestyle='--')
        plt.title('%s Backward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name))
        plt.xlabel('timelag (TRs)')
        plt.ylabel('sequenceness')
        plt.savefig(opj(path_out_all,'%s Backward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name)),
                    dpi = 400,bbox_inches = 'tight')
        plt.show()
        
        plt.figure(dpi=400,figsize=(10,5))
        x=np.arange(0,nbins,1)
        plt.xticks(range(len(x)),x,fontsize=10)
        l1=plt.plot(x,dif,color='turquoise',linestyle='-',marker = 'o')
        plt.axhline(y=dif_permutation,color='silver',linestyle='--')
        plt.axhline(y=(-dif_permutation),color='silver',linestyle='--')
        plt.title('%s Forward-Backward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name))
        plt.xlabel('timelag (TRs)')
        plt.ylabel('sequenceness')
        plt.savefig(opj(path_out_all,'%s Forward-Backward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name)),
                    dpi = 400,bbox_inches = 'tight')
        plt.show()
        
        
        sf_sequence_52 = sf[subi,1,0,:]
        temp_per = sf[subi,1,1:,1:]
        temp_per = abs(temp_per)
        temp_per = np.amax(temp_per,axis=1)
        sf_permutation = matlab_percentile(temp_per,95)
        
        sb_sequence_52 = sb[subi,1,0,:]
        temp_per = sb[subi,1,1:,1:]
        temp_per = abs(temp_per)
        temp_per = np.amax(temp_per,axis=1)
        sb_permutation = matlab_percentile(temp_per,95)
        
        dif = sf_sequence_52 - sb_sequence_52
        temp_per = sf[subi,1,1:,1:]-sb[subi,1,1:,1:]
        temp_per = abs(temp_per)
        temp_per = np.amax(temp_per,axis=1)
        dif_permutation = matlab_percentile(temp_per,95)
        
        # plot the results
        plt.figure(dpi=400,figsize=(10,5))
        x=np.arange(0,nbins,1)
        plt.xticks(range(len(x)),x,fontsize=10)
        l1=plt.plot(x,sf_sequence_52,color='turquoise',linestyle='-',marker = 'o')
        plt.axhline(y=sf_permutation,color='silver',linestyle='--')
        plt.axhline(y=(-sf_permutation),color='silver',linestyle='--')
        plt.title('%s Forward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name))
        plt.xlabel('timelag (TRs)')
        plt.ylabel('sequenceness')
        plt.savefig(opj(path_out_all,'%s Forward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name)),
                    dpi = 400,bbox_inches = 'tight')
        plt.show()
        
        plt.figure(dpi=400,figsize=(10,5))
        x=np.arange(0,nbins,1)
        plt.xticks(range(len(x)),x,fontsize=10)
        l1=plt.plot(x,sb_sequence_52,color='turquoise',linestyle='-',marker = 'o')
        plt.axhline(y=sb_permutation,color='silver',linestyle='--')
        plt.axhline(y=(-sb_permutation),color='silver',linestyle='--')
        plt.title('%s Backward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name))
        plt.xlabel('timelag (TRs)')
        plt.ylabel('sequenceness')
        plt.savefig(opj(path_out_all,'%s Backward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name)),
                    dpi = 400,bbox_inches = 'tight')
        plt.show()
        
        plt.figure(dpi=400,figsize=(10,5))
        x=np.arange(0,nbins,1)
        plt.xticks(range(len(x)),x,fontsize=10)
        l1=plt.plot(x,dif,color='turquoise',linestyle='-',marker = 'o')
        plt.axhline(y=dif_permutation,color='silver',linestyle='--')
        plt.axhline(y=(-dif_permutation),color='silver',linestyle='--')
        plt.title('%s Forward-Backward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name))
        plt.xlabel('timelag (TRs)')
        plt.ylabel('sequenceness')
        plt.savefig(opj(path_out_all,'%s Forward-Backward replay in sequential condition (%1.0f th TR peak) %s.png' % (subject,bold_delay_task,mask_name)),
                    dpi = 400,bbox_inches = 'tight')
        plt.show()
        
        
        
        
        
        
    
    # all subject mean
    
    sf_sequence_51 = sf[:,0,0,:].mean(axis=0)
    temp_per = sf[:,0,1:,1:].mean(axis=0)
    temp_per = abs(temp_per)
    temp_per = np.amax(temp_per,axis=1)
    sf_permutation = matlab_percentile(temp_per,95)
    
    sb_sequence_51 = sb[:,0,0,:].mean(axis=0)
    temp_per = sb[:,0,1:,1:].mean(axis=0)
    temp_per = abs(temp_per)
    temp_per = np.amax(temp_per,axis=1)
    sb_permutation = matlab_percentile(temp_per,95)
    
    dif = sf_sequence_51 - sb_sequence_51
    temp_per = sf[:,0,1:,1:]-sb[:,0,1:,1:]
    temp_per = temp_per.mean(axis=0)
    temp_per = abs(temp_per)
    temp_per = np.amax(temp_per,axis=1)
    dif_permutation = matlab_percentile(temp_per,95)
    
    
    # sns.relplot(x='Timelag(TR)', y='For Sequenceness', kind="line", data=sf_sequence);
    # plot the results
    plt.figure(dpi=400,figsize=(10,5))
    x=np.arange(0,nbins,1)
    plt.xticks(range(len(x)),x,fontsize=10)
    l1=plt.plot(x,sf_sequence_51,color='turquoise',linestyle='-',marker = 'o')
    plt.axhline(y=sf_permutation,color='silver',linestyle='--')
    plt.axhline(y=(-sf_permutation),color='silver',linestyle='--')
    plt.title('Forward replay in sequential condition (%1.0f th TR peak) %s' % (bold_delay_task,mask_name))
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path_out_all,'subject mean Forward replay in sequential condition (%1.0f th TR peak) %s.png' % (bold_delay_task,mask_name)),
                dpi = 400,bbox_inches = 'tight')
    plt.show()
    
    plt.figure(dpi=400,figsize=(10,5))
    x=np.arange(0,nbins,1)
    plt.xticks(range(len(x)),x,fontsize=10)
    l1=plt.plot(x,sb_sequence_51,color='turquoise',linestyle='-',marker = 'o')
    plt.axhline(y=sb_permutation,color='silver',linestyle='--')
    plt.axhline(y=(-sb_permutation),color='silver',linestyle='--')
    plt.title('Backward replay in sequential condition (%1.0f th TR peak) %s' % (bold_delay_task,mask_name))
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path_out_all,'subject mean Backward replay in sequential condition (%1.0f th TR peak) %s.png' % (bold_delay_task,mask_name)),
                dpi = 400,bbox_inches = 'tight')
    plt.show()
    
    plt.figure(dpi=400,figsize=(10,5))
    x=np.arange(0,nbins,1)
    plt.xticks(range(len(x)),x,fontsize=10)
    l1=plt.plot(x,dif,color='turquoise',linestyle='-',marker = 'o')
    plt.axhline(y=dif_permutation,color='silver',linestyle='--')
    plt.axhline(y=(-dif_permutation),color='silver',linestyle='--')
    plt.title('Forward-Backward replay in sequential condition (%1.0f th TR peak) %s' % (bold_delay_task,mask_name))
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path_out_all,'all subject mean Forward-Backward replay in sequential condition (%1.0f th TR peak) %s.png' % (bold_delay_task,mask_name)),
                dpi = 400, bbox_inches = 'tight')
    plt.show()
    
    # all subject mean
    
    sf_sequence_52 = sf[:,1,0,:].mean(axis=0)
    temp_per = sf[:,1,1:,1:].mean(axis=0)
    temp_per = abs(temp_per)
    temp_per = np.amax(temp_per,axis=1)
    sf_permutation = matlab_percentile(temp_per,95)
    
    sb_sequence_52 = sb[:,1,0,:].mean(axis=0)
    temp_per = sb[:,1,1:,1:].mean(axis=0)
    temp_per = abs(temp_per)
    temp_per = np.amax(temp_per,axis=1)
    sb_permutation = matlab_percentile(temp_per,95)
    
    dif = sf_sequence_52 - sb_sequence_52
    temp_per = sf[:,1,1:,1:]-sb[:,1,1:,1:]
    temp_per = temp_per.mean(axis=0)
    temp_per = abs(temp_per)
    temp_per = np.amax(temp_per,axis=1)
    dif_permutation = matlab_percentile(temp_per,95)
    
    
    # sns.relplot(x='Timelag(TR)', y='For Sequenceness', kind="line", data=sf_sequence);
    # plot the results
    plt.figure(dpi=400,figsize=(10,5))
    x=np.arange(0,nbins,1)
    plt.xticks(range(len(x)),x,fontsize=10)
    l1=plt.plot(x,sf_sequence_52,color='turquoise',linestyle='-',marker = 'o')
    plt.axhline(y=sf_permutation,color='silver',linestyle='--')
    plt.axhline(y=(-sf_permutation),color='silver',linestyle='--')
    plt.title('Forward replay 52 in sequential condition (%1.0f th TR peak) %s' % (bold_delay_task,mask_name))
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path_out_all,'subject mean Forward replay 52 in sequential condition (%1.0f th TR peak) %s.png' % (bold_delay_task,mask_name)),
                dpi = 400,bbox_inches = 'tight')
    plt.show()
    
    plt.figure(dpi=400,figsize=(10,5))
    x=np.arange(0,nbins,1)
    plt.xticks(range(len(x)),x,fontsize=10)
    l1=plt.plot(x,sb_sequence_52,color='turquoise',linestyle='-',marker = 'o')
    plt.axhline(y=sb_permutation,color='silver',linestyle='--')
    plt.axhline(y=(-sb_permutation),color='silver',linestyle='--')
    plt.title('Backward replay 52 in sequential condition (%1.0f th TR peak) %s' % (bold_delay_task,mask_name))
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path_out_all,'subject meanBackward replay 52 in sequential condition (%1.0f th TR peak) %s.png' % (bold_delay_task,mask_name)),
                dpi = 400,bbox_inches = 'tight')
    plt.show()
    
    plt.figure(dpi=400,figsize=(10,5))
    x=np.arange(0,nbins,1)
    plt.xticks(range(len(x)),x,fontsize=10)
    l1=plt.plot(x,dif,color='turquoise',linestyle='-',marker = 'o')
    plt.axhline(y=dif_permutation,color='silver',linestyle='--')
    plt.axhline(y=(-dif_permutation),color='silver',linestyle='--')
    plt.title('Forward-Backward 52 replay in sequential condition (%1.0f th TR peak) %s' % (bold_delay_task,mask_name))
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path_out_all,'subject mean Forward-Backward 52 replay in sequential condition (%1.0f th TR peak) %s.png' % (bold_delay_task,mask_name)),
                dpi = 400, bbox_inches = 'tight')
    plt.show()
    
    
    
    
    
    
    
    #mean subject and condition
    
    sf_sequence = sf[:,:,0,:].mean(axis=1).mean(axis=0)
    temp_per = sf[:,:,1:,1:].mean(axis=1).mean(axis=0)
    temp_per = abs(temp_per)
    temp_per = np.amax(temp_per,axis=1)
    sf_permutation = matlab_percentile(temp_per,95)
    
    sb_sequence = sb[:,:,0,:].mean(axis=1).mean(axis=0)
    temp_per = sb[:,:,1:,1:].mean(axis=1).mean(axis=0)
    temp_per = abs(temp_per)
    temp_per = np.amax(temp_per,axis=1)
    sb_permutation = matlab_percentile(temp_per,95)
    
    dif = sf_sequence_52 - sb_sequence_52
    temp_per = (sf[:,:,1:,1:]-sb[:,:,1:,1:]).mean(axis=0)
    temp_per = temp_per.mean(axis=0)
    temp_per = abs(temp_per)
    temp_per = np.amax(temp_per,axis=1)
    dif_permutation = matlab_percentile(temp_per,95)

  
    plt.figure(dpi=400,figsize=(10,5))
    x=np.arange(0,nbins,1)
    plt.xticks(range(len(x)),x,fontsize=10)
    l1=plt.plot(x,sf_sequence,color='turquoise',linestyle='-',marker = 'o')
    plt.axhline(y=sf_permutation,color='silver',linestyle='--')
    plt.axhline(y=(-sf_permutation),color='silver',linestyle='--')
    plt.title('all subject forward replay in sequential condition (%1.0f th TR peak) %s' % (bold_delay_task,mask_name))
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path_out_all,'all subject all condition forward replay in sequential condition (%1.0f th TR peak) %s.png' % (bold_delay_task,mask_name)),
                dpi = 400,bbox_inches = 'tight')
    plt.show()
    
    plt.figure(dpi=400,figsize=(10,5))
    x=np.arange(0,nbins,1)
    plt.xticks(range(len(x)),x,fontsize=10)
    l1=plt.plot(x,sb_sequence,color='turquoise',linestyle='-',marker = 'o')
    plt.axhline(y=sb_permutation,color='silver',linestyle='--')
    plt.axhline(y=(-sb_permutation),color='silver',linestyle='--')
    plt.title('all subject backward replay in sequential condition (%1.0f th TR peak) %s' % (bold_delay_task,mask_name))
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path_out_all,'all subject all condition backward replay in sequential condition (%1.0f th TR peak) %s.png' % (bold_delay_task,mask_name)),
                dpi = 400,bbox_inches = 'tight')
    plt.show()
    
    
    plt.figure(dpi=400)
    plt.xticks(fontsize=10)
    x=np.arange(0,nbins)
    l1=plt.plot(x,dif,color='turquoise',linestyle='-',marker = 'o')
    plt.axhline(y=dif_permutation,color='silver',linestyle='--')
    plt.axhline(y=(-dif_permutation),color='silver',linestyle='--')
    plt.title('Forward-Backward replay in sequential condition (%1.0f th TR peak) %s' % (bold_delay_task,mask_name))
    plt.xlabel('timelag (TRs)')
    plt.ylabel('squenceness')
    plt.savefig(opj(path_out_all,'all subject all condition forward-backward replay in sequential condition (%1.0f th TR peak) %s.png' % (bold_delay_task,mask_name)),
                dpi = 400,bbox_inches = 'tight')
    plt.show()

    # In[]
    # return pred_acc_mean_sub

# pred_acc_mean_sub_all = 
Parallel(n_jobs=60)(
                delayed(decoding)(mask_index, bold_delay_task)
                for bold_delay_task in bold_delay_task_list for mask_index in mask_index_list)

# In[STOP LOGGING]:
# a=np.hstack(pred_class_mean_all)
end = time.time()
total_time = (end - start) / 60
logging.info("total running time: %0.2f minutes" % total_time)
logging.shutdown()
# In[analysis]

# pred_acc = copy.deepcopy(pred_acc_mean_sub_all)
# pred_acc = np.hstack(pred_acc)
# pred_acc = np.transpose(pred_acc)
# pred_acc = pd.DataFrame(pred_acc,columns=['pred','subject','bold_delay_task','mask_name'])
# pred_acc_mean = pred_acc.groupby(['bold_delay_task','mask_name'])['pred'].mean()



