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
# import sys
import copy
from pprint import pformat
import numpy as np
# from nilearn.input_data import NiftiMasker
from nilearn import plotting, image, masking
import pandas as pd
# import math
from matplotlib import pyplot as plt
import warnings
# from sklearn.svm import SVC
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import RidgeClassifier
# from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn.preprocessing import StandardScaler
from bids.layout import BIDSLayout
from joblib import Parallel, delayed
import seaborn as sns
# DEAL WITH ERRORS, WARNING ETC
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

# In[DEFINITION OF ALL FUNCTIONS]:
    
# set the classlabel based on marker:
def classlabel(Marker):
    if Marker == 11 or Marker == 21:
        return 'girl'
    elif Marker == 12 or Marker == 22:
        return 'scissors'
    elif Marker == 13 or Marker == 23:
        return 'zebra'
    elif Marker == 14 or Marker == 24:
        return 'banana'

# fMRI event function
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
                (events["accuracy"] != 0),
                :,]
            # define the number of volumes per task run: 4 runs for VFL
            self.num_vol_run = [[],[],[],[]]
            self.num_vol_run[0] = num_vol_run_1
            self.num_vol_run[1] = num_vol_run_2
            self.num_vol_run[2] = num_vol_run_3
            self.num_vol_run[3] = num_vol_run_4
        # reset the indices of the data frame:
        self.events.reset_index(drop=True,inplace=True)
        self.participant = str(self.events["participant"][0])

        # sort all values by session and run:
        self.events.sort_values(by=["run"])
        # call further function upon initialization:
        self.define_trs()
        self.get_stats()

    def define_trs(self):
        # import relevant functions:
        import numpy as np
        # select all events onsets:
        self.event_onsets = self.events["onset"]
        # add the selected delay to account for the peak of the hrf
        self.bold_peaks = (self.event_onsets + (self.bold_delay * self.t_tr))  # delay is the time, not the TR
        # divide the expected time-point of bold peaks by the repetition time:
        self.peak_trs = self.bold_peaks / self.t_tr
        if self.task == 'VFL':
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
        # self.itis = np.repeat(
        #     np.array(self.events["interval_time"], dtype=float), self.interval
        # )
        self.stim_trs = np.repeat(np.array(self.stim_trs, dtype=int), self.interval)

    # def zscore(self, signal, fold_list):
    #     from nilearn.signal import clean
    #     import numpy as np
    #     # get boolean indices for all run indices in the run list:
    #     fold_indices = np.isin(self.fold, list(fold_list))
    #     # standardize data all runs in the run list:
    #     self.data_zscored = clean(
    #         signals=signal[self.trs[fold_indices]],
    #         # sessions=self.runs[fold_indices],
    #         t_r=1.3,
    #         detrend=False,
    #         standardize=True)
        
    def zscore(self, signal, fold_list):
        from nilearn.signal import clean
        import numpy as np
        # get boolean indices for all run indices in the run list:
        fold_indices = np.isin(self.fold, list(fold_list))
        seq_tr = np.tile(np.arange(1, self.interval + 1), self.num_trials)
        self.data_zscored = np.zeros([int(sum(fold_indices)),np.size(signal,axis=1)])
        self.signals_1 = [[] for i in range(self.interval)]
        for i in np.arange(1,self.interval+1):
            indexs = (seq_tr == i) #(tr)
            fold_seq = np.multiply(fold_indices,indexs)
            seq_fold_trials = fold_seq[fold_indices]
            self.signals_1[i-1] = signal[self.trs[fold_seq]]
            self.data_zscored[seq_fold_trials] = clean(
                signals=signal[self.trs[fold_seq]],
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
# name of the current project:
project = "fmrireplay"
path_root = None
sub_list = None
subject = None
# path to the project root:
project_name = "fmrireplay-decoding"
path_root = opj(os.getcwd().split(project)[0], "fmrireplay")
path_bids = opj(path_root, "BIDS")
path_bids_nf = opj(path_root, "BIDS-nofieldmap")
path_code = opj(path_bids_nf, "code", "decoding-TDLM")
path_fmriprep = opj(path_bids_nf, "derivatives", "fmrireplay-fmriprep")
path_masks = opj(path_bids_nf, "derivatives", "fmrireplay-masks")
path_glm_vfl = opj(path_bids_nf, "derivatives", "fmrireplay-glm-vfl")
path_level1_vfl = opj(path_glm_vfl, "l1pipeline")
path_decoding = opj(path_bids_nf, "derivatives", project_name)
path_behavior = opj(path_bids_nf,'derivatives','fmrireplay-behavior')
path_out_all = opj(path_decoding,'all_subject_results')

data_list_all=[] 
pred_acc_mean_sub_all = []
# define the subject id
layout = BIDSLayout(path_bids_nf)
sub_list = sorted(layout.get_subjects())
# sub_list = sub_list[0:41]
# delete_list = [30,29,27,25,22,15,13,11,4,0]
# delete_list = [30]
# [sub_list.pop(i) for i in delete_list]
subnum_list = copy.deepcopy(sub_list)
sub_template = ["sub-"] * len(sub_list)
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]
# sub-04 for test the code
subnum_list = list(map(int,subnum_list))

mask_list= ['mask_visual', 'mask_hippocampus', 'mask_entorhinal', 'mask_mtl', 'mask_vis_mtl', 'mask_temporal', 'mask_prefrontal']
mask_index_list=[0,1,2,3,4,5,6]
bold_delay_task_list = [1,2,3,4,5,6,7,8,9]

# for test
subject=sub_list[0]
mask_index=mask_index_list[4]
bold_delay_task=bold_delay_task_list[2]
path_out_all = opj(path_decoding,'all_subject_results')
# #         mask_run = mask_hpc[run-1]
# In[] 
# for subject in sub_list:
def decoding(subject):
# def decoding(subject,mask_index,bold_delay_task):
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
    
    # In[LOAD BEHAVIORAL DATA (THE EVENTS.TSV FILES OF THE SUBJECT)]:
    n_run = 4    
    # paths to all events files of the current subject:
    path_events = opj(path_behavior, subject,  "*tsv")
    # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
    path_events = sorted(glob.glob(path_events), key=lambda f: os.path.basename(f))
    logging.info("found %d event files" % len(path_events))
    logging.info("paths to events files (sorted):\n%s" % pformat(path_events))
 
    # paths to all events files of the current subject:
    path_events_vfl = opj(path_behavior, subject, "*vfl*events.tsv")
    # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
    path_events_vfl = sorted(glob.glob(path_events_vfl), key=lambda f: os.path.basename(f))
    # import events file and save data in dataframe:
    df_events_vfl = [pd.read_csv(path_events_vfl[f], sep="\t") for f in range(len(path_events_vfl))]
    df_events_vfl = pd.concat([df_events_vfl[0],df_events_vfl[1],df_events_vfl[2],df_events_vfl[3]])
    df_events_vfl['stim_label'] = df_events_vfl.apply(lambda x: classlabel(x.stimMarker), axis = 1) 
    del df_events_vfl['Unnamed: 0']
    del df_events_vfl['start']
    # df_vfl_event[f]['fold'] = df_vfl_event[f].apply(lambda x: cvfold(x['trials'],x['run']),axis=1)

    # In[CREATE PATHS TO THE MRI DATA]:
    # define path to input directories:
    mask_name_list = ['mask_visual', 'mask_hippocampus', 'mask_entorhinal', 'mask_mtl', 'mask_vis_mtl', 'mask_temporal', 'mask_prefrontal']
    # load mask files:
    def load_mask_file(mask_name):
        path_mask_task = opj(path_masks, mask_name, subject, "*", "*task*.nii.gz")
        path_mask_task = sorted(glob.glob(path_mask_task), key=lambda f: os.path.basename(f))
        mask = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_task[-4:])]
        return mask
    
    mask_visual = load_mask_file(mask_name_list[0])
    mask_hippocampus = load_mask_file(mask_name_list[1])
    mask_entorhinal = load_mask_file(mask_name_list[2])
    mask_mtl = load_mask_file(mask_name_list[3])
    mask_vis_mtl = load_mask_file(mask_name_list[4])
    mask_temporal = load_mask_file(mask_name_list[5])
    mask_prefrontal =load_mask_file(mask_name_list[6])

    # load the visual functional localizer mri task files:
    path_func_task_vfl = opj(path_level1_vfl, "smooth", subject, "*", "*task*nii.gz")
    path_func_task_vfl = sorted(glob.glob(path_func_task_vfl), key=lambda f: os.path.basename(f))
    # load smoothed functional mri data for all eight task runs:
    data_task_vfl = [image.load_img(i) for i in path_func_task_vfl]

    # load the anatomical mri file:
    path_anat = opj(path_fmriprep, subject, "anat", "%s_desc-preproc_T1w.nii.gz" % subject)
    path_anat = sorted(glob.glob(path_anat), key=lambda f: os.path.basename(f))
    anat = image.load_img(path_anat[0])

    # load the t-maps of the first-level glm:
    path_tmap_vfl = opj(path_level1_vfl, "contrasts", subject, "*", "spmT*.nii")
    path_tmap_vfl = sorted(glob.glob(path_tmap_vfl), key=lambda f: os.path.dirname(f))
    tmaps_vfl = [image.load_img(i).get_data().astype(float) for i in copy.deepcopy(path_tmap_vfl)]
    
    mask_list = [mask_visual, mask_hippocampus, mask_entorhinal, mask_mtl, mask_vis_mtl, mask_temporal, mask_prefrontal]
    mask_name_list= ['mask_visual', 'mask_hippocampus', 'mask_entorhinal', 'mask_mtl', 'mask_vis_mtl', 'mask_temporal', 'mask_prefrontal']
    mask = mask_list[mask_index]
    mask_name = mask_name_list[mask_index]
    # mask = mask_list[6]
    # mask_name = mask_name_list[6]
    # In[FEATURE SELECTION FOR VISUAL FUNCTIONAL LOCALIZER TASK]:
    taskname = 'VFL'
    ### FEATURE SELECTION: MASKS THE T-MAPS WITH THE ANATOMICAL MASKS
    # get combination of anatomical mask and t-map
    tmaps_masked = [np.multiply(i, j) for (i,j) in zip(copy.deepcopy(mask),copy.deepcopy(tmaps_vfl))]
    # masked tmap into image like object:
    tmaps_masked_img = [image.new_img_like(i, j) for (i, j) in zip(path_tmap_vfl, copy.deepcopy(tmaps_masked))]
    for i, path in enumerate(tmaps_masked_img):
        path_save = opj(path_out_masks, "%s_%s_run-%02d_tmap_masked.nii.gz" % (subject, taskname, i + 1))
        path.to_filename(path_save)
    ### FEATURE SELECTION: THRESHOLD THE MASKED T-MAPS
    # set the threshold:
    threshold = 3
    # threshold the masked tmap image:
    tmaps_masked_thresh_img = [image.threshold_img(i, threshold) for i in copy.deepcopy(tmaps_masked_img)]
    # extract data from the thresholded images
    tmaps_masked_thresh = [image.load_img(i).get_data().astype(float) for i in tmaps_masked_thresh_img]
    # calculate the number of tmap voxels:
    num_tmap_voxel = [np.size(i) for i in copy.deepcopy(tmaps_masked_thresh)]
    num_above_voxel = [np.count_nonzero(i) for i in copy.deepcopy(tmaps_masked_thresh)]
    num_below_voxel = [np.count_nonzero(i == 0) for i in copy.deepcopy(tmaps_masked_thresh)]

    # create a dataframe with the number of voxels
    df_thresh = pd.DataFrame(
        {   "id": [subject] * n_run,
            "run": np.arange(1, n_run + 1),
            "n_total": num_tmap_voxel,
            "n_above": num_above_voxel,
            "n_below": num_below_voxel,
        }
    )
    file_name = opj(path_out_data, "%s_%s_%s_thresholding.csv" % (subject, taskname, mask_name))
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
    for i, path in enumerate(masks_final_vfl):
        filename = "%s_run-%02d_tmap_masked_thresh.nii.gz" % (subject, i + 1)
        path_save = opj(path_out_masks, filename)
        path.to_filename(path_save)

    # In[DEFINE THE CLASSIFIERS]:
    
    class_labels = ["girl", "scissors", "zebra", "banana"]
    # create a dictionary with all values as independent instances:
    # see here: https://bit.ly/2J1DvZm
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
    
    class_labels = ["girl", "scissors", "zebra", "banana"]
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

    train_VFL_peak = TaskData(
        events=df_events_vfl,
        task="VFL",
        # trial_type="stimulus",
        bold_delay=bold_delay_task,
        interval=1,
        name="train-VFL_peak",
        num_vol_run_1 = np.size(data_task_vfl[0]._dataobj,axis=3),
        num_vol_run_2 = np.size(data_task_vfl[1]._dataobj,axis=3),
        num_vol_run_3 = np.size(data_task_vfl[2]._dataobj,axis=3),
        num_vol_run_4 = np.size(data_task_vfl[3]._dataobj,axis=3),
    )
    
    test_VFL_peak = TaskData(
        events=df_events_vfl,
        task="VFL",
        # trial_type="stimulus",
        bold_delay=bold_delay_task,
        interval=1,
        name="test-VFL_peak",
        num_vol_run_1 = np.size(data_task_vfl[0]._dataobj,axis=3),
        num_vol_run_2 = np.size(data_task_vfl[1]._dataobj,axis=3),
        num_vol_run_3 = np.size(data_task_vfl[2]._dataobj,axis=3),
        num_vol_run_4 = np.size(data_task_vfl[3]._dataobj,axis=3),
    )
    
    test_VFL_long = TaskData(
        events=df_events_vfl,
        task="VFL",
        # trial_type="stimulus",
        bold_delay=0,
        interval=7,
        name="test-VFL_long",
        num_vol_run_1 = np.size(data_task_vfl[0]._dataobj,axis=3),
        num_vol_run_2 = np.size(data_task_vfl[1]._dataobj,axis=3),
        num_vol_run_3 = np.size(data_task_vfl[2]._dataobj,axis=3),
        num_vol_run_4 = np.size(data_task_vfl[3]._dataobj,axis=3),
    )

    test_sets = [
        test_VFL_peak,
        test_VFL_long
    ]
    
    # In[SEPARATE RUN-WISE STANDARDIZATION (Z-SCORING) OF VISUAL FUNCTIONAL LOCALIZER]:
    # set the parameters for training
    n_fold = 8
    data_list = []
    runs = list([1,1,2,2,3,3,4,4])
    folds = list(range(1, n_fold + 1))
    mask_label = mask_name
    # training and test
    for fold, run in zip(folds, runs):
        # define the run indices for the training and test set:
        train_folds = [x for x in folds if x != fold]
        test_folds = [x for x in folds if x == fold]
        # get the feature selection mask of the current run:
        mask_run = masks_final_vfl[run-1]
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
            train_VFL_peak.zscore(signal=data_detrend, fold_list=train_folds)
            # for half trials training:
            # train_stim = copy.deepcopy((train_VFL_peak.stim[train_VFL_peak.fold == train_folds[0]],
            #                               train_VFL_peak.stim[train_VFL_peak.fold == train_folds[1]],
            #                               train_VFL_peak.stim[train_VFL_peak.fold == train_folds[2]]))
            # train_stim = pd.concat((pd.Series(train_stim[0]),pd.Series(train_stim[1]),pd.Series(train_stim[2])))
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
            # test_set = test_VFL_long
                logging.info("testing on test set %s" % test_set.name)
                test_set.zscore(signal=data_detrend, fold_list=test_folds)
                if test_set.data_zscored.size < 0:
                    continue
                # create dataframe containing classifier predictions:
                df_pred = test_set.predict(clf=clf, fold_list=test_folds)
                # add the current classifier as a new column:
                df_pred["classifier"] = np.repeat(clf_name, len(df_pred))
                # add a label that indicates the mask / training regime:
                df_pred["mask"] = np.repeat(mask_label, len(df_pred))
                # add a label that indicates the delay_tr:
                df_pred['delay_tr'] = np.repeat(bold_delay_task,len(df_pred))
                # melt the data frame:
                df_pred_melt = melt_df(df=df_pred, melt_columns=train_stim)
                # append dataframe to list of dataframe results:
                data_list.append(df_pred_melt)
                
                
# In[parallel function]
    return data_list
# run the decoding parallelly
data_list_all = Parallel(n_jobs=64)(
                delayed(decoding)(subject) for subject in sub_list)

# In[save the decoding probability]:
# copy the decoding results for analysis
pre_list = copy.deepcopy(data_list_all)
# delete_list = [71,62,53,44,35,26,17,8]
# [pre_list.pop(i) for i in delete_list]
# combine the results to one dataframe
pre_list_append = []
for i in range(len(pre_list)):
    pre_list_append = pre_list_append + pre_list[i]
pre_list_append = pd.concat([pre_list_append[i] for i in range(len(pre_list_append))], axis=0)
# save the decoding results
pre_list_append.to_csv(opj(path_out_all,'VFL_decoding.csv'),index=False)

# In[CALCULATE ACCURACY FOR PEAK]:
class_labels = ["girl", "scissors", "zebra", "banana"]
pred_class_mean = []
pre_list=copy.deepcopy(data_list_all)
# delete_list = [71,62,53,44,35,26,17,8]
# [pre_list.pop(i) for i in delete_list]
# combine the results to one dataframe
pre_list_append = []
for i in range(len(pre_list)):
    pre_list_append = pre_list_append + pre_list[i]
pre_list_append = pd.concat([pre_list_append[i] for i in range(len(pre_list_append))], axis=0)
# get the decoding results: specific variable in test-VFL_peak in one-vs-rest decoding 
pre_list_filter = pre_list_append[(pre_list_append["class"] != 'other' )
                                  & (pre_list_append["classifier"] != 'log_reg')
                                  & (pre_list_append["test_set"] == 'test-VFL_peak')
                                  # & (pre_list_append["stim"] == pre_list_append["class"])
                                    ]
# preset the variable
pre_list_filter['pred_acc'] = 0
# calculate the accuacy for each subject:
# use method 2 to calculate the accuracy: 
# The classifier with the largest probability among the four classifiers in a trial
for subnum in subnum_list:
    pre_list_filter_sub = pre_list_filter[pre_list_filter['participant'] == subnum]
    for i in np.unique(pre_list_filter_sub['tr']):
        pre_list_filter_sub_tr = pre_list_filter_sub[pre_list_filter_sub['tr'] == i]
        x = pre_list_filter_sub_tr[pre_list_filter_sub_tr['probability'] == pre_list_filter_sub_tr['probability'].max()]
        if (x['classifier'] == x['stim']).iloc[0]:
            pre_list_filter.loc[(pre_list_filter['participant'] == subnum) & (pre_list_filter['tr'] == i),'pred_acc'] = 1
        else:
            pre_list_filter.loc[(pre_list_filter['participant'] == subnum) & (pre_list_filter['tr'] == i),'pred_acc'] = 0
# accuracy for each partcicpant
pre_list_mean = pre_list_filter.groupby('participant')['pred_acc'].mean()
# deatach the groupby
pre_list_mean = pd.DataFrame(pre_list_mean)
# rename the variable
pre_list_mean = pre_list_mean.rename(columns={'pred_acc':'Accuracy'})
# plot the overall accuracy with mean and SE
sns.set_theme(style="whitegrid")
plt.figure(dpi=400,figsize=(3,5))
plt.xticks(fontsize=10)
sns.barplot(y="Accuracy", data=pre_list_mean)
plt.suptitle('All classifiers decoding accuracy of VFL-VFL (%1.0fth TR peak)' % bold_delay_task) 
plt.savefig(opj(path_out_all,'All classifiers decoding accuracy of VFL-VFL in %1.0fth TR peak.png' % bold_delay_task),
            dpi = 400,bbox_inches = 'tight')
plt.show()

# In[CALCULATE PREDICTION PROBABILITY FOR PEAK]:
class_labels = ["girl", "scissors", "zebra", "banana"]
pred_class_mean = []
pre_list=copy.deepcopy(data_list_all)
# delete_list = [71,62,53,44,35,26,17,8]
# [pre_list.pop(i) for i in delete_list]
# combine the results to one dataframe
pre_list_append = []
for i in range(len(pre_list)):
    pre_list_append = pre_list_append + pre_list[i]
pre_list_append = pd.concat([pre_list_append[i] for i in range(len(pre_list_append))], axis=0)
# get the decoding results: specific variable in test-VFL_peak in one-vs-rest decoding
pre_list_filter = pre_list_append[(pre_list_append["class"] != 'other') &
                                  (pre_list_append["classifier"] != 'log_reg') &
                                  (pre_list_append["test_set"] =='test-VFL_peak')]
# average the decoding probability for each classifier and each subject
pred_class_mean = [[np.mean(pre_list_filter[(pre_list_filter['stim'] == class_label)
                                  & (pre_list_filter['class'] == class_label)
                                  & (pre_list_filter['participant'] == participant)]['probability'])
                  for class_label in class_labels] for participant in subnum_list]
# transform to the matrix
pred_class_mean = pd.DataFrame(np.array(pred_class_mean),columns=[class_labels])
pred_class_mean_plot = pd.melt(pred_class_mean,value_vars=[class_labels],var_name=['Classifier'],value_name='Probability')
# plot the decoding probability for each classifier
sns.set_theme(style="whitegrid")
plt.figure(dpi=400,figsize=(10,5))
plt.xticks(fontsize=10)
sns.barplot(x="Classifier", y="Probability", data=pred_class_mean_plot)
plt.suptitle('decoding probability of VFL-VFL (%1.0fth TR peak)' % bold_delay_task) 
plt.savefig(opj(path_out_all,'decoding probability of VFL-VFL in %1.0fth TR peak.png' % bold_delay_task),
            dpi = 400,bbox_inches = 'tight')
plt.show()

# In[PREDICTION PROBABILITY AND ACCURACY FOR LONG TERM]:
pred_class_mean_ml = []
pre_list=copy.deepcopy(data_list_all)
# delete_list = [71,62,53,44,35,26,17,8]
# [pre_list.pop(i) for i in delete_list]
# combine the results to one dataframe
pre_list_append = []
for i in range(len(pre_list)):
    pre_list_append = pre_list_append + pre_list[i]
pre_list_append = pd.concat([pre_list_append[i] for i in range(len(pre_list_append))], axis=0)
# get the decoding results: specific variable in test-VFL_long in one-vs-rest decoding
pre_list_filter = pre_list_append[(pre_list_append["class"] != 'other') &
                                  (pre_list_append["classifier"] != 'log_reg') &
                                  (pre_list_append["test_set"] == 'test-VFL_long')]
# average the decoding probability for each classifier and each subject
pred_class_mean_ml = [[[np.mean(pre_list_filter[(pre_list_filter['stim'] == class_label) 
                                            & (pre_list_filter['class'] == class_label)
                                            & (pre_list_filter['seq_tr'] ==  seq_tr)
                                            & (pre_list_filter['participant'] == participant)]['probability']) 
                    for class_label in class_labels] for seq_tr in np.arange(1,8)] for participant in subnum_list]
pred_class_mean_ml = np.array(pred_class_mean_ml)
# plot the time courses of probabilistic classification evidence for all four stimulus classes
whole_TR = np.arange(1,8)
fig, ax = plt.subplots(nrows = 1, ncols = 4, sharey=True,figsize=(12,4),dpi=400)
for j, class_label in enumerate(class_labels):
    print(j)
    x_data = whole_TR
    y_data = pred_class_mean_ml.mean(axis=0)

    ax[j].set_ylim(ymax=1)
    ax[j].set_xticks(range(1,len(whole_TR)+1),labels=whole_TR)
    ax[j].set_yticks([0,0.2,0.4,0.6,0.8,1],labels=[0,20,40,60,80,100])
    for i in range(len(sub_list)):
        ax[j].plot(x_data,pred_class_mean_ml[i,:,j],color='darkgray',linestyle='-',linewidth=0.8)
    ax[j].plot(x_data,y_data[:,j],color='black',linestyle='-',linewidth=1.5)
    ax[j].axhline(y=0.25,color='silver',linestyle='--')
    ax[j].set_title('%s' % class_label) 
fig.text(0.5, 0, 'Time from stimulus onset (TRs)', ha='center')
fig.text(0, 0.5, 'Probability(%)', va='center', rotation='vertical')
plt.tight_layout(pad=2.5, w_pad=0.5, h_pad=2)
plt.suptitle('decoding probability of VFL-VFL (%1.0fth TR peak)' % bold_delay_task) 
plt.savefig(opj(path_out_all,'decoding probability of VFL-VFL (%1.0fth TR peak).png' % bold_delay_task),
            dpi = 400, bbox_inches = 'tight')
plt.show()

# In[]
#     return pred_acc_mean_sub
# pred_acc_mean_sub_all = Parallel(n_jobs=60)(
#                     delayed(decoding)(subject,mask_index,bold_delay_task)
#                     for bold_delay_task in bold_delay_task_list for mask_index in mask_index_list for subject in sub_list)
# In[analysis]
