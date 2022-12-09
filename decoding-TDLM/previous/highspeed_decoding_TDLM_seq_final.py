#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# SCRIPT INFORMATION:
# ======================================================================
# SCRIPT: DECODING ANALYSIS
# PROJECT: HIGHSPEED
# WRITTEN BY LENNART WITTKUHN, 2018 - 2019
# CONTACT: WITTKUHN AT MPIB HYPHEN BERLIN DOT MPG DOT DE
# MAX PLANCK RESEARCH GROUP NEUROCODE
# MAX PLANCK INSTITUTE FOR HUMAN DEVELOPMENT
# MAX PLANCK UCL CENTRE FOR COMPUTATIONAL PSYCHIATRY AND AGEING RESEARCH
# LENTZEALLEE 94, 14195 BERLIN, GERMANY
'''
========================================================================
IMPORT RELEVANT PACKAGES:
========================================================================
'''
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
from numpy.matlib import repmat
from matplotlib import pyplot as plt
import itertools
import warnings
from nilearn.signal import clean
import scipy
import scipy.io as scio
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from collections import Counter
import datalad.api as dl   # can not run in the code.
import nibabel as nib
import seaborn as sns
'''
========================================================================
DEAL WITH ERRORS, WARNING ETC.
========================================================================
'''
warnings.filterwarnings('ignore', message='numpy.dtype size changed*')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed*')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
'''
========================================================================
SETUP NECESSARY PATHS ETC:
========================================================================
'''
# get start time of the script:
start = time.time()
# name of the current project:
project = 'highspeed'
# initialize empty variables:
sub = None
path_tardis = None
path_server = None
path_local = None

# define paths without depending on the operating system (OS) platform:
# for Windows
#project name     
project_name = 'highspeed-decoding'
#current working directory in Nicodata
cwd1 = os.getcwd() 
#path_root
path_root = opj(cwd1, project_name)
# define the path to the cluster:
path_tardis = path_root   
# define the path to the server:
path_server = path_tardis
#define the path to the local computer:
path_local = opj(path_tardis, 'code', 'decoding')
#define the subject id:
sub = 'sub-14'
suball = ['sub-%s' %str(i).zfill(2) for i in range(1,41)]  #40 subjects

delete_sub = [39,36,30,23]
[suball.pop(i) for i in delete_sub]
'''
========================================================================
LOAD PROJECT PARAMETERS:
========================================================================
'''
path_params = glob.glob(opj(path_local, '*parameters.yaml'))[0]
with open(path_params, 'rb') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
f.close()

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
            self.itises = self.itis.reshape(self.real_trial, 13)

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


def matlab_percentile(x, p):
    p = np.asarray(p, dtype=float)
    n = len(x)
    p = (p-50)*n/(n-1) + 50
    p = np.clip(p, 0, 100)
    return np.percentile(x, p)

'''
========================================================================
DEFINE THE CLASSIFIERS
========================================================================
'''
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

# parameters for TDLM
# the list of all the sequences
uniquePerms = list(itertools.permutations([1,2,3,4,5],5))
# the number of sequences
nShuf = len(uniquePerms)
# all the sequences for the specific subject
real_sequence = [(1,2,3,4,5)]
# all possible sequences
all_sequence = np.array(uniquePerms)
# rest of sequences for the specific subject
set_y = set(map(tuple, real_sequence))
idx = [tuple(point) not in set_y for point in all_sequence]
rest_sequence = all_sequence[idx]
# recombine the real sequence and rest of sequences
new_all_sequence = np.vstack((real_sequence,rest_sequence))
# the number of timelags, including 32, 64, 128, 512and 2048ms
maxLag= 20
# the number of states (decoding models)
nstates = 5
# the number of subjects
# nSubj = len(suball)
nSubj = 36
nipi = 2
# the nan matrix for forward design matrix (36 subjects * 120 shuffles * 11 timelags)
sf = np.full([nSubj,len(new_all_sequence),maxLag+1],np.nan)
sb = np.full([nSubj,len(new_all_sequence),maxLag+1],np.nan)
# predefine GLM data frame 
betas_1GLM = None
betas_2GLM = None


'''
========================================================================
CREATE PATHS TO OUTPUT DIRECTORIES:
========================================================================
'''
# sub=suball[2]
pred_matrix_2048_list = []
pred_matrix_0512_list = []
seq_pred_matrix_2048_list = []
seq_pred_matrix_0512_list = []
serial_2048_list = []
serial_0512_list = []

for (subi,sub) in zip(range(len(suball[0:nSubj])),suball[0:nSubj]):
    print(subi,sub)
    subi=0
    sub='sub-01'
    path_decoding = opj(path_tardis, 'TDLM')
    path_out = opj(path_decoding, sub)
    path_out_figs = opj(path_out, 'plots')
    path_out_data = opj(path_out, 'data')
    path_out_logs = opj(path_out, 'logs')
    path_out_masks = opj(path_out, 'masks')
    ### CREATE OUTPUT DIRECTORIES IF THE DO NOT EXIST YET:
    for path in [path_out_figs, path_out_data, path_out_logs, path_out_masks]:
        if not os.path.exists(path):
            os.makedirs(path)
    ### LOAD BEHAVIORAL DATA (THE EVENTS.TSV FILES OF THE SUBJECT):
    # paths to all events files of the current subject:
    path_events = opj(cwd1,'highspeed-bids','forBIDS', sub, 'ses-*', 'func', '*tsv')
    #dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
    path_events = sorted(glob.glob(path_events), key=lambda f: os.path.basename(f))
    # import events file and save data in dataframe:
    df_events = pd.concat((pd.read_csv(f, sep='\t') for f in path_events),
                          ignore_index=True)
    
    ### CREATE PATHS TO THE MRI DATA:
    
    # define path to input directories:
    path_fmriprep = opj(cwd1, 'highspeed-fmriprep', 'fmriprep', sub)
    path_level1 = opj(cwd1, 'highspeed-glm', 'l1pipeline')
    path_masks = opj(cwd1, 'highspeed-masks', 'masks')
    
    paths = {
        'fmriprep': opj(cwd1, 'highspeed-fmriprep', 'fmriprep', sub),
        'level1': opj(cwd1, 'highspeed-glm', 'l1pipeline'),
        'masks': opj(cwd1, 'highspeed-masks', 'masks')
    }
    
    # load the visual mask task files:
    path_mask_vis_task = opj(path_masks, 'mask_visual', sub, '*', '*task-highspeed*.nii.gz')
    path_mask_vis_task = sorted(glob.glob(path_mask_vis_task), key=lambda f: os.path.basename(f))
    
    # load the hippocampus mask task files:
    path_mask_hpc_task = opj(path_masks, 'mask_hippocampus', sub, '*', '*task-highspeed*.nii.gz')
    path_mask_hpc_task = sorted(glob.glob(path_mask_hpc_task), key=lambda f: os.path.basename(f))

    # load the whole brain mask files:
    path_mask_whole_task = opj(path_fmriprep, '*', 'func', '*task-highspeed*T1w*brain_mask.nii.gz')
    path_mask_whole_task = sorted(glob.glob(path_mask_whole_task), key=lambda f: os.path.basename(f))

    # load the functional mri task files:
    path_func_task = opj(path_level1, 'smooth', sub, '*', '*task-highspeed*nii.gz')
    path_func_task = sorted(glob.glob(path_func_task), key=lambda f: os.path.basename(f))
    
    # define path to the functional resting state runs:
    path_rest = opj(path_masks, 'smooth', sub, '*', '*task-rest*nii.gz')
    path_rest = sorted(glob.glob(path_rest), key=lambda f: os.path.basename(f))

    # load the anatomical mri file:
    path_anat = opj(path_fmriprep, 'anat', '%s_desc-preproc_T1w.nii.gz' % sub)
    path_anat = sorted(glob.glob(path_anat), key=lambda f: os.path.basename(f))
 
    # load the confounds files:
    path_confs_task = opj(path_fmriprep, '*', 'func', '*task-highspeed*confounds_regressors.tsv')
    path_confs_task = sorted(glob.glob(path_confs_task), key=lambda f: os.path.basename(f))
 
    # load the spm.mat files:
    path_spm_mat = opj(path_level1, 'contrasts', sub, '*', 'SPM.mat')
    path_spm_mat = sorted(glob.glob(path_spm_mat), key=lambda f: os.path.dirname(f))

    # load the t-maps of the first-level glm:
    path_tmap = opj(path_level1, 'contrasts', sub, '*', 'spmT*.nii')
    path_tmap = sorted(glob.glob(path_tmap), key=lambda f: os.path.dirname(f))

    ### LOAD THE MRI DATA:
        
    anat = image.load_img(path_anat[0])
    # load visual mask:
    mask_vis = image.load_img(path_mask_vis_task[0]).get_data().astype(int)
    # load tmap data:
    tmaps = [image.load_img(i).get_data().astype(float) for i in copy.deepcopy(path_tmap)]
    # load hippocampus mask:
    mask_hpc = [image.load_img(i).get_data().astype(int) for i in copy.deepcopy(path_mask_hpc_task)]
    
    ### FEATURE SELECTION 
    # check if any value in the supposedly binary mask is bigger than 1:
    if np.any(mask_vis > 1):
        sys.exit("Values > 1 in the anatomical ROI!")
    # get combination of anatomical mask and t-map
    tmaps_masked = [np.multiply(mask_vis, i) for i in copy.deepcopy(tmaps)]
    # masked tmap into image like object:
    tmaps_masked_img = [image.new_img_like(i, j) for (i, j) in zip(path_tmap, copy.deepcopy(tmaps_masked))]
    
    for i, path in enumerate(tmaps_masked_img):
        path_save = opj(path_out_masks, '%s_run-%02d_tmap_masked.nii.gz' % (sub, i + 1))
        path.to_filename(path_save)
    
    ### FEATURE SELECTION: THRESHOLD THE MASKED T-MAPS
    
    # set the threshold:
    threshold = params['mri']['thresh']
    # threshold the masked tmap image:
    tmaps_masked_thresh_img = [image.threshold_img(i, threshold) for i in copy.deepcopy(tmaps_masked_img)]
    # extract data from the thresholded images
    tmaps_masked_thresh = [image.load_img(i).get_data().astype(float) for i in tmaps_masked_thresh_img]
    # calculate the number of tmap voxels:
    # all the voxels in the t-maps of brain
    num_tmap_voxel = [np.size(i) for i in copy.deepcopy(tmaps_masked_thresh)]
    # selected voxels in the t-maps
    num_above_voxel = [np.count_nonzero(i) for i in copy.deepcopy(tmaps_masked_thresh)]
    # rest voxels
    num_below_voxel = [np.count_nonzero(i == 0) for i in copy.deepcopy(tmaps_masked_thresh)]
    
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
    
    ### FEATURE SELECTION: BINARIZE THE THRESHOLDED MASKED T-MAPS
    # replace all NaNs with 0:
    tmaps_masked_thresh_bin = [np.where(np.isnan(i), 0, i) for i in copy.deepcopy(tmaps_masked_thresh)]
    # replace all other values with 1:
    tmaps_masked_thresh_bin = [np.where(i > 0, 1, i) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
    # turn the 3D-array into booleans:
    tmaps_masked_thresh_bin = [i.astype(bool) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
    # create image like object:
    masks_final = [image.new_img_like(path_func_task[0], i.astype(np.int)) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
    
    
    ### LOAD SMOOTHED FMRI DATA FOR ALL FUNCTIONAL TASK RUNS:
    # load smoothed functional mri data for all eight task runs:
    data_task = [image.load_img(i) for i in path_func_task]
    
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
    test_rep_cue = TaskData(
            events=df_events, condition='repetition', trial_type='cue',
            bold_delay=0, interval=5, name='test-rep_cue')
    test_sets = [
            test_odd_peak, test_odd_long, test_seq_long, test_rep_long
            # ,test_seq_cue, test_rep_cue
            ]
    # SEPARATE RUN-WISE STANDARDIZATION (Z-SCORING) OF ALL TASK CONDITIONS
      
    data_list = []
    runs = list(range(1, n_run+1))
    mask_label = 'cv'
    #for mask_label in ['cv', 'cv_hpc']:
    for run in runs:
        # define the run indices for the training and test set:
        train_runs = [x for x in runs if x != run]
        test_runs = [x for x in runs if x == run]
        # get the feature selection mask of the current run:
        if mask_label == 'cv':
            mask_run = masks_final[runs.index(run)]
        # elif mask_label == 'cv_hpc':
        #     mask_run = path_mask_hpc_task[runs.index(run)]
        # extract smoothed fMRI data from the mask for the cross-validation fold:
        masked_data = [masking.apply_mask(i, mask_run) for i in data_task]
        # detrend the masked fMRI data separately for each run:
        data_detrend = [detrend(i) for i in masked_data]
        # combine the detrended data of all runs:
        data_detrend = np.vstack(data_detrend)
        # loop through all classifiers in the classifier set:
        for clf_name, clf in clf_set.items():
            # print classifier:
            # fit the classifier to the training data:
            train_odd_peak.zscore(signals=data_detrend, run_list=train_runs)
            # get the example labels:
            train_stim = copy.deepcopy(train_odd_peak.stim[train_odd_peak.runs != run])
            # replace labels for single-label classifiers:
            if clf_name in class_labels:
                # replace all other labels with other
                train_stim = ['other' if x != clf_name else x for x in train_stim]
                # turn into a numpy array
                train_stim = np.array(train_stim, dtype=object)
            # check weights:
            #show_weights(array=train_stim)
            # train the classifier
            clf.fit(train_odd_peak.data_zscored, train_stim)
    
            # for test_set in test_sets:
            test_set = test_seq_long
            test_set.zscore(signals=data_detrend, run_list=test_runs)
            if test_set.data_zscored.size < 0:
                continue
            # create dataframe containing classifier predictions:
            df_pred = test_set.predict(clf=clf, run_list=test_runs)
            # add the current classifier as a new column:
            df_pred['classifier'] = np.repeat(clf_name, len(df_pred))
            # add a label that indicates the mask / training regime:
            df_pred['mask'] = np.repeat(mask_label, len(df_pred))
            # melt the data frame:
            df_pred_melt = melt_df(df=df_pred, melt_columns=train_stim)
            # append dataframe to list of dataframe results:
            data_list.append(df_pred_melt)
    
    # copy the data list 
    pre_list=copy.deepcopy(data_list)
    delete_list = [47,41,35,29,23,17,11,5]
    [pre_list.pop(i) for i in delete_list]
    pre_list_append_ovr = []
    pre_list_append_ovr = pd.concat([pre_list[i] for i in range(len(pre_list))], axis=0)
    prediction_method = 'ovr'
    
    # 2048ms prediction matrix
    pre_list_filter_2048 = pre_list_append_ovr[(pre_list_append_ovr["tITI"] == 2.048)&(pre_list_append_ovr["class"] != 'other')]
    pre_list_filter_2048['class_index'] = pre_list_filter_2048['class'].map(lambda a : stim_index(a))
    pre_list_filter_2048_sort = pre_list_filter_2048.sort_values(by=['class_index','trial'])
    pred_array_2048 = np.array(pre_list_filter_2048_sort.loc[:,'probability'])
    pred_matrix_2048 = np.reshape(pred_array_2048,(195,5),order='F') 
    pred_matrix_2048_list.append(pred_matrix_2048)
    test_seq_long.seq_2048 = []
    test_seq_long.trial_2048 = np.unique((np.argwhere(test_seq_long.itises == 2.048)[:,0]))
    test_seq_long.seq_2048 = test_seq_long.sequences[(test_seq_long.trial_2048),]
    
    #repeat the results of Nicolas's paper figure 3A
    serial_2048 = [pd.concat([pd.DataFrame(pred_matrix_2048[13*i:13*(i+1),int(test_seq_long.seq_2048[i,j])-1]) 
                         for i in range(len(test_seq_long.trial_2048))],axis=1) 
              for j in range(nstates)]
    serial_2048_list.append(serial_2048)
    seq_pred_matrix_2048 = np.transpose(np.vstack((np.reshape(serial_2048[0].values,195,order = "F"),
                                              np.reshape(serial_2048[1].values,195,order = "F"),
                                              np.reshape(serial_2048[2].values,195,order = "F"),
                                              np.reshape(serial_2048[3].values,195,order = "F"),
                                              np.reshape(serial_2048[4].values,195,order = "F"))))
    scipy.io.savemat('seq_pred_matrix_2048_%s.mat' % sub,{'seq_pred_matrix_2048':seq_pred_matrix_2048}) 
    seq_pred_matrix_2048_list.append(seq_pred_matrix_2048)
    
    
    # 512ms prediction matrix
    pre_list_filter_0512 = pre_list_append_ovr[(pre_list_append_ovr["tITI"] == 0.512)&(pre_list_append_ovr["class"] != 'other')]
    pre_list_filter_0512['class_index'] = pre_list_filter_0512['class'].map(lambda a : stim_index(a))
    pre_list_filter_0512_sort = pre_list_filter_0512.sort_values(by=['class_index','trial'])
    pred_array_0512 = np.array(pre_list_filter_0512_sort.loc[:,'probability'])
    pred_matrix_0512 = np.reshape(pred_array_0512,(195,5),order='F') 
    pred_matrix_0512_list.append(pred_matrix_0512)
    test_seq_long.seq_0512 = []
    test_seq_long.trial_0512 = np.unique((np.argwhere(test_seq_long.itises == 0.512)[:,0]))
    test_seq_long.seq_0512 = test_seq_long.sequences[(test_seq_long.trial_0512),]

    #repeat the results of Nicolas's paper figure 3A
    serial_0512 = [pd.concat([pd.DataFrame(pred_matrix_0512[13*i:13*(i+1),int(test_seq_long.seq_0512[i,j])-1]) 
                         for i in range(len(test_seq_long.trial_0512))],axis=1) 
              for j in range(nstates)]
    serial_0512_list.append(serial_0512)
    seq_pred_matrix_0512 = np.transpose(np.vstack((np.reshape(serial_0512[0].values,195,order = "F"),
                                              np.reshape(serial_0512[1].values,195,order = "F"),
                                              np.reshape(serial_0512[2].values,195,order = "F"),
                                              np.reshape(serial_0512[3].values,195,order = "F"),
                                              np.reshape(serial_0512[4].values,195,order = "F"))))      
    scipy.io.savemat('seq_pred_matrix_0512_%s.mat' % sub,{'seq_pred_matrix_0512':seq_pred_matrix_0512}) 
    seq_pred_matrix_0512_list.append(seq_pred_matrix_0512)

scipy.io.savemat('seq_pred_matrix_2048_list.mat',{'seq_pred_matrix_2048_list':seq_pred_matrix_2048_list}) 
scipy.io.savemat('seq_pred_matrix_0512_list.mat',{'seq_pred_matrix_0512_list':seq_pred_matrix_0512_list}) 


'''
### TDLM FOR SEQUENCE CONDITION TRIALS ###
'''
# load all subjects' sequence prediction data
seq_pred_matrix_0512_list1 = scio.loadmat('F://NicoData//0512prediction//seq_pred_matrix_0512_list.mat')
seq_pred_matrix_2048_list1 = scio.loadmat('F://NicoData//2048prediction//seq_pred_matrix_2048_list.mat')

seq_pred_matrix_0512_list2 = seq_pred_matrix_0512_list1['seq_pred_matrix_0512_list']
seq_pred_matrix_2048_list2 = seq_pred_matrix_2048_list1['seq_pred_matrix_2048_list']



pre_matrix_0512_mean = seq_pred_matrix_0512_list2.mean(axis=0)
## plot transition matrix space
plt.figure(dpi=400,figsize=(5,25))
x=np.arange(0,len(pre_matrix_0512_mean[:,1]),5)
plt.yticks(x,x,fontsize=10)
plt.imshow(pre_matrix_0512_mean, aspect='auto', interpolation='none',
            origin='lower',cmap='jet')
plt.title('Transition Matrix Space')
plt.xlabel('Time (TR)')
plt.ylabel('States')
plt.xticks(range(5),('Serial 1', 'Serial 2', 'Serial 3', 'Serial 4', 'Serial 5'))
plt.colorbar()
plt.show()   




# classifiers prediction probability 
serial_0512 = np.reshape(seq_pred_matrix_0512_list2,(36,13,15,5),order = 'F')
serial_2048 = np.reshape(seq_pred_matrix_2048_list2,(36,13,15,5),order = 'F')

scipy.io.savemat('serial_0512.mat',{'serial_0512':serial_0512}) 
scipy.io.savemat('serial_2048.mat',{'serial_2048':serial_2048}) 


# plot 2048ms probability
s1mean =  serial_2048.mean(axis=2).mean(axis=0)[:,0]
s2mean =  serial_2048.mean(axis=2).mean(axis=0)[:,1]
s3mean =  serial_2048.mean(axis=2).mean(axis=0)[:,2]
s4mean =  serial_2048.mean(axis=2).mean(axis=0)[:,3]
s5mean =  serial_2048.mean(axis=2).mean(axis=0)[:,4]
s1sem = serial_2048.mean(axis=2).std(axis=0)[:,0]/np.square(nSubj)
s2sem = (serial_2048.mean(axis=2).std(axis=0)[:,1])/np.square(nSubj)
s3sem = (serial_2048.mean(axis=2).std(axis=0)[:,2])/np.square(nSubj)
s4sem = (serial_2048.mean(axis=2).std(axis=0)[:,3])/np.square(nSubj)
s5sem = (serial_2048.mean(axis=2).std(axis=0)[:,4])/np.square(nSubj)
x=range(1,14)
plt.figure(dpi=400)
plt.xticks(fontsize=10)
plt.plot(x, s1mean, color='darkred', linestyle='-', linewidth=1, label='Serial event 1')
plt.fill_between(x,s1mean-s1sem,s1mean+s1sem,color='lightcoral',alpha=0.6, linewidth=5)
plt.plot(x, s2mean, color='orangered', linestyle='-', linewidth=1, label='Serial event 2')
plt.fill_between(x,s2mean-s2sem,s2mean+s2sem,color='lightsalmon',alpha=0.6, linewidth=5)
plt.plot(x, s3mean, color='gold', linestyle='-', linewidth=1, label='Serial event 3')
plt.fill_between(x,(s3mean-s3sem),(s3mean+s3sem),color='khaki',alpha=0.6, linewidth=5)
plt.plot(x, s4mean, color='darkgreen', linestyle='-', linewidth=1, label='Serial event 4')
plt.fill_between(x,(s4mean-s4sem),(s4mean+s4sem),color='palegreen',alpha=0.6, linewidth=5)
plt.plot(x, s5mean, color='dodgerblue', linestyle='-', linewidth=1, label='Serial event 5')
plt.fill_between(x,(s5mean-s5sem),(s5mean+s5sem),color='skyblue',alpha=0.6, linewidth=5)
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0) 
plt.xlabel('time(TR)') 
plt.ylabel('probablity')
plt.ylim(0,0.4)
# plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\Classifier Probability-%s-%s-%6.3f.png' % (sub, prediction_method, decoding_ipi)),
#         dpi = 400,bbox_inches = 'tight')
plt.show()

# plot 512ms probability
s1mean =  serial_0512.mean(axis=2).mean(axis=0)[:,0]
s2mean =  serial_0512.mean(axis=2).mean(axis=0)[:,1]
s3mean =  serial_0512.mean(axis=2).mean(axis=0)[:,2]
s4mean =  serial_0512.mean(axis=2).mean(axis=0)[:,3]
s5mean =  serial_0512.mean(axis=2).mean(axis=0)[:,4]
s1sem = serial_0512.mean(axis=2).std(axis=0)[:,0]/np.square(nSubj)
s2sem = (serial_0512.mean(axis=2).std(axis=0)[:,1])/np.square(nSubj)
s3sem = (serial_0512.mean(axis=2).std(axis=0)[:,2])/np.square(nSubj)
s4sem = (serial_0512.mean(axis=2).std(axis=0)[:,3])/np.square(nSubj)
s5sem = (serial_0512.mean(axis=2).std(axis=0)[:,4])/np.square(nSubj)
x=range(1,14)
plt.figure(dpi=400)
plt.xticks(fontsize=10)
plt.plot(x, s1mean, color='darkred', linestyle='-', linewidth=1, label='Serial event 1')
plt.fill_between(x,s1mean-s1sem,s1mean+s1sem,color='lightcoral',alpha=0.6, linewidth=5)
plt.plot(x, s2mean, color='orangered', linestyle='-', linewidth=1, label='Serial event 2')
plt.fill_between(x,s2mean-s2sem,s2mean+s2sem,color='lightsalmon',alpha=0.6, linewidth=5)
plt.plot(x, s3mean, color='gold', linestyle='-', linewidth=1, label='Serial event 3')
plt.fill_between(x,(s3mean-s3sem),(s3mean+s3sem),color='khaki',alpha=0.6, linewidth=5)
plt.plot(x, s4mean, color='darkgreen', linestyle='-', linewidth=1, label='Serial event 4')
plt.fill_between(x,(s4mean-s4sem),(s4mean+s4sem),color='palegreen',alpha=0.6, linewidth=5)
plt.plot(x, s5mean, color='dodgerblue', linestyle='-', linewidth=1, label='Serial event 5')
plt.fill_between(x,(s5mean-s5sem),(s5mean+s5sem),color='skyblue',alpha=0.6, linewidth=5)
plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0) 
plt.xlabel('time(TR)') 
plt.ylabel('probablity')
plt.ylim(0,0.4)
# plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\Classifier Probability-%s-%s-%6.3f.png' % (sub, prediction_method, decoding_ipi)),
#         dpi = 400,bbox_inches = 'tight')
plt.show()


# transition_matrix = np.full([nSubj,nstates,nstates],np.nan)
# TDLM for 512ms
# set the empty rp
rp = []
for (subi,sub) in zip(range(len(suball[0:nSubj])),suball[0:nSubj]):
    print(subi,sub)    
    #detect for each 2048 ms trial
    # loops for real sequence and permutation trials
    for i in range(len(new_all_sequence)):
        # real sequence
        rp = new_all_sequence[i,]
        # get the transition matrix
        T1 = TransM(rp)
        T2 = np.transpose(T1)
        nbins=maxLag+1
        # X1 = seq_pred_matrix[13*i:13*(i+1),:] #####
        X1 =seq_pred_matrix_0512_list2[subi,:,:]
        # timelag matrix 
        dm = scipy.linalg.toeplitz(X1[:,0],[np.zeros((nbins,1))]) #####
        dm = dm[:,1:]
        # 4 loops for another 4 states
        for k in range(1,5) : 
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
        design_mat_2 = np.transpose(np.vstack((np.reshape(T1,25,order = "F"), 
                                               np.reshape(T2,25,order = "F"), 
                                               np.reshape(np.eye(nstates),25,order = "F"),
                                               np.reshape(np.ones([nstates,nstates]),25,order = "F"))))
        betas_2GLM = np.dot(np.linalg.pinv(design_mat_2) , np.transpose(betasnbins64))
        # four different design matrix for regressor multiple to the temporal data
        # linear regression for the backward and forward replay 
        sf[subi,i,1:] = betas_2GLM[0,:]
        sb[subi,i,1:] = betas_2GLM[1,:]
    
  
   






### PLOT the FIGURE 

pre_matrix_0512_mean = seq_pred_matrix_0512_list2.mean(axis=0)
decoding_ipi = 0.512
prediction_method = 'ovr'


# plot the decoded state space
plt.figure(dpi=400,figsize=(5,25))
plt.xticks(fontsize=10)
plt.imshow(pre_matrix_0512_mean, aspect='auto', interpolation='none',
            origin='lower',cmap='hot')
plt.title('Decoded State Space')
plt.xlabel('States')
plt.ylabel('time points (TR)')
plt.xticks(range(5),('face', 'house', 'cat', 'shoe', 'chair'))
plt.colorbar()
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\decoded state space-%s-%6.3f.png' % (prediction_method, decoding_ipi)),
            dpi = 400,bbox_inches = 'tight')
plt.show() 




# mean the real sequence and permutation sequences respectively
sf_sequence = sf[:,0,:].mean(axis=0)
temp_per = sf[:,1:,1:].mean(axis=0)
temp_per = abs(temp_per)
temp_per = np.amax(temp_per,axis=1)
sf_permutation = matlab_percentile(temp_per,95)
# sf_sequence = pd.DataFrame({
#     'For Sequenceness': sf_sequence,
#     'Timelag(TR)': np.arange(0,nbins)})

sb_sequence = sb[:,0,:].mean(axis=0)
temp_per = sb[:,1:,1:].mean(axis=0)
temp_per = abs(temp_per)
temp_per = np.amax(temp_per,axis=1)
sb_permutation = matlab_percentile(temp_per,95)


dif = sf_sequence - sb_sequence
temp_per = sf[:,1:,1:]-sb[:,1:,1:]
temp_per = temp_per.mean(axis=0)
temp_per = abs(temp_per)
temp_per = np.amax(temp_per,axis=1)
dif_permutation = matlab_percentile(temp_per,95)


# sns.relplot(x='Timelag(TR)', y='For Sequenceness', kind="line", data=sf_sequence);
# plot the results
plt.figure(dpi=400,figsize=(10,5))
x=np.arange(0,nbins,1)
plt.xticks(range(len(x)),x,fontsize=10)
l1=plt.plot(x,sf_sequence,color='turquoise',linestyle='-',marker = 'o')
plt.axhline(y=sf_permutation,color='silver',linestyle='--')
plt.axhline(y=(-sf_permutation),color='silver',linestyle='--')
plt.title('Forward replay in sequential condition')
plt.xlabel('timelag (TRs)')
plt.ylabel('sequenceness')
# plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\Forward Replay-%s-%6.3f.png' % (prediction_method, decoding_ipi)),
#             dpi = 400,bbox_inches = 'tight')
plt.show()

plt.figure(dpi=400,figsize=(10,5))
x=np.arange(0,nbins,1)
plt.xticks(range(len(x)),x,fontsize=10)
l1=plt.plot(x,sb_sequence,color='turquoise',linestyle='-',marker = 'o')
plt.axhline(y=sb_permutation,color='silver',linestyle='--')
plt.axhline(y=(-sb_permutation),color='silver',linestyle='--')
plt.title('Backward replay in sequential condition')
plt.xlabel('timelag (TRs)')
plt.ylabel('sequenceness')
# plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\Backward Replay-%s-%6.3f.png' % (prediction_method, decoding_ipi)),
#             dpi = 400,bbox_inches = 'tight')
plt.show()

plt.figure(dpi=400)
plt.xticks(fontsize=10)
x=np.arange(0,nbins)
l1=plt.plot(x,dif,color='turquoise',linestyle='-',marker = 'o')
plt.axhline(y=dif_permutation,color='silver',linestyle='--')
plt.axhline(y=(-dif_permutation),color='silver',linestyle='--')
plt.title('Forward-Backward replay in sequential condition')
plt.xlabel('timelag (TRs)')
plt.ylabel('squenceness')
# plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\Forward-Backward replay-%s-%6.3f.png' % (prediction_method, decoding_ipi)),
#             dpi = 400,bbox_inches = 'tight')
plt.show()





        
        