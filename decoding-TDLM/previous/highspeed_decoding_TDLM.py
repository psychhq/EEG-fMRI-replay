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
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from collections import Counter
import datalad.api as dl   # can not run in the code.
import nibabel as nib
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
nSubj = 36
# the list of all the sequences
uniquePerms = list(itertools.permutations([1,2,3,4,5],5))
# the number of sequences
nShuf = len(uniquePerms)
# the number of timelags, including 32, 64, 128, 512and 2048ms
maxLag = 10
# the number of states (decoding models)
nstates = 5 #len(betas_de_re[0,])

    
'''
========================================================================
CREATE PATHS TO OUTPUT DIRECTORIES:
========================================================================
'''

# for n,sub in zip(range(len(suball)),suball):
sub=suball[1]

path_decoding = opj(path_tardis, 'TDLM')
path_out = opj(path_decoding, sub)
path_out_figs = opj(path_out, 'plots')
path_out_data = opj(path_out, 'data')
path_out_logs = opj(path_out, 'logs')
path_out_masks = opj(path_out, 'masks')
'''
========================================================================
CREATE OUTPUT DIRECTORIES IF THE DO NOT EXIST YET:
========================================================================
'''
for path in [path_out_figs, path_out_data, path_out_logs, path_out_masks]:
    if not os.path.exists(path):
        os.makedirs(path)
'''
========================================================================
SETUP LOGGING:
========================================================================
'''
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

'''
========================================================================
ADD BASIC SCRIPT INFORMATION TO THE LOGGER:
========================================================================
'''
logging.info('Running decoding script')
logging.info('operating system: %s' % sys.platform)
logging.info('project name: %s' % project)
logging.info('participant: %s' % sub)
logging.info('mask: %s' % mask)
logging.info('bold delay: %d secs' % bold_delay)
logging.info('smoothing kernel: %d mm' % smooth)

'''
========================================================================
LOAD BEHAVIORAL DATA (THE EVENTS.TSV FILES OF THE SUBJECT):
========================================================================
Load the events files for the subject. The 'oddball' data serves as the
training set of the classifier. The 'sequence' data constitutes the
test set.
'''
# paths to all events files of the current subject:
path_events = opj(cwd1,'highspeed-bids','forBIDS', sub, 'ses-*', 'func', '*tsv')
#dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
path_events = sorted(glob.glob(path_events), key=lambda f: os.path.basename(f))
logging.info('found %d event files' % len(path_events))
logging.info('paths to events files (sorted):\n%s' % pformat(path_events))
# import events file and save data in dataframe:
df_events = pd.concat((pd.read_csv(f, sep='\t') for f in path_events),
                      ignore_index=True)
'''
========================================================================
CREATE PATHS TO THE MRI DATA:
========================================================================
'''
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

project_name = 'fmrireplay-decoding'
path_root = opj(os.getcwd().split(project)[0] ,'fMRIreplay_hq')
path_bids = opj(path_root ,'fmrireplay-bids','BIDS')
path_code = opj(path_bids ,'code','decoding-TDLM')
path_fmriprep = opj(path_bids ,'derivatives','fmrireplay-fmriprep')
path_glm = opj(path_bids,'derivatives', project_name)
path_level1 = opj(path_glm, 'l1pipeline')
path_masks = opj(path_bids ,'derivatives','fmrireplay-masks', 'masks')
path_decoding = opj(path_bids,'derivatives','fmrireplay-decoding')

# path_behavior = opj(path_bids,'derivatives','fmrireplay-behavior')
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
#func_task = [nib.load(i) for i in path_func_task]

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
'''
========================================================================
LOAD THE MRI DATA:
========================================================================
'''
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


'''
========================================================================
FEATURE SELECTION
========================================================================
'''
#v= tmaps_masked[0]
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(v[:,0],v[:,1],v[:,2], zdir='z', c= 'red')
#plt.show()

# check if any value in the supposedly binary mask is bigger than 1:
if np.any(mask_vis > 1):
    logging.info('WARNING: detected values > 1 in the anatomical ROI!')
    sys.exit("Values > 1 in the anatomical ROI!")
# get combination of anatomical mask and t-map
tmaps_masked = [np.multiply(mask_vis, i) for i in copy.deepcopy(tmaps)]
# masked tmap into image like object:
tmaps_masked_img = [image.new_img_like(i, j) for (i, j) in zip(path_tmap, copy.deepcopy(tmaps_masked))]

for i, path in enumerate(tmaps_masked_img):
    path_save = opj(path_out_masks, '%s_run-%02d_tmap_masked.nii.gz' % (sub, i + 1))
    path.to_filename(path_save)

# plot masked t-maps
# logging.info('plotting masked tmaps with anatomical as background:')
# for i, path in enumerate(tmaps_masked_img):
#     logging.info('plotting masked tmap %d of %d' % (i+1, len(tmaps_masked_img)))
#     path_save = opj(path_out_figs, '%s_run-%02d_tmap_masked.png' % (sub, i+1))
#     plotting.plot_roi(path, anat, title=os.path.basename(path_save),
#                       output_file=path_save, colorbar=True)
'''
========================================================================
FEATURE SELECTION: THRESHOLD THE MASKED T-MAPS
We threshold the masked t-maps, selecting only voxels above AND
below this threshold. We then extract these data and count how
many voxels were above and / or below the threshold in total.
========================================================================
'''
# set the threshold:
threshold = params['mri']['thresh']
logging.info('thresholding t-maps with a threshold of %s' % str(threshold))
# threshold the masked tmap image:
tmaps_masked_thresh_img = [image.threshold_img(i, threshold) for i in copy.deepcopy(tmaps_masked_img)]

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

'''
========================================================================
FEATURE SELECTION: BINARIZE THE THRESHOLDED MASKED T-MAPS

========================================================================
'''
# replace all NaNs with 0:
tmaps_masked_thresh_bin = [np.where(np.isnan(i), 0, i) for i in copy.deepcopy(tmaps_masked_thresh)]
# replace all other values with 1:
tmaps_masked_thresh_bin = [np.where(i > 0, 1, i) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
# turn the 3D-array into booleans:
tmaps_masked_thresh_bin = [i.astype(bool) for i in copy.deepcopy(tmaps_masked_thresh_bin)]
# create image like object:
masks_final = [image.new_img_like(path_func_task[0], i.astype(np.int)) for i in copy.deepcopy(tmaps_masked_thresh_bin)]


'''
========================================================================
LOAD SMOOTHED FMRI DATA FOR ALL FUNCTIONAL TASK RUNS:
========================================================================
'''
# load smoothed functional mri data for all eight task runs:
logging.info('loading %d functional task runs ...' % len(path_func_task))
data_task = [image.load_img(i) for i in path_func_task]
logging.info('loading successful!')




'''
========================================================================
1. SPLIT THE EVENTS DATAFRAME FOR EACH TASK CONDITION
2. RESET THE INDICES OF THE DATAFRAMES
3. SORT THE ROWS OF ALL DATAFRAMES IN CHRONOLOGICAL ORDER
4. PRINT THE NUMBER OF TRIALS OF EACH TASK CONDITION
========================================================================
'''

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
SEPARATE RUN-WISE STANDARDIZATION (Z-SCORING) OF ALL TASK CONDITIONS
========================================================================
'''





data_list = []
runs = list(range(1, n_run+1))
mask_label = 'cv'

logging.info('starting leave-one-run-out cross-validation')
#for mask_label in ['cv', 'cv_hpc']:
logging.info('testing in mask %s' %(mask_label))
for run in runs:
    logging.info('testing on run %d of %d ...' % (run, len(runs)))
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
        logging.info('classifier: %s' % clf_name)
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
        logging.info('testing on test set %s' % test_set.name)
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

# 2048ms and specific classifier's prediction for 15 trials with different sequences with 13 TRs
pre_list=data_list
delete_list = [47,41,35,29,23,17,11,5]
[pre_list.pop(i) for i in delete_list]
pre_list_append = []
pre_list_append = pd.concat([pre_list[i] for i in range(len(pre_list))], axis=0)
pre_list_filter = pre_list_append[(pre_list_append["tITI"] == 2.048)&(pre_list_append["class"] != 'other')]

pre_list_filter_sort = pre_list_filter.sort_values(by=['class','trial'])

# get the prediction matrix 
pred_array = np.array(pre_list_filter_sort.loc[:,'probability'])
pred_matrix = np.reshape(pred_array,(195,5),order='F')



# get the 2048ms th trial's sequences
test_seq_long.seq2048 = []
test_seq_long.trial2048 = np.unique((np.argwhere(test_seq_long.itises == 2.048)[:,0]))
test_seq_long.seq2048 = test_seq_long.sequences[(test_seq_long.trial2048),]


'''
========================================================================
TDLM FOR SEQUENCE CONDITION TRIALS
========================================================================
'''
# parameters for TDLM
# # beta parameters of 10736 voexls for 5 decoding models
# betas_de = clf.coef_.T
# # intercept parameters for 5 decoding models
# intercepts_de = clf.intercept_.T
# the number of subjects
nSubj = 36
# # the list of all the sequences
# uniquePerms = list(itertools.permutations([1,2,3,4,5],5))
# # the number of sequences
# nShuf = len(uniquePerms)
# the number of timelags, including 32, 64, 128, 512and 2048ms
maxLag= 10
# the number of states (decoding models)
nstates = 5
# the nan matrix for forward design matrix (40 subjects * 15 trials * 120 shuffles * 2100 timelags)
sf = np.full([15,maxLag+1],np.nan)
# GLM predefine
betas_1GLM = None
betas_2GLM = None



# prediction for each classifiers, each trial for 13 TR.



logging.info('predict the sequence data')

# the probability of prediction of training model to sequence data
# classes_names = clf.classes_
# pred_resting_decoding_prob = [1/(1+np.exp(-(np.dot(i,betas_de_re) 
#                                     + repmat(intercepts_de_re, len(i),1)))) for i in data_rest_final]

# pred_resting_decoding_clas = [[''.join(clf.classes_[np.where(np.array(pred_resting_decoding_prob[i][j,:]) 
#                                                               == np.max(np.array(pred_resting_decoding_prob[i][j,:])))]) 
#                                 for j in range(len(pred_resting_decoding_prob[0]))] 
#                               for i in range(len(data_rest_final))]


plt.imshow(pred_matrix, aspect='auto', interpolation='none',
           origin='lower',cmap='hot')
plt.title('Decoded State Space')
plt.xlabel('States')
plt.ylabel('time points (TR)')
plt.xticks(range(5),('cat', 'chair', 'face', 'house', 'shoe'))
plt.colorbar()
plt.show()

rp = []
#detect for each  2048 ms trial
for i in range(len(test_seq_long.trial2048)):    
    # real sequence
    rp = test_seq_long.seq2048[i,]
    # get the integer of sequence
    rp = np.trunc(rp)
    # get the transition matrix
    T1 = TransM(rp)
    # get the probability of preditions
    
    
    #detect for each permutation
    # for j in uniquePerms: 
    nbins=maxLag+1
    X1 = pred_matrix[13*i:13*(i+1),:] #####
    # timelag matrix 
    dm = scipy.linalg.toeplitz(X1[:,0],[np.zeros((nbins,1))]) #####
    dm = dm[:,1:]
    
    # 4 loops
    for k in range(1,5) : 
        temp = scipy.linalg.toeplitz(X1[:,k],[np.zeros((nbins,1))]);
        temp = temp[:,1:]
        dm = np.hstack((dm,temp))
    
    # the next time point needed to be predicted 
    Y = X1
    # build a new framework for first GLM betas
    betas_1GLM = np.full([nstates*maxLag, nstates],np.nan)

    bins=maxLag;
    
    #detect for each timelag
    for l in range(0, maxLag): 
        temp_zinds = np.array(range(0,nstates*maxLag,bins)) + l
        # First GLM
        design_mat_1 = np.hstack((dm[:,temp_zinds],np.ones((len(dm[:,temp_zinds]),1))))
        temp = np.dot(np.linalg.pinv(design_mat_1),Y)
        betas_1GLM[temp_zinds,:] = temp[:-1,:]  ####?
    
    betasnbins64 = np.reshape(betas_1GLM,[maxLag,np.square(nstates)],order = "F") ######
    # Second GLM
    design_mat_2 = np.transpose(np.vstack((T1.flatten(), np.eye(nstates).flatten(),np.ones([nstates,nstates]).flatten())))
    betas_2GLM = np.dot(np.linalg.pinv(design_mat_2) , np.transpose(betasnbins64)); 
    # four different design matrix for regressor multiple to the temporal data
    # linear regression for the backward and forward replay
        
    sf[i,1:] = betas_2GLM[0,:]

# test = np.array(test_seq_long.sequences)


# mean the all the sequence
sf_mean_lag = sf.mean(axis=0)


#

x=np.arange(0,nbins)

l1=plt.plot(x,sf_mean_lag,'r--',label='sequential trial')
plt.plot(x,sf_mean_lag,'r--')
plt.title('replay in sequential condition')
plt.xlabel('timelag (TRs)')
plt.ylabel('squenceness')
plt.legend()
plt.show()



# concatenate all decoding dataframes in one final dataframe:
# df_all = pd.concat(data_list, sort=False)
# # create file path to save the dataframe:
# file_path = opj(path_out_data, '{}_decoding.csv'.format(sub))
# # save the final dataframe to a .csv-file:
# df_all.to_csv(file_path, sep=',', index=False)

'''
========================================================================
STOP LOGGING:
========================================================================
'''
end = time.time()
total_time = (end - start)/60
logging.info('total running time: %0.2f minutes' % total_time)
logging.shutdown()
