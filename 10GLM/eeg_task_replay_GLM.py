#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: FIRST LEVEL GLM IN TASK BASED ON EEG REPLAY
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITIVE NEUROSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY
# In[IMPORT BASIC LIBRARIES]:
import os
import copy
import warnings
import numpy as np
import pandas as pd
from os.path import join as opj
from joblib import Parallel, delayed
# import libraries for bids interaction:
from bids.layout import BIDSLayout
# import Nilearn
from nilearn import plotting, image, masking
from nilearn.image import load_img, resample_to_img
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix, check_design_matrix, FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img, cluster_level_inference
# filter out warnings related to the numpy package:
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")

# In[SET PATHS AND SUBJECTS]:
# initialize empty paths:
ana_conditions = 'eeg_cue_replay'
# path to the project root:
path_bids = opj(os.getcwd().split('fmrireplay')[0], 'fmrireplay')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_behavior = opj(path_bids, 'derivatives', 'behavior', 'sub_level')
path_onsets = opj(path_bids, 'derivatives', 'replay_onsets', 'sub_level')
path_mask = opj(path_bids, 'sourcedata/func_masks')
path_glm_l1 = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'level1')
path_glm_l2 = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'level2')
path_glm_masked = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'masked_results')
for path in [path_glm_l1, path_glm_l2]:
    if not os.path.exists(path):
        os.makedirs(path)

# In[SET SUBJECT LIST]:
# grab the list of subjects from the bids data set:
layout = BIDSLayout(path_bids)
# get all subject ids:
sub_list = sorted(layout.get_subjects())
loweeg = [45, 38, 36, 34, 28, 21, 15, 11, 5, 4, 2]
incomplete = [42, 22, 4]
extremeheadmotion = [38, 28, 16, 15, 13, 9, 5, 0]
delete_list = np.unique(
    np.hstack([loweeg, incomplete, extremeheadmotion]))[::-1]
[sub_list.pop(i) for i in delete_list]

# In[SET FUNCTIONS]:


def _orthogonalize(X):
    if X.size == X.shape[0]:
        return X
    from scipy.linalg import pinv
    for i in range(1, X.shape[1]):
        X[:, i] -= np.dot(np.dot(X[:, i], X[:, :i]), pinv(X[:, :i]))
    return X


# set all the nan as mean value in confound variables
def replace_nan(regressor_values):
    # calculate the mean value of the regressor:
    mean_value = regressor_values.mean(skipna=True)
    # replace all values containing nan with the mean value:
    regressor_values[regressor_values.isnull()] = mean_value
    # return list of the regressor values:
    return list(regressor_values)


# In[LEVEL 1 GLM FUNCTION]:
# the number of run
run_list = [1, 2, 3]
# time of repetition, in seconds:
time_repetition = 1.3
# set path templates
templates = dict(
    confounds=opj(path_fmriprep, 'sub-{subject}', 'func',
                  'sub-{subject}_task-replay_run-{run}_desc-confounds_timeseries.tsv'),
    events=opj(path_behavior, 'sub-{subject}',
               '{subject}_task-rep_run-{run}_events.tsv'),
    onset_events=opj(path_onsets, 'sub-{subject}',
                     'sub-{subject}_cue_replay_onsets_run-{run}_events.tsv'),
    func=opj(path_fmriprep, 'sub-{subject}', 'func',
             'sub-{subject}_task-replay_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
)
# mni template mask
mni_mask = opj(path_bids, 'sourcedata/glm_l1_masks/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii')


def l1_GLM_pm(subject):
    sub = 'sub-%s' % subject
    # create the result folder
    datasink = opj(path_glm_l1, sub)
    if not os.path.exists(datasink):
        os.makedirs(datasink)
    # set the design matrix
    func_runs = []
    design_matrices = []
    for i, run_id in enumerate(run_list):
        # load the data path
        confounds = templates['confounds'].format(subject=subject, run=run_id)
        events = templates['events'].format(subject=subject, run=str('%02.f' % run_id))
        onset_events = templates['onset_events'].format(subject=subject, run=str('%02.f' % run_id))
        func = templates['func'].format(subject=subject, run=run_id)
        # load the data
        confounds_file = pd.read_csv(confounds, sep='\t')
        events_file = pd.read_csv(events, sep='\t')
        onset_file = pd.read_csv(onset_events, sep='\t')
        func_run = load_img(func)
        # confound variables
        regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']
        regressors = [replace_nan(confounds_file[conf]) for conf in regressor_names]
        # mental simulation and wrong trial regressor
        events_file['onset'] = events_file['onset']
        events_file['trial_type'] = events_file['accuracy'].apply(lambda x: 'mental_simulation' if x == 1 else 'wrong')
        events = events_file.loc[:, ['trial_type', 'onset', 'duration']]
        # parameters of design matrix
        n_scans = func_run.shape[-1]
        frame_times = np.arange(n_scans) * time_repetition  # here are the corresponding frame times
        motion = np.transpose(np.array(regressors))
        design_matrix = make_first_level_design_matrix(
            frame_times, events, drift_model=None, add_regs=motion,
            add_reg_names=regressor_names, hrf_model='spm', oversampling=100)
        # concatenate the HRF-Convolved probability regressor
        onset_file.set_index(frame_times, inplace=True, drop=True)
        design_matrix = pd.concat((onset_file, design_matrix), axis=1)
        design_matrix_orth = pd.DataFrame((_orthogonalize(np.array(design_matrix.iloc[:, [0, 1]]))),
                                          columns=['HRF_onset_TR', 'mental_simulation'],
                                          index=np.arange(int(n_scans)) * time_repetition)
        design_matrix_1 = pd.concat(
            (design_matrix_orth, design_matrix.drop(['HRF_onset_TR', 'mental_simulation'], axis=1)), axis=1)
        # output the data
        design_matrices.append(design_matrix_1)
        func_runs.append(func_run)

    # check the design matrix if needed
    # plot_design_matrix(design_matrices[0], output_file=opj(path_glm_l1, 'design_matrix_example_1.eps'),rescale=True)

    # fit first level glm
    fmri_glm = FirstLevelModel(t_r=time_repetition, slice_time_ref=0.5, hrf_model='spm',
                               drift_model=None, high_pass=1 / 128,
                               mask_img=mni_mask,
                               smoothing_fwhm=6, verbose=0,
                               noise_model='ar1', minimize_memory=True)
    fmri_glm = fmri_glm.fit(run_imgs=func_runs, design_matrices=design_matrices)

    # construct the contrasts
    contrasts = {'replay_con': [], 'ms_con': [], }
    for design_matrix in design_matrices:
        contrast_matrix = np.eye(design_matrix.shape[1])
        basic_contrasts = dict([(column, contrast_matrix[i])
                                for i, column in enumerate(design_matrix.columns)])
        replay_con = basic_contrasts['HRF_onset_TR']
        ms_con = basic_contrasts['mental_simulation']
        for contrast_id in ['replay_con', 'ms_con']:
            contrasts[contrast_id].append(eval(contrast_id))
    # compute contrast
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        # Estimate the contrasts. Note that the model implicitly computes a fixed effect across the three sessions
        stats_map = fmri_glm.compute_contrast(contrast_val, stat_type='t', output_type='all')
        c_map = stats_map['effect_size']
        t_map = stats_map['stat']
        c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
        t_image_path = opj(datasink, '%s_tmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)
        t_map.to_filename(t_image_path)
        if index == 0:
            replay_beta = c_map
        elif index == 1:
            ms_beta = c_map

    return replay_beta, ms_beta


# run parallel functions
beta_list = Parallel(n_jobs=24)(delayed(l1_GLM_pm)(subject) for subject in sub_list)

# In[LEVEL 2 GLM AND THRESHOLDED THE T-MAP]:
# load the beta list
replay_beta_list = [beta_list[i][0] for i in range(len(sub_list))]
ms_beta_list = [beta_list[i][1] for i in range(len(sub_list))]

# RAW T-MAP
for (con, beta_map) in zip(['replay', 'ms'], [replay_beta_list, ms_beta_list]):
    second_level_input = beta_map
    design_matrix_l2 = pd.DataFrame(
        [1] * len(second_level_input),
        columns=['intercept'], )
    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(second_level_input, design_matrix=design_matrix_l2)
    # get the t map file
    t_map = second_level_model.compute_contrast(second_level_stat_type='t', output_type='stat')
    t_image_path = opj(path_glm_l2, '%s_tmap.nii.gz' % con)
    t_map.to_filename(t_image_path)
