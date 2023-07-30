#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: PPI IN TASK
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITIVE NEUROSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

# In[IMPORT BASIC LIBRARIES]:
import os
import copy
import warnings
import numpy as np
import pandas as pd
import pingouin as pg
from os.path import join as opj
from bids.layout import BIDSLayout
from joblib import Parallel, delayed
# import Nilearn
from nilearn import plotting, image, masking
from nilearn.image import load_img, resample_to_img, threshold_img
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix, check_design_matrix, FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.second_level import non_parametric_inference
from nilearn.glm import threshold_stats_img, cluster_level_inference

# filter out warnings related to the numpy package:
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")

# In[SET PATHS AND SUBJECTS]:
# initialize empty paths:
ana_conditions = 'eeg_rest_ppi'
# path to the project root:
path_bids = opj(os.getcwd().split('fmrireplay')[0], 'fmrireplay')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_behavior = opj(path_bids, 'derivatives', 'behavior', 'sub_level')
path_onsets = opj(path_bids, 'derivatives', 'replay_onsets', 'sub_level')
path_mask = opj(path_bids, 'sourcedata/func_masks')
path_glm_l1 = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'level1')
path_glm_l2 = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'level2')
path_glm_roi = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'ppi_roi')
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
# set all the nan as mean value in confound variables
def replace_nan(regressor_values):
    # calculate the mean value of the regressor:
    mean_value = regressor_values.mean(skipna=True)
    # replace all values containing nan with the mean value:
    regressor_values[regressor_values.isnull()] = mean_value
    # return list of the regressor values:
    return list(regressor_values)


# In[LEVEL 1 GLM FUNCTION]:
# the number of run and onset list
run_list = [1, 2]
cons_list = ['pre', 'post']
# time of repetition, in seconds:
time_repetition = 1.3
# list
ppi_beta_list = []
seed_beta_list = []
react_beta_list = []
# set path templates
templates = dict(
    confounds=opj(path_fmriprep, 'sub-{subject}', 'func',
                  'sub-{subject}_task-rest_run-{run}_desc-confounds_timeseries.tsv'),
    onset_events=opj(path_onsets, 'sub-{subject}',
                     'sub-{subject}_{cons}_rest_reactivation_events.tsv'),
    func=opj(path_fmriprep, 'sub-{subject}', 'func',
             'sub-{subject}_task-rest_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'), )

# 2mm brain template
mni_mask = opj(path_bids, 'sourcedata/glm_l1_masks/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii')
mni_template = image.load_img(mni_mask)
# hippocampus mask
hippocampus_mask = image.load_img(opj(path_mask, 'hippo_bin.nii'))
hippo_res_mask = resample_to_img(hippocampus_mask, mni_template, interpolation='nearest')


def l1_GLM_pm(subject):
    sub = 'sub-%s' % subject
    # create the datasink
    datasink = opj(path_glm_l1, sub)
    if not os.path.exists(datasink):
        os.makedirs(datasink)
    # set the design matrix
    func_runs = []
    design_matrices = []

    for cons, run_id in zip(cons_list, run_list):
        # load the data path
        confounds = templates['confounds'].format(subject=subject, run=run_id)
        onset_events = templates['onset_events'].format(subject=subject, cons=cons)
        func = templates['func'].format(subject=subject, run=run_id)

        # load the data
        confounds_file = pd.read_csv(confounds, sep='\t')
        onset_file = pd.read_csv(onset_events, sep='\t')
        func_run = load_img(func)
        func_run_res = resample_to_img(func_run, mni_template, interpolation='nearest')

        # confound variables
        regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                           'csf', 'white_matter', 'global_signal']
        regressors = [replace_nan(confounds_file[conf]) for conf in regressor_names]
        motion = np.transpose(np.array(regressors))

        regressor_names_hp = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        regressors_hp = [replace_nan(confounds_file[conf]) for conf in regressor_names_hp]
        motion_hp = np.transpose(np.array(regressors_hp))

        # extract the seed region time-series BOLD signal
        hippo_masker = NiftiMasker(mask_img=hippo_res_mask,
                                   standardize=True,
                                   detrend=True,
                                   smoothing_fwhm=6,
                                   t_r=time_repetition,
                                   high_pass=1 / 128)
        seed_ts = hippo_masker.fit_transform(func_run_res, confounds=motion_hp)

        # parameters of design matrix
        n_scans = func_run.shape[-1]
        frame_times = np.arange(n_scans) * time_repetition  # here are the corresponding frame times
        design_matrix = make_first_level_design_matrix(
            frame_times, None, drift_model=None,
            add_regs=motion, add_reg_names=regressor_names,
            hrf_model='spm', oversampling=100)

        # concatenate the HRF-Convolved probability regressor
        onset_file.columns = ['reactivation_onset_%s' % cons]
        onset_file.set_index(frame_times, inplace=True, drop=True)

        # create the seed regressor
        seed_ts_mean = np.mean(seed_ts, axis=1)
        seed_ts_mean_df = pd.DataFrame(seed_ts_mean, columns=['hippo_ts'])
        seed_ts_mean_df.set_index(frame_times, inplace=True, drop=True)
        seed_ts_mean_df[0:4] = 0

        # create the ppi regressor
        react_ppi = pd.DataFrame((seed_ts_mean * onset_file['reactivation_onset_%s' % cons]))
        react_ppi.columns = ['reactivation*seed']

        # concatenate ppi regressors
        design_matrix_2 = pd.concat((react_ppi, seed_ts_mean_df, onset_file, design_matrix), axis=1)
        # corrplot(design_matrix_2, subject, i)
        # output the data
        design_matrices.append(design_matrix_2)
        func_runs.append(func_run)

    # check the design matrix if needed
    # plotting.plot_design_matrix(design_matrix_2)#, output_file=opj(path_glm_l1, 'design_matrix_example_1.eps'),rescale=True)

    # fit first level glm

    def fmrifit(func, dm):
        fmri_glm = FirstLevelModel(t_r=time_repetition, slice_time_ref=0.5, hrf_model='spm',
                                   drift_model=None, high_pass=1 / 128,
                                   mask_img=mni_mask,
                                   smoothing_fwhm=6, verbose=0,
                                   noise_model='ar1', minimize_memory=True)
        fmri_glm = fmri_glm.fit(run_imgs=func, design_matrices=dm)
        return fmri_glm

    fmri_glm_pre = fmrifit(func=func_runs[0], dm=design_matrices[0])
    fmri_glm_post = fmrifit(func=func_runs[1], dm=design_matrices[1])

    # construct the contrasts
    contrasts = {'ppi_con': [], 'seed_con': [], 'reactivation_pre': [], }
    design_matrix = design_matrices[0]
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])
    ppi_con = basic_contrasts['reactivation*seed']
    seed_con = basic_contrasts['hippo_ts']
    reactivation_pre = basic_contrasts['reactivation_onset_pre']
    for contrast_id in ['ppi_con', 'seed_con', 'reactivation_pre']:
        contrasts[contrast_id].append(eval(contrast_id))
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        # Estimate the contrasts
        stats_map = fmri_glm_pre.compute_contrast(contrast_val, stat_type='t', output_type='all')
        c_map = stats_map['effect_size']
        c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)
        if index == 0:
            ppi_beta = c_map
            ppi_beta_list.append(ppi_beta)
        elif index == 1:
            seed_beta = c_map
            seed_beta_list.append(seed_beta)
        elif index == 2:
            react_beta = c_map
            react_beta_list.append(react_beta)

    # construct the contrasts
    contrasts = {'ppi_con': [], 'seed_con': [], 'reactivation_post': [], }
    design_matrix = design_matrices[1]
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])
    ppi_con = basic_contrasts['reactivation*seed']
    seed_con = basic_contrasts['hippo_ts']
    reactivation_post = basic_contrasts['reactivation_onset_post']
    for contrast_id in ['ppi_con', 'seed_con', 'reactivation_post']:
        contrasts[contrast_id].append(eval(contrast_id))
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        # Estimate the contrasts
        stats_map = fmri_glm_post.compute_contrast(contrast_val, stat_type='t', output_type='all')
        c_map = stats_map['effect_size']
        c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)
        if index == 0:
            ppi_beta = c_map
            ppi_beta_list.append(ppi_beta)
        elif index == 1:
            seed_beta = c_map
            seed_beta_list.append(seed_beta)
        elif index == 2:
            react_beta = c_map
            react_beta_list.append(react_beta)

    return ppi_beta_list, seed_beta_list, react_beta_list


# run parallel functions
beta_list = Parallel(n_jobs=64)(delayed(l1_GLM_pm)(subject) for subject in sub_list)

# In[LEVEL 2 GLM AND MASKED THE T-MAP]:

# load all subjects' beta list
# pre rest
ppi_pre = [beta_list[i][0][0] for i in range(len(sub_list))]
seed_pre = [beta_list[i][1][0] for i in range(len(sub_list))]
replay_pre = [beta_list[i][2][0] for i in range(len(sub_list))]
# post rest
ppi_post = [beta_list[i][0][1] for i in range(len(sub_list))]
seed_post = [beta_list[i][1][1] for i in range(len(sub_list))]
replay_post = [beta_list[i][2][1] for i in range(len(sub_list))]

# all important brain region masks
func_mask = image.load_img(opj(path_mask, 'mni_label_hip.nii'))
func_res_mask = resample_to_img(func_mask, mni_template, interpolation='nearest')
func_res_mask.to_filename(opj(path_glm_l2, 'mni_resample.nii'))

mtl_mask = image.load_img(opj(path_mask, 'MTL', 'MTL_resample.nii'))

# RAW T-MAP
# pre rest
for (con, beta_map) in zip(['ppi_pre', 'seed_pre', 'replay_pre'],
                           [ppi_pre, seed_pre, replay_pre]):
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
    if con == 'ppi_pre':
        # MASKED THE T-MAP
        t_map_mask_pre = threshold_img(t_map, threshold=3, cluster_threshold=10,
                                       two_sided=False, mask_img=mtl_mask, copy=True)
        # t_map_mask = image.new_img_like(t_map, (np.multiply(copy.deepcopy(mtl_mask.dataobj),
        #                                                     copy.deepcopy(t_map.dataobj))))
        path_t_map = opj(path_glm_l2, "%s_tmap_masked.nii.gz" % con)
        t_map_mask_pre.to_filename(path_t_map)

# post rest
for (con, beta_map) in zip(['ppi_post', 'seed_post', 'replay_post'],
                           [ppi_post, seed_post, replay_post]):
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
    if con == 'ppi_post':
        # MASKED THE T-MAP
        t_map_mask_post = threshold_img(t_map, threshold=3, cluster_threshold=10,
                                        two_sided=False, mask_img=mtl_mask, copy=True)
        path_t_map = opj(path_glm_l2, "%s_tmap_masked.nii.gz" % con)
        t_map_mask_post.to_filename(path_t_map)

# In[pair-wise t-test of pre- and post-rest]:

second_level_input = ppi_pre + ppi_post
# create the contrasts
n_sub = len(sub_list)
condition_effect = np.hstack(([-1] * n_sub, [1] * n_sub))
subject_effect = np.vstack((np.eye(n_sub), np.eye(n_sub)))
subjects = [f'S{i:02d}' for i in range(1, n_sub + 1)]
# design matrix
paired_design_matrix = pd.DataFrame(
    np.hstack((condition_effect[:, np.newaxis], subject_effect)),
    columns=['Post vs Pre'] + subjects)
# second level GLM
second_level_model_paired = SecondLevelModel().fit(
    second_level_input, design_matrix=paired_design_matrix)

# RAW T-MAP
t_map_paired = second_level_model_paired.compute_contrast('Post vs Pre', second_level_stat_type='t',
                                                          output_type='stat')
t_image_path = opj(path_glm_l2, 'paired_rest_eeg-tmap.nii.gz')
t_map_paired.to_filename(t_image_path)

# MASKED THE T-MAP
t_map_mask_pair = threshold_img(t_map_paired, threshold=3, cluster_threshold=10,
                                two_sided=False, mask_img=mtl_mask, copy=True)
path_t_map = opj(path_glm_l2, "ppi_tmap_pair_masked.nii.gz")
t_map_mask_pair.to_filename(path_t_map)

# In[ROI ANALYSIS]:

# probability entorhinal brain masks
entorhinal_mask = image.load_img(opj(path_mask, 'entorhinal', 'entorhinal.nii'))
entor_res_mask = resample_to_img(entorhinal_mask, mni_template, interpolation='nearest')

# probability pmotor brain masks
pmotor_mask = image.load_img(opj(path_mask, 'motor', 'Primary_motor_cortex.nii'))
pmotor_res_mask = resample_to_img(pmotor_mask, mni_template, interpolation='nearest')

# explore the results based on different probability threshold
# def roi_analysis(prob_thres):
prob_thres = 40
mask_entorhinal = np.asanyarray(copy.deepcopy(entor_res_mask).dataobj)
# replace all other values with 1:
mask_entorhinal = np.where(mask_entorhinal >= prob_thres, 100, mask_entorhinal)
mask_entorhinal = np.where(mask_entorhinal < prob_thres, 0, mask_entorhinal)
# turn the 3D-array into booleans:
mask_entorhinal = mask_entorhinal.astype(bool)
# create image like object:
mask_entorhinal_thres = image.new_img_like(mni_template, mask_entorhinal)

# pair-wise t test in entorhinal
beta_entor_ppi_pre = [np.mean(masking.apply_mask(ppi_pre[i], mask_entorhinal_thres)) for i in
                      range(len(ppi_pre))]
beta_entor_ppi_post = [np.mean(masking.apply_mask(ppi_post[i], mask_entorhinal_thres)) for i in
                       range(len(ppi_post))]
entor_t_test_pre = pg.ttest(x=beta_entor_ppi_pre, y=0,
                            paired=False, alternative='two-sided')
entor_t_test_post = pg.ttest(x=beta_entor_ppi_post, y=0,
                             paired=False, alternative='two-sided')
entor_t_test_paired = pg.ttest(x=beta_entor_ppi_post, y=beta_entor_ppi_pre,
                               paired=True, alternative='two-sided')

print("pre: entorhinal:", round(entor_t_test_pre.loc['T-test', 'T'], 4),
      round(entor_t_test_pre.loc['T-test', 'p-val'], 4), round(entor_t_test_pre.loc['T-test', 'cohen-d'], 4),
      "\n post: entorhinal:", round(entor_t_test_post.loc['T-test', 'T'], 4),
      round(entor_t_test_post.loc['T-test', 'p-val'], 4), round(entor_t_test_post.loc['T-test', 'cohen-d'], 4),
      "\n post-pre: entorhinal:", round(entor_t_test_paired.loc['T-test', 'T'], 4),
      round(entor_t_test_paired.loc['T-test', 'p-val'], 4), round(entor_t_test_paired.loc['T-test', 'cohen-d'], 4))

mask_pmotor = np.asanyarray(copy.deepcopy(pmotor_res_mask).dataobj)
# replace all other values with 1:
mask_pmotor = np.where(mask_pmotor >= prob_thres, 100, mask_pmotor)
mask_pmotor = np.where(mask_pmotor < prob_thres, 0, mask_pmotor)
# turn the 3D-array into booleans:
mask_pmotor = mask_pmotor.astype(bool)
# create image like object:
mask_pmotor_thres = image.new_img_like(mni_template, mask_pmotor)

# pair-wise t test in pmotor
beta_pmotor_ppi_pre = [np.mean(masking.apply_mask(ppi_pre[i], mask_pmotor_thres)) for i in
                       range(len(ppi_pre))]
beta_pmotor_ppi_post = [np.mean(masking.apply_mask(ppi_post[i], mask_pmotor_thres)) for i in
                        range(len(ppi_post))]
pmotor_t_test_pre = pg.ttest(x=beta_pmotor_ppi_pre, y=0,
                             paired=False, alternative='two-sided')
pmotor_t_test_post = pg.ttest(x=beta_pmotor_ppi_post, y=0,
                              paired=False, alternative='two-sided')
pmotor_t_test_paired = pg.ttest(x=beta_pmotor_ppi_post, y=beta_pmotor_ppi_pre,
                                paired=True, alternative='two-sided')

print("pre: pmotor:", round(pmotor_t_test_pre.loc['T-test', 'T'], 4),
      round(pmotor_t_test_pre.loc['T-test', 'p-val'], 4), round(pmotor_t_test_pre.loc['T-test', 'cohen-d'], 4),
      "\n post: pmotor:", round(pmotor_t_test_post.loc['T-test', 'T'], 4),
      round(pmotor_t_test_post.loc['T-test', 'p-val'], 4), round(pmotor_t_test_post.loc['T-test', 'cohen-d'], 4),
      "\n post-pre: pmotor:", round(pmotor_t_test_paired.loc['T-test', 'T'], 4),
      round(pmotor_t_test_paired.loc['T-test', 'p-val'], 4), round(pmotor_t_test_paired.loc['T-test', 'cohen-d'], 4))

# Parallel(n_jobs=64)(delayed(roi_analysis)(prob_thres) for prob_thres in range(1, 2))


# SAVE THE BETA FILE
# entorhinal cortex
df_beta_entor_pre = pd.DataFrame(beta_entor_ppi_pre, columns=['pre_rest'])
df_beta_entor_post = pd.DataFrame(beta_entor_ppi_post, columns=['post_rest'])
beta_entor = pd.concat((df_beta_entor_pre, df_beta_entor_post), axis=1)
beta_entor = beta_entor.melt(value_vars=['pre_rest', 'post_rest'],
                             var_name='session', value_name='beta_value')
beta_entor.to_csv(opj(path_glm_roi, 'beta_entor_eeg_rest_ppi.csv'))

# primary motor cortex
df_beta_pmotor_pre = pd.DataFrame(beta_pmotor_ppi_pre, columns=['pre_rest'])
df_beta_pmotor_post = pd.DataFrame(beta_pmotor_ppi_post, columns=['post_rest'])
beta_pmotor = pd.concat((df_beta_pmotor_pre, df_beta_pmotor_post), axis=1)
beta_pmotor = beta_pmotor.melt(value_vars=['pre_rest', 'post_rest'],
                               var_name='session', value_name='beta_value')
beta_pmotor.to_csv(opj(path_glm_roi, 'beta_pmotor_eeg_rest_ppi.csv'))
