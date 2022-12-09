#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: FIRST LEVEL GLM
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITIVE NEUROSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY
# In[1]: import basic libraries:
import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
from os.path import join as opj
from joblib import Parallel, delayed
# import libraries for bids interaction:
from bids.layout import BIDSLayout
# import Nilearn
from nilearn.image import load_img
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix, check_design_matrix, FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
# filter out warnings related to the numpy package:

warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")

# In[3]: SET PATHS AND SUBJECTS
# define paths depending on the operating system (OS) platform:
# initialize empty paths:
ana_conditions = 'raw_rest_pm-nodemean-8-middle'
sub_list = None
# path to the project root:
project_name = 'fmrireplay-glm-resting-l1'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_code = opj(path_bids, 'code', 'decoding-TDLM')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_behavior = opj(path_bids, 'derivatives', 'behavior')
# path_maxonset = opj(path_bids, 'derivatives', 'replay-onset', 'Max_onset')
# path_perm2 = opj(path_bids, 'derivatives', 'replay-onset', 'Perm_onset_2con')
# path_perm4 = opj(path_bids, 'derivatives', 'replay-onset', 'Perm_onset_4con')
path_raw = opj(path_bids, 'derivatives', 'resting-replay-onset', 'raw_replay')
path_glm_cor = opj(path_bids, 'derivatives', 'glm-l1', 'nilearn', 'correlation')
path_glm_l1 = opj(path_bids, 'derivatives', 'glm-l1', 'nilearn', ana_conditions)
path_glm_l2 = opj(path_bids, 'derivatives', 'glm-l2', 'nilearn', ana_conditions)
for path in [path_glm_l1, path_glm_l2]:
    if not os.path.exists(path):
        os.makedirs(path)
# In[4]:
# grab the list of subjects from the bids data set:
layout = BIDSLayout(path_bids)
# get all subject ids:
sub_list = sorted(layout.get_subjects())
# lowfmri = [37, 16, 15, 13, 11, 0]
loweeg = [45, 38, 36, 34, 28, 21, 15, 11, 5, 4, 2]
incomplete = [42, 22, 4]
extremeheadmotion = [38, 28, 16, 15, 13, 9, 5, 0]
delete_list = np.unique(
    np.hstack([loweeg, incomplete, extremeheadmotion]))[::-1]
[sub_list.pop(i) for i in delete_list]

# In[5]:
# time of repetition, in seconds:
time_repetition = 1.3
# number of dummy variables to remove from each run:
num_dummy = 0


def _orthogonalize(X):
    if X.size == X.shape[0]:
        return X
    from scipy.linalg import pinv
    for i in range(1, X.shape[1]):
        X[:, i] -= np.dot(np.dot(X[:, i], X[:, :i]), pinv(X[:, :i]))
    return X

    # corrplot(data=design_matrix, sub=sub)
def corrplot(data, sub, i):
    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="white")
    # Compute the correlation matrix
    corr = data.corr()
    # Generate a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15, 12))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=True, fmt=".2f")
    if not os.path.exists(path_glm_cor):
        os.makedirs(path_glm_cor)
    plt.savefig(opj(path_glm_cor, '%s run -%s correlation matrix' % (sub, i)), dpi=400)
    plt.show()


# In[]: GLM function

# # check the data files
# path_confounds = glob.glob(opj(path_fmriprep, '*', 'func', '*rest_run*confounds_timeseries.tsv'))
# path_func = glob.glob(opj(path_fmriprep, '*', 'func', '*rest*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))
# path_wholemask = glob.glob(opj(path_fmriprep, '*', 'func', '*rest*space-MNI152NLin2009cAsym*brain_mask.nii.gz'))
# path_onset_events = glob.glob(opj(path_raw, '*', '*_nodemean_middle_rest_replay_events.tsv'))
#
# replay_beta_list = []
# ms_beta_list = []
#
# # the number of run and onset list
# run_list = [1, 1, 2, 2]
#
# def l1_GLM_pm(subject):
#     # subject = sub_list[0]
#     sub = 'sub-%s' % subject
#     # path templates
#     templates = dict(
#         confounds=opj(path_fmriprep, 'sub-{subject}', 'func',
#                       'sub-{subject}_task-rest_run-{run}_desc-confounds_timeseries.tsv'),
#         onset_events=opj(path_raw, 'sub-{subject}',
#                          '{subject}_{onset_num}_nodemean_middle_rest_replay_events.tsv'),
#         func=opj(path_fmriprep, 'sub-{subject}', 'func',
#                  'sub-{subject}_task-rest_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
#     )
#
#     func_runs = []
#     design_matrices = []
#
#     for i, run_id in enumerate(run_list):
#         # i = 0
#         # run_id = 1
#         # print(i)
#         # print(run_id)
#         # load the data path
#         confounds = templates['confounds'].format(subject=subject, run=run_id)
#         onset_events = templates['onset_events'].format(subject=subject, run=str('%02.f' % run_id), onset_num=(i+1))
#         func = templates['func'].format(subject=subject, run=run_id)
#         # load the data
#         confounds_file = pd.read_csv(confounds, sep='\t')
#         onset_file = pd.read_csv(onset_events, sep='\t')
#         func_run = load_img(func)
#
#         # desired confound variables
#         confounds_file = confounds_file[num_dummy:]
#         regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']
#         # regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter',
#         #                    'csf_derivative1', 'csf_derivative1_power2', 'csf_power2', 'white_matter_derivative1',
#         #                    'white_matter_power2', 'white_matter_derivative1_power2', 'csf_wm']  # 15
#
#         # regressor_names = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
#         #                    'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
#         #                    'trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
#         #                    'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2',
#         #                    'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2',
#         #                    'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']
#
#         # set all the nan as mean value in confound variables
#         def replace_nan(regressor_values):
#             # calculate the mean value of the regressor:
#             mean_value = regressor_values.mean(skipna=True)
#             # replace all values containing nan with the mean value:
#             regressor_values[regressor_values.isnull()] = mean_value
#             # return list of the regressor values:
#             return list(regressor_values)
#
#         # create a nested list with regressor values
#         regressors = [replace_nan(confounds_file[conf]) for conf in regressor_names]
#
#         # parameters of design matrix
#         n_scans = func_run.shape[-1]
#         frame_times = np.arange(n_scans) * time_repetition  # here are the corresponding frame times
#         # confound variables
#         motion = np.transpose(np.array(regressors))
#         add_reg_names = regressor_names
#         # rest replay onset probability variable
#         name = ['HRF_onset_TR']
#         matrix = np.hstack((np.array(onset_file),motion))
#         #combine the two
#         name += add_reg_names
#         design_matrix = pd.DataFrame(matrix, columns=name, index=frame_times)
#         # plot the correlation matrix between variables
#         corrplot(data=design_matrix, sub=sub, i=i)
#         # check_design_matrix(design_matrix)
#         # plot_design_matrix(design_matrix)
#         # fit first level glm to estimate mean orientation
#         # mni_mask = r'/home/huangqi/Data/BIDS/sourcedata/glm_l1_masks/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
#         fmri_glm = FirstLevelModel(t_r=1.3, slice_time_ref=0.5, hrf_model='spm',
#                                    drift_model=None, high_pass=1 / 128,
#                                    # mask_img=mni_mask,
#                                    smoothing_fwhm=6, verbose=0, n_jobs=-2,
#                                    # standardize=True, signal_scaling=False,
#                                    noise_model='ar1', minimize_memory=True)
#         warnings.filterwarnings("ignore", category=FutureWarning)
#         fmri_glm = fmri_glm.fit(run_imgs=func_run, design_matrices=design_matrix)
#
#         contrasts = {'replay_con': [],}
#         contrast_matrix = np.eye(design_matrix.shape[1])
#         basic_contrasts = dict([(column, contrast_matrix[i])
#                                 for i, column in enumerate(design_matrix.columns)])
#         replay_con = basic_contrasts['HRF_onset_TR']
#         for contrast_id in ['replay_con']:
#             contrasts[contrast_id].append(eval(contrast_id))
#
#         print('Computing contrasts...')
#         datasink = opj(path_glm_l1, sub)
#         if not os.path.exists(datasink):
#             os.makedirs(datasink)
#
#         for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
#             # print('Contrast % 2i out of %i: %s' % (index + 1, len(contrasts), contrast_id))
#             # Estimate the contrasts. Note that the model implicitly computes a fixed effect across the three sessions
#             stats_map = fmri_glm.compute_contrast(contrast_val, stat_type='t', output_type='all')
#             c_map = stats_map['effect_size']
#             t_map = stats_map['stat']
#             z_map = stats_map['z_score']
#
#             c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
#             c_map.to_filename(c_image_path)
#
#             t_image_path = opj(datasink, '%s_tmap.nii.gz' % contrast_id)
#             t_map.to_filename(t_image_path)
#
#             z_image_path = opj(datasink, '%s_zmap.nii.gz' % contrast_id)
#             z_map.to_filename(z_image_path)
#
#             replay_beta_list.append(c_map)
#
#     return replay_beta_list
#     # return ms_beta_list
#
# warnings.filterwarnings("ignore", category=FutureWarning)
# beta_list = Parallel(n_jobs=33)(delayed(l1_GLM_pm)(subject) for subject in sub_list)
#
# # In[]:
# # get different condition beta maps
# pre_fwd = [beta_list[i][0] for i in range(len(sub_list))]
# pre_bwd = [beta_list[i][1] for i in range(len(sub_list))]
# post_fwd = [beta_list[i][2] for i in range(len(sub_list))]
# post_bwd = [beta_list[i][3] for i in range(len(sub_list))]
#
# for (con, beta_map) in zip(['pre_fwd', 'pre_bwd', 'post_fwd', 'post_bwd'],
#                            [pre_fwd, pre_bwd, post_fwd, post_bwd]):
#     second_level_input = beta_map
#     design_matrix_l2 = pd.DataFrame(
#         [1] * len(second_level_input),
#         columns=['intercept'],
#     )
#
#     second_level_model = SecondLevelModel()
#     second_level_model = second_level_model.fit(
#         second_level_input,
#         design_matrix=design_matrix_l2,
#     )
#     # z_map = second_level_model.compute_contrast(output_type='z_score')
#     # z_image_path = opj(path_glm_l2, '%s_zmap.nii.gz' % con)
#     # z_map.to_filename(z_image_path)
#     t_map = second_level_model.compute_contrast(second_level_stat_type='t', output_type='stat')
#     t_image_path = opj(path_glm_l2, '%s_tmap.nii.gz' % con)
#     t_map.to_filename(t_image_path)
#
#     t_map_thres, threshold = threshold_stats_img(stat_img=t_map,
#                                       alpha=0.01,
#                                       height_control='fdr',
#                                       cluster_threshold=0,
#                                       two_sided=True)
#     t_thres_image_path = opj(path_glm_l2, '%s_tmap_thres.nii.gz' % con)
#     t_map_thres.to_filename(t_thres_image_path)

# In[]: pre and post resting replay probability
# # check the data files
# path_confounds = glob.glob(opj(path_fmriprep, '*', 'func', '*rest_run*confounds_timeseries.tsv'))
# path_func = glob.glob(opj(path_fmriprep, '*', 'func', '*rest*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))
# path_wholemask = glob.glob(opj(path_fmriprep, '*', 'func', '*rest*space-MNI152NLin2009cAsym*brain_mask.nii.gz'))
# path_onset_events = glob.glob(opj(path_raw, '*', '*_nodemean_middle_rest_replay_events.tsv'))
# # list
# replay_beta_list = []
# ms_beta_list = []
# # the number of run and onset list
# run_list = [1, 2]
# cons_list = ['pre', 'post']
#
#
# def l1_GLM_pm(subject):
#     # subject = sub_list[0]
#     sub = 'sub-%s' % subject
#     # path templates
#     templates = dict(
#         confounds=opj(path_fmriprep, 'sub-{subject}', 'func',
#                       'sub-{subject}_task-rest_run-{run}_desc-confounds_timeseries.tsv'),
#         onset_events=opj(path_raw, 'sub-{subject}',
#                          '{subject}_{cons}_nodemean_middle_rest_replay_events.tsv'),
#         func=opj(path_fmriprep, 'sub-{subject}', 'func',
#                  'sub-{subject}_task-rest_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
#     )
#     # set lists
#     func_runs = []
#     design_matrices = []
#     # design matrix
#     for cons, run_id in zip(cons_list, run_list):
#         # load the data path
#         confounds = templates['confounds'].format(subject=subject, run=run_id)
#         onset_events = templates['onset_events'].format(subject=subject, run=str('%02.f' % run_id), cons=cons)
#         func = templates['func'].format(subject=subject, run=run_id)
#         # load the data
#         confounds_file = pd.read_csv(confounds, sep='\t')
#         onset_file = pd.read_csv(onset_events, sep='\t')
#         func_run = load_img(func)
#         # desired confound variables
#         confounds_file = confounds_file[num_dummy:]
#         regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']
#         # regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter',
#         #                    'csf_derivative1', 'csf_derivative1_power2', 'csf_power2', 'white_matter_derivative1',
#         #                    'white_matter_power2', 'white_matter_derivative1_power2', 'csf_wm']  # 15
#
#         # regressor_names = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
#         #                    'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
#         #                    'trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
#         #                    'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2',
#         #                    'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2',
#         #                    'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']
#         # set all the nan as mean value in confound variables
#         def replace_nan(regressor_values):
#             # calculate the mean value of the regressor:
#             mean_value = regressor_values.mean(skipna=True)
#             # replace all values containing nan with the mean value:
#             regressor_values[regressor_values.isnull()] = mean_value
#             # return list of the regressor values:
#             return list(regressor_values)
#         # create a nested list with regressor values
#         regressors = [replace_nan(confounds_file[conf]) for conf in regressor_names]
#         # parameters of design matrix
#         n_scans = func_run.shape[-1]
#         frame_times = np.arange(n_scans) * time_repetition  # here are the corresponding frame times
#         # confound variables
#         motion = np.transpose(np.array(regressors))
#         add_reg_names = regressor_names
#         # rest replay onset probability variable
#         name = ['replay_onset_%s' % cons]
#         matrix = np.hstack((np.array(onset_file), motion))
#         #combine the two parts
#         name += add_reg_names
#         design_matrix = pd.DataFrame(matrix, columns=name, index=frame_times)
#         # plot the correlation matrix between variables
#         # corrplot(data=design_matrix, sub=sub)
#         # check_design_matrix(design_matrix)
#         # plot_design_matrix(design_matrix)
#         func_runs.append(func_run)
#         design_matrices.append(design_matrix)
#
#     # fit first level glm to estimate mean orientation
#     def fmrifit(func, dm):
#         mni_mask = r'/home/huangqi/Data/BIDS/sourcedata/glm_l1_masks/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
#         fmri_glm = FirstLevelModel(t_r=1.3, slice_time_ref=0.5, hrf_model='spm',
#                                    drift_model=None, high_pass=1 / 128,
#                                    mask_img=mni_mask,
#                                    smoothing_fwhm=6, verbose=0, n_jobs=-2,
#                                    # standardize=True, signal_scaling=False,
#                                    noise_model='ar1', minimize_memory=True)
#         fmri_glm = fmri_glm.fit(run_imgs=func, design_matrices=dm)
#         return fmri_glm
#
#     fmri_glm_pre = fmrifit(func=func_runs[0], dm=design_matrices[0])
#     fmri_glm_post = fmrifit(func=func_runs[1], dm=design_matrices[1])
#
#     # pre-resting contrast
#     contrasts = {'replay_pre': [], } #contrast name
#     design_matrix = design_matrices[0]
#     contrast_matrix = np.eye(design_matrix.shape[1])
#     basic_contrasts = dict([(column, contrast_matrix[i])
#                             for i, column in enumerate(design_matrix.columns)])
#     replay_pre = basic_contrasts['replay_onset_pre'] # it is the regressor name, to form the contrast name
#     for contrast_id in ['replay_pre']: #contrast name
#         contrasts[contrast_id].append(eval(contrast_id))
#     print('Computing contrasts...')
#     datasink = opj(path_glm_l1, sub)
#     if not os.path.exists(datasink):
#         os.makedirs(datasink)
#     for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
#         # Estimate the contrasts
#         stats_map = fmri_glm_pre.compute_contrast(contrast_val, stat_type='t', output_type='all')
#         c_map = stats_map['effect_size']
#         t_map = stats_map['stat']
#         z_map = stats_map['z_score']
#
#         c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
#         c_map.to_filename(c_image_path)
#
#         t_image_path = opj(datasink, '%s_tmap.nii.gz' % contrast_id)
#         t_map.to_filename(t_image_path)
#
#         z_image_path = opj(datasink, '%s_zmap.nii.gz' % contrast_id)
#         z_map.to_filename(z_image_path)
#
#         replay_beta_list.append(c_map)
#
#     contrasts = {'replay_post': [], }
#     # pre-resting contrast
#     design_matrix = design_matrices[1]
#     contrast_matrix = np.eye(design_matrix.shape[1])
#     basic_contrasts = dict([(column, contrast_matrix[i])
#                             for i, column in enumerate(design_matrix.columns)])
#     replay_post = basic_contrasts['replay_onset_post']
#     for contrast_id in ['replay_post']:
#         contrasts[contrast_id].append(eval(contrast_id))
#     print('Computing contrasts...')
#     datasink = opj(path_glm_l1, sub)
#     if not os.path.exists(datasink):
#         os.makedirs(datasink)
#     for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
#         # Estimate the contrasts
#         stats_map = fmri_glm_post.compute_contrast(contrast_val, stat_type='t', output_type='all')
#         c_map = stats_map['effect_size']
#         t_map = stats_map['stat']
#         z_map = stats_map['z_score']
#
#         c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
#         c_map.to_filename(c_image_path)
#
#         t_image_path = opj(datasink, '%s_tmap.nii.gz' % contrast_id)
#         t_map.to_filename(t_image_path)
#
#         z_image_path = opj(datasink, '%s_zmap.nii.gz' % contrast_id)
#         z_map.to_filename(z_image_path)
#
#         replay_beta_list.append(c_map)
#
#     return replay_beta_list
#     # return ms_beta_list
#
# warnings.filterwarnings("ignore", category=FutureWarning)
# beta_list = Parallel(n_jobs=25)(delayed(l1_GLM_pm)(subject) for subject in sub_list)
#
# # get different condition beta maps
# pre = [beta_list[i][0] for i in range(len(sub_list))]
# post = [beta_list[i][1] for i in range(len(sub_list))]
#
# # simple t-test
# for (con, beta_map) in zip(['pre', 'post'], [pre, post]):
#     second_level_input = beta_map
#     design_matrix_l2 = pd.DataFrame(
#         [1] * len(second_level_input),
#         columns=['intercept'],
#         )
#
#     second_level_model = SecondLevelModel()
#     second_level_model = second_level_model.fit(
#         second_level_input,
#         design_matrix=design_matrix_l2,
#     )
#     # z_map = second_level_model.compute_contrast(output_type='z_score')
#     # z_image_path = opj(path_glm_l2, '%s_zmap.nii.gz' % con)
#     # z_map.to_filename(z_image_path)
#     t_map = second_level_model.compute_contrast(second_level_stat_type='t', output_type='stat')
#     t_image_path = opj(path_glm_l2, '%s_tmap.nii.gz' % con)
#     t_map.to_filename(t_image_path)
#
#     t_map_thres, threshold = threshold_stats_img(stat_img=t_map,
#                                                  alpha=0.01,
#                                                  height_control='fdr',
#                                                  cluster_threshold=0,
#                                                  two_sided=True)
#     t_thres_image_path = opj(path_glm_l2, '%s_tmap_thres.nii.gz' % con)
#     t_map_thres.to_filename(t_thres_image_path)
#
#
# #paired-wise t-test
# from nilearn.plotting import plot_design_matrix
# from matplotlib import pyplot as plt
# second_level_input = pre + post
# n_sub = len(sub_list)
# condition_effect = np.hstack(([-1] * n_sub, [1] * n_sub))
# subject_effect = np.vstack((np.eye(n_sub), np.eye(n_sub)))
# subjects = [f'S{i:02d}' for i in range(1, n_sub + 1)]
#
# paired_design_matrix = pd.DataFrame(
#     np.hstack((condition_effect[:, np.newaxis], subject_effect)),
#     columns=['Post vs Pre'] + subjects)
#
# second_level_model_paired = SecondLevelModel().fit(
#     second_level_input, design_matrix=paired_design_matrix)
#
# t_maps_paired = second_level_model_paired.compute_contrast('Post vs Pre',
#                                                               second_level_stat_type='t',
#                                                               output_type='stat')
# t_image_path = opj(path_glm_l2, '%s_paired-tmap_post-pre.nii.gz' % con)
# t_maps_paired.to_filename(t_image_path)
#
# t_maps_paired_thres, threshold = threshold_stats_img(stat_img=t_maps_paired,
#                                              alpha=0.01,
#                                              height_control='fdr',
#                                              cluster_threshold=0,
#                                              two_sided=True)
# t_thres_image_path = opj(path_glm_l2, '%s_paired-tmap_thres_post-pre.nii.gz' % con)
# t_maps_paired_thres.to_filename(t_thres_image_path)


# In[]: pre- and post-resting reactivation probability
start_time = time.time()
# check the data files
path_confounds = glob.glob(opj(path_fmriprep, '*', 'func', '*rest_run*confounds_timeseries.tsv'))
path_func = glob.glob(opj(path_fmriprep, '*', 'func', '*rest*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))
path_wholemask = glob.glob(opj(path_fmriprep, '*', 'func', '*rest*space-MNI152NLin2009cAsym*brain_mask.nii.gz'))
path_onset_events = glob.glob(opj(path_raw, '*', '*_middle_rest_reactivation_events.tsv'))
# list
reactivation_beta_list = []
# the number of run and onset list
run_list = [1, 2]
cons_list = ['pre', 'post']

def l1_GLM_pm(subject):
    # subject = sub_list[0]
    sub = 'sub-%s' % subject
    # path templates
    templates = dict(
        confounds=opj(path_fmriprep, 'sub-{subject}', 'func',
                      'sub-{subject}_task-rest_run-{run}_desc-confounds_timeseries.tsv'),
        onset_events=opj(path_raw, 'sub-{subject}',
                         '{subject}_{cons}_middle_rest_reactivation_events.tsv'),
        func=opj(path_fmriprep, 'sub-{subject}', 'func',
                 'sub-{subject}_task-rest_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
    )
    # set lists
    func_runs = []
    design_matrices = []
    # design matrix
    for cons, run_id in zip(cons_list, run_list):
        # load the data path
        confounds = templates['confounds'].format(subject=subject, run=run_id)
        onset_events = templates['onset_events'].format(subject=subject, run=str('%02.f' % run_id), cons=cons)
        func = templates['func'].format(subject=subject, run=run_id)
        # load the data
        confounds_file = pd.read_csv(confounds, sep='\t')
        onset_file = pd.read_csv(onset_events, sep='\t')
        func_run = load_img(func)
        # desired confound variables
        confounds_file = confounds_file[num_dummy:]
        regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']
        # regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter',
        #                    'csf_derivative1', 'csf_derivative1_power2', 'csf_power2', 'white_matter_derivative1',
        #                    'white_matter_power2', 'white_matter_derivative1_power2', 'csf_wm']  # 15

        # regressor_names = ['trans_x', 'trans_x_derivative1', 'trans_x_derivative1_power2', 'trans_x_power2',
        #                    'trans_y', 'trans_y_derivative1', 'trans_y_derivative1_power2', 'trans_y_power2',
        #                    'trans_z', 'trans_z_derivative1', 'trans_z_derivative1_power2', 'trans_z_power2',
        #                    'rot_x', 'rot_x_derivative1', 'rot_x_derivative1_power2', 'rot_x_power2',
        #                    'rot_y', 'rot_y_derivative1', 'rot_y_derivative1_power2', 'rot_y_power2',
        #                    'rot_z', 'rot_z_derivative1', 'rot_z_derivative1_power2', 'rot_z_power2']
        # set all the nan as mean value in confound variables
        def replace_nan(regressor_values):
            # calculate the mean value of the regressor:
            mean_value = regressor_values.mean(skipna=True)
            # replace all values containing nan with the mean value:
            regressor_values[regressor_values.isnull()] = mean_value
            # return list of the regressor values:
            return list(regressor_values)
        # create a nested list with regressor values
        regressors = [replace_nan(confounds_file[conf]) for conf in regressor_names]
        # parameters of design matrix
        n_scans = func_run.shape[-1]
        frame_times = np.arange(n_scans) * time_repetition  # here are the corresponding frame times
        # confound variables
        motion = np.transpose(np.array(regressors))
        add_reg_names = regressor_names
        # rest reactivation onset probability variable
        name = ['reactivation_onset_%s' % cons]
        matrix = np.hstack((np.array(onset_file), motion))
        #combine the two parts
        name += add_reg_names
        design_matrix = pd.DataFrame(matrix, columns=name, index=frame_times)
        # plot the correlation matrix between variables
        # corrplot(data=design_matrix, sub=sub)
        # check_design_matrix(design_matrix)
        # plot_design_matrix(design_matrix)
        func_runs.append(func_run)
        design_matrices.append(design_matrix)

    # fit first level glm to estimate mean orientation
    def fmrifit(func, dm):
        mni_mask = r'/home/huangqi/Data/BIDS/sourcedata/glm_l1_masks/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
        fmri_glm = FirstLevelModel(t_r=1.3, slice_time_ref=0.5, hrf_model='spm',
                                   drift_model=None, high_pass=1 / 128,
                                   mask_img=mni_mask,
                                   smoothing_fwhm=6, verbose=0, n_jobs=-2,
                                   # standardize=True, signal_scaling=False,
                                   noise_model='ar1', minimize_memory=True)
        fmri_glm = fmri_glm.fit(run_imgs=func, design_matrices=dm)
        return fmri_glm

    fmri_glm_pre = fmrifit(func=func_runs[0], dm=design_matrices[0])
    fmri_glm_post = fmrifit(func=func_runs[1], dm=design_matrices[1])

    # pre-resting contrast
    contrasts = {'reactivation_pre': [], } #contrast name
    design_matrix = design_matrices[0]
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])
    reactivation_pre = basic_contrasts['reactivation_onset_pre'] # it is the regressor name, to form the contrast name
    for contrast_id in ['reactivation_pre']: #contrast name
        contrasts[contrast_id].append(eval(contrast_id))
    print('sub-%s Computing contrasts...')
    datasink = opj(path_glm_l1, sub)
    if not os.path.exists(datasink):
        os.makedirs(datasink)
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        # Estimate the contrasts
        stats_map = fmri_glm_pre.compute_contrast(contrast_val, stat_type='t', output_type='all')
        c_map = stats_map['effect_size']
        t_map = stats_map['stat']
        z_map = stats_map['z_score']

        c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)

        t_image_path = opj(datasink, '%s_tmap.nii.gz' % contrast_id)
        t_map.to_filename(t_image_path)

        z_image_path = opj(datasink, '%s_zmap.nii.gz' % contrast_id)
        z_map.to_filename(z_image_path)

        reactivation_beta_list.append(c_map)

    contrasts = {'reactivation_post': [], }
    # pre-resting contrast
    design_matrix = design_matrices[1]
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])
    reactivation_post = basic_contrasts['reactivation_onset_post']
    for contrast_id in ['reactivation_post']:
        contrasts[contrast_id].append(eval(contrast_id))
    print('sub-%s Computing contrasts...' % subject)
    datasink = opj(path_glm_l1, sub)
    if not os.path.exists(datasink):
        os.makedirs(datasink)
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        # Estimate the contrasts
        stats_map = fmri_glm_post.compute_contrast(contrast_val, stat_type='t', output_type='all')
        c_map = stats_map['effect_size']
        t_map = stats_map['stat']
        z_map = stats_map['z_score']

        c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)

        t_image_path = opj(datasink, '%s_tmap.nii.gz' % contrast_id)
        t_map.to_filename(t_image_path)

        z_image_path = opj(datasink, '%s_zmap.nii.gz' % contrast_id)
        z_map.to_filename(z_image_path)

        reactivation_beta_list.append(c_map)

    return reactivation_beta_list
    # return ms_beta_list

warnings.filterwarnings("ignore", category=FutureWarning)
beta_list = Parallel(n_jobs=33)(delayed(l1_GLM_pm)(subject) for subject in sub_list)

# In[]:
# get different condition beta maps
pre = [beta_list[i][0] for i in range(len(sub_list))]
post = [beta_list[i][1] for i in range(len(sub_list))]

# simple t-test
for (con, beta_map) in zip(['pre', 'post'], [pre, post]):
    second_level_input = beta_map
    design_matrix_l2 = pd.DataFrame(
        [1] * len(second_level_input),
        columns=['intercept'],
        )

    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(
        second_level_input,
        design_matrix=design_matrix_l2,
    )
    # z_map = second_level_model.compute_contrast(output_type='z_score')
    # z_image_path = opj(path_glm_l2, '%s_zmap.nii.gz' % con)
    # z_map.to_filename(z_image_path)
    t_map = second_level_model.compute_contrast(second_level_stat_type='t', output_type='stat')
    t_image_path = opj(path_glm_l2, '%s_rea_tmap.nii.gz' % con)
    t_map.to_filename(t_image_path)

    t_map_thres, threshold = threshold_stats_img(stat_img=t_map,
                                                 alpha=0.01,
                                                 height_control='fdr',
                                                 cluster_threshold=0,
                                                 two_sided=True)
    t_thres_image_path = opj(path_glm_l2, '%s_rea_tmap_thres.nii.gz' % con)
    t_map_thres.to_filename(t_thres_image_path)


#paired-wise t-test
from nilearn.plotting import plot_design_matrix
from matplotlib import pyplot as plt
second_level_input = pre + post
n_sub = len(sub_list)
condition_effect = np.hstack(([-1] * n_sub, [1] * n_sub))
subject_effect = np.vstack((np.eye(n_sub), np.eye(n_sub)))
subjects = [f'S{i:02d}' for i in range(1, n_sub + 1)]

paired_design_matrix = pd.DataFrame(
    np.hstack((condition_effect[:, np.newaxis], subject_effect)),
    columns=['Post vs Pre'] + subjects)

second_level_model_paired = SecondLevelModel().fit(
    second_level_input, design_matrix=paired_design_matrix)

t_maps_paired = second_level_model_paired.compute_contrast('Post vs Pre',
                                                           second_level_stat_type='t',
                                                           output_type='stat')
t_image_path = opj(path_glm_l2, '%s_rea_paired-tmap_post-pre.nii.gz' % con)
t_maps_paired.to_filename(t_image_path)

t_maps_paired_thres, threshold = threshold_stats_img(stat_img=t_maps_paired,
                                                     alpha=0.01,
                                                     height_control='fdr',
                                                     cluster_threshold=0,
                                                     two_sided=True)
t_thres_image_path = opj(path_glm_l2, '%s_rea_paired-tmap_thres_post-pre.nii.gz' % con)
t_maps_paired_thres.to_filename(t_thres_image_path)

end_time = time.time()
run_time = round((end_time - start_time) / 60, 2)
print(f"Run time cost {run_time}")