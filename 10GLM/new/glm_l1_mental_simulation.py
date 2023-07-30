#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: GLM IN CUE REPLAY TASK
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITVIE NERUOSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY
# In[1]: import basic libraries:
import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
from os.path import join as opj
from joblib import Parallel, delayed
from bids.layout import BIDSLayout
from matplotlib import pyplot as plt
# import Nilearn
from nilearn.image import load_img
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix, check_design_matrix, FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn import plotting

warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")

# In[3]: SET PATHS AND SUBJECTS
# define paths depending on the operating system (OS) platform:
# initialize empty paths:
ana_conditions = 'raw_mental_simulation'
sub_list = None
# path to the project root:
project_name = 'fmrireplay-glm-replay-l1'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_code = opj(path_bids, 'code', 'decoding-TDLM')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_behavior = opj(path_bids, 'derivatives', 'behavior')
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
# the number of run
run_list = [1, 2, 3]
# time of repetition, in seconds:
time_repetition = 1.3
# number of dummy variables to remove from each run:
num_dummy = 0

replay_beta_list = []
ms_beta_list = []


def _orthogonalize(X):
    if X.size == X.shape[0]:
        return X
    from scipy.linalg import pinv
    for i in range(1, X.shape[1]):
        X[:, i] -= np.dot(np.dot(X[:, i], X[:, :i]), pinv(X[:, :i]))
    return X


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

# check the data files
path_confounds = glob.glob(opj(path_fmriprep, '*', 'func', '*replay*confounds_timeseries.tsv'))
path_events = glob.glob(opj(path_behavior, '*', '*task-rep*events.tsv'))
path_func = glob.glob(opj(path_fmriprep, '*', 'func', '*replay*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))
path_wholemask = glob.glob(opj(path_fmriprep, '*', 'func', '*replay*space-MNI152NLin2009cAsym*brain_mask.nii.gz'))


def l1_GLM_pm(subject):
    # subject = sub_list[0]
    sub = 'sub-%s' % subject
    # path templates
    templates = dict(
        confounds=opj(path_fmriprep, 'sub-{subject}', 'func',
                      'sub-{subject}_task-replay_run-{run}_desc-confounds_timeseries.tsv'),
        events=opj(path_behavior, 'sub-{subject}',
                   '{subject}_task-rep_run-{run}_events.tsv'),
        func=opj(path_fmriprep, 'sub-{subject}', 'func',
                 'sub-{subject}_task-replay_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
        wholemask=opj(path_fmriprep, 'sub-{subject}', 'func',
                      'sub-{subject}_task-replay_run-{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'),
    )

    func_runs = []
    design_matrices = []
    for i, run_id in enumerate(run_list):
        # load the data path
        confounds = templates['confounds'].format(subject=subject, run=run_id)
        events = templates['events'].format(subject=subject, run=str('%02.f' % run_id))
        func = templates['func'].format(subject=subject, run=run_id)
        # load the data
        confounds_file = pd.read_csv(confounds, sep='\t')
        events_file = pd.read_csv(events, sep='\t')
        func_run = load_img(func)

        # desired confound variables
        confounds_file = confounds_file[num_dummy:]
        regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']

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

        # desired mental simulation variables
        events_file['onset'] = events_file['onset'] - num_dummy * time_repetition
        events_file['trial_type'] = events_file['accuracy'].apply(lambda x: 'mental_simulation' if x == 1 else 'wrong')
        events = events_file.loc[:, ['trial_type', 'onset', 'duration']]

        scan_tr = np.size(func_run, axis=3)
        # parameters of design matrix
        n_scans = func_run.shape[-1]
        frame_times = np.arange(n_scans) * time_repetition  # here are the corresponding frame times
        motion = np.transpose(np.array(regressors))
        add_reg_names = regressor_names
        hrf_model = 'spm'
        design_matrix = make_first_level_design_matrix(
            frame_times, events, drift_model=None,
            add_regs=motion, add_reg_names=add_reg_names,
            hrf_model=hrf_model, oversampling=100)
        # check_design_matrix(design_matrix)
        # plot_design_matrix(design_matrix)

        # output the data
        design_matrices.append(design_matrix)
        func_runs.append(func_run)

    # fit first level glm to estimate mean orientation
    mni_mask = r'/home/huangqi/Data/BIDS/sourcedata/glm_l1_masks/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii'
    fmri_glm = FirstLevelModel(t_r=1.3, slice_time_ref=0.5, hrf_model='spm',
                               drift_model=None, high_pass=1 / 128,
                               mask_img=mni_mask,
                               smoothing_fwhm=6, verbose=0, n_jobs=-2,
                               # standardize=True, signal_scaling=False,
                               noise_model='ar1', minimize_memory=True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    fmri_glm = fmri_glm.fit(run_imgs=func_runs, design_matrices=design_matrices)

    contrasts = {'ms_con': [], }
    for design_matrix in design_matrices:
        contrast_matrix = np.eye(design_matrix.shape[1])
        basic_contrasts = dict([(column, contrast_matrix[i])
                                for i, column in enumerate(design_matrix.columns)])
        ms_con = basic_contrasts['mental_simulation']
        for contrast_id in ['ms_con']:
            contrasts[contrast_id].append(eval(contrast_id))

    print('Computing contrasts...')
    datasink = opj(path_glm_l1, sub)
    if not os.path.exists(datasink):
        os.makedirs(datasink)

    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('Contrast % 2i out of %i: %s' % (index + 1, len(contrasts), contrast_id))
        # Estimate the contrasts. Note that the model implicitly computes a fixed effect across the three sessions
        stats_map = fmri_glm.compute_contrast(contrast_val, stat_type='t', output_type='all')
        c_map = stats_map['effect_size']
        t_map = stats_map['stat']

        c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)

        t_image_path = opj(datasink, '%s_tmap.nii.gz' % contrast_id)
        t_map.to_filename(t_image_path)

        ms_beta_list.append(c_map)

    return ms_beta_list


warnings.filterwarnings("ignore", category=FutureWarning)
beta_list = Parallel(n_jobs=18)(delayed(l1_GLM_pm)(subject) for subject in sub_list)

# In[]:
ms_list = [beta_list[i][0] for i in range(len(sub_list))]

for (con, beta_map) in zip(['ms'], [ms_list]):
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

    t_map = second_level_model.compute_contrast(second_level_stat_type='t', output_type='stat')
    t_image_path = opj(path_glm_l2, '%s_tmap.nii.gz' % con)
    t_map.to_filename(t_image_path)

    t_map_thres, threshold = threshold_stats_img(stat_img=t_map,
                                                 alpha=0.01,
                                                 height_control='fdr',
                                                 cluster_threshold=0,
                                                 two_sided=True)
    t_thres_image_path = opj(path_glm_l2, '%s_tmap_thres.nii.gz' % con)
    t_map_thres.to_filename(t_thres_image_path)