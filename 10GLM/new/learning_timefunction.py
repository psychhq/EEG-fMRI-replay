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
import copy
import warnings
import numpy as np
import pandas as pd
import operator
import statsmodels.api as sm
import numpy as np
from os.path import join as opj
from joblib import Parallel, delayed
from bids.layout import BIDSLayout
from matplotlib import pyplot as plt
import math
# import Nilearn
from nilearn import masking
from nilearn.image import load_img, index_img, threshold_img, resample_to_img, new_img_like
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix, check_design_matrix, FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img
from nilearn import plotting

warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")

# In[SET PATHS AND SUBJECTS]:
# initialize empty paths:
ana_conditions = 'learning_single_trial'
# path to the project root:
path_bids = opj(os.getcwd().split('fmrireplay')[0], 'fmrireplay')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_behavior = opj(path_bids, 'derivatives', 'behavior', 'sub_level')
path_onsets = opj(path_bids, 'derivatives', 'replay_onsets', 'sub_level')
path_mask = opj(path_bids, 'sourcedata/func_masks')
path_glm_l1 = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'level1')
path_glm_l2 = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'level2')
path_glm_masked = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'masked_results')
for path in [path_glm_l1, path_glm_l2, path_glm_masked]:
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


# In[5]:

# set all the nan as mean value in confound variables
def replace_nan(regressor_values):
    # calculate the mean value of the regressor:
    mean_value = regressor_values.mean(skipna=True)
    # replace all values containing nan with the mean value:
    regressor_values[regressor_values.isnull()] = mean_value
    # return list of the regressor values:
    return list(regressor_values)


def get_reg_index(design_matrix, target_name):
    target_index = []
    for i, reg_name in enumerate(design_matrix.columns):
        if target_name in reg_name:
            target_index.append(i)
    if len(target_index) == 0:
        print("The {} don't have regressor.".format(target_name))
    return target_index


def set_contrasts(design_matrix):
    regressor = design_matrix.columns
    contrast_name = list(set([reg for reg in regressor if 'onset' in reg]))
    contrast_name.sort()

    # base contrast
    contrasts_set = {}
    for contrast_id in contrast_name:
        contrast_index = get_reg_index(design_matrix, contrast_id)
        if len(contrast_index) == 0:
            continue
        contrast_vector = np.zeros(design_matrix.shape[1])
        contrast_vector[contrast_index] = 1
        contrasts_set[contrast_id] = contrast_vector
    return contrasts_set


# In[GLM function]
# the number of run
run_list = [1, 2, 3]
# label and class list for event identifying
c_map_list = []
# time of repetition, in seconds:
time_repetition = 1.3
# set path templates
templates = dict(
    confounds=opj(path_fmriprep, 'sub-{subject}', 'func',
                  'sub-{subject}_task-learning_run-{run}_desc-confounds_timeseries.tsv'),
    events_learning=opj(path_behavior, 'sub-{subject}',
                        '{subject}_task-learning_run-{run}_events.tsv'),
    func=opj(path_fmriprep, 'sub-{subject}', 'func',
             'sub-{subject}_task-learning_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
    wholemask=opj(path_fmriprep, 'sub-{subject}', 'func',
                  'sub-{subject}_task-learning_run-{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'),
)
# mni template mask
mni_mask = opj(path_bids, 'sourcedata/glm_l1_masks/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii')


def l1_GLM_pm(subject):
    # subject = sub_list[2]
    sub = 'sub-%s' % subject
    # create the result folder
    datasink = opj(path_glm_l1, sub)
    if not os.path.exists(datasink):
        os.makedirs(datasink)
    func_runs = []
    design_matrices = []
    for i, run_id in enumerate(run_list):
        # load the data path
        confounds = templates['confounds'].format(subject=subject, run=run_id)
        events_learning = templates['events_learning'].format(subject=subject, run=str('%02.f' % run_id))
        func = templates['func'].format(subject=subject, run=run_id)
        # load the data
        confounds_file = pd.read_csv(confounds, sep='\t')
        events_learning_file = pd.read_csv(events_learning, sep='\t')
        end_time_tr = (math.ceil((events_learning_file.loc[0, 'end']) / time_repetition))
        func_run = index_img(load_img(func), slice(0, end_time_tr))
        n_scans = func_run.shape[-1]
        # confound variables
        regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']
        regressors = [replace_nan(confounds_file[conf]) for conf in regressor_names]
        # event types and onset for learning sessions
        events = events_learning_file.melt(id_vars=['duration', 'trial'], value_vars=['onset_a', 'onset_b'],
                                           var_name='image_type', value_name='onset')
        events['trial_type'] = pd.DataFrame(
            map(lambda x, y: '%s%0.f' % (x, (y + 1)), events['image_type'], events['trial']))
        events = events.loc[:, ['trial_type', 'onset', 'duration']]
        # parameters of design matrix
        frame_times = np.arange(n_scans) * time_repetition  # here are the corresponding frame times
        motion = np.transpose(np.array(regressors))[0:end_time_tr, :]
        design_matrix = make_first_level_design_matrix(
            frame_times, events, drift_model=None,
            add_regs=motion, add_reg_names=regressor_names,
            hrf_model='spm', oversampling=100)
        # output the data
        design_matrices.append(design_matrix)
        func_runs.append(func_run)

    # check the design matrix if needed
    # plot_design_matrix(design_matrices[0], output_file=opj(path_glm_l1, 'design_matrix_example_1.png'), rescale=True)

    # fit first level glm
    for design_matrix, func_run in zip(design_matrices, func_runs):
        fmri_glm = FirstLevelModel(t_r=time_repetition, slice_time_ref=0.5, hrf_model='spm',
                                   drift_model=None, high_pass=1 / 128, n_jobs=-1,
                                   mask_img=mni_mask, smoothing_fwhm=6, verbose=0,
                                   noise_model='ar1', minimize_memory=True)
        fmri_glm = fmri_glm.fit(run_imgs=func_run, design_matrices=design_matrix)
        # set the contrasts for each trial
        contrasts = set_contrasts(design_matrix)
        # compute contrast
        for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
            # Estimate the contrasts. Note that the model implicitly computes a fixed effect across the three sessions
            stats_map = fmri_glm.compute_contrast(contrast_val, stat_type='t', output_type='all')
            c_map = stats_map['effect_size']
            t_map = stats_map['stat']
            c_map.to_filename(opj(datasink, '%s_cmap.nii.gz' % contrast_id))
            t_map.to_filename(opj(datasink, '%s_tmap.nii.gz' % contrast_id))
            c_map_list.append(c_map)

    return c_map_list


# run parallel functions
beta_list = Parallel(n_jobs=64)(delayed(l1_GLM_pm)(subject) for subject in sub_list)

# In[LEVEL 2 GLM AND THRESHOLDED THE T-MAP]:
mni_template = load_img(mni_mask)
hippocampus_mask = load_img(opj(path_mask, 'hippo_bin.nii'))
hippo_res_mask = resample_to_img(hippocampus_mask, mni_template, interpolation='nearest')


def hippo_line(i):
    image_a_list = beta_list[i][0:8] + beta_list[i][18:26] + beta_list[i][36:44]
    image_b_list = beta_list[i][9:17] + beta_list[i][27:35] + beta_list[i][45:53]

    hippo_a_beta = np.array([np.mean(masking.apply_mask(image_a_list[j], mask_hippocampus_thres)) for j in range(24)])
    hippo_b_beta = np.array([np.mean(masking.apply_mask(image_b_list[j], mask_hippocampus_thres)) for j in range(24)])

    return hippo_a_beta, hippo_b_beta


hippo_beta_list = Parallel(n_jobs=64)(delayed(hippo_line)(i) for i in range(len(sub_list)))
# beta value matrix
hippo_a_beta_array = np.transpose(np.array([hippo_beta_list[i][0] for i in range(len(sub_list))]))
hippo_b_beta_array = np.transpose(np.array([hippo_beta_list[i][1] for i in range(len(sub_list))]))

hippo_a_beta_df = pd.DataFrame([hippo_beta_list[i][0] for i in range(len(sub_list))]).reset_index(drop=False)
hippo_b_beta_df = pd.DataFrame([hippo_beta_list[i][1] for i in range(len(sub_list))]).reset_index(drop=False)

a_df_melt = pd.melt(hippo_a_beta_df, id_vars='index', value_vars=np.arange(0, 24),
                    var_name='Time', value_name='beta_value').rename(columns={'index': 'subject'})
b_df_melt = pd.melt(hippo_b_beta_df, id_vars='index', value_vars=np.arange(0, 24),
                    var_name='Time', value_name='beta_value').rename(columns={'index': 'subject'})

md = sm.MixedLM.from_formula("beta_value ~ Time", a_df_melt, groups=a_df_melt['subject'])
mdf = md.fit()
print(mdf.summary())
a_df_melt.to_csv(opj(path_glm_l2, 'image_a_beta.csv'))
b_df_melt.to_csv(opj(path_glm_l2, 'image_b_beta.csv'))
# average the beta across subject
hippo_a_beta_mean = np.mean(hippo_a_beta_array, axis=0)
hippo_b_beta_mean = np.mean(hippo_b_beta_array, axis=0)

# plot the beta value of hippocampus in learning session for first image and second image
x_value = np.arange(1, 25)
fig, ax = plt.subplots()

ax.plot(x_value, hippo_a_beta_mean, color='red')
ax.plot(x_value, hippo_b_beta_mean, color='blue')

ax.set_xlabel('trials in learning session')
ax.set_ylabel('beta value in hippocampus')

plt.show()

# plot the linear regression
x_value = sm.add_constant(x_value)

# Create a linear regression object and fit the model
model_a = sm.OLS(hippo_a_beta_array, x_value).fit()
model_b = sm.OLS(hippo_b_beta_array, x_value).fit()

# Print the model summary
print(model_a.summary())
print(model_b.summary())

import statsmodels.api as sm
import seaborn as sns
import numpy as np

# Generate some random data
X = np.arange(1, 25)  # independent variable (25 time points)
y = hippo_a_beta_array  # dependent variable (33 subjects, 25 time points)
y2 = hippo_b_beta_array
# Compute the mean and standard error across subjects at each time point
y_mean = np.mean(y, axis=0)
y_std_err = np.std(y, axis=0) / np.sqrt(y.shape[0])

y2_mean = np.mean(y2, axis=0)
y2_std_err = np.std(y2, axis=0) / np.sqrt(y2.shape[0])

# Add a constant to the independent variable
X = sm.add_constant(X)

# Create a linear regression object and fit the model
model = sm.OLS(y_mean, X).fit()
model2 = sm.OLS(y2_mean, X).fit()

# Print the model summary
print(model.summary())
print(model2.summary())

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from seaborn.utils import desaturate

# 设置字体为Arial
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
# 创建子图
fig, ax = plt.subplots(dpi=400, figsize=(8,4))
# 绘制回归图和线图
sns.regplot(x=X[:, 1], y=y_mean, color='#dd4e75', ci=95, scatter=True, scatter_kws={'s': 60}, label='First image')
sns.regplot(x=X[:, 1], y=y2_mean, color='#45bb9c', ci=95, scatter=True, scatter_kws={'s': 60}, label='Second image')
sns.lineplot(x=X[:, 1], y=model.predict(X), color='#dd4e75', label='Linear Trend of First image')
sns.lineplot(x=X[:, 1], y=model2.predict(X), color='#45bb9c', label='Linear Trend of Second image')
# 填充置信区间
# plt.fill_between(X[:,1], y_mean - y_std_err, y_mean + y_std_err, alpha=0.4, color='#dd4e75', label='95% CI')
# plt.fill_between(X[:,1], y2_mean - y2_std_err, y2_mean + y2_std_err, alpha=0.4, color='#45bb9c', label='95% CI')
sns.despine(ax=ax, top=True, right=True)
ax.tick_params(axis='both', which='major', length=6, direction='in')
ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(opj(path_glm_l2, 'learning effect.pdf'),
            format='pdf')
plt.show()
