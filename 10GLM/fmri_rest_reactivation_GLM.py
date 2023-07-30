#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: FIRST LEVEL GLM
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITIVE NEUROSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY
# In[1]: import basic libraries:
import os
import copy
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from os.path import join as opj
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from bids.layout import BIDSLayout
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
# import Nilearn
from nilearn import plotting, image, masking
from nilearn.image import load_img, resample_to_img
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix, check_design_matrix, FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm import threshold_stats_img, cluster_level_inference
# filter out warnings related to the numpy package:
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")

# In[SET PATHS AND SUBJECTS]:
# initialize empty paths:
ana_conditions = 'fmri_rest'
# path to the project root:
path_bids = opj(os.getcwd().split('fmrireplay')[0], 'fmrireplay')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_behavior = opj(path_bids, 'derivatives', 'behavior', 'sub_level')
path_onsets = opj(path_bids, 'derivatives', 'replay_onsets', 'sub_level')
path_mask = opj(path_bids, 'sourcedata/func_masks')
path_glm_l1 = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'level1')
path_glm_l2 = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'level2')
path_glm_masked = opj(path_bids, 'derivatives', 'glm', ana_conditions, 'masked_results')
path_glm_masked_eeg = opj(path_bids, 'derivatives', 'glm', 'eeg_rest', 'masked_results')
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


# In[SET FUNCTIONS]:
# set all the nan as mean value in confound variables
def replace_nan(regressor_values):
    # calculate the mean value of the regressor:
    mean_value = regressor_values.mean(skipna=True)
    # replace all values containing nan with the mean value:
    regressor_values[regressor_values.isnull()] = mean_value
    # return list of the regressor values:
    return list(regressor_values)


# In[LEVEL 1 GLM FUNCTION -- rest reactivation probability]:
# time of repetition, in seconds:
time_repetition = 1.3
# the number of run and onset list
run_list = [1, 2]
cons_list = ['pre', 'post']
# list
reactivation_beta = []
# path templates
templates = dict(
    confounds=opj(path_fmriprep, 'sub-{subject}', 'func',
                  'sub-{subject}_task-rest_run-{run}_desc-confounds_timeseries.tsv'),
    onset_events=opj(path_onsets, 'sub-{subject}',
                     'sub-{subject}_{cons}_rest_reactivation_fmri.csv'),
    func=opj(path_fmriprep, 'sub-{subject}', 'func',
             'sub-{subject}_task-rest_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
)
# fit first level glm to estimate mean orientation
mni_mask = opj(path_bids, 'sourcedata/glm_l1_masks/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii')


def l1_GLM_fMRI(subject):
    sub = 'sub-%s' % subject
    datasink = opj(path_glm_l1, sub)
    if not os.path.exists(datasink):
        os.makedirs(datasink)
    # set lists
    func_runs = []
    design_matrices = []
    # design matrix
    for cons, run_id in zip(cons_list, run_list):
        # load the data path
        confounds = templates['confounds'].format(subject=subject, run=run_id)
        onset_events = templates['onset_events'].format(subject=subject, cons=cons)
        func = templates['func'].format(subject=subject, run=run_id)
        # load the data
        func_run = load_img(func)
        n_scans = func_run.shape[-1]
        confounds_file = pd.read_csv(confounds, sep='\t')
        onset_file = pd.read_csv(onset_events, sep=',')
        # desired confound variables
        regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']
        # create a nested list with regressor values
        regressors = [replace_nan(confounds_file[conf]) for conf in regressor_names]
        # confound variables
        motion = np.transpose(np.array(regressors))
        # frame_times for scan
        frame_times = np.arange(n_scans) * time_repetition  # here are the corresponding frame times
        design_matrix = make_first_level_design_matrix(
            frame_times, events=None, drift_model=None,
            add_regs=motion, add_reg_names=regressor_names,
            hrf_model='spm', oversampling=100)
        # concatenate the HRF-Convolved probability regressor
        onset_file.set_index(frame_times, inplace=True, drop=True)
        design_matrix = pd.concat((onset_file['probability'], design_matrix), axis=1)
        name = {'probability': 'reactivation_onset_%s' % cons, }
        design_matrix = design_matrix.rename(columns=name)
        plot_design_matrix(design_matrix)
        func_runs.append(func_run)
        design_matrices.append(design_matrix)

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

    # pre-rest contrast
    contrasts = {'reactivation_pre': [], }  # contrast name
    design_matrix = design_matrices[0]
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])
    reactivation_pre = basic_contrasts['reactivation_onset_pre']  # it is the regressor name, to form the contrast name
    for contrast_id in ['reactivation_pre']:  # contrast name
        contrasts[contrast_id].append(eval(contrast_id))
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        # Estimate the contrasts
        stats_map = fmri_glm_pre.compute_contrast(contrast_val, stat_type='t', output_type='all')
        c_map = stats_map['effect_size']
        c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)
        reactivation_beta.append(c_map)

    # post-rest contrast
    contrasts = {'reactivation_post': [], }
    design_matrix = design_matrices[1]
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])
    reactivation_post = basic_contrasts['reactivation_onset_post']
    for contrast_id in ['reactivation_post']:
        contrasts[contrast_id].append(eval(contrast_id))
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        # Estimate the contrasts
        stats_map = fmri_glm_post.compute_contrast(contrast_val, stat_type='t', output_type='all')
        c_map = stats_map['effect_size']
        c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)
        reactivation_beta.append(c_map)
    return reactivation_beta


# run parallel functions
beta_list = Parallel(n_jobs=64)(delayed(l1_GLM_fMRI)(subject) for subject in sub_list)

# In[LEVEL 2 GLM]:

# get different condition beta maps
pre_fmri = [beta_list[i][0] for i in range(len(sub_list))]
post_fmri = [beta_list[i][1] for i in range(len(sub_list))]

# # 2mm brain template
mni_template = image.load_img(mni_mask)
# # all important brain region masks
func_mask = image.load_img(opj(path_mask, 'MTL', 'MTL_resample.nii'))
func_res_mask = resample_to_img(func_mask, mni_template, interpolation='nearest')
func_res_mask.to_filename(opj(path_glm_l2, 'mni_resample.nii'))

# RAW T-MAP
for (con, beta_map) in zip(['pre', 'post'], [pre_fmri, post_fmri]):
    second_level_input = beta_map
    design_matrix_l2 = pd.DataFrame(
        [1] * len(second_level_input),
        columns=['intercept'], )
    # second level model
    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(second_level_input, design_matrix=design_matrix_l2, )
    # t-map of second level model
    t_map = second_level_model.compute_contrast(second_level_stat_type='t', output_type='stat')
    t_image_path = opj(path_glm_l2, '%s_rest_fmri_tmap.nii.gz' % con)
    t_map.to_filename(t_image_path)

    # MULTIPLE COMPARISON
    # voxel-level multiple comparison
    t_map_thres, threshold = threshold_stats_img(stat_img=t_map, mask_img=func_res_mask,
                                                 threshold=3.365, height_control=None,
                                                 two_sided=False, cluster_threshold=10, )
    t_thres_image_path = opj(path_glm_l2, '%s_rest_fmri_tmap_thres.nii.gz' % con)
    t_map_thres.to_filename(t_thres_image_path)

# In[Paired comparison between pre- and post-rest]
# pair-wise t-test of pre- and post-rest
second_level_input = pre_fmri + post_fmri
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
t_image_path = opj(path_glm_l2, 'paired_rest_fmri_rea-tmap.nii.gz')
t_map_paired.to_filename(t_image_path)

# MULTIPLE COMPARISON
# voxel-level multiple comparison

t_map_paired_thres, threshold = threshold_stats_img(stat_img=t_map_paired, threshold=3.365, mask_img=func_res_mask,
                                                    height_control=None, two_sided=False, cluster_threshold=0, )
print('post-pre threshold:', threshold)
t_paired_thres_image_path = opj(path_glm_l2, 'paired_rest_fmri_tmap_thres.nii.gz')
t_map_paired_thres.to_filename(t_paired_thres_image_path)

# In[ROI ANALYSIS(HIPPOCAMPUS AND PRIMARY MOTOR CORTEX)]:

# probability hippocampus brain masks
hippocampus_mask = image.load_img(opj(path_mask, 'hippocampus', 'Hippocampus.nii'))
hippo_res_mask = resample_to_img(hippocampus_mask, mni_template, interpolation='nearest')

# probability primary motor cortex brain masks
pmotor_mask = image.load_img((opj(path_mask, 'motor', 'Primary_motor_cortex.nii')))
pmotor_res_mask = resample_to_img(pmotor_mask, mni_template, interpolation='nearest')

# explore the results based on different probability threshold
# def roi_analysis(prob_thres):
prob_thres = 1
# transform to the matrix
mask_hippocampus = np.asanyarray(copy.deepcopy(hippo_res_mask).dataobj)
mask_motor = np.asanyarray(copy.deepcopy(pmotor_res_mask).dataobj)
# replace all other values with 1:
mask_hippocampus = np.where(mask_hippocampus >= prob_thres, 100, mask_hippocampus)
mask_hippocampus = np.where(mask_hippocampus < prob_thres, 0, mask_hippocampus)
# turn the 3D-array into booleans:
mask_hippocampus = mask_hippocampus.astype(bool)
# create image like object:
mask_hippocampus_thres = image.new_img_like(mni_template, mask_hippocampus)

# replace all other values with 1:
mask_motor = np.where(mask_motor >= prob_thres, 100, mask_motor)
mask_motor = np.where(mask_motor < prob_thres, 0, mask_motor)
# turn the 3D-array into booleans:
mask_motor = mask_motor.astype(bool)
# create image like object:
mask_motor_thres = image.new_img_like(mni_template, mask_motor)

# extract hippocampus beta value from beta map
beta_hippo_pre = [np.mean(masking.apply_mask(beta_list[i][0], mask_hippocampus_thres)) for i in
                  range(len(beta_list))]
beta_hippo_post = [np.mean(masking.apply_mask(beta_list[i][1], mask_hippocampus_thres)) for i in
                   range(len(beta_list))]

# pair-wise t test in hippocampus
hippo_t_pre = pg.ttest(x=beta_hippo_pre, y=0, paired=False, alternative='two-sided')
hippo_t_post = pg.ttest(x=beta_hippo_post, y=0, paired=False, alternative='two-sided')
hippo_t_test = pg.ttest(x=beta_hippo_post, y=beta_hippo_pre, paired=True, alternative='two-sided')

print("Hippocampus:", prob_thres, hippo_t_pre.loc['T-test', 'T'], hippo_t_pre.loc['T-test', 'p-val'])
print("Hippocampus:", prob_thres, hippo_t_post.loc['T-test', 'T'], hippo_t_post.loc['T-test', 'p-val'])
print("Hippocampus:", prob_thres, hippo_t_test.loc['T-test', 'T'], hippo_t_test.loc['T-test', 'p-val'])

# extract motor cortex beta value from beta map
beta_motor_pre = [np.mean(masking.apply_mask(beta_list[i][0], mask_motor_thres)) for i in range(len(beta_list))]
beta_motor_post = [np.mean(masking.apply_mask(beta_list[i][1], mask_motor_thres)) for i in range(len(beta_list))]
# pair-wise t test in motor cortex
motor_t_pre = pg.ttest(x=beta_motor_pre, y=0, paired=False, alternative='two-sided')
motor_t_post = pg.ttest(x=beta_motor_post, y=0, paired=False, alternative='two-sided')
motor_t_test = pg.ttest(x=beta_motor_post, y=beta_motor_pre, paired=True, alternative='two-sided')
print("Motor cortex:", prob_thres, motor_t_pre.loc['T-test', 'T'], motor_t_pre.loc['T-test', 'p-val'])
print("Motor cortex:", prob_thres, motor_t_post.loc['T-test', 'T'], motor_t_post.loc['T-test', 'p-val'])
print("Motor cortex:", prob_thres, motor_t_test.loc['T-test', 'T'], motor_t_test.loc['T-test', 'p-val'])

# prob_thres==1 is the best result
if prob_thres == 1:
    # hippocampus
    df_beta_hippo_pre = pd.DataFrame(beta_hippo_pre, columns=['pre_rest'])
    df_beta_hippo_post = pd.DataFrame(beta_hippo_post, columns=['post_rest'])
    beta_hippocampus = pd.concat((df_beta_hippo_pre, df_beta_hippo_post), axis=1)
    beta_hippocampus = beta_hippocampus.melt(value_vars=['pre_rest', 'post_rest'],
                                             var_name='session', value_name='beta_value')
    beta_hippocampus.to_csv(opj(path_glm_masked, 'beta_hippocampus_rest_fmri_rea.csv'))
    # primary motor cortex
    df_beta_motor_pre = pd.DataFrame(beta_motor_pre, columns=['pre_rest'])
    df_beta_motor_post = pd.DataFrame(beta_motor_post, columns=['post_rest'])
    beta_pmotor = pd.concat((df_beta_motor_pre, df_beta_motor_post), axis=1)
    beta_pmotor = beta_pmotor.melt(value_vars=['pre_rest', 'post_rest'],
                                   var_name='session', value_name='beta_value')
    beta_pmotor.to_csv(opj(path_glm_masked, 'beta_pmotor_rest_fmri_rea.csv'))

# Parallel(n_jobs=64)(delayed(roi_analysis)(prob_thres) for prob_thres in range(1, 100))

# In[Statistical inference of ROI analysis]:
# load the data
beta_hippocampus = pd.read_csv(opj(path_glm_masked, 'beta_hippocampus_rest_fmri_rea.csv'))
beta_hippocampus['id'] = np.hstack((np.arange(0, 33), np.arange(0, 33)))
beta_pmotor = pd.read_csv(opj(path_glm_masked, 'beta_pmotor_rest_fmri_rea.csv'))
beta_pmotor['id'] = np.hstack((np.arange(0, 33), np.arange(0, 33)))
# pivot the df
beta_hippocampus = pd.pivot(beta_hippocampus, index='id', columns='session', values='beta_value')
beta_pmotor = pd.pivot(beta_pmotor, index='id', columns='session', values='beta_value')
# one sample t-test for hippocampus
t_test_hippo_pre = pg.ttest(x=beta_hippocampus.loc[0:34, 'pre_rest'],
                            y=0, paired=False, alternative='two-sided')
t_test_hippo_post = pg.ttest(x=beta_hippocampus.loc[0:34, 'post_rest'],
                             y=0, paired=False, alternative='two-sided')
# paired t test
t_test_hippo = pg.ttest(x=beta_hippocampus.loc[0:34, 'post_rest'],
                        y=beta_hippocampus.loc[0:34, 'pre_rest'],
                        paired=True, alternative='two-sided')
t_test_pmotor = pg.ttest(x=beta_pmotor.loc[0:34, 'post_rest'],
                         y=beta_pmotor.loc[0:34, 'pre_rest'],
                         paired=True, alternative='two-sided')

# In[THE RELATIONSHIP BETWEEN BETA VALUE OF EEG AND FMRI IN PRE- AND POST-REST IN HIPPOCAMPUS ROI]
# load the beta values from EEG and fMRI
eeg_beta = pd.read_csv(opj(path_glm_masked_eeg, 'beta_hippocampus_rest_eeg_rea.csv'))
fmri_beta = pd.read_csv(opj(path_glm_masked, 'beta_hippocampus_rest_fmri_rea.csv'))
# rename the columns
eeg_beta = eeg_beta.rename(columns={"Unnamed: 0": 'id', 'beta_value': 'EEG'})
fmri_beta = fmri_beta.rename(columns={"Unnamed: 0": 'id', 'beta_value': 'fMRI'})
# concatnate the two dataframes
eeg_fmri_roi = pd.merge(eeg_beta, fmri_beta, on=['id', 'session'])
eeg_fmri_roi['id'] = np.hstack((np.arange(0, 33), np.arange(0, 33)))
# melt the dataframes
eeg_fmri_roi_2 = eeg_fmri_roi.melt(id_vars=['id', 'session'], value_vars=['EEG', 'fMRI'],
                                   var_name='EEG_fMRI', value_name='beta_value')
eeg_fmri_roi_2['id'] = np.hstack((np.arange(0, 33), np.arange(0, 33), np.arange(0, 33), np.arange(0, 33)))
# plot the ROI
# point plot
sns.catplot(data=eeg_fmri_roi_2, x="EEG_fMRI", y="beta_value", hue="session",
            palette={"pre_rest": "lightcoral", "post_rest": "cornflowerblue"},
            markers=["^", "o"], linestyles=["-", "--"], dodge=True,
            kind="point")
plt.savefig(opj(path_bids, 'derivatives', 'glm', 'ROI_analysis', 'EEG-fMRI_ROI_point.svg'))
# bar plot
sns.catplot(data=eeg_fmri_roi_2, x="session", y="beta_value", hue="EEG_fMRI",
            palette={"EEG": "lightcoral", "fMRI": "cornflowerblue"},
            kind="bar", errorbar='se')
plt.savefig(opj(path_bids, 'derivatives', 'glm', 'ROI_analysis', 'EEG-fMRI_ROI_bar.svg'))
plt.show()
# calculate the 2 level repeated measurement ANOVA
model = AnovaRM(eeg_fmri_roi_2, 'beta_value', 'id', within=['session', 'EEG_fMRI'])
results = model.fit()
print(results)
# another way to 2*2 ANOVA analysis
aov = pg.rm_anova(eeg_fmri_roi_2, dv='beta_value', within=['EEG_fMRI', 'session'], subject='id', effsize='np2')
print(aov)
# linear model to 2*2 analysis
model = ols('beta_value ~ C(session) + C(EEG_fMRI) + C(EEG_fMRI)*C(session) + C(id)', data=eeg_fmri_roi_2).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(model.summary())
# Conduct simple effect analysis for factor1 at level X of factor2
se_eeg_session = pg.rm_anova(eeg_fmri_roi_2[eeg_fmri_roi_2['EEG_fMRI'] == 'EEG'], dv='beta_value',
                             within=['session'], subject='id')
se_fmri_session = pg.rm_anova(eeg_fmri_roi_2[eeg_fmri_roi_2['EEG_fMRI'] == 'fMRI'], dv='beta_value',
                              within=['session'], subject='id')
se_pre_modality = pg.rm_anova(eeg_fmri_roi_2[eeg_fmri_roi_2['session'] == 'pre_rest'], dv='beta_value',
                              within=['EEG_fMRI'], subject='id')
se_post_modality = pg.rm_anova(eeg_fmri_roi_2[eeg_fmri_roi_2['session'] == 'post_rest'], dv='beta_value',
                               within=['EEG_fMRI'], subject='id')
# output the results
print('pre-post rest in EEG level: F = %.3f, p = %.3f; \n '
      'pre-post rest in fMRI level: F = %.3f, p = %.3f; \n '
      'EEG-fMRI in pre rest level: F = %.3f, p = %.3f; \n '
      'EEG-fMRI in post rest level: F = %.3f, p = %.3f' % (
          se_eeg_session['F'], se_eeg_session['p-unc'],
          se_fmri_session['F'], se_fmri_session['p-unc'],
          se_pre_modality['F'], se_pre_modality['p-unc'],
          se_post_modality['F'], se_post_modality['p-unc']))
# Compute the robust correlation coefficient between EEG and fMRI in post rest
corr_post = pg.corr(x=fmri_beta[fmri_beta['session'] == 'post_rest']['fMRI'],
                    y=eeg_beta[eeg_beta['session'] == 'post_rest']['EEG'],
                    alternative='two-sided', method="percbend")
print(corr_post)
# Compute the robust correlation coefficient between EEG and fMRI in pre rest
corr_pre = pg.corr(x=fmri_beta[fmri_beta['session'] == 'pre_rest']['fMRI'],
                   y=eeg_beta[eeg_beta['session'] == 'pre_rest']['EEG'],
                   alternative='two-sided', method="percbend")
print(corr_pre)
