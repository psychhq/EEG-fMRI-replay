#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: IDENTIFY THE FMRI-BASED REACTIVATION RPOBABILITY ONSETS IN REST
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITIVE NEUROSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

# In[IMPORT RELEVANT PACKAGES]:
import os
import copy
import logging
import warnings
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
from os.path import join as opj
from bids.layout import BIDSLayout
from joblib import Parallel, delayed
from pandas.core.common import SettingWithCopyWarning
# DEAL WITH ERRORS, WARNING ETC
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

# In[SETUP NECESSARY PATHS ETC]:
# path to the project root:
path_bids = opj(os.getcwd().split('fmrireplay')[0], 'fmrireplay')
path_code = opj(path_bids, 'code', 'decoding_task_rest')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_masks = opj(path_bids, 'derivatives', 'masks')
path_level1_vfl = opj(path_bids, "derivatives", "glm-vfl", "l1pipeline")
path_decoding = opj(path_bids, 'derivatives', 'decoding', 'sub_level')
path_out = opj(path_bids, 'derivatives', 'replay_onsets', 'sub_level')

for path in [path_out, path_decoding]:
    if not os.path.exists(path):
        os.makedirs(path)

# In[SETUP NECESSARY PATHS ETC]:
# load the learning sequence
learning_sequence = pd.read_csv(opj(path_bids, 'sourcedata', 'sequence.csv'), sep=',')

# define the subject id
layout = BIDSLayout(path_bids)
sub_list = sorted(layout.get_subjects())
# delete the subjects
loweeg = [45, 38, 36, 34, 28, 21, 15, 11, 5, 4, 2]
incomplete = [42, 22, 4]
extremeheadmotion = [38, 28, 16, 15, 13, 9, 5, 0]

delete_list = np.unique(
    np.hstack([loweeg, incomplete, extremeheadmotion]))[::-1]
[sub_list.pop(i) for i in delete_list]

# create a new subject 'sub-*' list
subnum_list = copy.deepcopy(sub_list)
sub_template = ["sub-"] * len(sub_list)
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]
subnum_list = list(map(int, subnum_list))

# In[IDENTIFY THE REACTIVATION PROBABILITY ONSET OF FMRI IN REST]:


def rest_rea_fmri(subject):
    # temporal prediction probability
    pred_pre = pd.read_csv(opj(path_decoding, subject, 'data','%s fMRI task-rest_run-1 decoding probability in rest.csv' % subject),sep=',')
    pred_post = pd.read_csv(opj(path_decoding, subject, 'data','%s fMRI task-rest_run-2 decoding probability in rest.csv' % subject),sep=',')
    # select useful columns
    pred_pre = pred_pre.loc[:, ['tr', 'class', 'probability']]
    pred_post = pred_post.loc[:, ['tr', 'class', 'probability']]
    # transform the data from long to wide
    pre_rest_prob = pred_pre.pivot(index='tr', columns='class', values='probability')
    post_rest_prob = pred_post.pivot(index='tr', columns='class', values='probability')
    # rename the columns
    rename_dic = {'girl': 'A', 'scissors': 'B', 'zebra': 'C', 'banana': 'D'}
    column_name_list = ['A', 'B', 'C', 'D']
    # cut the rest duration in data
    pre_rest_prob = pre_rest_prob.rename(columns=rename_dic).reindex(columns=column_name_list).reset_index()
    pre_rest_prob['probability'] = np.sum((pre_rest_prob['A'], pre_rest_prob['B'],
                                           pre_rest_prob['C'], pre_rest_prob['D']), axis=0)
    pre_rest_prob.loc[0:3, ['A', 'B', 'C', 'D', 'probability']] = 0
    pre_rest_prob.loc[235:, ['A', 'B', 'C', 'D', 'probability']] = 0
    # cut the rest duration in data
    post_rest_prob = post_rest_prob.rename(columns=rename_dic).reindex(columns=column_name_list).reset_index()
    post_rest_prob['probability'] = np.sum((post_rest_prob['A'], post_rest_prob['B'],
                                             post_rest_prob['C'], post_rest_prob['D']), axis=0)
    post_rest_prob.loc[0:3, ['A', 'B', 'C', 'D', 'probability']] = 0
    post_rest_prob.loc[235:, ['A', 'B', 'C', 'D', 'probability']] = 0
    # save the fmri reactivation probability in rest
    pre_rest_prob.to_csv(
        opj(path_out, subject, '%s_%s_rest_reactivation_fmri.csv' % (subject, 'pre')), sep=',',
        index=False)
    post_rest_prob.to_csv(
        opj(path_out, subject, '%s_%s_rest_reactivation_fmri.csv' % (subject, 'post')), sep=',',
        index=False)

    pre_rest_prob_mean = pre_rest_prob[4:235]['probability'].mean()/4
    post_rest_prob_mean = post_rest_prob[4:235]['probability'].mean()/4
    return pre_rest_prob_mean, post_rest_prob_mean


prob_list = Parallel(n_jobs=-2)(delayed(rest_rea_fmri)(subject) for subject in sub_list)

# In[the difference of fmri-based reactivation probability between pre- and post-rest]:

pre_prob_list = pd.DataFrame([prob_list[i][0] for i in range(len(sub_list))], columns=['pre-rest-fmri'])
post_prob_list = pd.DataFrame([prob_list[i][1] for i in range(len(sub_list))], columns=['post-rest-fmri'])
prob_df = pd.concat((pre_prob_list, post_prob_list), axis=1)
# calculate the difference between post and pre rest
prob_df['difference'] = prob_df.loc[:, 'pre-rest-fmri'] - prob_df.loc[:, 'post-rest-fmri']
# calculate the standardization score for difference values
prob_df['z-difference'] = stats.zscore(prob_df['difference'])
# paired t test between pre rest and post rest
paired_t_test_bwd = pg.ttest(x=prob_df['pre-rest-fmri'], y=prob_df['post-rest-fmri'], paired=True, alternative='less')
# output the csv file
prob_df.to_csv(opj(path_bids, 'derivatives', 'replay_onsets', 'rest_reativation_fmri.csv'), sep=',')
