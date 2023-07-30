# -*- coding: utf-8 -*-
"""

# # SCRIPT: DECODING CORRELATION BETWEEN EEG AND FMRI
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITIVE NEUROSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

"""
# In[IMPORT RELEVANT PACKAGES]:
import os
import glob
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy import stats
from scipy.stats import gamma
from os.path import join as opj
from scipy.stats import pearsonr
from bids.layout import BIDSLayout
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from pandas.core.common import SettingWithCopyWarning
from statsmodels.stats.multitest import multipletests

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# In[SETUP NECESSARY PATHS ETC]:
# path to the project root:
path_bids = opj(os.getcwd().split('fmrireplay')[0], 'fmrireplay')
path_code = opj(path_bids, 'code', 'decoding_task_rest')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_behavior_input = opj(path_bids, 'sourcedata', 'behavior-rawdata')
path_decoding = opj(path_bids, 'derivatives', 'decoding', 'VFL_decoding')
path_cue_eeg = opj(path_bids, 'sourcedata', 'replay_onsets', 'cue_replay_prob')
path_pre_rest = opj(path_bids, 'sourcedata', 'replay_onsets', 'pre_rest_replay_onsets')
path_post_rest = opj(path_bids, 'sourcedata', 'replay_onsets', 'post_rest_replay_onsets')
path_decode_fmri = opj(path_bids, 'derivatives', 'decoding', 'sub_level')
path_decode_fmri_rest = opj(path_bids, 'derivatives', 'replay_onsets', 'sub_level')
path_output = opj(path_bids, 'derivatives', 'decoding', 'decoding_correlation')

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
subject = None
# some parameters
long_interval = 8
# the number of timelags
maxLag = long_interval
# just for plot
nbins = maxLag + 1
# the number of subjects
nSubj = len(sub_list)
# the nan matrix for forward design matrix (2 directions * 8 timelags conditions)
cross = np.full([2, maxLag], np.nan)


# In[PRESET THE FUNCTIONS]:
# scan numbers of cue replay task runs
def confound_tr(path_fmriprep, subject):
    from os.path import join as opj
    import pandas as pd
    run_list = [1, 2, 3]
    nums_scan = []
    templates = dict(
        confounds=opj(path_fmriprep, 'sub-{subject}', 'func',
                      'sub-{subject}_task-replay_run-{run}_desc-confounds_timeseries.tsv'),
    )
    for run_id in run_list:
        confounds = templates['confounds'].format(subject=subject, run=run_id)
        confounds_file = pd.read_csv(confounds, sep='\t')
        num_scan = confounds_file.shape[0]
        nums_scan.append(num_scan)
    return nums_scan


def pre_rest_start(path_behavior_input, subject):
    # get the path of VFL files
    path_behavior_vfl = opj(path_behavior_input, 'VFL', 'VFL_%s_*.csv' % subject)
    # create the path to the data file
    path_vfl_event = sorted(glob.glob(path_behavior_vfl), key=lambda f: os.path.basename(f))
    # read the visual functional localizer trial data
    df_vfl_event = pd.read_csv(path_vfl_event[0], sep=',')
    # select useful columns
    df_vfl_event = df_vfl_event[['key_resp_6.rt', 'key_resp_6.started',
                                 'text_4.started', 'subject_id']]
    abs_start_pre = float(df_vfl_event.loc[1, ['text_4.started']]) - \
                    (float(df_vfl_event.loc[0, ['key_resp_6.rt']]) +
                     float(df_vfl_event.loc[0, ['key_resp_6.started']]))
    return abs_start_pre


def post_rest_start(path_behavior_input, subject):
    # load the behavioral data file
    behavior_path = opj(path_behavior_input, 'replay', 'cue_replay_%s_*.csv' % subject)
    # create the path to the data file
    behavior_path = sorted(glob.glob(behavior_path), key=lambda f: os.path.basename(f))
    # read the visual functional localizer trial data
    behavior_file = pd.read_csv(behavior_path[0], sep=',')
    # select useful columns
    behavior_file = behavior_file[['key_resp_10.rt', 'key_resp_10.started',
                                   'polygon_fixation_rest.started']]
    # post resting start timepoint
    abs_start_post = float(behavior_file.loc[3, ['polygon_fixation_rest.started']]) - \
                     (float(behavior_file.loc[1, ['key_resp_10.rt']]) +
                      float(behavior_file.loc[1, ['key_resp_10.started']]))
    return abs_start_post


def behavior_start(path_behavior_input, subject):
    sub = 'sub-%s' % subject
    # load the behavioral data file
    behavior_path = opj(path_behavior_input, 'replay', 'cue_replay_%s_*.csv' % subject)
    # create the path to the data file
    behavior_path = sorted(glob.glob(behavior_path), key=lambda f: os.path.basename(f))
    # read the visual functional localizer trial data
    behavior_file = [pd.read_csv(f, sep=',') for f in behavior_path]
    # select useful columns
    behavior_file = [
        behavior_file[f][['runs_replay.thisRepN', 'trials_replay.thisN', 'text.started',
                          'text_cue_replay.started']] for f in range(len(behavior_path))]
    # delete unrelated rows by NaN value
    behavior_file = [
        behavior_file[f].dropna(axis=0, how='any', subset=['text_cue_replay.started']).reset_index(drop=True)
        for f in range(len(behavior_path))]
    # rename the columns(variables)
    rename_beh = {'runs_replay.thisRepN': 'run',
                  'trials_replay.thisN': 'trials',
                  'text.started': 'start',
                  'text_cue_replay.started': 'onset'}
    behavior_file = [behavior_file[f].rename(columns=rename_beh) for f in range(len(behavior_path))]
    # set some new values for all first session
    behavior_file[0]['session'] = 1
    behavior_file[0]['run'] = behavior_file[0]['run'] + 1
    # if one file from one subject
    if len(behavior_path) == 1:
        behavior_file = behavior_file[0]
        # calculate onset time
        onset = [behavior_file[(behavior_file['run'] == i)]['onset'].sub(
            float(behavior_file[(behavior_file['run'] == i)
                                & (behavior_file['trials'] == 0)]['start'])) for i in np.arange(1, 4)]
        onset = pd.concat([onset[0], onset[1], onset[2]], axis=0)
        behavior_file['onset'] = onset
    # if there are two files from one subject
    elif len(behavior_path) == 2:
        # set some new values for the second session
        behavior_file[1]['session'] = 2
        behavior_file[1]['run'] = behavior_file[1]['run'] + behavior_file[0]['run'].iloc[-1] + 1
        # calculate onset time for each run
        if sub == 'sub-07' or sub == 'sub-32':
            for f in range(len(behavior_path)):
                if f == 0:
                    onset1 = [behavior_file[f][(behavior_file[f]['run'] == i)]['onset'].sub(
                        float(behavior_file[f][(behavior_file[f]['run'] == i)
                                               & (behavior_file[f]['trials'] == 0)]['start'])) for i in np.arange(1, 3)]
                    onset = pd.concat([onset1[0], onset1[1]], axis=0)
                    behavior_file[f]['onset'] = onset
                elif f == 1:
                    onset2 = behavior_file[f][(behavior_file[f]['run'] == 3)]['onset'].sub(
                        float(behavior_file[f][(behavior_file[f]['run'] == 3)
                                               & (behavior_file[f]['trials'] == 0)]['start']))
                    behavior_file[f]['onset'] = onset2
        elif sub == 'sub-40':
            for f in range(len(behavior_path)):
                if f == 0:
                    onset1 = [behavior_file[f][(behavior_file[f]['run'] == i)]['onset'].sub(
                        float(behavior_file[f][(behavior_file[f]['run'] == i)
                                               & (behavior_file[f]['trials'] == 0)]['start'])) for i in np.arange(1, 2)]
                    behavior_file[f]['onset'] = onset1[0]
                elif f == 1:
                    onset2 = [behavior_file[f][(behavior_file[f]['run'] == i)]['onset'].sub(
                        float(behavior_file[f][(behavior_file[f]['run'] == i)
                                               & (behavior_file[f]['trials'] == 0)]['start'])) for i in np.arange(2, 4)]
                    onset = pd.concat([onset2[0], onset2[1]], axis=0)
                    behavior_file[f]['onset'] = onset
        temp = pd.concat([behavior_file[0], behavior_file[1]])
        behavior_file = temp
    # devide behavior event into different files by runs
    class_run_beh = behavior_file['run'].unique()
    behavior_event = [behavior_file[behavior_file['run'] == i].reset_index(drop=True) for i in class_run_beh]
    # get the absolute time onset for each run
    abs_start = np.array([behavior_event[f].loc[0, ['onset']] for f in range(len(behavior_event))])
    return abs_start


def cor(x, y):
    from scipy import stats
    correlation = stats.pearsonr(x, y)[0]
    return correlation


def cor_all(x, y):
    from scipy import stats
    correlation = stats.pearsonr(x, y)[0]
    p = stats.pearsonr(x, y)[1]
    return correlation, p


def _gamma_difference_hrf(tr, oversampling=100, time_length=32., onset=0.,
                          delay=6, undershoot=16., dispersion=1.,
                          u_dispersion=1., ratio=0.167):
    dt = tr / oversampling
    time_stamps = np.linspace(0, time_length,
                              np.rint(float(time_length) / dt).astype(int))
    time_stamps -= onset

    # define peak and undershoot gamma functions
    peak_gamma = gamma.pdf(
        time_stamps,
        delay / dispersion,
        loc=dt,
        scale=dispersion)
    undershoot_gamma = gamma.pdf(
        time_stamps,
        undershoot / u_dispersion,
        loc=dt,
        scale=u_dispersion)

    # calculate the hrf
    hrf = peak_gamma - ratio * undershoot_gamma
    hrf /= hrf.sum()
    return hrf


def spm_hrf(tr, oversampling=100, time_length=32., onset=0.):
    return _gamma_difference_hrf(tr, oversampling, time_length, onset)


def matlab_percentile(x, p):
    p = np.asarray(p, dtype=float)
    n = len(x)
    p = (p - 50) * n / (n - 1) + 50
    p = np.clip(p, 0, 100)
    return np.percentile(x, p)


# detect the autocorrelation between hippocampus and visual cortex
def TDLM_cross(probability_matrix):
    # transition direction
    rp = [1, 2]

    # transition matrix
    def TransM_cross(x):
        # create an empty matrix
        transition_matrix = np.zeros([2, 2])
        # transition
        for a in range(1):
            transition_matrix[x[a] - 1][x[a + 1] - 1] = 1
        return transition_matrix

    # get the transition matrix
    T1 = TransM_cross(rp)
    T2 = np.transpose(T1)
    # probability data
    X1 = np.array(probability_matrix)

    # detect cross-correlation for each time lag
    def Crosscorr(X1, T, l):
        orig = np.dot(X1[0:-(2 * l), :], T)
        proj = X1[l:-l, :]
        # Scale variance
        corrtemp = np.full(np.size(proj, axis=1), np.nan)
        for iseq in range(np.size(proj, axis=1)):
            if (np.nansum(orig[:, iseq]) != 0) & (np.nansum(proj[:, iseq]) != 0):
                corrtemp[iseq] = pearsonr(orig[:, iseq], proj[:, iseq]).statistic
        sf = np.nanmean(corrtemp)
        return sf

    # run for each time lag and save coefficients
    cross = np.full([2, maxLag], np.nan)
    for l in range(1, maxLag + 1):
        cross[0, l - 1] = Crosscorr(X1=X1, T=T1, l=l)
        cross[1, l - 1] = Crosscorr(X1=X1, T=T2, l=l)
    return cross


# In[THE CORRELATION OF DECODING ACCURACY]:

# load the data
EEG_accuracy = pd.read_csv(opj(path_decoding, 'eeg_acc.txt'), sep=',', names=['EEG_accuracy'])
fMRI_accuracy = pd.read_csv(opj(path_decoding, 'VFL_accuracy.csv'), sep=',', header=0, names=['fMRI_accuracy'])

# percbend correlation (robust correlation)
corr_perc = pg.corr(x=EEG_accuracy.iloc[:, 0], y=fMRI_accuracy.iloc[:, 0],
                    alternative='two-sided', method="percbend")
print(corr_perc)

# In[THE CORRELATION OF DECODING PROBABILITY IN VFL]:

# load the data
EEG_probability = pd.read_csv(opj(path_decoding, 'eeg_prob.txt'), sep=',', header=0)
fMRI_probability = pd.read_csv(opj(path_decoding, 'VFL_probability.csv'), sep=',', header=0)
# the robust correlation
# vfl_prob_corr = [pg.corr(x=EEG_probability.iloc[:, i], y=fMRI_probability.iloc[:, i],
#                          alternative='two-sided', method="percbend") for i in np.arange(0, 4)]
# pearson correlation
vfl_prob_corr = [pg.corr(x=EEG_probability.iloc[:, i], y=fMRI_probability.iloc[:, i],
                         alternative='two-sided', method="pearson") for i in np.arange(0, 4)]
# print the results
print('face probability: correlation: %.3f, p-value: %.3f; \n'
      'scissor probability: correlation: %.3f, p-value: %.3f; \n'
      'zebra probability: correlation: %.3f, p-value: %.3f; \n'
      'banana probability: correlation: %.3f, p-value: %.3f' % (vfl_prob_corr[0]['r'], vfl_prob_corr[0]['p-val'],
                                                                vfl_prob_corr[1]['r'], vfl_prob_corr[1]['p-val'],
                                                                vfl_prob_corr[2]['r'], vfl_prob_corr[2]['p-val'],
                                                                vfl_prob_corr[3]['r'], vfl_prob_corr[3]['p-val']))


# In[THE CROSS-CORRELATION BETWEEN DECODING PROBABILITY OF EEG AND FMRI IN TASK]:
def eeg_cue_decode(subject, tr=1.3):
    print('=====sub-%s task start!=====' % subject)
    # set the subject string
    sub = 'sub-%s' % subject
    # load the onset data file
    eeg_task = pd.read_csv(opj(path_cue_eeg, 'sub_%s_cue_replay_prob.txt' % subject), sep=',')
    fmri_task = pd.read_csv(opj(path_decode_fmri, sub, 'data', 'sub-%s fMRI decoding probability.csv' % subject),
                            sep=',')
    eeg_task['EEG_probability'] = np.sum((eeg_task['A'], eeg_task['B'],
                                          eeg_task['C'], eeg_task['D']), axis=0)
    # EEG trials
    eeg_trial = np.unique(eeg_task['points_TrialNum'])
    # fMRI trials
    fmri_trial = np.unique(fmri_task['trials'])
    # Calculate the intersection of trials
    unique_trial = set(eeg_trial).intersection(fmri_trial)
    # select these trials
    eeg_task_1 = eeg_task[eeg_task['points_TrialNum'].isin(unique_trial)]
    fmri_task_1 = fmri_task[fmri_task['trials'].isin(unique_trial)]
    # select mental simulation part
    sub_eeg_task = pd.DataFrame()
    for i in unique_trial:
        # select all the time points in each trials
        test = eeg_task_1[eeg_task_1['points_TrialNum'] == i].iloc[0:910, :].reset_index(
            drop=True, inplace=False).reset_index(drop=False)
        test['seq_tr'] = 1 + test['index'] // 130
        # get the specific time point BOLD signal
        slice_time_ref = 0.5
        start_time = slice_time_ref * tr * 100
        end_time = (7 - 1 + slice_time_ref) * tr * 100
        frame_times = np.linspace(start_time, end_time, 7)
        # interpolation
        for variable in ['A', 'B', 'C', 'D', 'EEG_probability']:
            f = interp1d(test['index'], test[variable])
            temp_decoding = pd.DataFrame(f(frame_times).T, columns=[variable])
            if variable == 'A':
                eeg_decoding_res = temp_decoding
            else:
                eeg_decoding_res = pd.concat((eeg_decoding_res, temp_decoding), axis=1)
        # concatenate the resampled decoding probability in EEG
        sub_eeg_task = pd.concat((sub_eeg_task, eeg_decoding_res), axis=0)
    # reset the index of df
    sub_eeg_task = sub_eeg_task.reset_index(drop=True)

    # rename the fmri df
    rename_dic = {'girl': 'A', 'scissors': 'B', 'zebra': 'C', 'banana': 'D'}
    sub_fmri_task = fmri_task_1.rename(columns=rename_dic)
    # calculate the probability
    sub_fmri_task['fMRI_probability'] = np.sum((sub_fmri_task['A'], sub_fmri_task['B'],
                                                sub_fmri_task['C'], sub_fmri_task['D']), axis=0)
    # extract the fmri decoding probability
    sub_fmri_task = sub_fmri_task.loc[:, ['A', 'B', 'C', 'D', 'fMRI_probability']]
    # concatenate the EEG and fMRI decoding probability
    sub_decode_task = [sub_eeg_task.reset_index(drop=True), sub_fmri_task.reset_index(drop=True)]

    return sub_decode_task


sub_prob_task = Parallel(n_jobs=-1)(delayed(eeg_cue_decode)(subject) for subject in sub_list)

# calculate the corelation between EEG and fMRI probability for each subjects
task_prob_corr_p = np.array([float(pg.corr(x=sub_prob_task[i][0].loc[:, 'EEG_probability'],
                                           y=sub_prob_task[i][1].loc[:, 'fMRI_probability'],
                                           alternative='two-sided', method="pearson")['r'])
                             for i in range(len(sub_list))])
mean_corr = np.mean(task_prob_corr_p)
# t test
task_test = pg.ttest(x=task_prob_corr_p, y=0, paired=False, alternative='two-sided')
print('cue replay task: mean correlation: %.3f. correlation against 0: %.3f, p-value:%.3f' % (
    mean_corr, task_test['T'], task_test['p-val']))
# robust correlation
# task_prob_corr_p = np.array(
#     [pg.corr(x=sub_prob_task[i][0].loc[:, 'EEG_probability'],
#              y=sub_prob_task[i][1].loc[:, 'fMRI_probability'],
#              alternative='two-sided', method="percbend")['p-val']
#      for i in range(len(sub_list))])
#
# mean_corr = np.mean(task_prob_corr_p)
# sem_corr = np.std(task_prob_corr_p)/np.sqrt(33)
# task_test = pg.ttest(x=task_prob_corr_p, y=0, paired=False, alternative='two-sided')
# combine EEG and fMRI probability
eeg_fmri_prob = [pd.concat((sub_prob_task[i][0].loc[:, 'EEG_probability'],
                            sub_prob_task[i][1].loc[:, 'fMRI_probability']), axis=1)
                 for i in range(len(sub_list))]
# calculate the cross correlation
pre_prob_corr = [TDLM_cross(eeg_fmri_prob[i]) for i in range(len(sub_list))]
pre_prob_corr_all = np.array(pre_prob_corr)
# calculate the mean and standard deviation for pre rest
s1mean = (pre_prob_corr_all[:, 0, :] - pre_prob_corr_all[:, 1, :]).mean(axis=0)
s1std = (pre_prob_corr_all[:, 0, :] - pre_prob_corr_all[:, 1, :]).std(axis=0)
s1sem = (pre_prob_corr_all[:, 0, :] - pre_prob_corr_all[:, 1, :]).std(axis=0) / np.sqrt(nSubj)
# the difference of cross correlation
cross_diff = pre_prob_corr_all[:, 0, :] - pre_prob_corr_all[:, 1, :]
# statistical inference
stats_df = pd.DataFrame(index=np.arange(0, 8), columns=['t', 'p'])
for i in range(8):
    cross_test = pg.ttest(x=cross_diff[:, i], y=0, alternative='two-sided', paired=False)
    stats_df.loc[i, 't'] = float(cross_test['T'].values)
    stats_df.loc[i, 'p'] = float(cross_test['p-val'].values)
# multiple comparison correction
corrected_p_values = multipletests(list(stats_df['p'].values), method='fdr_bh')
# output the results
data_file = pd.DataFrame({'mean_sequence': s1mean,
                          'std_sequence': s1std})
data_file.to_csv(opj(path_output, 'cue_replay_task_cross_correlation.csv'), index=False)
# plot the cross-correlation sequenceness
x = np.arange(0, maxLag, 1)
plt.figure(dpi=400, figsize=(7, 5))
plt.xticks(x, x + 1, fontsize=10)
plt.plot(x, s1mean, color='darkred', linestyle='-', linewidth=1)
plt.fill_between(x, s1mean - s1sem, s1mean + s1sem,
                 color='lightcoral', alpha=0.5, linewidth=1)
plt.axhline(y=0, color='silver', linestyle='--')
plt.title('Cross Correlation: EEG to fMRI decoding in cue replay task')
plt.xlabel('lag (TRs)')
plt.ylabel('fwd minus bkw sequenceness')
plt.savefig(opj(path_output, 'Cross Correlation: EEG to fMRI decoding in cue replay task.svg'),
            dpi=400, bbox_inches='tight')
plt.show()


# In[THE CROSS-CORRELATION BETWEEN DECODING PROBABILITY OF EEG AND FMRI IN REST]:
def eeg_rest_decode(subject, tr=1.3):
    print('=====sub-%s rest start!=====' % subject)
    # get the absolute time start point for pre resting and post resting state
    abs_start_pre = pre_rest_start(path_behavior_input, subject)
    abs_start_post = post_rest_start(path_behavior_input, subject)
    # get the confound file to calculate the TR scans
    scan_tr_pre = confound_tr(path_fmriprep, subject)[0]
    scan_tr_post = confound_tr(path_fmriprep, subject)[1]
    # load the onset data file
    eeg_pre_rest = pd.read_csv(opj(path_pre_rest, 'sub_%s_preRest_2_prob.txt' % subject), sep=',')
    eeg_post_rest = pd.read_csv(opj(path_post_rest, 'sub_%s_postRest_2_prob.txt' % subject), sep=',')
    # pack above
    abs_start = [abs_start_pre, abs_start_post]
    scan_tr = [scan_tr_pre, scan_tr_post]
    # calculate the probability
    eeg_pre_rest['eeg_probability'] = np.sum((eeg_pre_rest['A'], eeg_pre_rest['B'],
                                              eeg_pre_rest['C'], eeg_pre_rest['D']), axis=0)

    eeg_post_rest['eeg_probability'] = np.sum((eeg_post_rest['A'], eeg_post_rest['B'],
                                               eeg_post_rest['C'], eeg_post_rest['D']), axis=0)

    sub_decoding = []
    for i, file in enumerate([eeg_pre_rest, eeg_post_rest]):
        # set the relative replay event onset
        file['onset'] = file.loc[:, ['points_time']] / 1000 + abs_start[i]
        file['onset'] = file['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        file = file.loc[:, ['onset', 'A', 'B', 'C', 'D', 'eeg_probability']]
        # get the maximum time point in a run
        max_time = int(scan_tr[i]) * tr
        # create the time frame list
        scan_list = pd.DataFrame(np.linspace(0, max_time, int(max_time * 100) + 1), columns=['onset'])
        scan_list['onset'] = scan_list['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        scan_list['A'] = 0
        scan_list['B'] = 0
        scan_list['C'] = 0
        scan_list['D'] = 0
        scan_list['eeg_probability'] = 0
        # concatenate scan list and onset_run_event
        temp = pd.concat((file, scan_list), axis=0)
        tempa = temp.drop_duplicates(subset='onset').sort_values(by='onset').reset_index(drop=True)
        # get the specific time point BOLD signal
        slice_time_ref = 0.5
        start_time = slice_time_ref * tr
        end_time = (scan_tr[i] - 1 + slice_time_ref) * tr
        frame_times = np.linspace(start_time, end_time, scan_tr[i])
        for variable in ['A', 'B', 'C', 'D', 'eeg_probability']:
            # interpolation
            f = interp1d(tempa['onset'], tempa[variable])
            temp_decoding = pd.DataFrame(f(frame_times).T, columns=[variable])
            if variable == 'A':
                eeg_decoding_res = temp_decoding
            else:
                eeg_decoding_res = pd.concat((eeg_decoding_res, temp_decoding), axis=1)
        # output the EEG decoding probability
        eeg_decoding_res = eeg_decoding_res[4:235]  # 5 dummy variables
        sub_decoding.append(eeg_decoding_res)
    # load the fmri decoding probability
    # pre rest
    fmri_pre_rest = pd.read_csv(
        opj(path_decode_fmri_rest, 'sub-%s' % subject, 'sub-%s_pre_rest_reactivation_fmri.csv' % subject), sep=',')
    fmri_pre_rest = fmri_pre_rest.iloc[4:235, 1:6]  # 4 dummy variables
    names = {'probability': 'fmri_probability'}
    fmri_pre_rest = fmri_pre_rest.rename(columns=names)
    sub_decoding.append(fmri_pre_rest)
    # post rest
    fmri_post_rest = pd.read_csv(
        opj(path_decode_fmri_rest, 'sub-%s' % subject, 'sub-%s_post_rest_reactivation_fmri.csv' % subject), sep=',')
    fmri_post_rest = fmri_post_rest.iloc[4:235, 1:6]
    fmri_post_rest = fmri_post_rest.rename(columns=names)
    sub_decoding.append(fmri_post_rest)

    return sub_decoding


# run Parallel function
sub_prob = Parallel(n_jobs=-2)(delayed(eeg_rest_decode)(subject) for subject in sub_list)

# sort out the probability for pre- and post-rest
pre_prob = [pd.concat((sub_prob[i][0].loc[:, 'eeg_probability'],
                       sub_prob[i][2].loc[:, 'fmri_probability']), axis=1)
            for i in range(len(sub_list))]
post_prob = [pd.concat((sub_prob[i][1].loc[:, 'eeg_probability'],
                        sub_prob[i][3].loc[:, 'fmri_probability']), axis=1)
             for i in range(len(sub_list))]

# pre rest
pre_prob_corr_p = np.array([float(pg.corr(x=pre_prob[i].loc[:, 'eeg_probability'],
                                          y=pre_prob[i].loc[:, 'fmri_probability'],
                                          alternative='two-sided', method="pearson")['r'])
                            for i in range(len(sub_list))])
mean_pre = np.mean(task_prob_corr_p)
pre_rest_test = pg.ttest(x=pre_prob_corr_p, y=0, paired=False, alternative='two-sided')
print('pre-rest: mean correlation: %.3f.  correlation against 0: %.3f, p-value:%.3f' % (
    mean_pre, pre_rest_test['T'], pre_rest_test['p-val']))

# post rest
post_prob_corr_p = np.array([float(pg.corr(x=post_prob[i].loc[:, 'eeg_probability'],
                                           y=post_prob[i].loc[:, 'fmri_probability'],
                                           alternative='two-sided', method="pearson")['r'])
                             for i in range(len(sub_list))])
# correlation between EEG and fMRI without time lag
mean_post = np.mean(post_prob_corr_p)
post_rest_test = pg.ttest(x=post_prob_corr_p, y=0, paired=False, alternative='two-sided')
print('post-rest: mean correlation: %.3f. correlation against 0: %.3f, p-value:%.3f' % (
    mean_post, post_rest_test['T'], post_rest_test['p-val']))

# calculate the cross correlation
pre_prob_corr = [TDLM_cross(pre_prob[i]) for i in range(len(sub_list))]
post_prob_corr = [TDLM_cross(post_prob[i]) for i in range(len(sub_list))]
pre_prob_corr_all = np.array(pre_prob_corr)
post_prob_corr_all = np.array(post_prob_corr)

pre_cross_t = [pg.ttest(x=(pre_prob_corr_all[:, 0, :] - pre_prob_corr_all[:, 1, :])[:, i],
                        y=0, paired=False, alternative='two-sided') for i in range(8)]
post_cross_t = [pg.ttest(x=(post_prob_corr_all[:, 0, :] - post_prob_corr_all[:, 1, :])[:, i],
                         y=0, paired=False, alternative='two-sided') for i in range(8)]

pre_cross_diff = pre_prob_corr_all[:, 0, :] - pre_prob_corr_all[:, 1, :]
# multiple comparison correction
stats_df_pre = pd.DataFrame(index=np.arange(0, 8), columns=['t', 'p'])
for i in range(8):
    cross_test = pg.ttest(x=pre_cross_diff[:, i], y=0, alternative='two-sided', paired=False)
    stats_df_pre.loc[i, 't'] = float(cross_test['T'].values)
    stats_df_pre.loc[i, 'p'] = float(cross_test['p-val'].values)
corrected_p_values_pre = multipletests(list(stats_df_pre['p'].values), method='fdr_bh')

post_cross_diff = post_prob_corr_all[:, 0, :] - post_prob_corr_all[:, 1, :]
# multiple comparison correction
stats_df_post = pd.DataFrame(index=np.arange(0, 8), columns=['t', 'p'])
for i in range(8):
    cross_test = pg.ttest(x=post_cross_diff[:, i], y=0, alternative='two-sided', paired=False)
    stats_df_post.loc[i, 't'] = float(cross_test['T'].values)
    stats_df_post.loc[i, 'p'] = float(cross_test['p-val'].values)
corrected_p_values_post = multipletests(list(stats_df_post['p'].values), method='fdr_bh')

# calculate the mean and standard deviation for pre rest
s1mean = (pre_prob_corr_all[:, 0, :] - pre_prob_corr_all[:, 1, :]).mean(axis=0)
s1std = (pre_prob_corr_all[:, 0, :] - pre_prob_corr_all[:, 1, :]).std(axis=0)
s1sem = (pre_prob_corr_all[:, 0, :] - pre_prob_corr_all[:, 1, :]).std(axis=0) / np.sqrt(nSubj)

# output the results
data_file = pd.DataFrame({'mean_sequence': s1mean,
                          'std_sequence': s1std})
data_file.to_csv(opj(path_output, 'pre_rest_cross_correlation.csv'), index=False)
# plot the cross-correlation sequenceness
x = np.arange(0, maxLag, 1)
plt.figure(dpi=400, figsize=(7, 5))
plt.xticks(x, x + 1, fontsize=10)
plt.plot(x, s1mean, color='darkred', linestyle='-', linewidth=1)
plt.fill_between(x, s1mean - s1sem, s1mean + s1sem,
                 color='lightcoral', alpha=0.5, linewidth=1)
plt.axhline(y=0, color='silver', linestyle='--')
plt.title('Cross Correlation: EEG to fMRI decoding in pre rest')
plt.xlabel('lag (TRs)')
plt.ylabel('fwd minus bkw sequenceness')
plt.savefig(opj(path_output, 'Cross Correlation: EEG to fMRI decoding in pre rest.svg'),
            dpi=400, bbox_inches='tight')
plt.show()

# calculate the mean and standard deviation for post rest
s1mean = (post_prob_corr_all[:, 0, :] - post_prob_corr_all[:, 1, :]).mean(axis=0)
s1std = (post_prob_corr_all[:, 0, :] - post_prob_corr_all[:, 1, :]).std(axis=0)
s1sem = (post_prob_corr_all[:, 0, :] - post_prob_corr_all[:, 1, :]).std(axis=0) / np.sqrt(nSubj)
# output the results
data_file = pd.DataFrame({'mean_sequence': s1mean,
                          'std_sequence': s1std})
data_file.to_csv(opj(path_output, 'post_rest_cross_correlation.csv'), index=False)
# plot the cross-correlation sequenceness
x = np.arange(0, maxLag, 1)
plt.figure(dpi=400, figsize=(7, 5))
plt.xticks(x, x + 1, fontsize=10)
plt.plot(x, s1mean, color='darkred', linestyle='-', linewidth=1)
plt.fill_between(x, s1mean - s1sem, s1mean + s1sem,
                 color='lightcoral', alpha=0.5, linewidth=1)
plt.axhline(y=0, color='silver', linestyle='--')
plt.title('Cross Correlation: EEG to fMRI decoding in post rest')
plt.xlabel('lag (TRs)')
plt.ylabel('fwd minus bkw sequenceness')
plt.savefig(opj(path_output, 'Cross Correlation: EEG to fMRI decoding in post rest.svg'),
            dpi=400, bbox_inches='tight')
plt.show()

# In[Comparison among three conditions]
# concatenate three conditions' correlation
all_correlation = np.vstack((pre_prob_corr_p, post_prob_corr_p, task_prob_corr_p))
all_correlation = pd.DataFrame(np.transpose(all_correlation), columns=['pre_rest', 'post_rest', 'task'])
all_correlation = all_correlation.melt(value_vars=['pre_rest', 'post_rest', 'task'],
                                       var_name='session', value_name='correlation')
all_correlation['id'] = np.hstack((np.arange(0, 33), np.arange(0, 33), np.arange(0, 33)))

# one-way ANOVA
all_anova = pg.rm_anova(data=all_correlation, dv='correlation', within='session', subject='id', detailed=True,
                        effsize='np2')
print(all_anova)
# plot the correlation
sns.set_theme(style="whitegrid")
plt.figure(dpi=400, figsize=(10, 5))
plt.xticks(fontsize=10)
sns.barplot(x="session", y="correlation", data=all_correlation)
plt.suptitle('three condition correlation')
plt.savefig(opj(path_output, 'three condition correlation.svg'), format='svg')
plt.show()

# t-test for each pair
pre_post_test = pg.ttest(x=pre_prob_corr_p, y=post_prob_corr_p, alternative='two-sided', paired=True)
task_post_test = pg.ttest(x=task_prob_corr_p, y=post_prob_corr_p, alternative='two-sided', paired=True)
task_pre_test = pg.ttest(x=pre_prob_corr_p, y=task_prob_corr_p, alternative='two-sided', paired=True)

print('pre-post: t = %.3f, p = %.3f; \n pre-task: t = %.3f, p = %.3f; \n post-task: t = %.3f, p = %.3f' % (
    pre_post_test['T'], pre_post_test['p-val'],
    task_pre_test['T'], task_pre_test['p-val'],
    task_post_test['T'], task_post_test['p-val']))
