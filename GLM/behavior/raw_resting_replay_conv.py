# In[]:
import glob
import os
import warnings
from os.path import join as opj
import numpy as np
import pandas as pd
from scipy import stats
from pandas.core.common import SettingWithCopyWarning
from bids.layout import BIDSLayout
from scipy.stats import gamma
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from matplotlib import mlab as mlab

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# name of the current project:
sub_list = None
subject = None
# path to the project root:
project_name = 'resting-replay-onset'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_onset_pre = opj(path_bids, 'sourcedata', 'onsets-pre-resting')
path_onset_post = opj(path_bids, 'sourcedata', 'onsets-post-resting')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_behavior_input = opj(path_bids, 'sourcedata', 'fmrireplay-rawbehavior')
path_onset_output = opj(path_bids, 'derivatives', project_name)
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

tr = 1.3


# In[]: resting state replay probability onset

def confound_tr(path_fmriprep, subject):
    from os.path import join as opj
    import pandas as pd
    run_list = [1, 2]
    nums_scan = []
    templates = dict(
        confounds=opj(path_fmriprep, 'sub-{subject}', 'func',
                      'sub-{subject}_task-rest_run-{run}_desc-confounds_timeseries.tsv'), )
    for run_id in run_list:
        confounds = templates['confounds'].format(subject=subject, run=run_id)
        confounds_file = pd.read_csv(confounds, sep='\t')
        num_scan = confounds_file.shape[0]
        nums_scan.append(num_scan)
    return nums_scan


def pre_resting_start(path_behavior_input, subject):
    # test for the specific subjects
    # subject = sub_list[0]
    # get the sub-*
    sub = "sub-%s" % subject
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


def post_resting_start(path_behavior_input, subject):
    # test for the specific subjects
    # subject = sub_list[0]
    # get the sub-*
    sub = 'sub-%s' % subject
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


# In[]:
def conv(subject):
    # subject = sub_list[4]
    print('=====sub-%s start!=====' % subject)
    # set the subject string
    sub = 'sub-%s' % subject
    # get the absolute time start point for pre resting and post resting state
    abs_start_pre = pre_resting_start(path_behavior_input, subject)
    abs_start_post = post_resting_start(path_behavior_input, subject)
    # get the confound file to calculate the TR scans
    scan_tr_pre = confound_tr(path_fmriprep, subject)[0]
    scan_tr_post = confound_tr(path_fmriprep, subject)[1]
    # load the onset data file
    onset_file_fwd_pre = pd.read_csv(opj(path_onset_pre, 'sub_%s_preRest_2_fwd_onset_raw.txt' % subject)
                                     , sep=',')
    onset_file_bwd_pre = pd.read_csv(opj(path_onset_pre, 'sub_%s_preRest_2_bwd_onset_raw.txt' % subject)
                                     , sep=',')
    onset_file_fwd_post = pd.read_csv(opj(path_onset_post, 'sub_%s_postRest_2_fwd_onset_raw.txt' % subject)
                                      , sep=',')
    onset_file_bwd_post = pd.read_csv(opj(path_onset_post, 'sub_%s_postRest_2_bwd_onset_raw.txt' % subject)
                                      , sep=',')
    abs_start = [abs_start_pre, abs_start_pre, abs_start_post, abs_start_post]
    scan_tr = [scan_tr_pre, scan_tr_pre, scan_tr_post, scan_tr_post]
    save_list = []
    #     save_list.append(resample_onset)
    # print(stats.pearsonr(save_list[0].iloc[:,0], save_list[1].iloc[:,0]))
    # print(stats.pearsonr(save_list[2].iloc[:,0], save_list[3].iloc[:,0]))
    for i, file in enumerate([onset_file_fwd_pre, onset_file_bwd_pre, onset_file_fwd_post, onset_file_bwd_post]):
        # i = 0
        # file = onset_file_fwd_pre
        # set the probability of replay event
        file['probability'] = file.iloc[:, 4]
        # set the relative replay event onset
        file['onset'] = file.loc[:, ['points_time']] / 1000 + abs_start[i]
        file['onset'] = file['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        file = file.loc[:, ['onset', 'probability']]
        # onset_run_event[i]['probability'] = mlab.detrend_mean(onset_run_event[i]['probability'], axis=0)
        # get the maximum time point in a run
        max_time = int(scan_tr[i]) * tr
        # create the time frame list
        scan_list = pd.DataFrame(np.linspace(0, max_time, int(max_time * 100) + 1), columns=['onset'])
        scan_list['onset'] = scan_list['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        scan_list['probability'] = 0
        # concatenate scan list and onset_run_event
        temp = pd.concat((file, scan_list), axis=0)
        tempa = temp.drop_duplicates(subset='onset').sort_values(by='onset').reset_index(drop=True)
        save_list.append(tempa)

    prob_pre = (save_list[0].iloc[:, 1] + save_list[1].iloc[:, 1])
    tempa_pre = pd.concat((save_list[0].iloc[:, 0], prob_pre), axis=1)
    prob_post = (save_list[2].iloc[:, 1] + save_list[3].iloc[:, 1])
    tempa_post = pd.concat((save_list[2].iloc[:, 0], prob_post), axis=1)

    for i, cons, tempa in zip([0, 2], ['pre', 'post'], [tempa_pre, tempa_post]):
        # spm HRF kernel
        hkernel = spm_hrf(tr=1.3, oversampling=100)
        # convolve the HRF kernel and probability event
        tempa['conv_reg'] = np.array(np.convolve(tempa['probability'], hkernel)[:tempa['probability'].size])
        # interpolation
        f = interp1d(tempa['onset'], tempa['conv_reg'])
        # get the specific time point BOLD signal
        slice_time_ref = 0.5
        start_time = slice_time_ref * tr
        end_time = (scan_tr[i] - 1 + slice_time_ref) * tr
        frame_times = np.linspace(start_time, end_time, scan_tr[i])
        # resample BOLD signal
        resample_onset = pd.DataFrame(f(frame_times).T, columns=['HRF_onset_TR'])
        # mean BOLD signal in one TR
        # resample_onset = pd.DataFrame([tempa[(tempa['onset'] >= ts * tr) &
        #                                      (tempa['onset'] < (ts + 1) * tr)]['conv_reg'].mean()
        #                                for ts in range(scan_tr[i])], columns=['HRF_onset_TR'])
        # save the data
        file_name = '%s_%s_nodemean_middle_rest_replay_events.tsv' % (subject, cons)
        write_path = opj(path_onset_output, 'raw_replay', sub)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        resample_onset.to_csv(opj(write_path, file_name), sep='\t', index=0)


Parallel(n_jobs=64)(delayed(conv)(subject) for subject in sub_list)


# In[]:  resting state reactivation probability onset

def conv(subject):
    # subject = sub_list[3]
    print('=====sub-%s start!=====' % subject)
    # set the subject string
    sub = 'sub-%s' % subject
    # get the absolute time start point for pre resting and post resting state
    abs_start_pre = pre_resting_start(path_behavior_input, subject)
    abs_start_post = post_resting_start(path_behavior_input, subject)
    # get the confound file to calculate the TR scans
    scan_tr_pre = confound_tr(path_fmriprep, subject)[0]
    scan_tr_post = confound_tr(path_fmriprep, subject)[1]
    # load the onset data file
    onset_file_react_pre = pd.read_csv(opj(path_onset_pre, 'sub_%s_preRest_2_prob.txt' % subject)
                                       , sep=',')
    onset_file_react_post = pd.read_csv(opj(path_onset_post, 'sub_%s_postRest_2_prob.txt' % subject)
                                        , sep=',')

    abs_start = [abs_start_pre, abs_start_post]
    scan_tr = [scan_tr_pre, scan_tr_post]
    save_list = []
    #     save_list.append(resample_onset)
    onset_file_react_pre['probability'] = np.mean((onset_file_react_pre['A'], onset_file_react_pre['B'],
                                                   onset_file_react_pre['C'], onset_file_react_pre['D']), axis=0)


    onset_file_react_post['probability'] = np.mean((onset_file_react_post['A'], onset_file_react_post['B'],
                                                    onset_file_react_post['C'], onset_file_react_post['D']), axis=0)
    # print(stats.pearsonr(onset_file_react_post.iloc[:,3], onset_file_react_post.iloc[:,7]))

    # print(stats.pearsonr(save_list[2].iloc[:,0], save_list[3].iloc[:,0]))
    for i, file in enumerate([onset_file_react_pre, onset_file_react_post]):
        # i = 0
        # file = onset_file_fwd_pre
        # set the relative replay event onset
        file['onset'] = file.loc[:, ['points_time']] / 1000 + abs_start[i]
        file['onset'] = file['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        file = file.loc[:, ['onset', 'probability']]
        # onset_run_event[i]['probability'] = mlab.detrend_mean(onset_run_event[i]['probability'], axis=0)
        # get the maximum time point in a run
        max_time = int(scan_tr[i]) * tr
        # create the time frame list
        scan_list = pd.DataFrame(np.linspace(0, max_time, int(max_time * 100) + 1), columns=['onset'])
        scan_list['onset'] = scan_list['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        scan_list['probability'] = 0
        # concatenate scan list and onset_run_event
        temp = pd.concat((file, scan_list), axis=0)
        tempa = temp.drop_duplicates(subset='onset').sort_values(by='onset').reset_index(drop=True)
        # spm HRF kernel
        hkernel = spm_hrf(tr=1.3, oversampling=100)
        # convolve the HRF kernel and probability event
        tempa['conv_reg'] = np.array(np.convolve(tempa['probability'], hkernel)[:tempa['probability'].size])
        # interpolation
        f = interp1d(tempa['onset'], tempa['conv_reg'])
        # get the specific time point BOLD signal
        slice_time_ref = 0.5
        start_time = slice_time_ref * tr
        end_time = (scan_tr[i] - 1 + slice_time_ref) * tr
        frame_times = np.linspace(start_time, end_time, scan_tr[i])
        # resample BOLD signal
        resample_onset = pd.DataFrame(f(frame_times).T, columns=['HRF_onset_TR'])
        # mean BOLD signal in one TR
        # resample_onset = pd.DataFrame([tempa[(tempa['onset'] >= ts * tr) &
        #                                      (tempa['onset'] < (ts + 1) * tr)]['conv_reg'].mean()
        #                                for ts in range(scan_tr[i])], columns=['HRF_onset_TR'])
        # save the data
        consname = {0: {'cons': 'pre'}, 1: {'cons': 'post'}, }
        file_name = '%s_%s_middle_rest_reactivation_events.tsv' % (subject, consname[i]['cons'])
        write_path = opj(path_onset_output, 'raw_replay', sub)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        resample_onset.to_csv(opj(write_path, file_name), sep='\t', index=0)


Parallel(n_jobs=64)(delayed(conv)(subject) for subject in sub_list)
