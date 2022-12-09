# In[]:
import glob
import os
import warnings
from os.path import join as opj
import numpy as np
import pandas as pd
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
project_name = 'replay-onset'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_onset = opj(path_bids, 'sourcedata', 'onsets', 'merge')
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


# In[]:

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
    print('=====sub-%s start!=====' % subject)
    # subject = sub_list[0]
    # set the subject string
    sub = 'sub-%s' % subject
    # get the absolute time start point
    abs_start = behavior_start(path_behavior_input, subject)
    # get the confound file to calculate the TR scans
    scan_tr = confound_tr(path_fmriprep, subject)
    # load the onset data file
    onset_path = opj(path_onset, 'sub_%s_replayonsets_raw.txt' % subject)
    onset_file = pd.read_csv(onset_path, sep=',')
    # set the probability of replay event: fwd-fwd bwd-bwd
    # onset_file.loc[:, ['probability']] = np.nan
    # onset_file.loc[(onset_file.isfwdTrial == 1), 'probability'] = onset_file.loc[
    #     (onset_file.isfwdTrial == 1), 'prob_fwd_lag30']
    # onset_file.loc[(onset_file.isfwdTrial == 0), 'probability'] = onset_file.loc[
    #     (onset_file.isfwdTrial == 0), 'prob_bwd_lag30']
    # set the probability of replay event
    onset_file.loc[:, ['probability']] = np.nan
    onset_file['probability'] = onset_file['prob_fwd_lag30'] + onset_file['prob_bwd_lag30']
    # set the relative replay event onset
    onset_file.loc[:, ['onset']] = np.nan
    onset_file['onset'] = onset_file.loc[:, ['points_time']] / 1000
    onset_file['onset'] = onset_file['onset'].apply(
        lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
    # devide replay event into different files by runs
    class_run = onset_file['points_runNum'].unique()
    class_run.sort()
    onset_run_data = [onset_file[onset_file['points_runNum'] == i] for i in class_run]

    # rearrange the replay onset event
    onset_run_event = [
        onset_run_data[i].loc[:, ['onset', 'probability']] for i in range(len(onset_run_data))]
    for i in range(len(onset_run_event)):
        onset_run_event[i]['onset'] = onset_run_event[i]['onset'] + abs_start[i]
        onset_run_event[i]['onset'] = onset_run_event[i]['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        # onset_run_event[i]['probability'] = mlab.detrend_mean(onset_run_event[i]['probability'], axis=0)
        # get the maximum time point in a run
        max_time = int(scan_tr[i]) * tr
        # create the time frame list
        scan_list = pd.DataFrame(np.linspace(0, max_time, int(max_time * 100) + 1), columns=['onset'])
        scan_list['onset'] = scan_list['onset'].apply(
            lambda x: float(str(x).split('.')[0] + '.' + str(x).split('.')[1][:2]))
        scan_list['probability'] = 0
        # concatenate scan list and onset_run_event
        temp = pd.concat((onset_run_event[i], scan_list), axis=0)
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
        file_name = '%s_4_nodemean_middle_cue_replay_onset_run-%02.0f_events.tsv' % (subject, (i + 1))
        write_path = opj(path_onset_output, 'raw_replay', sub)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        resample_onset.to_csv(opj(write_path, file_name), sep='\t', index=0)


Parallel(n_jobs=64)(delayed(conv)(subject) for subject in sub_list)
