#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: IDENTIFY REPLAY AND REACTIVATION (FUNCTIONS)
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITIVE NEUROSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY


# In[DEFINE SOME FUNCTIONS]:


def confound_tr_task(path_fmriprep, subject):
    import os
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


def confound_tr_rest(path_fmriprep, subject):
    import os
    from os.path import join as opj
    import pandas as pd
    run_list = [1, 2]
    nums_scan = []
    templates = dict(
        confounds=opj(path_fmriprep, 'sub-{subject}', 'func',
                      'sub-{subject}_task-rest_run-{run}_desc-confounds_timeseries.tsv'),
    )
    for run_id in run_list:
        confounds = templates['confounds'].format(subject=subject, run=run_id)
        confounds_file = pd.read_csv(confounds, sep='\t')
        num_scan = confounds_file.shape[0]
        nums_scan.append(num_scan)
    return nums_scan


def behavior_start(path_behavior, subject):
    import os
    from os.path import join as opj
    import numpy as np
    import pandas as pd
    import glob

    sub = 'sub-%s' % subject
    # load the behavioral data file
    beh_path = opj(path_behavior, 'replay', 'cue_replay_%s_*.csv' % subject)
    # create the path to the data file
    beh_path = sorted(glob.glob(beh_path), key=lambda f: os.path.basename(f))
    # read the visual functional localizer trial data
    beh_event = [pd.read_csv(f, sep=',') for f in beh_path]
    # select useful columns
    beh_event = [
        beh_event[f][['runs_replay.thisRepN', 'trials_replay.thisN', 'text.started',
                          'text_cue_replay.started']] for f in range(len(beh_path))]
    # delete unrelated rows by NaN value
    beh_event = [
        beh_event[f].dropna(axis=0, how='any', subset=['text_cue_replay.started']).reset_index(drop=True)
        for f in range(len(beh_path))]
    # rename the columns(variables)
    rename_beh = {'runs_replay.thisRepN': 'run',
                  'trials_replay.thisN': 'trials',
                  'text.started': 'start',
                  'text_cue_replay.started': 'onset'}
    beh_event = [beh_event[f].rename(columns=rename_beh) for f in range(len(beh_path))]
    # set some new values for all first session
    beh_event[0]['session'] = 1
    beh_event[0]['run'] = beh_event[0]['run'] + 1
    # if one file from one subject
    if len(beh_path) == 1:
        beh_event = beh_event[0]
        # calculate onset time
        onset = [beh_event[(beh_event['run'] == i)]['onset'].sub(
            float(beh_event[(beh_event['run'] == i)
                                & (beh_event['trials'] == 0)]['start'])) for i in np.arange(1, 4)]
        onset = pd.concat([onset[0], onset[1], onset[2]], axis=0)
        beh_event['onset'] = onset
    # if there are two files from one subject
    elif len(beh_path) == 2:
        # set some new values for the second session
        beh_event[1]['session'] = 2
        beh_event[1]['run'] = beh_event[1]['run'] + beh_event[0]['run'].iloc[-1] + 1
        # calculate onset time for each run
        if sub == 'sub-07' or sub == 'sub-32':
            for f in range(len(beh_path)):
                if f == 0:
                    onset1 = [beh_event[f][(beh_event[f]['run'] == i)]['onset'].sub(
                        float(beh_event[f][(beh_event[f]['run'] == i)
                                               & (beh_event[f]['trials'] == 0)]['start'])) for i in np.arange(1, 3)]
                    onset = pd.concat([onset1[0], onset1[1]], axis=0)
                    beh_event[f]['onset'] = onset
                elif f == 1:
                    onset2 = beh_event[f][(beh_event[f]['run'] == 3)]['onset'].sub(
                        float(beh_event[f][(beh_event[f]['run'] == 3)
                                               & (beh_event[f]['trials'] == 0)]['start']))
                    beh_event[f]['onset'] = onset2
        elif sub == 'sub-40':
            for f in range(len(beh_path)):
                if f == 0:
                    onset1 = [beh_event[f][(beh_event[f]['run'] == i)]['onset'].sub(
                        float(beh_event[f][(beh_event[f]['run'] == i)
                                               & (beh_event[f]['trials'] == 0)]['start'])) for i in np.arange(1, 2)]
                    beh_event[f]['onset'] = onset1[0]
                elif f == 1:
                    onset2 = [beh_event[f][(beh_event[f]['run'] == i)]['onset'].sub(
                        float(beh_event[f][(beh_event[f]['run'] == i)
                                               & (beh_event[f]['trials'] == 0)]['start'])) for i in np.arange(2, 4)]
                    onset = pd.concat([onset2[0], onset2[1]], axis=0)
                    beh_event[f]['onset'] = onset
        temp = pd.concat([beh_event[0], beh_event[1]])
        beh_event = temp
    # devide behavior event into different files by runs
    class_run_beh = beh_event['run'].unique()
    behavior_event = [beh_event[beh_event['run'] == i].reset_index(drop=True) for i in class_run_beh]
    # get the absolute time onset for each run
    abs_start = np.array([behavior_event[f].loc[0, ['onset']] for f in range(len(behavior_event))])
    return abs_start


def pre_rest_start(path_behavior, subject):
    import os
    from os.path import join as opj
    import numpy as np
    import pandas as pd
    import glob
    # get the sub-*
    sub = "sub-%s" % subject
    # get the path of VFL files
    path_behavior_vfl = opj(path_behavior, 'VFL', 'VFL_%s_*.csv' % subject)
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


def post_rest_start(path_behavior, subject):
    import os
    from os.path import join as opj
    import numpy as np
    import pandas as pd
    import glob
    # get the sub-*
    sub = 'sub-%s' % subject
    # load the behavioral data file
    behavior_path = opj(path_behavior, 'replay', 'cue_replay_%s_*.csv' % subject)
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
    import numpy as np
    from scipy.stats import gamma

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
