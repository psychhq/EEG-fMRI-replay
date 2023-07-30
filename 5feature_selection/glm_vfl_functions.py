#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======================================================================
# DEFINE FUNCTION: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
# ======================================================================

def get_subject_info(events, confounds):
    """
    FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
    :param events: list with paths to events files
    :param confounds: list with paths to confounds files
    :return: Bunch object with event onsets, durations and regressors
    """

    # import libraries (needed to be done in the function):
    import pandas as pd
    from nipype.interfaces.base import Bunch

    # event types we consider:
    vfl_event_spec = {
        'correct': {'accuracy': 1},
        'wrong': {'accuracy': 0}
    }

    # read the events and confounds files of the current run:
    # events = selectfiles_results.outputs.events[0]
    # confounds = selectfiles_results.outputs.confounds[1]
    run_events = pd.read_csv(events, sep="\t")
    run_confounds = pd.read_csv(confounds, sep="\t")

    # time of repetition, in seconds:
    time_repetition = 1.3
    # number of dummy variables to remove from each run:
    num_dummy = 4

    # delete dummy variable time
    run_events['onset'] = run_events['onset'] - num_dummy * time_repetition
    run_confounds = run_confounds[num_dummy:]

    # nuisance regressors
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
    regressors = [replace_nan(run_confounds[conf]) for conf in regressor_names]

    # set some empty lists
    onsets = []
    durations = []
    event_names = []

    # create the event lists
    for event in vfl_event_spec:
        onset_list = list(
            run_events['onset']
            [(run_events['task'] == 'VFL') &
             (run_events['accuracy'] == vfl_event_spec[event]['accuracy'])])
        duration_list = list(
            run_events['duration']
            [(run_events['task'] == 'VFL') &
             (run_events['accuracy'] == vfl_event_spec[event]['accuracy'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []):
            event_names.append(event)
            onsets.append(onset_list)
            durations.append(duration_list)

    # create a bunch for each run:
    subject_info = Bunch(
        conditions=event_names, onsets=onsets, durations=durations,  # event regressors
        regressor_names=regressor_names, regressors=regressors)  # confound regressors
    return subject_info, sorted(event_names)

'''
======================================================================
DEFINE FUNCTION: FUNCTION TO PLOT THE CONTRASTS AGAINST ANATOMICAL
======================================================================
'''

def plot_stat_maps(anat, stat_map, thresh):
    """
    :param anat:
    :param stat_map:
    :param thresh:
    :return:
    """
    # import libraries (needed to be done in the function):
    from nilearn.plotting import plot_stat_map
    from os import path as op

    out_path = op.join(op.dirname(stat_map), 'contrast_thresh_%s.png' % thresh)
    plot_stat_map(
        stat_map, title=('Threshold: %s' % thresh),
        bg_img=anat, threshold=thresh, dim=-1, display_mode='ortho',
        output_file=out_path)
    return out_path

def leave_one_out(subject_info, event_names, data_func, run):
    """
    Select subsets of lists in leave-one-out fashion:
    :param subject_info: list of subject info bunch objects
    :param data_func: list of functional runs
    :param run: current run
    :return: return list of subject info and data excluding the current run
    """
    # create new list with event_names of all runs except current run:
    event_names = [info for i, info in enumerate(event_names) if i != run]
    # check the number of trials in correct trial and wrong trial
    num_events = [len(i) for i in event_names]
    max_events = event_names[num_events.index(max(num_events))]

    # create list of contrasts:
    stim = 'correct'
    contrast1 = (stim, 'T', max_events, [1 if stim in s else 0 for s in max_events])
    contrasts = [contrast1]

    # create new list with subject info of all runs except current run:
    subject_info = [info for i, info in enumerate(subject_info) if i != run]
    # create new list with functional data of all runs except current run:
    data_func = [info for i, info in enumerate(data_func) if i != run]

    # return the new lists
    return subject_info, data_func, contrasts
