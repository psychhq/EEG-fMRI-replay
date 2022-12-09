#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[2]:
# ======================================================================
# DEFINE FUNCTION: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
# ======================================================================
def get_subject_info_max(events, confounds, onset_events):
    import pandas as pd
    import numpy as np
    from nipype.interfaces.base import Bunch

    # read the events and confounds files of the current run:
    # events = selectfiles_results.outputs.events[0]
    # confounds = selectfiles_results.outputs.confounds[0]
    # onset_events = selectfiles_results.outputs.onset_events[0]
    run_events = pd.read_csv(events, sep="\t")
    run_confounds = pd.read_csv(confounds, sep="\t")
    run_replay_onset_events = pd.read_csv(onset_events, sep='\t')

    subject = (events.split('behavior')[1])[1:7]
    # time of repetition, in seconds:
    time_repetition = 1.3
    # number of dummy variables to remove from each run:
    num_dummy = 4

    # delete dummy variable time
    run_events['onset'] = run_events['onset'] - num_dummy * time_repetition
    run_replay_onset_events['onsets'] = run_replay_onset_events['onsets'] - num_dummy * time_repetition

    # reset the markers of wrong trials
    for i in range(run_events[run_events['accuracy'] == 0].shape[0]):
        run_events.loc[run_events[run_events['accuracy'] == 0].index[i], 'Marker'] = 53

    # get the dummy variables
    run_confounds = run_confounds[num_dummy:]
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

    replay_event_onset = {
        'fwd_fwd': {'isfwdTrial': 1},
        'bwd_bwd': {'isfwdTrial': 0},
    }

    for event2 in replay_event_onset:
        onset_list = list(
            run_replay_onset_events['onsets']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
        duration_list = list(
            run_replay_onset_events['duration']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []):
            event_names.append(event2)
            onsets.append(onset_list)
            durations.append(duration_list)

    # event types we consider:
    rep_event_spec = {
        'forward_correct': {'accuracy': 1, 'Marker': 51},
        'backward_correct': {'accuracy': 1, 'Marker': 52},
        'wrong': {'accuracy': 0, 'Marker': 53},
    }
    for event1 in rep_event_spec:
        onset_list = list(
            run_events['onset']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        duration_list = list(
            run_events['duration']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []):
            event_names.append(event1)
            onsets.append(onset_list)
            durations.append(duration_list)

    subject_info = Bunch(
        # event regressors
        conditions=event_names,
        onsets=onsets,
        durations=durations,
        # confound regressors
        regressor_names=regressor_names,
        regressors=regressors
    )

    return subject_info


# In[3]:
# ======================================================================
# DEFINE FUNCTION: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
# ======================================================================
def get_subject_info_max_pm(events, confounds, onset_events):
    import pandas as pd
    import numpy as np
    from nipype.interfaces.base import Bunch

    # read the events and confounds files of the current run:
    # events = selectfiles_results.outputs.events[0]
    # confounds = selectfiles_results.outputs.confounds[0]
    # onset_events = selectfiles_results.outputs.onset_events[0]
    run_events = pd.read_csv(events, sep="\t")
    run_confounds = pd.read_csv(confounds, sep="\t")
    run_replay_onset_events = pd.read_csv(onset_events, sep='\t')

    subject = (events.split('behavior')[1])[1:7]
    # time of repetition, in seconds:
    time_repetition = 1.3
    # number of dummy variables to remove from each run:
    num_dummy = 4

    # delete dummy variable time
    run_events['onset'] = run_events['onset'] - num_dummy * time_repetition
    run_replay_onset_events['onsets'] = run_replay_onset_events['onsets'] - num_dummy * time_repetition

    # reset the markers of wrong trials
    for i in range(run_events[run_events['accuracy'] == 0].shape[0]):
        run_events.loc[run_events[run_events['accuracy'] == 0].index[i], 'Marker'] = 53

    # get the dummy variables
    run_confounds = run_confounds[num_dummy:]
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
    modulation = []

    replay_event_onset = {
        'fwd_fwd': {'isfwdTrial': 1},
        'bwd_bwd': {'isfwdTrial': 0},
    }

    for event2 in replay_event_onset:
        onset_list = list(
            run_replay_onset_events['onsets']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
        duration_list = list(
            run_replay_onset_events['duration']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
        modulation_list = list(
            run_replay_onset_events['probability']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []) & (modulation_list != []):
            event_names.append(event2)
            onsets.append(onset_list)
            durations.append(duration_list)
            modulation.append(modulation_list)

    # event types we consider:
    rep_event_spec = {
        'forward_correct': {'accuracy': 1, 'Marker': 51},
        'backward_correct': {'accuracy': 1, 'Marker': 52},
        'wrong': {'accuracy': 0, 'Marker': 53},
    }

    for event1 in rep_event_spec:
        onset_list = list(
            run_events['onset']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        duration_list = list(
            run_events['duration']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []):
            event_names.append(event1)
            onsets.append(onset_list)
            durations.append(duration_list)

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    replay_event_prob = {
        'fwd_prob': {'isfwdTrial': 1},
        'bwd_prob': {'isfwdTrial': 0},
    }
    for param_name in replay_event_prob:
        pmod_params_list = list(
            run_replay_onset_events['probability']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_prob[param_name]['isfwdTrial'])])
        if (pmod_params_list != []):
            pmod_names.append(param_name)
            pmod_params.append(pmod_params_list)
            pmod_polys.append(1)

    # create a bunch for each regressor:
    fwd_prob = Bunch(name=[pmod_names[0]], param=[pmod_params[0]], poly=[pmod_polys[0]])
    bwd_prob = Bunch(name=[pmod_names[1]], param=[pmod_params[1]], poly=[pmod_polys[1]])

    if subject == 'sub-48':
        subject_info = Bunch(
            # event regressors
            conditions=event_names,
            onsets=onsets,
            durations=durations,
            # parametric modulation
            pmod=[fwd_prob, bwd_prob, None, None],
            orth=['No', 'No', 'No', 'No'],
            # confound regressors
            regressor_names=regressor_names,
            regressors=regressors
        )
    elif subject != 'sub-48':
        subject_info = Bunch(
            # event regressors
            conditions=event_names,
            onsets=onsets,
            durations=durations,
            # parametric modulation
            pmod=[fwd_prob, bwd_prob, None, None, None],
            orth=['No', 'No', 'No', 'No', 'No'],
            # confound regressors
            regressor_names=regressor_names,
            regressors=regressors
        )

    return subject_info


# In[2]:
# ======================================================================
# DEFINE FUNCTION: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
# ======================================================================
def get_subject_info_perm2(events, confounds, onset_events):
    import pandas as pd
    import numpy as np
    from nipype.interfaces.base import Bunch

    # read the events and confounds files of the current run:
    # events = selectfiles_results.outputs.events[0]
    # confounds = selectfiles_results.outputs.confounds[0]
    # onset_events = selectfiles_results.outputs.onset_events[0]
    run_events = pd.read_csv(events, sep="\t")
    run_confounds = pd.read_csv(confounds, sep="\t")
    run_replay_onset_events = pd.read_csv(onset_events, sep='\t')

    subject = (events.split('behavior')[1])[1:7]
    # time of repetition, in seconds:
    time_repetition = 1.3
    # number of dummy variables to remove from each run:
    num_dummy = 4

    # delete dummy variable time
    run_events['onset'] = run_events['onset'] - num_dummy * time_repetition
    run_replay_onset_events['onsets'] = run_replay_onset_events['onsets'] - num_dummy * time_repetition

    # reset the markers of wrong trials
    for i in range(run_events[run_events['accuracy'] == 0].shape[0]):
        run_events.loc[run_events[run_events['accuracy'] == 0].index[i], 'Marker'] = 53

    # get the dummy variables
    run_confounds = run_confounds[num_dummy:]
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

    replay_event_onset = {
        'fwd_fwd': {'isfwdTrial': 1},
        'bwd_bwd': {'isfwdTrial': 0},
    }
    for event2 in replay_event_onset:
        onset_list = list(
            run_replay_onset_events['onsets']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
        duration_list = list(
            run_replay_onset_events['duration']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []):
            event_names.append(event2)
            onsets.append(onset_list)
            durations.append(duration_list)
        # event types we consider:
        rep_event_spec = {
            'forward_correct': {'accuracy': 1, 'Marker': 51},
            'backward_correct': {'accuracy': 1, 'Marker': 52},
            'wrong': {'accuracy': 0, 'Marker': 53},
        }

    for event1 in rep_event_spec:
        onset_list = list(
            run_events['onset']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        duration_list = list(
            run_events['duration']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []):
            event_names.append(event1)
            onsets.append(onset_list)
            durations.append(duration_list)

    subject_info = Bunch(
        # event regressors
        conditions=event_names,
        onsets=onsets,
        durations=durations,
        # confound regressors
        regressor_names=regressor_names,
        regressors=regressors
    )

    return subject_info


# In[3]:
# ======================================================================
# DEFINE FUNCTION: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
# ======================================================================
def get_subject_info_perm2_pm(events, confounds, onset_events):
    import pandas as pd
    import numpy as np
    from nipype.interfaces.base import Bunch

    # read the events and confounds files of the current run:
    # events = selectfiles_results.outputs.events[0]
    # confounds = selectfiles_results.outputs.confounds[0]
    # onset_events = selectfiles_results.outputs.onset_events[0]
    run_events = pd.read_csv(events, sep="\t")
    run_confounds = pd.read_csv(confounds, sep="\t")
    run_replay_onset_events = pd.read_csv(onset_events, sep='\t')

    subject = (events.split('behavior')[1])[1:7]
    # time of repetition, in seconds:
    time_repetition = 1.3
    # number of dummy variables to remove from each run:
    num_dummy = 4

    # delete dummy variable time
    run_events['onset'] = run_events['onset'] - num_dummy * time_repetition
    run_replay_onset_events['onsets'] = run_replay_onset_events['onsets'] - num_dummy * time_repetition

    # reset the markers of wrong trials
    for i in range(run_events[run_events['accuracy'] == 0].shape[0]):
        run_events.loc[run_events[run_events['accuracy'] == 0].index[i], 'Marker'] = 53

    # get the dummy variables
    run_confounds = run_confounds[num_dummy:]
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
    modulation = []

    replay_event_onset = {
        'fwd_fwd': {'isfwdTrial': 1},
        'bwd_bwd': {'isfwdTrial': 0},
    }
    for event2 in replay_event_onset:
        onset_list = list(
            run_replay_onset_events['onsets']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
        duration_list = list(
            run_replay_onset_events['duration']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
        modulation_list = list(
            run_replay_onset_events['probability']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []) & (modulation_list != []):
            event_names.append(event2)
            onsets.append(onset_list)
            durations.append(duration_list)
            modulation.append(modulation_list)

    # event types we consider:
    rep_event_spec = {
        'forward_correct': {'accuracy': 1, 'Marker': 51},
        'backward_correct': {'accuracy': 1, 'Marker': 52},
        'wrong': {'accuracy': 0, 'Marker': 53},
    }

    for event1 in rep_event_spec:
        onset_list = list(
            run_events['onset']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        duration_list = list(
            run_events['duration']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []):
            event_names.append(event1)
            onsets.append(onset_list)
            durations.append(duration_list)

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    replay_event_prob = {
        'fwd_prob': {'isfwdTrial': 1},
        'bwd_prob': {'isfwdTrial': 0},
    }

    for param_name in replay_event_prob:
        pmod_params_list = list(
            run_replay_onset_events['probability']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_prob[param_name]['isfwdTrial'])])
        if pmod_params_list != []:
            pmod_names.append(param_name)
            pmod_params.append(pmod_params_list)
            pmod_polys.append(1)

    # create a bunch for each regressor:
    fwd_prob = Bunch(name=[pmod_names[0]], param=[pmod_params[0]], poly=[pmod_polys[0]])
    bwd_prob = Bunch(name=[pmod_names[1]], param=[pmod_params[1]], poly=[pmod_polys[1]])

    if subject == 'sub-48':
        subject_info = Bunch(
            # event regressors
            conditions=event_names,
            onsets=onsets,
            durations=durations,
            # parametric modulation
            pmod=[fwd_prob, bwd_prob, None, None],
            orth=['No', 'No', 'No', 'No'],
            # confound regressors
            regressor_names=regressor_names,
            regressors=regressors
        )

    elif subject != 'sub-48':
        subject_info = Bunch(
            # event regressors
            conditions=event_names,
            onsets=onsets,
            durations=durations,
            # parametric modulation
            pmod=[fwd_prob, bwd_prob, None, None, None],
            orth=['No', 'No', 'No', 'No', 'No'],
            # confound regressors
            regressor_names=regressor_names,
            regressors=regressors
        )

    return subject_info


# In[2]:
# ======================================================================
# DEFINE FUNCTION: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
# ======================================================================
def get_subject_info_perm4(events, confounds, onset_events):
    import pandas as pd
    import numpy as np
    from nipype.interfaces.base import Bunch

    # read the events and confounds files of the current run:
    # events = selectfiles_results.outputs.events[0]
    # confounds = selectfiles_results.outputs.confounds[0]
    # onset_events = selectfiles_results.outputs.onset_events[0]
    run_events = pd.read_csv(events, sep="\t")
    run_confounds = pd.read_csv(confounds, sep="\t")
    run_replay_onset_events = pd.read_csv(onset_events, sep='\t')

    subject = (events.split('behavior')[1])[1:7]
    # time of repetition, in seconds:
    time_repetition = 1.3
    # number of dummy variables to remove from each run:
    num_dummy = 4

    # delete dummy variable time
    run_events['onset'] = run_events['onset'] - num_dummy * time_repetition
    run_replay_onset_events['onsets'] = run_replay_onset_events['onsets'] - num_dummy * time_repetition

    # reset the markers of wrong trials
    for i in range(run_events[run_events['accuracy'] == 0].shape[0]):
        run_events.loc[run_events[run_events['accuracy'] == 0].index[i], 'Marker'] = 53

    # get the dummy variables
    run_confounds = run_confounds[num_dummy:]
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

    replay_event_onset = {
        'fwd_fwd': {'isfwdTrial': 1, 'isfwdReplay': 1},
        'fwd_bwd': {'isfwdTrial': 1, 'isfwdReplay': 0},
        'bwd_fwd': {'isfwdTrial': 0, 'isfwdReplay': 1},
        'bwd_bwd': {'isfwdTrial': 0, 'isfwdReplay': 0},
    }

    for event2 in replay_event_onset:
        onset_list = list(
            run_replay_onset_events['onsets']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial']) &
             (run_replay_onset_events['isfwdReplay'] == replay_event_onset[event2]['isfwdReplay'])])
        duration_list = list(
            run_replay_onset_events['duration']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial']) &
             (run_replay_onset_events['isfwdReplay'] == replay_event_onset[event2]['isfwdReplay'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []):
            event_names.append(event2)
            onsets.append(onset_list)
            durations.append(duration_list)

    # event types we consider:
    rep_event_spec = {
        'forward_correct': {'accuracy': 1, 'Marker': 51},
        'backward_correct': {'accuracy': 1, 'Marker': 52},
        'wrong': {'accuracy': 0, 'Marker': 53},
    }

    for event1 in rep_event_spec:
        onset_list = list(
            run_events['onset']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        duration_list = list(
            run_events['duration']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []):
            event_names.append(event1)
            onsets.append(onset_list)
            durations.append(duration_list)

    subject_info = Bunch(
        # event regressors
        conditions=event_names,
        onsets=onsets,
        durations=durations,
        # confound regressors
        regressor_names=regressor_names,
        regressors=regressors
    )

    return subject_info


# In[3]:
# ======================================================================
# DEFINE FUNCTION: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
# ======================================================================
def get_subject_info_perm4_pm(events, confounds, onset_events):
    import pandas as pd
    import numpy as np
    from nipype.interfaces.base import Bunch

    # read the events and confounds files of the current run:
    # events = selectfiles_results.outputs.events[0]
    # confounds = selectfiles_results.outputs.confounds[0]
    # onset_events = selectfiles_results.outputs.onset_events[0]
    run_events = pd.read_csv(events, sep="\t")
    run_confounds = pd.read_csv(confounds, sep="\t")
    run_replay_onset_events = pd.read_csv(onset_events, sep='\t')

    subject = (events.split('behavior')[1])[1:7]
    # time of repetition, in seconds:
    time_repetition = 1.3
    # number of dummy variables to remove from each run:
    num_dummy = 4

    # delete dummy variable time
    run_events['onset'] = run_events['onset'] - num_dummy * time_repetition
    run_replay_onset_events['onsets'] = run_replay_onset_events['onsets'] - num_dummy * time_repetition

    # reset the markers of wrong trials
    for i in range(run_events[run_events['accuracy'] == 0].shape[0]):
        run_events.loc[run_events[run_events['accuracy'] == 0].index[i], 'Marker'] = 53

    # get the dummy variables
    run_confounds = run_confounds[num_dummy:]
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
    modulation = []

    replay_event_onset = {
        'fwd_fwd': {'isfwdTrial': 1, 'isfwdReplay': 1},
        'fwd_bwd': {'isfwdTrial': 1, 'isfwdReplay': 0},
        'bwd_fwd': {'isfwdTrial': 0, 'isfwdReplay': 1},
        'bwd_bwd': {'isfwdTrial': 0, 'isfwdReplay': 0},
    }

    for event2 in replay_event_onset:
        onset_list = list(
            run_replay_onset_events['onsets']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial']) &
             (run_replay_onset_events['isfwdReplay'] == replay_event_onset[event2]['isfwdReplay'])])
        duration_list = list(
            run_replay_onset_events['duration']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial']) &
             (run_replay_onset_events['isfwdReplay'] == replay_event_onset[event2]['isfwdReplay'])])
        modulation_list = list(
            run_replay_onset_events['probability']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial']) &
             (run_replay_onset_events['isfwdReplay'] == replay_event_onset[event2]['isfwdReplay'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []) & (modulation_list != []):
            event_names.append(event2)
            onsets.append(onset_list)
            durations.append(duration_list)
            modulation.append(modulation_list)

    # event types we consider:
    rep_event_spec = {
        'forward_correct': {'accuracy': 1, 'Marker': 51},
        'backward_correct': {'accuracy': 1, 'Marker': 52},
        'wrong': {'accuracy': 0, 'Marker': 53},
    }

    for event1 in rep_event_spec:
        onset_list = list(
            run_events['onset']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        duration_list = list(
            run_events['duration']
            [(run_events['task'] == 'REP') &
             (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []):
            event_names.append(event1)
            onsets.append(onset_list)
            durations.append(duration_list)

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    replay_event_prob = {
        'fwd_fwd_prob': {'isfwdTrial': 1, 'isfwdReplay': 1},
        'fwd_bwd_prob': {'isfwdTrial': 1, 'isfwdReplay': 0},
        'bwd_fwd_prob': {'isfwdTrial': 0, 'isfwdReplay': 1},
        'bwd_bwd_prob': {'isfwdTrial': 0, 'isfwdReplay': 0},
    }
    for param_name in replay_event_prob:
        pmod_params_list = list(
            run_replay_onset_events['probability']
            [(run_replay_onset_events['isfwdTrial'] == replay_event_prob[param_name]['isfwdTrial']) &
             (run_replay_onset_events['isfwdReplay'] == replay_event_prob[param_name]['isfwdReplay'])])
        if (pmod_params_list != []):
            pmod_names.append(param_name)
            pmod_params.append(pmod_params_list)
            pmod_polys.append(1)

    # create a bunch for each regressor:
    fwd_fwd_prob = Bunch(name=[pmod_names[0]], param=[pmod_params[0]], poly=[pmod_polys[0]])
    fwd_bwd_prob = Bunch(name=[pmod_names[1]], param=[pmod_params[1]], poly=[pmod_polys[1]])
    bwd_fwd_prob = Bunch(name=[pmod_names[2]], param=[pmod_params[2]], poly=[pmod_polys[2]])
    bwd_bwd_prob = Bunch(name=[pmod_names[3]], param=[pmod_params[3]], poly=[pmod_polys[3]])


    if subject != 'sub-48':
        subject_info = Bunch(
            # event regressors
            conditions=event_names,
            onsets=onsets,
            durations=durations,
            # parametric modulation
            pmod=[fwd_fwd_prob, fwd_bwd_prob, bwd_fwd_prob, bwd_bwd_prob, None, None, None],
            orth=['No', 'No', 'No', 'No', 'No', 'No', 'No'],
            # confound regressors
            regressor_names=regressor_names,
            regressors=regressors
        )
    elif subject == 'sub-48':
        subject_info = Bunch(
            # event regressors
            conditions=event_names,
            onsets=onsets,
            durations=durations,
            # parametric modulation
            pmod=[fwd_fwd_prob, fwd_bwd_prob, bwd_fwd_prob, bwd_bwd_prob, None, None],
            orth=['No', 'No', 'No', 'No', 'No', 'No'],
            # confound regressors
            regressor_names=regressor_names,
            regressors=regressors
        )

    return subject_info


# def get_subject_info_backup(events, confounds, onset_events):
#     """
#     FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
#     :param onset_events: list with paths to cue replay onset events files
#     :param events: list with paths to events files
#     :param confounds: list with paths to confounds files
#     :return: Bunch object with event onsets, durations and regressors
#     """
#
#     # import libraries (needed to be done in the function):
#     import pandas as pd
#     import numpy as np
#     from nipype.interfaces.base import Bunch
#
#     # read the events and confounds files of the current run:
#     # events = selectfiles_results.outputs.events[0]
#     # confounds = selectfiles_results.outputs.confounds[0]
#     # onset_events = selectfiles_results.outputs.onset_events[0]
#     run_events = pd.read_csv(events, sep="\t")
#     run_confounds = pd.read_csv(confounds, sep="\t")
#     run_replay_onset_events = pd.read_csv(onset_events, sep='\t')
#
#     subject = (events.split('behavior')[1])[1:7]
#     # time of repetition, in seconds:
#     time_repetition = 1.3
#     # number of dummy variables to remove from each run:
#     num_dummy = 4
#
#     # delete dummy variable time
#     run_events['onset'] = run_events['onset'] - num_dummy * time_repetition
#     run_replay_onset_events['onsets'] = run_replay_onset_events['onsets'] - num_dummy * time_repetition
#
#     # reset the markers of wrong trials
#     for i in range(run_events[run_events['accuracy'] == 0].shape[0]):
#         run_events.loc[run_events[run_events['accuracy'] == 0].index[i], 'Marker'] = 53
#     # delete the trials with nan rating scores
#     # for i in range(run_events[run_events['rating'] == 'None'].shape[0]):
#     #     run_events.loc[run_events[run_events['rating'] == 'None'].index[i], 'rating'] = np.nan
#     # run_events['rating'] = pd.to_numeric(run_events['rating'], errors='coerce')
#
#     # get the dummy variables
#     run_confounds = run_confounds[num_dummy:]
#
#     # original head motion code
#     # define confounds to include as regressors:
#     # confounds = ['trans', 'rot', 'a_comp_cor', 'framewise_displacement']
#     # search for confounds of interest in the confounds data frame:
#     # regressor_names = [col for col in run_confounds.columns if
#     #                    any([conf in col for conf in confounds])]
#
#     # Qi's setting
#     # regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
#     regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']
#
#     # Nico's setting
#     # regressor_names =['trans_x','trans_y','trans_z','rot_x','rot_y','rot_z',
#     #                   'a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_04','a_comp_cor_05',
#     #                   'framewise_displacement']
#
#     # set all the nan as mean value in confound variables
#     def replace_nan(regressor_values):
#         # calculate the mean value of the regressor:
#         mean_value = regressor_values.mean(skipna=True)
#         # replace all values containing nan with the mean value:
#         regressor_values[regressor_values.isnull()] = mean_value
#         # return list of the regressor values:
#         return list(regressor_values)
#
#     # create a nested list with regressor values
#     regressors = [replace_nan(run_confounds[conf]) for conf in regressor_names]
#
#     # set some empty lists
#     onsets = []
#     durations = []
#     event_names = []
#
#     if subject != 'sub-48':
#         # event types we consider:
#         rep_event_spec = {
#             'forward_correct': {'accuracy': 1, 'Marker': 51},
#             'backward_correct': {'accuracy': 1, 'Marker': 52},
#             'wrong': {'accuracy': 0, 'Marker': 53},
#         }
#         for event1 in rep_event_spec:
#             onset_list = list(
#                 run_events['onset']
#                 [(run_events['task'] == 'REP') &
#                  (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
#                  (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
#             duration_list = list(
#                 run_events['duration']
#                 [(run_events['task'] == 'REP') &
#                  (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
#                  (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
#             # it is not necessarily to append if it is an empty list
#             if (onset_list != []) & (duration_list != []):
#                 event_names.append(event1)
#                 onsets.append(onset_list)
#                 durations.append(duration_list)
#     elif subject == 'sub-48':
#         # event types we consider:
#         rep_event_spec = {
#             'forward_correct': {'accuracy': 1, 'Marker': 51},
#             'backward_correct': {'accuracy': 1, 'Marker': 52},
#         }
#         for event1 in rep_event_spec:
#             onset_list = list(
#                 run_events['onset']
#                 [(run_events['task'] == 'REP') &
#                  (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
#                  (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
#             duration_list = list(
#                 run_events['duration']
#                 [(run_events['task'] == 'REP') &
#                  (run_events['Marker'] == rep_event_spec[event1]['Marker']) &
#                  (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
#             # it is not necessarily to append if it is an empty list
#             if (onset_list != []) & (duration_list != []):
#                 event_names.append(event1)
#                 onsets.append(onset_list)
#                 durations.append(duration_list)
#
#     replay_event_onset = {
#         'fwd_fwd': {'isfwdTrial': 1},
#         'bwd_bwd': {'isfwdTrial': 0},
#     }
#     for event2 in replay_event_onset:
#         onset_list = list(
#             run_replay_onset_events['onsets']
#             [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
#         duration_list = list(
#             run_replay_onset_events['duration']
#             [(run_replay_onset_events['isfwdTrial'] == replay_event_onset[event2]['isfwdTrial'])])
#         # it is not necessarily to append if it is an empty list
#         if (onset_list != []) & (duration_list != []):
#             event_names.append(event2)
#             onsets.append(onset_list)
#             durations.append(duration_list)
#
#     subject_info = Bunch(
#         # event regressors
#         conditions=event_names,
#         onsets=onsets,
#         durations=durations,
#         # confound regressors
#         regressor_names=regressor_names,
#         regressors=regressors
#     )
#
#     return subject_info
