# ======================================================================
# DEFINE FUNCTION: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
# ======================================================================
def get_subject_info_max_pm_orth(events, confounds, onset_events):
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
    num_dummy = 0

    # delete dummy variable time
    run_events['onset'] = run_events['onset'] - num_dummy * time_repetition
    run_replay_onset_events['onsets'] = run_replay_onset_events['onsets'] - num_dummy * time_repetition

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

    # event types we consider:
    rep_event_spec = {
        'mental_simulation': {'accuracy': 1},
        'wrong': {'accuracy': 0},
    }

    for event1 in rep_event_spec:
        onset_list = list(
            run_events['onset']
            [(run_events['task'] == 'REP') &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        duration_list = list(
            run_events['duration']
            [(run_events['task'] == 'REP') &
             (run_events['accuracy'] == rep_event_spec[event1]['accuracy'])])
        # it is not necessarily to append if it is an empty list
        if (onset_list != []) & (duration_list != []):
            event_names.append(event1)
            onsets.append(onset_list)
            durations.append(duration_list)

    event2 = 'replay'
    onset_list = list(
        run_replay_onset_events['onsets'])
    duration_list = list(
        run_replay_onset_events['duration'])
    modulation_list = list(
        run_replay_onset_events['probability'])

    event_names.append(event2)
    onsets.append(onset_list)
    durations.append(duration_list)
    modulation.append(modulation_list)

    pmod_names = []
    pmod_params = []
    pmod_polys = []

    param_name = 'prob'
    pmod_params_list = list(
        run_replay_onset_events['probability'])
    if (pmod_params_list != []):
        pmod_names.append(param_name)
        pmod_params.append(pmod_params_list)
        pmod_polys.append(1)

    # create a bunch for each regressor:
    prob = Bunch(name=pmod_names, param=pmod_params, poly=pmod_polys)

    if event_names == ['mental_simulation', 'wrong', 'replay']:
        pmod=[None, None, prob]
        orth=['Yes', 'Yes', 'Yes']
    elif event_names == ['mental_simulation', 'replay']:
        pmod=[None, prob]
        orth=['Yes', 'Yes']
    else:
        raise Exception('The conditions are not expected.')

    subject_info = Bunch(
        # event regressors
        conditions=event_names, onsets=onsets, durations=durations,
        # parametric modulation
        pmod=pmod,
        orth=orth,
        # confound regressors
        regressor_names=regressor_names, regressors=regressors
    )

    return subject_info