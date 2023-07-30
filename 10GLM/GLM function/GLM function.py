def l1_GLM_pm(subject):
    sub = 'sub-%s' % subject
    # create the result folder
    datasink = opj(path_glm_l1, sub)
    if not os.path.exists(datasink):
        os.makedirs(datasink)
    # set the design matrix
    func_runs = []
    design_matrices = []
    for i, run_id in enumerate(run_list):
        # load the data path
        confounds = templates['confounds'].format(subject=subject, run=run_id)
        events = templates['events'].format(subject=subject, run=str('%02.f' % run_id))
        onset_events = templates['onset_events'].format(subject=subject, run=str('%02.f' % run_id))
        func = templates['func'].format(subject=subject, run=run_id)
        # load the data
        confounds_file = pd.read_csv(confounds, sep='\t')
        events_file = pd.read_csv(events, sep='\t')
        onset_file = pd.read_csv(onset_events, sep='\t')
        func_run = load_img(func)
        # confound variables
        regressor_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']
        regressors = [replace_nan(confounds_file[conf]) for conf in regressor_names]
        # mental simulation and wrong trial regressor
        events_file['onset'] = events_file['onset']
        events_file['trial_type'] = events_file['accuracy'].apply(lambda x: 'mental_simulation' if x == 1 else 'wrong')
        events = events_file.loc[:, ['trial_type', 'onset', 'duration']]
        # parameters of design matrix
        n_scans = func_run.shape[-1]
        frame_times = np.arange(n_scans) * time_repetition  # here are the corresponding frame times
        motion = np.transpose(np.array(regressors))
        design_matrix = make_first_level_design_matrix(
            frame_times, events, drift_model=None, add_regs=motion,
            add_reg_names=regressor_names, hrf_model='spm', oversampling=100)
        # concatenate the HRF-Convolved probability regressor
        onset_file.set_index(frame_times, inplace=True, drop=True)
        design_matrix = pd.concat((onset_file, design_matrix), axis=1)
        design_matrix_orth = pd.DataFrame((_orthogonalize(np.array(design_matrix.iloc[:, [0, 1]]))),
                                          columns=['HRF_onset_TR', 'mental_simulation'],
                                          index=np.arange(int(n_scans)) * time_repetition)
        design_matrix_1 = pd.concat(
            (design_matrix_orth, design_matrix.drop(['HRF_onset_TR', 'mental_simulation'], axis=1)), axis=1)
        # output the data
        design_matrices.append(design_matrix_1)
        func_runs.append(func_run)

    # check the design matrix if needed
    # plot_design_matrix(design_matrices[0], output_file=opj(path_glm_l1, 'design_matrix_example_1.eps'),rescale=True)

    # fit first level glm
    mni_mask = opj(path_bids, 'sourcedata/glm_l1_masks/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii')
    fmri_glm = FirstLevelModel(t_r=time_repetition, slice_time_ref=0.5, hrf_model='spm',
                               drift_model=None, high_pass=1 / 128,
                               mask_img=mni_mask,
                               smoothing_fwhm=6, verbose=0,
                               noise_model='ar1', minimize_memory=True)
    fmri_glm = fmri_glm.fit(run_imgs=func_runs, design_matrices=design_matrices)

    # construct the contrasts
    contrasts = {'replay_con': [], 'ms_con': [], }
    for design_matrix in design_matrices:
        contrast_matrix = np.eye(design_matrix.shape[1])
        basic_contrasts = dict([(column, contrast_matrix[i])
                                for i, column in enumerate(design_matrix.columns)])
        replay_con = basic_contrasts['HRF_onset_TR']
        ms_con = basic_contrasts['mental_simulation']
        for contrast_id in ['replay_con', 'ms_con']:
            contrasts[contrast_id].append(eval(contrast_id))
    # compute contrast
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        # Estimate the contrasts. Note that the model implicitly computes a fixed effect across the three sessions
        stats_map = fmri_glm.compute_contrast(contrast_val, stat_type='t', output_type='all')
        c_map = stats_map['effect_size']
        c_image_path = opj(datasink, '%s_cmap.nii.gz' % contrast_id)
        c_map.to_filename(c_image_path)
        if index == 0:
            replay_beta = c_map
        elif index == 1:
            ms_beta = c_map

    return replay_beta, ms_beta