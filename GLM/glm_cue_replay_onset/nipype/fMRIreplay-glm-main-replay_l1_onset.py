#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: FIRST LEVEL GLM
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITVIE NERUOSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY
# In[1]:
'''
======================================================================
IMPORT RELEVANT PACKAGES
======================================================================
'''
# import basic libraries:
import os
import glob
import time
import warnings
import numpy as np
from os.path import join as opj

# import nipype libraries:
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.utils.profiler import log_nodes_cb
from nipype import config, logging

# import spm and matlab interfaces:
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.spm.model import (
    Level1Design, EstimateModel, EstimateContrast, ThresholdStatistics,
    Threshold)
from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces import spm
from nipype.interfaces.spm import Smooth

# import fsl interfaces:`
from niflow.nipype1.workflows.fmri.fsl import create_susan_smooth
from nipype.interfaces.fsl.utils import ExtractROI

# import libraries for bids interaction:
from bids.layout import BIDSLayout

# import custom functions:
from fMRIreplay_glm_functions_l1_onset import (get_subject_info_max, get_subject_info_max_pm,
                                               get_subject_info_perm2, get_subject_info_perm2_pm,
                                               get_subject_info_perm4, get_subject_info_perm4_pm)

from glm_onset_pm_orth_function import get_subject_info_max_pm_orth
from nipype.algorithms.misc import Gunzip

# In[2]:
'''
======================================================================
ENVIRONMENT SETTINGS (DEALING WITH ERRORS AND WARNINGS):
======================================================================
'''
# set the fsl output type environment variable:
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
# deal with nipype-related warnings:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# inhibit CTF lock
os.environ['MCR_INHIBIT_CTF_LOCK'] = '1'
# filter out warnings related to the numpy package:
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")

# In[3]:
'''
======================================================================
SET PATHS AND SUBJECTS
======================================================================
'''
# define paths depending on the operating system (OS) platform:
# initialize empty paths:
sub_list = None
# path to the project root:
project_name = 'fmrireplay-glm-replay-l1'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_code = opj(path_bids, 'code', 'decoding-TDLM')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep')
path_behavior = opj(path_bids, 'derivatives', 'behavior')
path_maxonset = opj(path_bids, 'derivatives', 'replay-onset', 'Max_onset')
path_perm2 = opj(path_bids, 'derivatives', 'replay-onset', 'Perm_onset_2con')
path_perm4 = opj(path_bids, 'derivatives', 'replay-onset', 'Perm_onset_4con')

# set the path to MATLAB
spm.SPMCommand().set_mlab_paths(paths='/home/huangqi/MATLAB/spm12')
# test for spm
# spm.SPMCommand().version

# In[4]:
# grab the list of subjects from the bids data set:
layout = BIDSLayout(path_bids)
# get all subject ids:
sub_list = sorted(layout.get_subjects())
# lowfmri = [37, 16, 15, 13, 11, 0]
loweeg = [45, 38, 36, 34, 28, 21, 15, 11, 5, 4, 2]
incomplete = [42, 22, 4]
extremeheadmotion = [38, 28, 16, 15, 13, 9, 5, 0]
delete_list = np.unique(
    np.hstack([loweeg, incomplete, extremeheadmotion]))[::-1]
[sub_list.pop(i) for i in delete_list]

# In[5]:
'''
======================================================================
DEFINE SETTINGS
======================================================================
'''
# time of repetition, in seconds:
time_repetition = 1.3
# total number of runs:
num_runs = 3
# smoothing kernel, in mm:
fwhm = 6
# number of dummy variables to remove from each run:
num_dummy = 4


# In[]: GLM function

def onset_glm_1st(templates, get_subject_info, contrast_list, path_glm_l1):
    # start cue
    start_time = time.time()
    # print("Training set",set_id," ",ifold," start!")
    # In[6]:
    """
    ======================================================================
    DEFINE NODE: INFOSOURCE
    ======================================================================
    """
    # define the infosource node that collects the data:
    infosource = Node(IdentityInterface(
        fields=['subject_id']), name='infosource')
    # let the node iterate (parallelize) over all subjects:
    infosource.iterables = [('subject_id', sub_list)]

    # In[7]:
    '''
    ======================================================================
    DEFINE SELECTFILES NODE
    ======================================================================
    '''
    # define the selectfiles node:
    selectfiles = Node(SelectFiles(templates, sort_filelist=True),
                       name='selectfiles')
    # selectfiles.inputs.run_id = runs

    # selectfiles.inputs.subject_id = '11'
    # selectfiles_results = selectfiles.run()
    # selectfiles.run().outputs

    # In[9]:
    '''
    ======================================================================
    DEFINE CREATE_SUSAN_SMOOTH WORKFLOW NODE
    ======================================================================
    '''
    # define the susan smoothing node and specify the smoothing fwhm:
    # susan = create_susan_smooth()
    # # set the smoothing kernel:
    # susan.inputs.inputnode.fwhm = fwhm

    gunzip_func = MapNode(Gunzip(), name='gunzip_func', iterfield=['in_file'])
    smooth = Node(Smooth(fwhm=[fwhm, fwhm, fwhm]), name="smooth")

    # In[10]:
    '''
    ======================================================================
    DEFINE NODE: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
    ======================================================================
    '''
    subject_info = MapNode(Function(
        input_names=['events', 'confounds', 'onset_events'],
        output_names=['subject_info'],
        function=get_subject_info),
        name='subject_info', iterfield=['events', 'confounds', 'onset_events'])
    # check the correct trials from VFL task and transform to the event list
    # subject_info.inputs.events = selectfiles_results.outputs.events
    # subject_info.inputs.confounds = selectfiles_results.outputs.confounds
    # subject_info.inputs.onset_events = selectfiles_results.outputs.onset_events
    # subject_info_results = subject_info.run()

    # In[11]:
    '''
    ======================================================================
    DEFINE NODE: REMOVE DUMMY VARIABLES (USING FSL ROI)
    ======================================================================
    '''
    # remove the first 4 TR in one run (5.2s blank screen with 1.3s/TR)
    # function: extract region of interest (ROI) from an image
    trim = MapNode(ExtractROI(), name='trim', iterfield=['in_file'])
    # define index of the first selected volume (i.e., minimum index):
    trim.inputs.t_min = num_dummy
    # define the number of volumes selected starting at the minimum index:
    trim.inputs.t_size = -1
    # define the fsl output type:
    trim.inputs.output_type = 'NIFTI'

    # In[13]:
    '''
    ======================================================================
    DEFINE NODE: SPECIFY SPM MODEL (GENERATE SPM-SPECIFIC MODEL)
    ======================================================================
    '''
    # function: makes a model specification compatible with spm designers
    # adds SPM specific options to SpecifyModel
    l1model = Node(SpecifySPMModel(), name="l1model")
    # input: concatenate runs to a single session (boolean, default: False):
    l1model.inputs.concatenate_runs = False
    # input: units of event onsets and durations (secs or scans):
    l1model.inputs.input_units = 'secs'
    # input: units of design event onsets and durations (secs or scans):
    l1model.inputs.output_units = 'secs'
    # input: time of repetition (a float):
    l1model.inputs.time_repetition = time_repetition
    # high-pass filter cutoff in secs (a float, default = 128 secs):
    l1model.inputs.high_pass_filter_cutoff = 128

    # In[14]:
    '''
    ======================================================================
    DEFINE NODE: LEVEL 1 DESIGN (GENERATE AN SPM DESIGN MATRIX)
    ======================================================================
    '''
    # mask_img = r'/home/huangqi/Data/BIDS/sourcedata/glm_l1_masks/tpl-MNI152NLin2009cAsym_res-03_desc-brain_mask.nii'
    # check whether loading the mask appropriately
    # import nilearn.image as img
    # a=img.load_img(mask_img)

    # function: generate an SPM design matrix
    l1design = Node(Level1Design(), name="l1design")
    # input: (a dictionary with keys which are 'hrf' or 'fourier' or
    # 'fourier_han' or 'gamma' or 'fir' and with values which are any value)
    l1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
    # input: units for specification of onsets ('secs' or 'scans'):
    l1design.inputs.timing_units = 'secs'
    # input: interscan interval / repetition time in secs (a float):
    l1design.inputs.interscan_interval = time_repetition
    # input: Model serial correlations AR(1), FAST or none:
    l1design.inputs.model_serial_correlations = 'AR(1)'
    # input: number of time-bins per scan in secs (an integer):
    l1design.inputs.microtime_resolution = 23  # the number of slices / multiband accelerate
    # input: the onset/time-bin in seconds for alignment (a float):
    l1design.inputs.microtime_onset = 12
    # l1design.inputs.mask_image = mask_img
    # l1design.inputs.flags = {'mthresh': float('-inf'),'volt': 1}

    # In[15]:
    '''
    ======================================================================
    DEFINE NODE: ESTIMATE MODEL (ESTIMATE THE PARAMETERS OF THE MODEL)
    ======================================================================
    '''
    # function: use spm_spm to estimate the parameters of a model
    l1estimate = Node(EstimateModel(), name="l1estimate")
    # input: (a dictionary with keys which are 'Classical' or 'Bayesian2'
    # or 'Bayesian' and with values which are any value)
    l1estimate.inputs.estimation_method = {'Classical': 1}

    # In[16]:
    '''
    ======================================================================
    DEFINE NODE: ESTIMATE CONTRASTS (ESTIMATES THE CONTRASTS)
    ======================================================================
    '''
    # function: use spm_contrasts to estimate contrasts of interest
    l1contrasts = Node(EstimateContrast(contrasts=contrast_list), name="l1contrasts")
    # node input: overwrite previous results:
    l1contrasts.overwrite = True

    # In[20]:
    '''
    ======================================================================
    CREATE DATASINK NODE (OUTPUT STREAM):
    ======================================================================
    '''
    # create a node of the function:
    l1datasink = Node(DataSink(), name='datasink')
    # assign the path to the base directory:
    l1datasink.inputs.base_directory = opj(path_glm_l1, 'l1pipeline')
    # create a list of substitutions to adjust the file paths of datasink:
    substitutions = [('_subject_id_', 'sub-')]
    # assign the substitutions to the datasink command:
    l1datasink.inputs.substitutions = substitutions
    # determine whether to store output in parameterized form:
    l1datasink.inputs.parameterization = True

    # In[21]:
    '''
    ======================================================================
    DEFINE THE LEVEL 1 ANALYSIS SUB-WORKFLOW AND CONNECT THE NODES:
    ======================================================================
    '''
    # initiation of the 1st-level analysis workflow:
    l1analysis = Workflow(name='l1analysis')
    # connect the 1st-level analysis components
    l1analysis.connect(l1model, 'session_info', l1design, 'session_info')
    l1analysis.connect(l1design, 'spm_mat_file', l1estimate, 'spm_mat_file')
    l1analysis.connect(l1estimate, 'spm_mat_file', l1contrasts, 'spm_mat_file')
    l1analysis.connect(l1estimate, 'beta_images', l1contrasts, 'beta_images')
    l1analysis.connect(l1estimate, 'residual_image', l1contrasts, 'residual_image')

    # In[22]:
    '''
    ======================================================================
    ENABLE LOGGING:
    ======================================================================
    '''
    # initiation of the 1st-level analysis workflow:
    l1pipeline = Workflow(name='l1pipeline')
    # stop execution of the workflow if an error is encountered:
    l1pipeline.config = {'execution': {'stop_on_first_crash': True,
                                       'hash_method': 'timestamp'}}
    # define the base directory of the workflow:
    l1pipeline.base_dir = opj(path_glm_l1, 'work')

    # In[23]:
    '''
    ======================================================================
    CONNECT WORKFLOW NODES AND INPUT AND OUTPUT STREAM FOR THE LEVEL 1 SPM ANALYSIS SUB-WORKFLOW::
    ======================================================================
    '''
    # connect infosource to selectfiles node:
    l1pipeline.connect(infosource, 'subject_id', selectfiles, 'subject_id')
    # generate subject specific events and regressors to subject_info:
    l1pipeline.connect(selectfiles, 'events', subject_info, 'events')
    l1pipeline.connect(selectfiles, 'confounds', subject_info, 'confounds')
    l1pipeline.connect(selectfiles, 'onset_events', subject_info, 'onset_events')
    # connect functional files to smoothing workflow:
    # l1pipeline.connect(selectfiles, 'func', susan, 'inputnode.in_files')
    # l1pipeline.connect(selectfiles, 'wholemask', susan, 'inputnode.mask_file')

    # for SPM smoothing
    l1pipeline.connect([(selectfiles, gunzip_func, [('func', 'in_file')]),
                        (gunzip_func, smooth, [('out_file', 'in_files')]),
                        (smooth, trim, [('smoothed_files', 'in_file')])])
    l1pipeline.connect(smooth, 'smoothed_files', l1datasink, 'smooth')

    # # connect smoothed functional data to datasink:
    # l1pipeline.connect(susan, 'outputnode.smoothed_files', l1datasink, 'smooth')
    # # connect smoothed functional data to the trimming node:
    # l1pipeline.connect(susan, 'outputnode.smoothed_files', trim, 'in_file')
    # connect regressors to the level 1 model specification node:
    l1pipeline.connect(subject_info, 'subject_info', l1analysis, 'l1model.subject_info')
    # connect smoothed and trimmed data to the level 1 model specification:
    l1pipeline.connect(trim, 'roi_file', l1analysis, 'l1model.functional_runs')
    # connect all output results of the level 1 analysis to the datasink:
    l1pipeline.connect(l1analysis, 'l1estimate.beta_images', l1datasink, 'estimates.@beta_images')
    l1pipeline.connect(l1analysis, 'l1estimate.residual_image', l1datasink, 'estimates.@residual_image')
    l1pipeline.connect(l1analysis, 'l1contrasts.spm_mat_file', l1datasink, 'contrasts.@spm_mat')
    l1pipeline.connect(l1analysis, 'l1contrasts.spmT_images', l1datasink, 'contrasts.@spmT')
    l1pipeline.connect(l1analysis, 'l1contrasts.con_images', l1datasink, 'contrasts.@con')

    # In[25]:
    '''
    ======================================================================
    WRITE GRAPH AND EXECUTE THE WORKFLOW
    ======================================================================
    '''
    # write the graph:
    l1pipeline.write_graph(graph2use='colored', simple_form=True)
    l1pipeline.run(plugin='MultiProc')
    # l1pipeline.run('MultiProc', plugin_args={'n_procs': 30})

    end_time = time.time()
    run_time = round((end_time - start_time) / 60, 2)
    print(f"Run time cost {run_time}")


# In[8]:
# check the data files
path_confounds = glob.glob(opj(path_fmriprep, '*', 'func', '*replay*confounds_timeseries.tsv'))
path_events = glob.glob(opj(path_behavior, '*', '*task-rep*events.tsv'))
path_func = glob.glob(opj(path_fmriprep, '*', 'func', '*replay*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'))
path_wholemask = glob.glob(opj(path_fmriprep, '*', 'func', '*replay*space-MNI152NLin2009cAsym*brain_mask.nii.gz'))
path_maxonset_events = glob.glob(opj(path_maxonset, '*', '*replay_onset*events.tsv'))
path_permonset2_events = glob.glob(opj(path_perm2, '*', '*replay_onset*events.tsv'))
path_permonset4_events = glob.glob(opj(path_perm4, '*', '*replay_onset*events.tsv'))

# In[9]:
session_list = ['maxonset_events', 'maxonset_events_pm',
                'permonset2_events', 'permonset2_events_pm',
                'permonset4_events', 'permonset4_events_pm',
                'maxonset_events_pm_orth']

for session in session_list:
    if session == 'maxonset_events':
        # path templates
        templates = dict(
            confounds=opj(path_fmriprep, 'sub-{subject_id}', 'func', '*replay*confounds_timeseries.tsv'),
            events=opj(path_behavior, 'sub-{subject_id}', '*task-rep*events.tsv'),
            onset_events=opj(path_maxonset, 'sub-{subject_id}', '*replay_onset*events.tsv'),
            func=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                     '*replay*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
            wholemask=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                          '*replay*space-MNI152NLin2009cAsym*brain_mask.nii.gz'), )
        # subject info function
        get_subject_info = get_subject_info_max
        # contrasts for max cue replay onset first level GLM
        # condition_names = ['forward_correct', 'backward_correct', 'wrong', 'fwd_fwd', 'bwd_bwd']
        cont01 = ['allreplay', 'T', ['forward_correct', 'backward_correct'], [1, 1]]
        cont02 = ['fwd_replay', 'T', ['fwd_fwd'], [1]]
        cont03 = ['bwd_replay', 'T', ['bwd_bwd'], [1]]
        cont04 = ['fwd_bwd_replay', 'T', ['fwd_fwd', 'bwd_bwd'], [1, 1]]
        contrast_list = [cont01, cont02, cont03, cont04]
        # output path of GLM l1
        path_glm_l1 = opj(path_bids, 'derivatives', 'glm-l1-onset', 'maxonset')
        onset_glm_1st(templates=templates, get_subject_info=get_subject_info,
                      contrast_list=contrast_list, path_glm_l1=path_glm_l1)

    elif session == 'maxonset_events_pm':
        templates = dict(
            confounds=opj(path_fmriprep, 'sub-{subject_id}', 'func', '*replay*confounds_timeseries.tsv'),
            events=opj(path_behavior, 'sub-{subject_id}', '*task-rep*events.tsv'),
            onset_events=opj(path_maxonset, 'sub-{subject_id}', '*replay_onset*events.tsv'),
            func=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                     '*replay*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
            wholemask=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                          '*replay*space-MNI152NLin2009cAsym*brain_mask.nii.gz'),
        )
        get_subject_info = get_subject_info_max_pm
        # contrasts for max cue replay onset first level GLM
        # condition_names = ['forward_correct','backward_correct', 'wrong',
        #                    'fwd_fwd','fwd_fwdxfwd_prob^1', 'bwd_bwd', 'bwd_bwdxbwd_prob^1']
        cont01 = ['allreplay', 'T', ['forward_correct', 'backward_correct'], [1, 1]]
        cont02 = ['fwd_replay', 'T', ['fwd_fwd'], [1]]
        cont03 = ['bwd_replay', 'T', ['bwd_bwd'], [1]]
        cont04 = ['fwd_bwd_replay', 'T', ['fwd_fwd', 'bwd_bwd'], [1, 1]]
        cont05 = ['fwd_prob', 'T', ['fwd_fwdxfwd_prob^1'], [1]]
        cont06 = ['bwd_prob', 'T', ['bwd_bwdxbwd_prob^1'], [1]]
        cont07 = ['fwd_bwd_prob', 'T', ['fwd_fwdxfwd_prob^1', 'bwd_bwdxbwd_prob^1'], [1, 1]]
        cont08 = ['all_in', 'T', ['fwd_fwd', 'bwd_bwd',
                                  'fwd_fwdxfwd_prob^1', 'bwd_bwdxbwd_prob^1'], [1, 1, 1, 1]]
        contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07, cont08]
        # output path of GLM l1
        path_glm_l1 = opj(path_bids, 'derivatives', 'glm-l1-onset', 'maxonset-pm')
        onset_glm_1st(templates=templates, get_subject_info=get_subject_info,
                      contrast_list=contrast_list, path_glm_l1=path_glm_l1)

    elif session == 'permonset2_events':
        templates = dict(
            confounds=opj(path_fmriprep, 'sub-{subject_id}', 'func', '*replay*confounds_timeseries.tsv'),
            events=opj(path_behavior, 'sub-{subject_id}', '*task-rep*events.tsv'),
            onset_events=opj(path_perm2, 'sub-{subject_id}', '*replay_onset*events.tsv'),
            func=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                     '*replay*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
            wholemask=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                          '*replay*space-MNI152NLin2009cAsym*brain_mask.nii.gz'),
        )
        get_subject_info = get_subject_info_perm2
        # contrasts for max cue replay onset first level GLM
        # condition_names = ['forward_correct', 'backward_correct', 'wrong', 'fwd_fwd', 'bwd_bwd']
        cont01 = ['allreplay', 'T', ['forward_correct', 'backward_correct'], [1, 1]]
        cont02 = ['fwd_replay', 'T', ['fwd_fwd'], [1]]
        cont03 = ['bwd_replay', 'T', ['bwd_bwd'], [1]]
        cont04 = ['fwd_bwd_replay', 'T', ['fwd_fwd', 'bwd_bwd'], [1, 1]]
        contrast_list = [cont01, cont02, cont03, cont04]
        # output path of GLM l1
        path_glm_l1 = opj(path_bids, 'derivatives', 'glm-l1-onset', 'permonset2')
        onset_glm_1st(templates=templates, get_subject_info=get_subject_info,
                      contrast_list=contrast_list, path_glm_l1=path_glm_l1)

    elif session == 'permonset2_events_pm':
        templates = dict(
            confounds=opj(path_fmriprep, 'sub-{subject_id}', 'func', '*replay*confounds_timeseries.tsv'),
            events=opj(path_behavior, 'sub-{subject_id}', '*task-rep*events.tsv'),
            onset_events=opj(path_perm2, 'sub-{subject_id}', '*replay_onset*events.tsv'),
            func=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                     '*replay*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
            wholemask=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                          '*replay*space-MNI152NLin2009cAsym*brain_mask.nii.gz'),
        )
        get_subject_info = get_subject_info_perm2_pm
        # contrasts for max cue replay onset first level GLM
        # condition_names = ['forward_correct','backward_correct', 'wrong',
        #                    'fwd_fwd','fwd_fwdxfwd_prob^1', 'bwd_bwd', 'bwd_bwdxbwd_prob^1']
        cont01 = ['allreplay', 'T', ['forward_correct', 'backward_correct'], [1, 1]]
        cont02 = ['fwd_replay', 'T', ['fwd_fwd'], [1]]
        cont03 = ['bwd_replay', 'T', ['bwd_bwd'], [1]]
        cont04 = ['fwd_bwd_replay', 'T', ['fwd_fwd', 'bwd_bwd'], [1, 1]]
        cont05 = ['fwd_prob', 'T', ['fwd_fwdxfwd_prob^1'], [1]]
        cont06 = ['bwd_prob', 'T', ['bwd_bwdxbwd_prob^1'], [1]]
        cont07 = ['fwd_bwd_prob', 'T', ['fwd_fwdxfwd_prob^1', 'bwd_bwdxbwd_prob^1'], [1, 1]]
        cont08 = ['all_in', 'T', ['fwd_fwd', 'bwd_bwd',
                                  'fwd_fwdxfwd_prob^1', 'bwd_bwdxbwd_prob^1'], [1, 1, 1, 1]]
        contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07, cont08]
        # output path of GLM l1
        path_glm_l1 = opj(path_bids, 'derivatives', 'glm-l1-onset', 'permonset2-pm')
        onset_glm_1st(templates=templates, get_subject_info=get_subject_info,
                      contrast_list=contrast_list, path_glm_l1=path_glm_l1)

    elif session == 'permonset4_events':
        templates = dict(
            confounds=opj(path_fmriprep, 'sub-{subject_id}', 'func', '*replay*confounds_timeseries.tsv'),
            events=opj(path_behavior, 'sub-{subject_id}', '*task-rep*events.tsv'),
            onset_events=opj(path_perm4, 'sub-{subject_id}', '*replay_onset*events.tsv'),
            func=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                     '*replay*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
            wholemask=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                          '*replay*space-MNI152NLin2009cAsym*brain_mask.nii.gz'),
        )
        get_subject_info = get_subject_info_perm4
        # contrasts for max cue replay onset first level GLM
        # condition_names = ['forward_correct', 'backward_correct', 'wrong', 'fwd_fwd', 'fwd_bwd', 'bwd_fwd', 'bwd_bwd']
        cont01 = ['allreplay', 'T', ['forward_correct', 'backward_correct'], [1, 1]]
        cont02 = ['fwd_replay', 'T', ['fwd_fwd', 'bwd_fwd'], [1, 1]]
        cont03 = ['bwd_replay', 'T', ['bwd_bwd', 'fwd_bwd'], [1, 1]]
        cont04 = ['fwd_bwd_replay', 'T', ['fwd_fwd', 'bwd_fwd', 'bwd_bwd', 'fwd_bwd'], [1, 1, 1, 1]]
        contrast_list = [cont01, cont02, cont03, cont04]
        # output path of GLM l1
        path_glm_l1 = opj(path_bids, 'derivatives', 'glm-l1-onset', 'permonset4')
        onset_glm_1st(templates=templates, get_subject_info=get_subject_info,
                      contrast_list=contrast_list, path_glm_l1=path_glm_l1)
    elif session == 'permonset4_events_pm':
        templates = dict(
            confounds=opj(path_fmriprep, 'sub-{subject_id}', 'func', '*replay*confounds_timeseries.tsv'),
            events=opj(path_behavior, 'sub-{subject_id}', '*task-rep*events.tsv'),
            onset_events=opj(path_perm4, 'sub-{subject_id}', '*replay_onset*events.tsv'),
            func=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                     '*replay*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
            wholemask=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                          '*replay*space-MNI152NLin2009cAsym*brain_mask.nii.gz'),
        )
        get_subject_info = get_subject_info_perm4_pm
        # contrasts for max cue replay onset first level GLM
        # condition_names = ['forward_correct','backward_correct', 'wrong',
        #                    'fwd_fwd','fwd_fwdxfwd_prob^1', 'bwd_bwd', 'bwd_bwdxbwd_prob^1']
        cont01 = ['allreplay', 'T', ['forward_correct', 'backward_correct'], [1, 1]]
        cont02 = ['fwd_replay', 'T', ['fwd_fwd', 'fwd_bwd'], [1, 1]]
        cont03 = ['bwd_replay', 'T', ['fwd_bwd', 'bwd_bwd'], [1, 1]]
        cont04 = ['fwd_bwd_replay', 'T', ['fwd_fwd', 'fwd_bwd', 'bwd_fwd', 'bwd_bwd'], [1, 1, 1, 1]]
        cont05 = ['fwd_prob', 'T', ['fwd_fwdxfwd_fwd_prob^1', 'bwd_fwdxbwd_fwd_prob^1'], [1, 1]]
        cont06 = ['bwd_prob', 'T', ['fwd_bwdxfwd_bwd_prob^1', 'bwd_bwdxbwd_bwd_prob^1'], [1, 1]]
        cont07 = ['fwd_bwd_prob', 'T', ['fwd_fwdxfwd_fwd_prob^1', 'bwd_fwdxbwd_fwd_prob^1',
                                        'fwd_bwdxfwd_bwd_prob^1', 'bwd_bwdxbwd_bwd_prob^1'], [1, 1, 1, 1]]
        contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07]
        # output path of GLM l1
        path_glm_l1 = opj(path_bids, 'derivatives', 'glm-l1-onset', 'permonset4-pm')
        onset_glm_1st(templates=templates, get_subject_info=get_subject_info,
                      contrast_list=contrast_list, path_glm_l1=path_glm_l1)

    elif session == 'maxonset_events_pm_orth':
        templates = dict(
            confounds=opj(path_fmriprep, 'sub-{subject_id}', 'func', '*replay*confounds_timeseries.tsv'),
            events=opj(path_behavior, 'sub-{subject_id}', '*task-rep*events.tsv'),
            onset_events=opj(path_maxonset, 'sub-{subject_id}', '*replay_onset*events.tsv'),
            func=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                     '*replay*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'),
            wholemask=opj(path_fmriprep, 'sub-{subject_id}', 'func',
                          '*replay*space-MNI152NLin2009cAsym*brain_mask.nii.gz'),
        )
        get_subject_info = get_subject_info_max_pm_orth
        # contrasts for max cue replay onset first level GLM
        # condition_names = ['forward_correct','backward_correct', 'wrong',
        #                    'fwd_fwd','fwd_fwdxfwd_prob^1', 'bwd_bwd', 'bwd_bwdxbwd_prob^1']
        cont01 = ['mental_simulation', 'T', ['mental_simulation'], [1]]
        cont02 = ['replay', 'T', ['replay'], [1]]
        cont03 = ['replay_prob', 'T', ['replayxprob^1'], [1]]
        contrast_list = [cont01, cont02, cont03]
        # output path of GLM l1
        path_glm_l1 = opj(path_bids, 'derivatives', 'glm-l1-onset', 'maxonset-pm-orth')

    onset_glm_1st(templates=templates, get_subject_info=get_subject_info,
                  contrast_list=contrast_list, path_glm_l1=path_glm_l1)
