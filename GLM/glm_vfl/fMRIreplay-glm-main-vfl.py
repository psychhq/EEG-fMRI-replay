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
# import sys
import warnings
from os.path import join as opj
import numpy as np
# import nipype libraries:
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode
# from nipype.utils.profiler import log_nodes_cb
# from nipype import config, logging
# import spm and matlab interfaces:
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.spm.model import (
    Level1Design, EstimateModel, EstimateContrast, ThresholdStatistics,
    Threshold)
from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces import spm
# import fsl interfaces:`
from niflow.nipype1.workflows.fmri.fsl import create_susan_smooth
from nipype.interfaces.fsl.utils import ExtractROI
# import libraries for bids interaction:
from bids.layout import BIDSLayout
# import freesurfer interfaces:
# import custom functions:
from fMRIreplay_glm_functions_vfl import (
    get_subject_info, plot_stat_maps, leave_one_out)

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
project_name = 'fmrireplay-glm-vfl'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_behavior = opj(path_bids, 'derivatives', 'behavior')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep', 'fmriprep')
path_glm = opj(path_bids, 'derivatives', 'glm-vfl')

path_spm = '/home/huangqi/MATLAB/spm12'
path_matlab = '/usr/local/MATLAB/R2020b/bin/matlab -nodesktop -nosplash'
spm.SPMCommand.set_mlab_paths(paths=path_spm, matlab_cmd=path_matlab)
MatlabCommand.set_default_paths(path_spm)
MatlabCommand.set_default_matlab_cmd(path_matlab)

# In[4]:
# grab the list of subjects from the bids data set:
layout = BIDSLayout(path_bids)
# get all subject ids:
sub_list = sorted(layout.get_subjects())
# create a template to add the "sub-" prefix to the ids
sub_template = ['sub-'] * len(sub_list)
# add the prefix to all ids:
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]
# run workflow for specific subject
# delete the subjects
# lowfmri = [37, 16, 15, 13, 11, 0]
loweeg = [45, 38, 36, 34, 28, 21, 15, 11, 5, 4]
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
num_runs = 4
# total number of folds for leave-one-out-run:
num_fold = 8
# smoothing kernel, in mm:
fwhm = 6
# number of dummy variables to remove from each run:
num_dummy = 4

# In[6]:
'''
======================================================================
DEFINE NODE: INFOSOURCE
======================================================================
'''
# define the infosource node that collects the data:
infosource = Node(IdentityInterface(
    fields=['subject_id']), name='infosource')
# let the node iterate (paralellize) over all subjects:
infosource.iterables = [('subject_id', sub_list)]

# In[7]:
'''
======================================================================
DEFINE SELECTFILES NODE
======================================================================
'''
# check the data files
path_confounds = glob.glob(opj(path_fmriprep, '*', 'func', '*vfl*confounds_timeseries.tsv'))
path_events = glob.glob(opj(path_behavior, '*', '*vfl*events.tsv'))
path_func = glob.glob(opj(path_fmriprep, '*', 'func', '*vfl*space-T1w*preproc_bold.nii.gz'))
path_anat = glob.glob(opj(path_fmriprep, '*', 'anat', '*_desc-preproc_T1w.nii.gz'))
path_wholemask = glob.glob(opj(path_fmriprep, '*', 'func', '*vfl*space-T1w*brain_mask.nii.gz'))

# In[8]:
# define all relevant files paths:
templates = dict(
    confounds=opj(path_fmriprep, '{subject_id}', 'func', '*vfl*confounds_timeseries.tsv'),
    events=opj(path_behavior, '{subject_id}', '*vfl*events.tsv'),
    func=opj(path_fmriprep, '{subject_id}', 'func', '*vfl*space-T1w*preproc_bold.nii.gz'),
    anat=opj(path_fmriprep, '{subject_id}', 'anat', '{subject_id}_desc-preproc_T1w.nii.gz'),
    wholemask=opj(path_fmriprep, '{subject_id}', 'func', '*vfl*space-T1w*brain_mask.nii.gz'),
)
# define the selectfiles node:
selectfiles = Node(SelectFiles(templates, sort_filelist=True),
                   name='selectfiles')
# set expected thread and memory usage for the node:
# selectfiles.interface.num_threads = 1
# selectfiles.interface.mem_gb = 0.1
# selectfiles.inputs.subject_id = 'sub-04'
# selectfiles_results = selectfiles.run()
# selectfiles.run().outputs

# In[9]:
'''
======================================================================
DEFINE CREATE_SUSAN_SMOOTH WORKFLOW NODE
======================================================================
'''
# define the susan smoothing node and specify the smoothing fwhm:
susan = create_susan_smooth()
# set the smoothing kernel:
susan.inputs.inputnode.fwhm = fwhm

# In[10]:
'''
======================================================================
DEFINE NODE: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
======================================================================
'''
subject_info = MapNode(Function(
    input_names=['events', 'confounds'],
    output_names=['subject_info', 'event_names'],
    function=get_subject_info),
    name='subject_info', iterfield=['events', 'confounds'])
# check the correct trials from VFL task and transform to the event list
# subject_info.inputs.events = selectfiles_results.outputs.events
# subject_info.inputs.confounds = selectfiles_results.outputs.confounds
# subject_info_results = subject_info.run()
# subject_info_results.outputs

# In[11]:
'''
======================================================================
DEFINE NODE: REMOVE DUMMY VARIABLES (USING FSL ROI)
======================================================================
'''
# remove the first 4 TR in one run (5.2s blank screen with 1.3TR)
# function: extract region of interest (ROI) from an image
trim = MapNode(ExtractROI(), name='trim', iterfield=['in_file'])
# define index of the first selected volume (i.e., minimum index):
trim.inputs.t_min = num_dummy
# define the number of volumes selected starting at the minimum index:
trim.inputs.t_size = -1
# define the fsl output type:
trim.inputs.output_type = 'NIFTI'

# In[12]:
'''
======================================================================
DEFINE NODE: LEAVE-ONE-RUN-OUT SELECTION OF DATA
======================================================================
'''
leave_one_run_out = Node(Function(
    input_names=['subject_info', 'event_names', 'data_func', 'run'],
    output_names=['subject_info', 'data_func', 'contrasts'],
    function=leave_one_out),
    name='leave_one_run_out')
# define the number of rows as an iterable:
leave_one_run_out.iterables = ('run', range(num_runs))

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
# function: generate an SPM design matrix
l1design = Node(Level1Design(), name="l1design")
# input: (a dictionary with keys which are 'hrf' or 'fourier' or
# 'fourier_han' or 'gamma' or 'fir' and with values which are any value)
l1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}  # ??
# input: units for specification of onsets ('secs' or 'scans'):
l1design.inputs.timing_units = 'secs'
# input: interscan interval / repetition time in secs (a float):
l1design.inputs.interscan_interval = time_repetition
# input: Model serial correlations AR(1), FAST or none:
l1design.inputs.model_serial_correlations = 'AR(1)'
# input: number of time-bins per scan in secs (an integer):
l1design.inputs.microtime_resolution = 32  # the number of slices / multiband accelerate
# input: the onset/time-bin in seconds for alignment (a float):
l1design.inputs.microtime_onset = 1
# specify the mask for better entorhinal cortex signal
# l1design.inputs.mask_image = mask_img
# l1design.inputs.flags = {'mthresh':float(0.4)}


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
l1contrasts = Node(EstimateContrast(), name="l1contrasts")
# input: list of contrasts with each contrast being a list of the form:
# [('name', 'stat', [condition list], [weight list], [session list])]:
# l1contrasts.inputs.contrasts = l1contrasts_list
# node input: overwrite previous results:
l1contrasts.overwrite = True

# In[17]:
'''
======================================================================
DEFINE NODE: FUNCTION TO PLOT CONTRASTS
======================================================================
'''
plot_contrasts = MapNode(Function(
    input_names=['anat', 'stat_map', 'thresh'],
    output_names=['out_path'],
    function=plot_stat_maps),
    name='plot_contrasts', iterfield=['thresh'])
# input: plot data with set of different thresholds:
plot_contrasts.inputs.thresh = [None, 1, 2, 3]

# In[18]:
'''
======================================================================
DEFINE NODE: THRESHOLD
======================================================================
'''
# function: Topological FDR thresholding based on cluster extent/size.
# Smoothness is estimated from GLM residuals but is assumed to be the
# same for all the voxels.
thresh = Node(Threshold(), name="thresh")
# input: whether to use FWE (Bonferroni) correction for initial threshold
# (a boolean, nipype default value: True):
thresh.inputs.use_fwe_correction = False
# input: whether to use FDR over cluster extent probabilities (boolean)
thresh.inputs.use_topo_fdr = True
# input: value for initial thresholding (defining clusters):
thresh.inputs.height_threshold = 0.05
# input: is the cluster forming threshold a stat value or p-value?
# ('p-value' or 'stat', nipype default value: p-value):
thresh.inputs.height_threshold_type = 'p-value'
# input: which contrast in the SPM.mat to use (an integer):
thresh.inputs.contrast_index = 1
# input: p threshold on FDR corrected cluster size probabilities (float):
thresh.inputs.extent_fdr_p_threshold = 0.05
# input: minimum cluster size in voxels (an integer, default = 0):
thresh.inputs.extent_threshold = 0

# In[19]:
'''
======================================================================
DEFINE NODE: THRESHOLD STATISTICS
======================================================================
'''
# function: Given height and cluster size threshold calculate
# theoretical probabilities concerning false positives
thresh_stat = Node(ThresholdStatistics(), name="thresh_stat")
# input: which contrast in the SPM.mat to use (an integer):
thresh_stat.inputs.contrast_index = 1

# In[20]:
'''
======================================================================
CREATE DATASINK NODE (OUTPUT STREAM):
======================================================================
'''
# create a node of the function:
l1datasink = Node(DataSink(), name='datasink')
# assign the path to the base directory:
l1datasink.inputs.base_directory = opj(path_glm, 'l1pipeline')
# create a list of substitutions to adjust the file paths of datasink:
substitutions = [('_subject_id_', '')]
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
l1pipeline.base_dir = opj(path_glm, 'work')


# In[23]:
'''
======================================================================
CONNECT WORKFLOW NODES:
======================================================================
'''
# connect infosource to selectfiles node:
l1pipeline.connect(infosource, 'subject_id', selectfiles, 'subject_id')
# generate subject specific events and regressors to subject_info:
l1pipeline.connect(selectfiles, 'events', subject_info, 'events')
l1pipeline.connect(selectfiles, 'confounds', subject_info, 'confounds')
# connect functional files to smoothing workflow:
l1pipeline.connect(selectfiles, 'func', susan, 'inputnode.in_files')
l1pipeline.connect(selectfiles, 'wholemask', susan, 'inputnode.mask_file')
l1pipeline.connect(susan, 'outputnode.smoothed_files', l1datasink, 'smooth')
# connect smoothed functional data to the trimming node:
l1pipeline.connect(susan, 'outputnode.smoothed_files', trim, 'in_file')

# In[24]:
'''
======================================================================
INPUT AND OUTPUT STREAM FOR THE LEVEL 1 SPM ANALYSIS SUB-WORKFLOW:
======================================================================
'''
# connect regressors to the subsetting node::
l1pipeline.connect(subject_info, 'subject_info', leave_one_run_out, 'subject_info')
# connect event_names to the subsetting node:
l1pipeline.connect(subject_info, 'event_names', leave_one_run_out, 'event_names')
# connect smoothed and trimmed data to subsetting node:
l1pipeline.connect(trim, 'roi_file', leave_one_run_out, 'data_func')
# connect regressors to the level 1 model specification node:
l1pipeline.connect(leave_one_run_out, 'subject_info', l1analysis, 'l1model.subject_info')
# connect smoothed and trimmed data to the level 1 model specification:
l1pipeline.connect(leave_one_run_out, 'data_func', l1analysis, 'l1model.functional_runs')
# connect l1 contrast specification to contrast estimation
l1pipeline.connect(leave_one_run_out, 'contrasts', l1analysis, 'l1contrasts.contrasts')
# connect the anatomical image to the plotting node:
l1pipeline.connect(selectfiles, 'anat', plot_contrasts, 'anat')
# connect spm t-images to the plotting node:
l1pipeline.connect(l1analysis, 'l1contrasts.spmT_images', plot_contrasts, 'stat_map')
# connect the t-images and spm mat file to the threshold node:
l1pipeline.connect(l1analysis, 'l1contrasts.spmT_images', thresh, 'stat_image')
l1pipeline.connect(l1analysis, 'l1contrasts.spm_mat_file', thresh, 'spm_mat_file')
# connect all output results of the level 1 analysis to the datasink:
l1pipeline.connect(l1analysis, 'l1estimate.beta_images', l1datasink, 'estimates.@beta_images')
l1pipeline.connect(l1analysis, 'l1estimate.residual_image', l1datasink, 'estimates.@residual_image')
l1pipeline.connect(l1analysis, 'l1contrasts.spm_mat_file', l1datasink, 'contrasts.@spm_mat')
l1pipeline.connect(l1analysis, 'l1contrasts.spmT_images', l1datasink, 'contrasts.@spmT')
l1pipeline.connect(l1analysis, 'l1contrasts.con_images', l1datasink, 'contrasts.@con')
l1pipeline.connect(plot_contrasts, 'out_path', l1datasink, 'contrasts.@out_path')
l1pipeline.connect(thresh, 'thresholded_map', l1datasink, 'thresh.@threshhold_map')
l1pipeline.connect(thresh, 'pre_topo_fdr_map', l1datasink, 'thresh.@pre_topo_fdr_map')

# In[25]:
'''
======================================================================
WRITE GRAPH AND EXECUTE THE WORKFLOW
======================================================================
'''
# write the graph:
l1pipeline.write_graph(graph2use='colored', simple_form=True)
l1pipeline.run(plugin='MultiProc')
