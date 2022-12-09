#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: FIRST LEVEL GLM
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITVIE NERUOSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

# # ======================================================================
# # IMPORT RELEVANT PACKAGES
# # ======================================================================

# In[171]:


# import basic libraries:
import os
import glob
import sys
import warnings
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
# import fsl interfaces:`
from nipype.workflows.fmri.fsl import create_susan_smooth
from nipype.interfaces.fsl.utils import ExtractROI
# import libraries for bids interaction:
from bids.layout import BIDSLayout
# import freesurfer interfaces:
# import custom functions:
# from fMRIreplay_glm_functions_asd import (
#     get_subject_info, plot_stat_maps, leave_one_out)


# # ======================================================================
# # ENVIRONMENT SETTINGS (DEALING WITH ERRORS AND WARNINGS):
# # ======================================================================

# In[64]:


# set the fsl output type environment variable:
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
# deal with nipype-related warnings:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# inhibit CTF lock
os.environ['MCR_INHIBIT_CTF_LOCK'] = '1'
# filter out warnings related to the numpy package:
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")


# # ======================================================================
# # SET PATHS AND SUBJECTS
# # ======================================================================

# In[192]:
# define paths depending on the operating system (OS) platform:
project = 'fMRIreplay'
# initialize empty paths:
path_root = None
sub_list = None
# path to the project root:
project_name = 'fmrireplay-glm-replay'
path_root = opj(os.getcwd().split(project)[0] ,'fMRIreplay_hq')
path_bids = opj(path_root ,'fmrireplay-bids','BIDS')
path_bids_nf = opj(path_root ,'fmrireplay-bids','BIDS-nofieldmap')
path_behavior = opj(path_bids_nf ,'derivatives','fmrireplay-behavior')
path_fmriprep = opj(path_bids_nf ,'derivatives','fmrireplay-fmriprep')
path_glm = opj(path_bids_nf,'derivatives',project_name)

#SPM setting
path_matlab = '/home/user01/Downloads/spm12/run_spm12.sh /usr/local/MATLAB/MATLAB_Runtime/v97/ script'
spm.SPMCommand.set_mlab_paths(matlab_cmd=path_matlab, use_mcr=True)

# In[121]:
# grab the list of subjects from the bids data set:
layout = BIDSLayout(path_bids_nf)
# get all subject ids:
sub_list = sorted(layout.get_subjects())
# create a template to add the "sub-" prefix to the ids
sub_template = ['sub-'] * len(sub_list)
# add the prefix to all ids:
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]
# if user defined to run specific subject
# sub_list = sub_list[int(sys.argv[1]):int(sys.argv[2])]

# # ======================================================================
# # DEFINE PBS CLUSTER JOB TEMPLATE (NEEDED WHEN RUNNING ON THE CLUSTER):
# # ======================================================================

# In[ ]:


# job_template = """
# #PBS -l walltime=10:00:00
# #PBS -j oe
# #PBS -o /home/user01/fMRIreplay_hq/fmrireplay-bids/BIDS/derivatives/fmrireplay-glm/logs/glm
# #PBS -m n
# #PBS -v FSLOUTPUTTYPE=NIFTI_GZ
# """

# source /etc/bash_completion.d/virtualenvwrapper
# workon fmrireplay-glm
# module load fsl/5.0
# module load matlab/R2017b
# module load freesurfer/6.0.0
# # ======================================================================
# # DEFINE SETTINGS
# # ======================================================================

# In[122]:

# time of repetition, in seconds:
time_repetition = 1.3
# total number of runs:
num_runs = 3
# total number of folds for leave-one-out-run:
num_fold = 8
# smoothing kernel, in mm:
fwhm = 6
# number of dummy variables to remove from each run:
num_dummy = 0

# # ======================================================================
# # DEFINE NODE: INFOSOURCE
# # ======================================================================

# In[210]:


# sub_list = ['sub-04','sub-05','sub-06','sub-07']


# In[211]:


# define the infosource node that collects the data:
infosource = Node(IdentityInterface(
    fields=['subject_id']), name='infosource')
# let the node iterate (paralellize) over all subjects:
infosource.iterables = [('subject_id', sub_list)]


# # ======================================================================
# # DEFINE SELECTFILES NODE
# # ======================================================================

# In[212]:


# check the data files
path_confounds = glob.glob(opj(path_fmriprep,'*','func', '*replay*confounds_timeseries.tsv'))
path_events = glob.glob(opj(path_behavior ,'*', '*rep*events.tsv')) 
path_func = glob.glob(opj(path_fmriprep ,'*','func', '*replay*space-T1w*preproc_bold.nii.gz'))
path_anat = glob.glob(opj(path_fmriprep ,'*','anat', '*_desc-preproc_T1w.nii.gz'))
path_wholemask = glob.glob(opj(path_fmriprep ,'*','func', '*replay*space-T1w*brain_mask.nii.gz'))

# In[214]:


# define all relevant files paths:
templates = dict(
    confounds=opj(path_fmriprep,'{subject_id}' , 'func','*replay*confounds_timeseries.tsv'),
    events=opj(path_behavior,'{subject_id}', '*rep*events.tsv'),
    func=opj(path_fmriprep,'{subject_id}', 'func','*replay*space-T1w*preproc_bold.nii.gz'),
    anat=opj(path_fmriprep,'{subject_id}', 'anat','{subject_id}_desc-preproc_T1w.nii.gz'),
    wholemask=opj(path_fmriprep,'{subject_id}','func','*replay*space-T1w*brain_mask.nii.gz'),
)
# define the selectfiles node:
selectfiles = Node(SelectFiles(templates, sort_filelist=True),
                   name='selectfiles')
# set expected thread and memory usage for the node:
selectfiles.interface.num_threads = 1
selectfiles.interface.mem_gb = 0.1
# selectfiles.inputs.subject_id = 'sub-04'
# selectfiles_results = selectfiles.run()
# selectfiles.run().outputs


# # ======================================================================
# # DEFINE CREATE_SUSAN_SMOOTH WORKFLOW NODE
# # ======================================================================

# In[227]:


# define the susan smoothing node and specify the smoothing fwhm:
susan = create_susan_smooth()
# set the smoothing kernel:
susan.inputs.inputnode.fwhm = fwhm
# set expected thread and memory usage for the nodes:
susan.get_node('inputnode').interface.num_threads = 1
susan.get_node('inputnode').interface.mem_gb = 0.1
susan.get_node('median').interface.num_threads = 1
susan.get_node('median').interface.mem_gb = 3
susan.get_node('mask').interface.num_threads = 1
susan.get_node('mask').interface.mem_gb = 3
susan.get_node('meanfunc2').interface.num_threads = 1
susan.get_node('meanfunc2').interface.mem_gb = 3
susan.get_node('merge').interface.num_threads = 1
susan.get_node('merge').interface.mem_gb = 3
susan.get_node('multi_inputs').interface.num_threads = 1
susan.get_node('multi_inputs').interface.mem_gb = 3
susan.get_node('smooth').interface.num_threads = 1
susan.get_node('smooth').interface.mem_gb = 3
susan.get_node('outputnode').interface.num_threads = 1
susan.get_node('outputnode').interface.mem_gb = 0.1


# # ======================================================================
# # DEFINE NODE: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
# # ======================================================================

# In[228]:


# subject_info = MapNode(Function(
#     input_names=['events', 'confounds'],
#     output_names=['subject_info', 'event_names'],
#     function=get_subject_info),
#     name='subject_info', iterfield=['events', 'confounds'])
# # set expected thread and memory usage for the node:
# subject_info.interface.num_threads = 1
# subject_info.interface.mem_gb = 0.1
# # check the correct trials from asd task and transform to the event list
# # subject_info.inputs.events = selectfiles_results.outputs.events
# # subject_info.inputs.confounds = selectfiles_results.outputs.confounds
# # subject_info_results = subject_info.run()
# # subject_info_results.outputs


# # # ======================================================================
# # # DEFINE NODE: REMOVE DUMMY VARIABLES (USING FSL ROI)
# # # ======================================================================

# # In[230]:


# # function: extract region of interest (ROI) from an image
# trim = MapNode(ExtractROI(), name='trim', iterfield=['in_file'])
# # define index of the first selected volume (i.e., minimum index):
# trim.inputs.t_min = num_dummy
# # define the number of volumes selected starting at the minimum index:
# trim.inputs.t_size = -1
# # define the fsl output type:
# trim.inputs.output_type = 'NIFTI'
# # set expected thread and memory usage for the node:
# trim.interface.num_threads = 1
# trim.interface.mem_gb = 3


# # # ======================================================================
# # # DEFINE NODE: LEAVE-ONE-RUN-OUT SELECTION OF DATA
# # # ======================================================================

# # In[231]:


# leave_one_run_out = Node(Function(
#     input_names=['subject_info', 'event_names', 'data_func', 'run'],
#     output_names=['subject_info', 'data_func', 'contrasts'],
#     function=leave_one_out),
#     name='leave_one_run_out')
# # define the number of rows as an iterable:
# leave_one_run_out.iterables = ('run', range(num_runs))


# # # ======================================================================
# # # DEFINE NODE: SPECIFY SPM MODEL (GENERATE SPM-SPECIFIC MODEL)
# # # ======================================================================

# # In[232]:


# # function: makes a model specification compatible with spm designers
# # adds SPM specific options to SpecifyModel
# l1model = Node(SpecifySPMModel(), name="l1model")
# # input: concatenate runs to a single session (boolean, default: False):
# l1model.inputs.concatenate_runs = False
# # input: units of event onsets and durations (secs or scans):
# l1model.inputs.input_units = 'secs'
# # input: units of design event onsets and durations (secs or scans):
# l1model.inputs.output_units = 'secs'
# # input: time of repetition (a float):
# l1model.inputs.time_repetition = time_repetition
# # high-pass filter cutoff in secs (a float, default = 128 secs):
# l1model.inputs.high_pass_filter_cutoff = 128


# # # ======================================================================
# # # DEFINE NODE: LEVEL 1 DESIGN (GENERATE AN SPM DESIGN MATRIX)
# # # ======================================================================

# # In[233]:


# # function: generate an SPM design matrix
# l1design = Node(Level1Design(), name="l1design")
# # input: (a dictionary with keys which are 'hrf' or 'fourier' or
# # 'fourier_han' or 'gamma' or 'fir' and with values which are any value)
# l1design.inputs.bases = {'hrf': {'derivs': [0, 0]}} #??
# # input: units for specification of onsets ('secs' or 'scans'):
# l1design.inputs.timing_units = 'secs'
# # input: interscan interval / repetition time in secs (a float):
# l1design.inputs.interscan_interval = time_repetition
# # input: Model serial correlations AR(1), FAST or none:
# l1design.inputs.model_serial_correlations = 'AR(1)'
# # input: number of time-bins per scan in secs (an integer):???
# l1design.inputs.microtime_resolution = 16
# # input: the onset/time-bin in seconds for alignment (a float):???
# l1design.inputs.microtime_onset = 1
# # set expected thread and memory usage for the node:
# l1design.interface.num_threads = 1
# l1design.interface.mem_gb = 2


# # # ======================================================================
# # # DEFINE NODE: ESTIMATE MODEL (ESTIMATE THE PARAMETERS OF THE MODEL)
# # # ======================================================================

# # In[234]:

# # function: use spm_spm to estimate the parameters of a model
# l1estimate = Node(EstimateModel(), name="l1estimate")
# # input: (a dictionary with keys which are 'Classical' or 'Bayesian2'
# # or 'Bayesian' and with values which are any value)
# l1estimate.inputs.estimation_method = {'Classical': 1}
# # set expected thread and memory usage for the node:
# l1estimate.interface.num_threads = 1
# l1estimate.interface.mem_gb = 2

# # # ======================================================================
# # # DEFINE NODE: ESTIMATE CONTRASTS (ESTIMATES THE CONTRASTS)
# # # ======================================================================

# # In[235]:

# # function: use spm_contrasts to estimate contrasts of interest
# l1contrasts = Node(EstimateContrast(), name="l1contrasts")
# # input: list of contrasts with each contrast being a list of the form:
# # [('name', 'stat', [condition list], [weight list], [session list])]:
# # l1contrasts.inputs.contrasts = l1contrasts_list
# # node input: overwrite previous results:
# l1contrasts.overwrite = True
# # set expected thread and memory usage for the node:
# l1contrasts.interface.num_threads = 1
# l1contrasts.interface.mem_gb = 1.5

# # # ======================================================================
# # # DEFINE NODE: FUNCTION TO PLOT CONTRASTS
# # # ======================================================================

# # In[236]:

# plot_contrasts = MapNode(Function(
#     input_names=['anat', 'stat_map', 'thresh'],
#     output_names=['out_path'],
#     function=plot_stat_maps),
#     name='plot_contrasts', iterfield=['thresh'])
# # input: plot data with set of different thresholds:
# plot_contrasts.inputs.thresh = [None, 1, 1.96, 2, 3]
# # set expected thread and memory usage for the node:
# plot_contrasts.interface.num_threads = 1
# plot_contrasts.interface.mem_gb = 0.2

# # # ======================================================================
# # # DEFINE NODE: THRESHOLD
# # # ======================================================================

# # In[237]:

# # function: Topological FDR thresholding based on cluster extent/size.
# # Smoothness is estimated from GLM residuals but is assumed to be the
# # same for all of the voxels.
# thresh = Node(Threshold(), name="thresh")
# # input: whether to use FWE (Bonferroni) correction for initial threshold
# # (a boolean, nipype default value: True):
# thresh.inputs.use_fwe_correction = False
# # input: whether to use FDR over cluster extent probabilities (boolean)
# thresh.inputs.use_topo_fdr = True
#  # input: value for initial thresholding (defining clusters):
# thresh.inputs.height_threshold = 0.05
# # input: is the cluster forming threshold a stat value or p-value?
# # ('p-value' or 'stat', nipype default value: p-value):
# thresh.inputs.height_threshold_type = 'p-value'
# # input: which contrast in the SPM.mat to use (an integer):
# thresh.inputs.contrast_index = 1
# # input: p threshold on FDR corrected cluster size probabilities (float):
# thresh.inputs.extent_fdr_p_threshold = 0.05
# # input: minimum cluster size in voxels (an integer, default = 0):
# thresh.inputs.extent_threshold = 0
# # set expected thread and memory usage for the node:
# thresh.interface.num_threads = 1
# thresh.interface.mem_gb = 0.2

# # # ======================================================================
# # # DEFINE NODE: THRESHOLD STATISTICS
# # # ======================================================================

# # In[238]:


# # function: Given height and cluster size threshold calculate
# # theoretical probabilities concerning false positives
# thresh_stat = Node(ThresholdStatistics(), name="thresh_stat")
# # input: which contrast in the SPM.mat to use (an integer):
# thresh_stat.inputs.contrast_index = 1


# # ======================================================================
# # CREATE DATASINK NODE (OUTPUT STREAM):
# # ======================================================================

# In[239]:


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
# set expected thread and memory usage for the node:
l1datasink.interface.num_threads = 1
l1datasink.interface.mem_gb = 0.2


# # ======================================================================
# # DEFINE THE LEVEL 1 ANALYSIS SUB-WORKFLOW AND CONNECT THE NODES:
# # ======================================================================

# In[240]:


# # initiation of the 1st-level analysis workflow:
# l1analysis = Workflow(name='l1analysis')
# # connect the 1st-level analysis components
# l1analysis.connect(l1model, 'session_info', l1design, 'session_info')
# l1analysis.connect(l1design, 'spm_mat_file', l1estimate, 'spm_mat_file')
# l1analysis.connect(l1estimate, 'spm_mat_file', l1contrasts, 'spm_mat_file')
# l1analysis.connect(l1estimate, 'beta_images', l1contrasts, 'beta_images')
# l1analysis.connect(l1estimate, 'residual_image', l1contrasts, 'residual_image')


# # ======================================================================
# # ENABLE LOGGING:
# # ======================================================================

# In[241]:


# initiation of the 1st-level analysis workflow:
l1pipeline = Workflow(name='l1pipeline')
# stop execution of the workflow if an error is encountered:
l1pipeline.config = {'execution': {'stop_on_first_crash': True,
                                   'hash_method': 'timestamp'}}
# define the base directory of the workflow:
l1pipeline.base_dir = opj(path_glm, 'work')


# # ======================================================================
# # CONNECT WORKFLOW NODES:
# # ======================================================================

# In[242]:


# connect infosource to selectfiles node:
l1pipeline.connect(infosource, 'subject_id', selectfiles, 'subject_id')
# generate subject specific events and regressors to subject_info:
# l1pipeline.connect(selectfiles, 'events', subject_info, 'events')
# l1pipeline.connect(selectfiles, 'confounds', subject_info, 'confounds')
# connect functional files to smoothing workflow:
l1pipeline.connect(selectfiles, 'func', susan, 'inputnode.in_files')
l1pipeline.connect(selectfiles, 'wholemask', susan, 'inputnode.mask_file')
l1pipeline.connect(susan, 'outputnode.smoothed_files', l1datasink, 'smooth')
# # connect smoothed functional data to the trimming node:
# l1pipeline.connect(susan, 'outputnode.smoothed_files', trim, 'in_file')


# # ======================================================================
# # INPUT AND OUTPUT STREAM FOR THE LEVEL 1 SPM ANALYSIS SUB-WORKFLOW:
# # ======================================================================

# In[244]:


# # connect regressors to the subsetting node::
# l1pipeline.connect(subject_info, 'subject_info', leave_one_run_out, 'subject_info')
# # connect event_names to the subsetting node:
# l1pipeline.connect(subject_info, 'event_names', leave_one_run_out, 'event_names')
# # connect smoothed and trimmed data to subsetting node:
# l1pipeline.connect(trim, 'roi_file', leave_one_run_out, 'data_func')
# # connect regressors to the level 1 model specification node:
# l1pipeline.connect(leave_one_run_out, 'subject_info', l1analysis, 'l1model.subject_info')
# # connect smoothed and trimmed data to the level 1 model specification:
# l1pipeline.connect(leave_one_run_out, 'data_func', l1analysis, 'l1model.functional_runs')
# # connect l1 contrast specification to contrast estimation
# l1pipeline.connect(leave_one_run_out, 'contrasts', l1analysis, 'l1contrasts.contrasts')
# # connect the anatomical image to the plotting node:
# l1pipeline.connect(selectfiles, 'anat', plot_contrasts, 'anat')
# # connect spm t-images to the plotting node:
# l1pipeline.connect(l1analysis, 'l1contrasts.spmT_images', plot_contrasts, 'stat_map')
# # connect the t-images and spm mat file to the threshold node:
# l1pipeline.connect(l1analysis, 'l1contrasts.spmT_images', thresh, 'stat_image')
# l1pipeline.connect(l1analysis, 'l1contrasts.spm_mat_file', thresh, 'spm_mat_file')
# # connect all output results of the level 1 analysis to the datasink:
# l1pipeline.connect(l1analysis, 'l1estimate.beta_images', l1datasink, 'estimates.@beta_images')
# l1pipeline.connect(l1analysis, 'l1estimate.residual_image', l1datasink, 'estimates.@residual_image')
# l1pipeline.connect(l1analysis, 'l1contrasts.spm_mat_file', l1datasink, 'contrasts.@spm_mat')
# l1pipeline.connect(l1analysis, 'l1contrasts.spmT_images', l1datasink, 'contrasts.@spmT')
# l1pipeline.connect(l1analysis, 'l1contrasts.con_images', l1datasink, 'contrasts.@con')
# l1pipeline.connect(plot_contrasts, 'out_path', l1datasink, 'contrasts.@out_path')
# l1pipeline.connect(thresh, 'thresholded_map', l1datasink, 'thresh.@threshhold_map')
# l1pipeline.connect(thresh, 'pre_topo_fdr_map', l1datasink, 'thresh.@pre_topo_fdr_map')


# # ======================================================================
# # WRITE GRAPH AND EXECUTE THE WORKFLOW
# # ======================================================================

# In[245]:


# write the graph:
l1pipeline.write_graph(graph2use='colored', simple_form=True)
l1pipeline.run(plugin='MultiProc')
#w set the maximum resources the workflow can utilize:
# args_dict = {'status_callback' : log_nodes_cb}
# execute the workflow depending on the operating system:
# if 'darwin' in sys.platform:
#     # will execute the workflow using all available cpus:
#     l1pipeline.run(plugin='MultiProc')
# elif 'linux' in sys.platform:
#     l1pipeline.run(plugin='PBS', plugin_args=dict(template=job_template))
