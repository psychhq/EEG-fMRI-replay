#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# SCRIPT INFORMATION:
# ======================================================================
# SCRIPT: CREATE BINARIZED MASKS FROM SEGMENTED FUNCTIONAL IMAGES
# PROJECT: HIGHSPEED
# WRITTEN BY LENNART WITTKUHN, 2018 - 2020
# CONTACT: WITTKUHN AT MPIB HYPHEN BERLIN DOT MPG DOT DE
# MAX PLANCK RESEARCH GROUP NEUROCODE
# MAX PLANCK INSTITUTE FOR HUMAN DEVELOPMENT
# MAX PLANCK UCL CENTRE FOR COMPUTATIONAL PSYCHIATRY AND AGEING RESEARCH
# LENTZEALLEE 94, 14195 BERLIN, GERMANY
# ======================================================================
# IMPORT RELEVANT PACKAGES
# ======================================================================
# import basic libraries:
import os
import sys
import glob
import warnings
from os.path import join as opj
# import nipype libraries:
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode
# import libraries for bids interaction:
from bids.layout import BIDSLayout
# import freesurfer interfaces:
from nipype.interfaces.freesurfer import Binarize
# import fsl interfaces:
from niflow.nipype1.workflows.fmri.fsl import create_susan_smooth
import datalad.api as dl
# ======================================================================
# ENVIRONMENT SETTINGS (DEALING WITH ERRORS AND WARNINGS):
# ======================================================================
# set the fsl output type environment variable:
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
# set the freesurfer subject directory:
os.environ['SUBJECTS_DIR'] = '/opt/software/freesurfer/6.0.0/subjects'
# deal with nipype-related warnings:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
# filter out warnings related to the numpy package:
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")
# ======================================================================
# DEFINE PBS CLUSTER JOB TEMPLATE (NEEDED WHEN RUNNING ON THE CLUSTER):
# ======================================================================
job_template = """
#PBS -l walltime=5:00:00
#PBS -j oe
#PBS -o $HOME/highspeed/highspeed-masks/logs
#PBS -m n
#PBS -v FSLOUTPUTTYPE=NIFTI_GZ
source /etc/bash_completion.d/virtualenvwrapper
workon highspeed-masks
module load fsl/5.0
module load freesurfer/6.0.0
"""
# ======================================================================
# DEFINE PATHS AND SUBJECTS
# ======================================================================
path_root = None
sub_list = None
# path to the project root:
project_name = 'highspeed-masks'
path_root = os.getenv('PWD').split(project_name)[0] + project_name
# grab the list of subjects from the bids data set:
path_bids = opj(path_root, 'bids')
dl.get(opj('bids', 'participants.json'))
dl.get(glob.glob(opj(path_root, 'bids', 'sub-*', 'ses-*', '*', '*.json')))
dl.get(glob.glob(opj(path_root, 'bids', 'sub-*', 'ses-*', '*.json')))
dl.get(glob.glob(opj(path_root, 'bids', '*.json')))
layout = BIDSLayout(path_bids)
# get all subject ids:
sub_list = sorted(layout.get_subjects())
# create a template to add the "sub-" prefix to the ids
sub_template = ['sub-'] * len(sub_list)
# add the prefix to all ids:
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]
# if user defined to run specific subject
sub_list = sub_list[int(sys.argv[1]):int(sys.argv[2])]
# ======================================================================
# DEFINE NODE: INFOSOURCE
# ======================================================================
# define the infosource node that collects the data:
infosource = Node(IdentityInterface(
    fields=['subject_id']), name='infosource')
# let the node iterate (parallelize) over all subjects:
infosource.iterables = [('subject_id', sub_list)]
# ======================================================================
# DEFINE SELECTFILES NODE
# ======================================================================
path_func = opj(path_root, 'fmriprep', 'fmriprep', '*', '*',
        'func', '*space-T1w*preproc_bold.nii.gz')
path_func_parc = opj(path_root, 'fmriprep', 'fmriprep', '*',
        '*', 'func', '*space-T1w*aparcaseg_dseg.nii.gz')
path_wholemask = opj(path_root, 'fmriprep', 'fmriprep', '*',
        '*', 'func', '*space-T1w*brain_mask.nii.gz')
dl.get(glob.glob(path_func))
dl.get(glob.glob(path_func_parc))
dl.get(glob.glob(path_wholemask))
# define all relevant files paths:
templates = dict(
    func=opj(path_root, 'fmriprep', 'fmriprep', '{subject_id}', '*',
             'func', '*space-T1w*preproc_bold.nii.gz'),
    func_parc=opj(path_root, 'fmriprep', 'fmriprep', '{subject_id}',
                  '*', 'func', '*space-T1w*aparcaseg_dseg.nii.gz'),
    wholemask=opj(path_root, 'fmriprep', 'fmriprep', '{subject_id}',
                  '*', 'func', '*space-T1w*brain_mask.nii.gz'),
)
# define the selectfiles node:
selectfiles = Node(SelectFiles(templates, sort_filelist=True),
                   name='selectfiles')
# set expected thread and memory usage for the node:
selectfiles.interface.num_threads = 1
selectfiles.interface.estimated_memory_gb = 0.1
# selectfiles.inputs.subject_id = 'sub-20'
# selectfiles_results = selectfiles.run()
# ======================================================================
# DEFINE CREATE_SUSAN_SMOOTH WORKFLOW NODE
# ======================================================================
# define the susan smoothing node and specify the smoothing fwhm:
susan = create_susan_smooth()
# set the smoothing kernel:
susan.inputs.inputnode.fwhm = 4
# set expected thread and memory usage for the nodes:
susan.get_node('inputnode').interface.num_threads = 1
susan.get_node('inputnode').interface.estimated_memory_gb = 0.1
susan.get_node('median').interface.num_threads = 1
susan.get_node('median').interface.estimated_memory_gb = 3
susan.get_node('mask').interface.num_threads = 1
susan.get_node('mask').interface.estimated_memory_gb = 3
susan.get_node('meanfunc2').interface.num_threads = 1
susan.get_node('meanfunc2').interface.estimated_memory_gb = 3
susan.get_node('merge').interface.num_threads = 1
susan.get_node('merge').interface.estimated_memory_gb = 3
susan.get_node('multi_inputs').interface.num_threads = 1
susan.get_node('multi_inputs').interface.estimated_memory_gb = 3
susan.get_node('smooth').interface.num_threads = 1
susan.get_node('smooth').interface.estimated_memory_gb = 3
susan.get_node('outputnode').interface.num_threads = 1
susan.get_node('outputnode').interface.estimated_memory_gb = 0.1
# ======================================================================
# DEFINE BINARIZE NODE
# ======================================================================
mask_visual_labels = [
    1005, 2005,  # cuneus
    1011, 2011,  # lateral occipital
    1021, 2021,  # pericalcarine
    1029, 2029,  # superioparietal
    1013, 2013,  # lingual
    1008, 2008,  # inferioparietal
    1007, 2007,  # fusiform
    1009, 2009,  # inferiotemporal
    1016, 2016,  # parahippocampal
    1015, 2015,  # middle temporal
]
mask_hippocampus_labels = [
    17, 53,  # left and right hippocampus
]
mask_mtl_labels = [
    17, 53,  # left and right hippocampus
    1016, 2016,  # parahippocampal
    1006, 2006,  # ctx-entorhinal
]
# function: use freesurfer mri_binarize to threshold an input volume
mask_visual = MapNode(
        interface=Binarize(), name='mask_visual', iterfield=['in_file'])
# input: match instead of threshold
mask_visual.inputs.match = mask_visual_labels
# optimize the efficiency of the node:
mask_visual.plugin_args = {'qsub_args': '-l nodes=1:ppn=1', 'overwrite': True}
mask_visual.plugin_args = {'qsub_args': '-l mem=100MB', 'overwrite': True}
# function: use freesurfer mri_binarize to threshold an input volume
mask_hippocampus = MapNode(
        interface=Binarize(), name='mask_hippocampus', iterfield=['in_file'])
# input: match instead of threshold
mask_hippocampus.inputs.match = mask_hippocampus_labels
# optimize the efficiency of the node:
mask_hippocampus.plugin_args = {
    'qsub_args': '-l nodes=1:ppn=1', 'overwrite': True}
mask_hippocampus.plugin_args = {
    'qsub_args': '-l mem=100MB', 'overwrite': True}
# function: use freesurfer mri_binarize to threshold an input volume
mask_mtl = MapNode(
        interface=Binarize(), name='mask_mtl', iterfield=['in_file'])
# input: match instead of threshold
mask_mtl.inputs.match = mask_mtl_labels
# optimize the efficiency of the node:
mask_mtl.plugin_args = {'qsub_args': '-l nodes=1:ppn=1', 'overwrite': True}
mask_mtl.plugin_args = {'qsub_args': '-l mem=100MB', 'overwrite': True}
# ======================================================================
# CREATE DATASINK NODE
# ======================================================================
# create a node of the function:
datasink = Node(DataSink(), name='datasink')
# assign the path to the base directory:
datasink.inputs.base_directory = opj(path_root, 'masks')
# create a list of substitutions to adjust the filepaths of datasink:
substitutions = [('_subject_id_', '')]
# assign the substitutions to the datasink command:
datasink.inputs.substitutions = substitutions
# determine whether to store output in parameterized form:
datasink.inputs.parameterization = True
# set expected thread and memory usage for the node:
datasink.interface.num_threads = 1
datasink.interface.estimated_memory_gb = 0.2
# ======================================================================
# DEFINE WORKFLOW PIPELINE:
# ======================================================================
# initiation of the 1st-level analysis workflow:
wf = Workflow(name='masks')
# stop execution of the workflow if an error is encountered:
wf.config = {'execution': {'stop_on_first_crash': True}}
# define the base directory of the workflow:
wf.base_dir = opj(path_root, 'work')
# connect infosource to selectfiles node:
wf.connect(infosource, 'subject_id', selectfiles, 'subject_id')
# connect functional files to smoothing workflow:
wf.connect(selectfiles, 'func', susan, 'inputnode.in_files')
wf.connect(selectfiles, 'wholemask', susan, 'inputnode.mask_file')
wf.connect(susan, 'outputnode.smoothed_files', datasink, 'smooth')
# connect segmented functional files to visual mask node
wf.connect(selectfiles, 'func_parc', mask_visual, 'in_file')
wf.connect(mask_visual, 'binary_file', datasink, 'mask_visual.@binary')
wf.connect(mask_visual, 'count_file', datasink, 'mask_visual.@count')
# connect segmented functional files to hippocampus node
wf.connect(selectfiles, 'func_parc', mask_hippocampus, 'in_file')
wf.connect(mask_hippocampus, 'binary_file', datasink, 'mask_hippocampus.@binary')
wf.connect(mask_hippocampus, 'count_file', datasink, 'mask_hippocampus.@count')
# connect segmented functional files to mtl node
wf.connect(selectfiles, 'func_parc', mask_mtl, 'in_file')
wf.connect(mask_mtl, 'binary_file', datasink, 'mask_mtl.@binary')
wf.connect(mask_mtl, 'count_file', datasink, 'mask_mtl.@count')
# ======================================================================
# WRITE GRAPH AND EXECUTE THE WORKFLOW
# ======================================================================
# write the graph:
wf.write_graph(graph2use='colored', simple_form=True)
# set the maximum resources the workflow can utilize:
# args_dict = {'status_callback' : log_nodes_cb}
# execute the workflow depending on the operating system:
if 'darwin' in sys.platform:
    # will execute the workflow using all available cpus:
    wf.run(plugin='MultiProc')
elif 'linux' in sys.platform:
    wf.run(plugin='PBS', plugin_args=dict(template=job_template))
