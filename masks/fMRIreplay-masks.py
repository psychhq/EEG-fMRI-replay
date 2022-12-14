#!/usr/bin/env python
# coding: utf-8

# SCRIPT INFORMATION:
# SCRIPT: CREATE BINARIZED MASKS FROM SEGMENTED FUNCTIONAL IMAGES
# PROJECT: FMRIREPLAY
# WRITTEN BY QI HUANG, 2022
# CONTACT: STATE KEY LABORATORY OF COGNITVIE NERUOSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

# In[1]:
'''
======================================================================
IMPORT RELEVANT PACKAGES
======================================================================
'''
# import basic libraries:
import os
import sys
import glob
import warnings
import numpy as np
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

# In[2]:
'''
======================================================================
ENVIRONMENT SETTINGS (DEALING WITH ERRORS AND WARNINGS):
======================================================================
'''
# set the fsl output type environment variable:
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
# set the freesurfer subject directory:
os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer/subjects'
# deal with nipype-related warnings:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
# filter out warnings related to the numpy package:
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")
# In[3]:
'''
======================================================================
DEFINE PATHS AND SUBJECTS
======================================================================
'''
sub_list = None
# grab the list of subjects from the bids data set:
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_fmriprep = opj(path_bids, 'derivatives', 'fmriprep', 'fmriprep')
path_masks = opj(path_bids, 'derivatives', 'masks')
layout = BIDSLayout(path_bids) #, derivatives=True)
# get all subject ids:
sub_list = sorted(layout.get_subjects())
# create a template to add the "sub-" prefix to the ids
sub_template = ['sub-'] * len(sub_list)
# add the prefix to all ids:
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]

loweeg = [45, 38, 36, 34, 28, 21, 15, 11, 5, 4]
incomplete = [42, 22, 4]
extremeheadmotion = [38, 28, 16, 15, 13, 9, 5, 0]
delete_list = np.unique(
    np.hstack([loweeg, incomplete, extremeheadmotion]))[::-1]
[sub_list.pop(i) for i in delete_list]

# In[4]:

# define the infosource node that collects the data:
infosource = Node(IdentityInterface(
    fields=['subject_id']), name='infosource')
# let the node iterate (parallelize) over all subjects:
infosource.iterables = [('subject_id', sub_list)]

# In[5]:

# check the data files
path_func = glob.glob(opj(path_fmriprep, '*', 'func', '*space-T1w*preproc_bold.nii.gz'))
path_func_parc = glob.glob(opj(path_fmriprep, '*', 'func', '*space-T1w*aparcaseg_dseg.nii.gz'))
path_wholemask = glob.glob(opj(path_fmriprep, '*', 'func', '*space-T1w*brain_mask.nii.gz'))

# define the templates
templates = dict(
    func=opj(path_fmriprep, '{subject_id}',
             'func', '*space-T1w*preproc_bold.nii.gz'),
    func_parc=opj(path_fmriprep, '{subject_id}',
                  'func', '*space-T1w*aparcaseg_dseg.nii.gz'),
    wholemask=opj(path_fmriprep, '{subject_id}',
                  'func', '*space-T1w*brain_mask.nii.gz'),
)
# define the selectfiles node:
selectfiles = Node(SelectFiles(templates, sort_filelist=True),
                   name='selectfiles')

# selectfiles.inputs.subject_id = 'sub-004'
# selectfiles_results = selectfiles.run()
# In[6]:
'''
======================================================================
DEFINE CREATE_SUSAN_SMOOTH WORKFLOW NODE
======================================================================
'''
# define the susan smoothing node and specify the smoothing fwhm:
susan = create_susan_smooth()
# set the smoothing kernel:
susan.inputs.inputnode.fwhm = 6

# In[6]:
'''
======================================================================
DEFINE BINARIZE NODE
======================================================================
'''
mask_visual_labels = [
    1005, 2005,  # cuneus
    1011, 2011,  # lateral occipital
    1021, 2021,  # pericalcarine
    1029, 2029,  # superioparietal
    1013, 2013,  # lingual
    1008, 2008,  # inferioparietal
    1007, 2007,  # fusiform
]
mask_hippocampus_labels = [
    17, 53,  # left and right hippocampus
]
mask_entorhinal_labels = [
    1006, 2006,  # left entorhinal right entorhinal
]
mask_mtl_labels = [
    17, 53,  # left and right hippocampus
    1016, 2016,  # parahippocampal
    1006, 2006,  # ctx-entorhinal
]
mask_vis_mtl_labels = [
    # mtl
    17, 53,  # left and right hippocampus
    1006, 2006,  # left entorhinal right entorhinal
    1016, 2016,  # parahippocampal
    # original visual cortex
    1005, 2005,  # cuneus
    1011, 2011,  # lateral occipital
    1021, 2021,  # pericalcarine
    1029, 2029,  # superioparietal
    1013, 2013,  # lingual
    1008, 2008,  # inferioparietal
    1007, 2007,  # fusiform
    # mtl
    1009, 2009,  # inferiotemporal
    1015, 2015,  # middle temporal
]
mask_temporal_labels = [
    17, 53,  # left and right hippocampus
    1006, 2006,  # left entorhinal right entorhinal
    1016, 2016,  # left parahippocampal right parahippocampal

    1007, 2007,  # left fusiform right fusiform
    1009, 2009,  # left inferior temporal right inferior temporal
    1015, 2015,  # left middle temporal right middle temporal

    1030, 2030,  # left superior temporal right superior temporal
    1034, 2034,  # left transverse temporal right transverse temporal
    1018, 2018,  # left pars opercularis
    1019, 2019,  # left pars orbitalis
    1020, 2020,  # left pars triangularis
]
mask_prefrontal_labels = [
    1002, 2002,  # left caudal anterior cingulate
    1003, 2003,  # left caudal middle frontal
    1012, 2012,  # left lateral orbitofrontal
    1014, 2014,  # left medial orbitofrontal
    # 1017,2017,# left paracentral
    # 1018,2018,# left pars opercularis
    # 1019,2019,# left pars orbitalis
    # 1020,2020,# left pars triangularis
    # 1024,2024,# left precentral
    1026, 2026,  # left rostral anterior cingulate
    1027, 2027,  # left rostral middle frontal
    1028, 2028,  # left superior frontal
]

# binarize the brain mask
# function: use freesurfer mri_binarize to threshold an input volume
mask_visual = MapNode(interface=Binarize(), name='mask_visual', iterfield=['in_file'])
# input: match instead of threshold, it
mask_visual.inputs.match = mask_visual_labels
# optimize the efficiency of the node:
mask_visual.plugin_args = {'qsub_args': '-l nodes=1:ppn=1', 'overwrite': True}
mask_visual.plugin_args = {'qsub_args': '-l mem=1000MB', 'overwrite': True}

# function: use freesurfer mri_binarize to threshold an input volume
mask_hippocampus = MapNode(interface=Binarize(), name='mask_hippocampus', iterfield=['in_file'])
# input: match instead of threshold
mask_hippocampus.inputs.match = mask_hippocampus_labels
# optimize the efficiency of the node:
mask_hippocampus.plugin_args = {'qsub_args': '-l nodes=1:ppn=1', 'overwrite': True}
mask_hippocampus.plugin_args = {'qsub_args': '-l mem=1000MB', 'overwrite': True}

# function: use freesurfer mri_binarize to threshold an input volume
mask_entorhinal = MapNode(interface=Binarize(), name='mask_entorhinal', iterfield=['in_file'])
# input: match instead of threshold
mask_entorhinal.inputs.match = mask_entorhinal_labels
# optimize the efficiency of the node:
mask_entorhinal.plugin_args = {'qsub_args': '-l nodes=1:ppn=1', 'overwrite': True}
mask_entorhinal.plugin_args = {'qsub_args': '-l mem=1000MB', 'overwrite': True}

# function: use freesurfer mri_binarize to threshold an input volume
mask_mtl = MapNode(interface=Binarize(), name='mask_mtl', iterfield=['in_file'])
# input: match instead of threshold
mask_mtl.inputs.match = mask_mtl_labels
# optimize the efficiency of the node:
mask_mtl.plugin_args = {'qsub_args': '-l nodes=1:ppn=1', 'overwrite': True}
mask_mtl.plugin_args = {'qsub_args': '-l mem=1000MB', 'overwrite': True}

# function: use freesurfer mri_binarize to threshold an input volume
mask_vis_mtl = MapNode(interface=Binarize(), name='mask_vis_mtl', iterfield=['in_file'])
# input: match instead of threshold
mask_vis_mtl.inputs.match = mask_vis_mtl_labels
# optimize the efficiency of the node:
mask_vis_mtl.plugin_args = {'qsub_args': '-l nodes=1:ppn=1', 'overwrite': True}
mask_vis_mtl.plugin_args = {'qsub_args': '-l mem=1000MB', 'overwrite': True}

# function: use freesurfer mri_binarize to threshold an input volume
mask_temporal = MapNode(interface=Binarize(), name='mask_temporal', iterfield=['in_file'])
# input: match instead of threshold
mask_temporal.inputs.match = mask_temporal_labels
# optimize the efficiency of the node:
mask_temporal.plugin_args = {'qsub_args': '-l nodes=1:ppn=1', 'overwrite': True}
mask_temporal.plugin_args = {'qsub_args': '-l mem=1000MB', 'overwrite': True}

# function: use freesurfer mri_binarize to threshold an input volume
mask_prefrontal = MapNode(interface=Binarize(), name='mask_prefrontal', iterfield=['in_file'])
# input: match instead of threshold
mask_prefrontal.inputs.match = mask_prefrontal_labels
# optimize the efficiency of the node:
mask_prefrontal.plugin_args = {'qsub_args': '-l nodes=1:ppn=1', 'overwrite': True}
mask_prefrontal.plugin_args = {'qsub_args': '-l mem=1000MB', 'overwrite': True}

# In[7]:
'''
======================================================================
CREATE DATASINK NODE
======================================================================
'''
# create a node of the function:
datasink = Node(DataSink(), name='datasink')
# assign the path to the base directory:
datasink.inputs.base_directory = path_masks
# create a list of substitutions to adjust the filepaths of datasink:
substitutions = [('_subject_id_', '')]
# assign the substitutions to the datasink command:
datasink.inputs.substitutions = substitutions
# determine whether to store output in parameterized form:
datasink.inputs.parameterization = True

# In[ ]:
# initiation of the 1st-level analysis workflow:
wf = Workflow(name='masks')
# stop execution of the workflow if an error is encountered:
wf.config = {'execution': {'stop_on_first_crash': True}}
# define the base directory of the workflow:
wf.base_dir = opj(path_masks, 'work')
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
# connect segmented functional files to entorhinal node
wf.connect(selectfiles, 'func_parc', mask_entorhinal, 'in_file')
wf.connect(mask_entorhinal, 'binary_file', datasink, 'mask_entorhinal.@binary')
wf.connect(mask_entorhinal, 'count_file', datasink, 'mask_entorhinal.@count')
# connect segmented functional files to mtl node
wf.connect(selectfiles, 'func_parc', mask_mtl, 'in_file')
wf.connect(mask_mtl, 'binary_file', datasink, 'mask_mtl.@binary')
wf.connect(mask_mtl, 'count_file', datasink, 'mask_mtl.@count')
# connect segmented functional files to vis-mtl node
wf.connect(selectfiles, 'func_parc', mask_vis_mtl, 'in_file')
wf.connect(mask_vis_mtl, 'binary_file', datasink, 'mask_vis_mtl.@binary')
wf.connect(mask_vis_mtl, 'count_file', datasink, 'mask_vis_mtl.@count')
# connect segmented functional files to temporal node
wf.connect(selectfiles, 'func_parc', mask_temporal, 'in_file')
wf.connect(mask_temporal, 'binary_file', datasink, 'mask_temporal.@binary')
wf.connect(mask_temporal, 'count_file', datasink, 'mask_temporal.@count')
# connect segmented functional files to prefrontal node
wf.connect(selectfiles, 'func_parc', mask_prefrontal, 'in_file')
wf.connect(mask_prefrontal, 'binary_file', datasink, 'mask_prefrontal.@binary')
wf.connect(mask_prefrontal, 'count_file', datasink, 'mask_prefrontal.@count')
# In[173]:
'''
======================================================================
WRITE GRAPH AND EXECUTE THE WORKFLOW
======================================================================
'''
# write the graph:
wf.write_graph(graph2use='colored', simple_form=True)
# run the workflow
wf.run(plugin='MultiProc')
# In[]:
# check the smoothing data
# from nilearn import image, plotting
# plotting.plot_epi(
#     'fmean.nii.gz', title="mean (no smoothing)", display_mode='z',
#     cmap='gray', cut_coords=(-45, -30, -15, 0, 15));
# plotting.plot_epi(
#     'smean.nii.gz', title="mean (susan smoothed)", display_mode='z',
#     cmap='gray', cut_coords=(-45, -30, -15, 0, 15));
