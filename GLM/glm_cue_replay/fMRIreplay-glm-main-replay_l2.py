#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: SECOND LEVEL GLM
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
import sys
import time
import warnings
import numpy as np
import copy
import pandas as pd
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
from nipype.interfaces.spm import OneSampleTTestDesign
from nipype.interfaces.spm import EstimateModel, EstimateContrast
from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces import spm
# import libraries for bids interaction:
from bids.layout import BIDSLayout
from IPython.display import Image

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

# start cue
start_time = time.time()
# In[3]:
'''
======================================================================
SET PATHS AND SUBJECTS
======================================================================
'''
path_list =['maxonset','maxonset-pm','permonset2',
            'permonset2-pm','permonset4','permonset4-pm']

contrast_1st_onset_list = [['0001', '0002','0003','0004'],
                           ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008'],
                           ['0001', '0002','0003','0004'],
                           ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008'],
                           ['0001', '0002','0003','0004'],
                           ['0001', '0002', '0003', '0004', '0005', '0006', '0007']]


# define paths depending on the operating system (OS) platform:
project = 'fmrireplay'
# initialize empty paths:
sub_list = None
# path to the project root:
project_name = 'fmrireplay-resting-TDLM'
path_bids = opj(os.getcwd().split('BIDS')[0], 'BIDS')
path_code = opj(path_bids, 'code', 'GLM')

# SPM setting
spm.SPMCommand().set_mlab_paths(paths='/home/huangqi/MATLAB/spm12')


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
# delete_list = np.unique(
#     np.hstack([loweeg, incomplete, extremeheadmotion]))[::-1]
[sub_list.pop(i) for i in delete_list]
# sub_list=sub_list[0:1]
# create a template to add the "sub-" prefix to the ids
subnum_list = copy.deepcopy(sub_list)
sub_template = ['sub-'] * len(sub_list)
# add the prefix to all ids:
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]
# run workflow for specific subject
# sub_list = sub_list[8]
# runs=[1,2,3]

for i in range(6):
    # In[6]:
    path_glm_l1 = opj(path_bids, 'derivatives', 'glm-l1-onset', path_list[i])
    path_glm_l2 = opj(path_bids, 'derivatives', 'glm-l2-onset', path_list[i])
    # define the infosource node that collects the data:
    contrast_1st_onset = contrast_1st_onset_list[i]
    '''
    ======================================================================
    DEFINE NODE: INFOSOURCE
    ======================================================================
    '''

    infosource = Node(IdentityInterface(
        fields=['contrast_id', 'subj_id']), name='infosource')
    # let the node iterate (paralellize) over all subjects:
    infosource.iterables = [('contrast_id', contrast_1st_onset)]
    infosource.inputs.subj_id = sub_list

    # In[7]:
    '''
    ======================================================================
    DEFINE SELECTFILES NODE
    ======================================================================
    '''
    # check the data files
    path_contrasts = glob.glob(opj(path_glm_l1, 'l1pipeline', 'contrasts', '*', 'con_*.nii'))
    # In[8]:
    # define all relevant files paths:
    templates = {'contrasts': opj(path_glm_l1, 'l1pipeline', 'contrasts', '{subj_id}', 'con_{contrast_id}.nii')}
    # define the selectfiles node:
    selectfiles = MapNode(SelectFiles(templates, sort_filelist=True),
                          name='selectfiles', iterfield=['subj_id'])
    # list of contrast identifiers
    # selectfiles.iterables = [('subject_id',sub_list)]
    # selectfiles.inputs.run_id = runs
    # set expected thread and memory usage for the node:
    # selectfiles.inputs.subject_id = sub_list[0]
    # selectfiles.inputs.contrast_id = contrast_1st_list[0]
    # selectfiles_results = selectfiles.run()
    # selectfiles.run().outputs

    # In[9]:
    '''
    ======================================================================
    DEFINE NODE: ONE SAMPLE T TEST DESIGN
    ======================================================================
    '''
    # Initiate the OneSampleTTestDesign node here
    onesamplettestdes = Node(OneSampleTTestDesign(),
                             name="onesampttestdes")
    # onesamplettestdes.inputs.explicit_mask_file = opj(path_fmriprep,'mni_icbm152_nlin_asym_09c/1mm_brainmask.nii.gz''

    # In[10]:
    '''
    ======================================================================
    DEFINE NODE: LEVEL 2 BETA ESTIMATE
    ======================================================================
    '''
    # Initiate the EstimateModel and the EstimateContrast node here
    level2estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                          name="level2estimate")

    # In[11]:
    '''
    ======================================================================
    DEFINE NODE: LEVEL 2 BETA ESTIMATE CONTRAST
    ======================================================================
    '''
    level2conestimate = Node(EstimateContrast(group_contrast=True),
                             name="level2conestimate")
    cont01 = ['Group', 'T', ['mean'], [1]]
    level2conestimate.inputs.contrasts = [cont01]

    # In[12]:
    '''
    ======================================================================
    DEFINE NODE: LEVEL 2 THRESHOLD
    ======================================================================
    '''
    level2thresh = Node(Threshold(contrast_index=1,
                                  use_topo_fdr=True,
                                  use_fwe_correction=False,
                                  extent_threshold=0,
                                  height_threshold=0.01,
                                  height_threshold_type='p-value',
                                  extent_fdr_p_threshold=0.05),
                        name="level2thresh")

    # In[13]:
    '''
    ======================================================================
    DEFINE NODE: DATA SINK
    ======================================================================
    '''
    # Initiate DataSink node here

    l2datasink = Node(DataSink(), name="datasink")
    l2datasink.inputs.base_directory = opj(path_glm_l2, 'l2pipeline')
    # create a list of substitutions to adjust the file paths of datasink:
    substitutions = [('_cont_id_', 'con_')]
    # assign the substitutions to the datasink command:
    l2datasink.inputs.substitutions = substitutions
    # determine whether to store output in parameterized form:
    l2datasink.inputs.parameterization = True

    # In[14]:
    '''
    ======================================================================
    DEFINE NODE: CONNECT THE WORKFLOW
    ======================================================================
    '''
    l2analysis = Workflow(name='l2analysis')
    l2analysis.connect([(infosource, selectfiles, [('contrast_id', 'contrast_id'),
                                                   ('subj_id', 'subj_id')])
                        ])
    l2analysis.connect([(selectfiles, onesamplettestdes, [('contrasts', 'in_files')])
                        ])

    # Connect OneSampleTTestDesign, EstimateModel and EstimateContrast here
    l2analysis.connect([(onesamplettestdes, level2estimate, [('spm_mat_file', 'spm_mat_file')]),
                        (level2estimate, level2conestimate, [('spm_mat_file', 'spm_mat_file'),
                                                             ('beta_images', 'beta_images'),
                                                             ('residual_image', 'residual_image')])
                        ])
    # Connect the Threshold node to the EstimateContrast node here
    l2analysis.connect([(level2conestimate, level2thresh, [('spm_mat_file', 'spm_mat_file'),
                                                           ('spmT_images', 'stat_image'),
                                                           ])
                        ])

    # Connect nodes to datasink here
    l2analysis.connect([(level2conestimate, l2datasink, [('spm_mat_file', '2ndLevel.@spm_mat'),
                                                         ('spmT_images', '2ndLevel.@T'),
                                                         ('con_images', '2ndLevel.@con')])])
    # Connect nodes to datasink here
    l2analysis.connect([(level2thresh, l2datasink, [('thresholded_map', '2ndLevel.@threshold')])])

    # In[25]:
    '''
    ======================================================================
    WRITE GRAPH AND EXECUTE THE WORKFLOW
    ======================================================================
    '''
    # write the graph:
    # Create 1st-level analysis output graph
    l2analysis.write_graph(graph2use='colored', format='png', simple_form=True)
    # Visualize the graph
    # Image(filename= 'graph.png')
    l2analysis.run(plugin='MultiProc')

    end_time = time.time()
    run_time = round((end_time - start_time) / 60, 2)
    print(f"Run time cost {run_time} min")

# %%bash
# TEMPLATE='/opt/spm12-r7219/spm12_mcr/spm12/tpm/TPM.nii'
#
# # Extract the first volume with `fslroi`
# fslroi $TEMPLATE GM_PM.nii.gz 0 1
#
# # Threshold the probability mask at 10%
# fslmaths GM_PM.nii -thr 0.10 -bin /output/datasink_handson/GM_mask.nii.gz
#
# # Unzip the mask and delete the GM_PM.nii file
# gunzip /output/datasink_handson/GM_mask.nii.gz
# rm GM_PM.nii.gz
# Let's take a look at this mask:
#
# import nibabel as nb
# mask = nb.load('/output/datasink_handson/GM_mask.nii')
# mask.orthoview()
