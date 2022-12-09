#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# SCRIPT INFORMATION:
# ======================================================================
# SCRIPT: PLOT MASKS FOR ILLUSTRATION
# PROJECT: HIGHSPEED
# WRITTEN BY LENNART WITTKUHN, 2020
# CONTACT: WITTKUHN AT MPIB HYPHEN BERLIN DOT MPG DOT DE
# MAX PLANCK RESEARCH GROUP NEUROCODE
# MAX PLANCK INSTITUTE FOR HUMAN DEVELOPMENT
# MAX PLANCK UCL CENTRE FOR COMPUTATIONAL PSYCHIATRY AND AGEING RESEARCH
# LENTZEALLEE 94, 14195 BERLIN, GERMANY
# ======================================================================
# IMPORT RELEVANT PACKAGES
# ======================================================================
from nilearn import plotting
from nilearn import image
import os
from os.path import join as opj
import glob
import re
import matplotlib.pyplot as plt
import statistics

sub = 'sub-04'
path_root = os.getcwd()
path_anat = opj(path_root, 'data', 'bids', sub, '*', 'anat', '*.nii.gz')
anat = sorted(glob.glob(path_anat))[0]
path_patterns = opj(path_root, 'data', 'decoding', sub, 'data', '*.nii.gz')
path_patterns = sorted(glob.glob(path_patterns))
path_patterns = [x for x in path_patterns if 'hpc' not in x]
path_union = opj(path_root, 'data', 'decoding', sub, 'masks', '*union.nii.gz')
path_union = glob.glob(path_union)

project = "fMRIreplay"
path_root = opj(os.getcwd().split(project)[0], "fMRIreplay_hq")
path_bids = opj(path_root, "fmrireplay-bids", "BIDS")
path_anat = opj(path_bids, sub, "anat", "*.nii.gz")
anat = sorted(glob.glob(path_anat))[0]
path_decoding = opj(path_bids, "derivatives", "fmrireplay-decoding")
path_patterns = opj(path_decoding, sub, 'mask', '*.nii.gz')
path_patterns = [x for x in path_patterns if 'hpc' not in x]
path_union = opj(path_root, 'data', 'decoding', sub, 'masks', '*union.nii.gz')
path_union = glob.glob(path_union)


# path_fmriprep = opj(path_root, 'derivatives', 'fmriprep')
# path_masks = opj(path_root, 'derivatives', 'decoding', sub)
# path_anat = opj(path_fmriprep, sub, 'anat', sub + '_desc-preproc_T1w.nii.gz')
# path_visual = opj(path_masks, 'masks', '*', '*.nii.gz')
# vis_mask = glob.glob(path_visual)[0]
# vis_mask_smooth = image.smooth_img(vis_mask, 4)

plotting.plot_roi(path_union[0], bg_img=anat,
                  cut_coords=[30, 10, 15],
                  title="Region-of-interest", black_bg=True,
                  display_mode='ortho', cmap='red_transparent',
                  draw_cross=False)

# calculate average patterns across all trials:
mean_patterns = [image.mean_img(i) for i in path_patterns]
# check the shape of the mean patterns (should be 3D):
[print(image.get_data(i).shape) for i in mean_patterns]
# extract labels of patterns
labels = [re.search('union_(.+?).nii.gz', i).group(1) for i in path_patterns]


# function used to plot individual patterns:
def plot_patterns(pattern, name):
    display = plotting.plot_stat_map(
            pattern,
            bg_img=anat,
            #cut_coords=[30, 29, -6],
            title=name,
            black_bg=True,
            colorbar=True,
            display_mode='ortho',
            draw_cross=False
            )
    path_save = opj(path_root, 'figures', 'pattern_{}.pdf').format(name)
    display.savefig(filename=path_save)
    display.close()
    return(display)


# plot individual patterns and save coordinates:
coords = []
for pattern, name in zip(mean_patterns, labels):
    display = plot_patterns(pattern, name)
    coords.append(display.cut_coords)
# mean_coords = [sum(x)/len(x) for x in zip(*coords)]
mean_coords = [statistics.median(x) for x in zip(*coords)]

# create subplot with all patterns using mean coordinates:
fig, axes = plt.subplots(nrows=len(path_patterns), ncols=1, figsize=(14, 20))
for pattern, name, ax in zip(mean_patterns, labels,  axes):
    display = plotting.plot_stat_map(
            pattern, bg_img=anat, cut_coords=mean_coords, title=name,
            black_bg=True, colorbar=True, display_mode='ortho',
            draw_cross=False, axes=ax, symmetric_cbar=True, vmax=1)
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig(opj(path_root, 'figures', 'pattern_all.pdf'), bbox_inches='tight')
