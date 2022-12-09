#!/usr/bin/env python
# coding: utf-8

# # SCRIPT: FMRI-DECODING
# # PROJECT: FMRIREPLAY
# # WRITTEN BY QI HUANG 2022
# # CONTACT: STATE KEY LABORATORY OF COGNITVIE NERUOSCIENCE AND LEARNING, BEIJING NORMAL UNIVERSITY

# In[IMPORT RELEVANT PACKAGES]:
import glob
import os
import logging
import time
from os.path import join as opj
import copy
import numpy as np
from nilearn import image, masking
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import warnings
from scipy.stats import pearsonr  
from bids.layout import BIDSLayout
from joblib import Parallel, delayed
# import seaborn as sns
# import collections
# DEAL WITH ERRORS, WARNING ETC
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

# In[DEFINITION OF ALL FUNCTIONS]:
    
# set the classlabel based on marker
def classlabel(Marker):
    if Marker == 11 or Marker == 21 or Marker == 41 or Marker == 51:
        return 'girl'
    elif Marker == 12 or Marker == 22 or Marker == 42  or Marker == 52:
        return 'scissors'
    elif Marker == 13 or Marker == 23 or Marker == 43:
        return 'zebra'
    elif Marker == 14 or Marker == 24 or Marker == 44:
        return 'banana'

class TaskData:
    def __init__(
        self,
        events,
        confounds,
        task,
        name,
        num_vol_run_1,
        num_vol_run_2,
        num_vol_run_3,
        num_vol_run_4,
        # trial_type,
        bold_delay,
        interval,
        ):
        # define name of the task data subset:
        self.name = name
        # define the task condition the task data subset if from:
        self.task = task
        # define the delay (in seconds) by which onsets are moved:
        self.bold_delay = bold_delay
        # define the number of TRs from event onset that should be selected:
        self.interval = interval
        # define the repetition time (TR) of the mri data acquisition:
        self.t_tr = 1.3

        # select events: upright stimulus, correct answer trials only:
        if task == "VFL":
            self.events = events.loc[
                (events["task"] == 'VFL') &
                (events["accuracy"] == 1),
                :,
            ]
            # define the number of volumes per task run:
            self.num_vol_run = [[],[],[],[]]
            self.num_vol_run[0] = num_vol_run_1
            self.num_vol_run[1] = num_vol_run_2
            self.num_vol_run[2] = num_vol_run_3
            self.num_vol_run[3] = num_vol_run_4
            
        if task == 'REP':
            self.events = events.loc[
                (events["task"] == 'REP')
                & (events["accuracy"] == 1),
                :,
            ]
            # define the number of volumes per task run:
            self.num_vol_run = [[],[],[]]
            self.num_vol_run[0] = num_vol_run_1
            self.num_vol_run[1] = num_vol_run_2
            self.num_vol_run[2] = num_vol_run_3       
            
        self.confounds = confounds
        # reset the indices of the data frame:
        self.events.reset_index(drop=True,inplace=True)
        # sort all values by session and run:
        self.events.sort_values(by=["run"])
        # call further function upon initialization:
        self.define_trs_pre()
        self.rmheadmotion()
        self.define_trs_post()
        self.get_stats()
        
    def define_trs_pre(self):
        # import relevant functions:
        import numpy as np
        # select all events onsets:
        self.event_onsets = self.events["onset"]
        # add the selected delay to account for the peak of the hrf
        self.bold_peaks = (self.event_onsets + (self.bold_delay * self.t_tr))  # delay is the time, not the TR
        # divide the expected time-point of bold peaks by the repetition time:
        self.peak_trs = self.bold_peaks / self.t_tr
        if self.task == 'VFL' or self.task == 'learning' or self.task == 'REP':
            # add the number of run volumes to the tr indices:
            run_volumes = [sum(self.num_vol_run[0:int(self.events["run"].iloc[row]-1)])
                           for row in range(len(self.events["run"]))]
            run_volumes = pd.Series(run_volumes)
        # add the number of volumes of each run:
        trs = round(self.peak_trs + run_volumes)  # (run-1)*run_trs + this_run_peak_trs
        # copy the relevant trs as often as specified by the interval:
        a = np.transpose(np.tile(trs, (self.interval, 1)))
        # create same-sized matrix with trs to be added:
        b = np.full((len(trs), self.interval), np.arange(self.interval))
        # assign the final list of trs:
        self.trs = np.array(np.add(a, b).flatten(), dtype=int)
        # save the TRs of the stimulus presentations
        self.stim_trs = round(self.event_onsets / self.t_tr + run_volumes)

    def rmheadmotion(self): 
        self.fd  = self.confounds.loc[:,'framewise_displacement'].fillna(0).reset_index(drop=True)
        self.extras = np.array(self.fd[self.fd.values >0.2].index)
        self.rmheadmotions = np.zeros(len(self.stim_trs))
        for i in range(len(self.stim_trs)): 
            if any(extra in np.arange(self.stim_trs[i],(self.stim_trs[i]+self.interval)) for extra in self.extras):
                self.rmheadmotions[i] = 1
            else:
                self.rmheadmotions[i] = 0     
        self.rmheadmotions = self.rmheadmotions.astype(bool)
        self.events_rm = self.events.loc[~self.rmheadmotions,:]
        self.events_rm.reset_index(drop=True,inplace=True)
        # sort all values by session and run:
        self.events_rm.sort_values(by=["run"])

    def define_trs_post(self):
        # import relevant functions:
        import numpy as np
        # select all events onsets:
        self.event_onsets = self.events_rm["onset"]
        # add the selected delay to account for the peak of the hrf
        self.bold_peaks = (self.event_onsets + (self.bold_delay * self.t_tr))  # delay is the time, not the TR
        # divide the expected time-point of bold peaks by the repetition time:
        self.peak_trs = self.bold_peaks / self.t_tr
        if self.task == 'VFL' or self.task == 'learning' or self.task == 'REP':
            # add the number of run volumes to the tr indices:
            run_volumes = [sum(self.num_vol_run[0:int(self.events_rm["run"].iloc[row]-1)])
                           for row in range(len(self.events_rm["run"]))]
            self.run_volumes = pd.Series(run_volumes)
        # add the number of volumes of each run:
        self.trs_round = round(self.peak_trs + run_volumes)  # (run-1)*run_trs + this_run_peak_trs
        # copy the relevant trs as often as specified by the interval:
        self.a = np.transpose(np.tile(self.trs_round, (self.interval, 1)))
        # create same-sized matrix with trs to be added:
        self.b = np.full((len(self.trs_round), self.interval), np.arange(self.interval))
        # assign the final list of trs:
        self.trs = np.array(np.add(self.a, self.b).flatten(), dtype=int)
        # save the TRs of the stimulus presentations
        self.stim_trs = round(self.event_onsets / self.t_tr + run_volumes)
        
    def get_stats(self):
        import numpy as np
        self.num_trials = len(self.events_rm)
        self.runs = np.repeat(
            np.array(self.events_rm["run"], dtype=int), self.interval)
        self.trials = np.repeat(
            np.array(self.events_rm["trials"], dtype=int), self.interval)
        self.sess = np.repeat(
            np.array(self.events_rm["session"], dtype=int), self.interval)
        self.stim = np.repeat(
            np.array(self.events_rm["stim_label"], dtype=object), self.interval)
        self.fold = np.repeat(
            np.array(self.events_rm["fold"], dtype=object), self.interval)
        if self.task == 'REP':
            self.marker = np.repeat(
                np.array(self.events_rm["Marker"], dtype=object), self.interval)
        self.stim_trs = np.repeat(np.array(self.stim_trs, dtype=int), self.interval)
 
    def zscore(self, signals, fold_list):
        from nilearn.signal import clean
        import numpy as np
        # get boolean indices for all run indices in the run list:
        fold_indices = np.isin(self.fold, list(fold_list))
        # standardize data all runs in the run list:
        self.data_zscored = clean(
            signals=signals[self.trs[fold_indices]],
            # sessions=self.runs[fold_indices],
            t_r=1.3,
            detrend=False,
            standardize=True)

    def predict(self, clf, fold_list):
        # import packages:
        import pandas as pd
        # get classifier class predictions:
        pred_class = clf.predict(self.data_zscored)
        # get classifier probabilistic predictions:
        pred_proba = clf.predict_proba(self.data_zscored)
        # get the classes of the classifier:
        classes_names = clf.classes_
        # get boolean indices for all run indices in the run list:
        fold_indices = np.isin(self.fold, list(fold_list))
        # create a dataframe with the probabilities of each class:
        df = pd.DataFrame(pred_proba, columns=classes_names)
        # get the number of predictions made:
        num_pred = len(df)
        # get the number of trials in the test set:
        num_trials = int(num_pred / self.interval)
        # add the predicted class label to the dataframe:
        df["pred_label"] = pred_class
        df['marker'] = self.marker[fold_indices]
        # add the true stimulus label to the dataframe:
        df["stim"] = self.stim[fold_indices]
        # add the volume number (TR) to the dataframe:
        df["tr"] = self.trs[fold_indices]
        # add the sequential TR to the dataframe:
        df["seq_tr"] = np.tile(np.arange(1, self.interval + 1), num_trials)
        # add the counter of trs on which the stimulus was presented
        df["stim_tr"] = self.stim_trs[fold_indices]
        # add the trial number to the dataframe:
        df["trials"] = self.trials[fold_indices]
        # add the run number to the dataframe:
        df["run"] = self.runs[fold_indices]
        # add the session number to the dataframe:
        df["session"] = self.sess[fold_indices]
        df["fold"] = self.fold[fold_indices]
        # add the inter trial interval to the dataframe:
        # df["tITI"] = self.itis[fold_indices]
        # add the participant id to the dataframe:
        df["participant"] = np.repeat(self.events["participant"].unique(), num_pred)
        # add the name of the classifier to the dataframe:
        df["test_set"] = np.repeat(self.name, num_pred)
        # return the dataframe:
        return df

def detrend(data):
    from nilearn.signal import clean
    data_detrend = clean(signals=data, t_r=1.3, detrend=True, standardize=False)
    return data_detrend

def standardize(data):
    from nilearn.signal import clean
    data_standardize = clean(signals=data, t_r=1.3, detrend=False, standardize=True)
    return data_standardize

def TransM_cross(x):
    # create an empty matrix
    transition_matrix = np.zeros([2,2])
    # transition
    for a in range(1):
        transition_matrix[x[a]-1][x[a+1]-1] = 1
    return(transition_matrix)

# detect the autocorrelation between hippocampus and visual cortex
def TDLM_cross(probability_matrix):
    rp = []
    # print(isub,sub)    
    #detect for each 2048 ms trial
    # loops for real sequence and permutation trials
    # real sequence
    rp = [1,2]
    # get the transition matrix
    T1 = TransM_cross(rp)
    T2 = np.transpose(T1)
    nbins = maxLag+1
    # X1 = seq_pred_matrix[13*i:13*(i+1),:] #####
    X1 = np.array(probability_matrix)
    #detect for each timelag
    def Crosscorr(X1,T,l):
    # rd is samples by states 
    # T is the transition matrix of interest
    # T2 is the 2-step transition matrix, or can be []
    # lag is how many samples the data should be shifted by
        nSamples = np.size(X1,axis=0)
        nStates = np.size(X1,axis=1)
        orig = np.dot(X1[0:(-(2*l)),:],T)
        proj = X1[l:-l,:] #?

        ## Scale variance
        corrtemp = np.full(np.size(proj,axis=1),np.nan)
        for iseq in range(np.size(proj,axis=1)):
            if (np.nansum(orig[:,iseq])!=0) & (np.nansum(proj[:,iseq])!=0):
                corrtemp[iseq] = pearsonr(orig[:,iseq],proj[:,iseq]).statistic
        sf = np.nanmean(corrtemp)
        return sf

    for l in range(1,maxLag+1):
        cross[0,l-1] = Crosscorr(X1=X1, T=T1, l=l);
        cross[1,l-1] = Crosscorr(X1=X1, T=T2, l=l);

# In[SETUP NECESSARY PATHS ETC]:
# name of the current project:
project = "fmrireplay"
path_root = None
sub_list = None
subject = None
# path to the project root:
project_name = "fmrireplay-decoding"
path_root = opj(os.getcwd().split(project)[0], project)
path_bids = opj(path_root, "fmrireplay-bids", "BIDS")
path_bids_nf = opj(path_root, "BIDS-nofieldmap")
path_code = opj(path_bids_nf, "code", "decoding-TDLM")
path_fmriprep = opj(path_bids_nf, "derivatives", "fmrireplay-fmriprep")
path_masks = opj(path_bids_nf, "derivatives", "fmrireplay-masks")
path_glm_vfl = opj(path_bids_nf, "derivatives", "fmrireplay-glm-vfl")
path_level1_vfl = opj(path_glm_vfl, "l1pipeline")
path_glm_rep = opj(path_bids_nf, "derivatives", "fmrireplay-glm-replay")
path_level1_rep = opj(path_glm_rep, "l1pipeline")
path_decoding = opj(path_bids_nf, "derivatives", project_name)
path_behavior = opj(path_bids_nf,'derivatives','fmrireplay-behavior')
# load the learning sequence
learning_sequence = pd.read_csv(opj(path_behavior,'sequence.csv'), sep=',')

# define the subject id
layout = BIDSLayout(path_bids_nf)
sub_list = sorted(layout.get_subjects())
# choose the specific subjects
# sub_list = sub_list[0:40]
lowfmri = [37, 16, 15, 13, 11, 0]
loweeg = [45, 38, 36, 34, 28, 21, 15, 11, 5, 4]
incomplete = [42, 22, 4]
extremeheadmotion = [38, 28, 16, 15, 13, 9, 5, 0]
delete_list = np.unique(
    np.hstack([lowfmri, loweeg, incomplete, extremeheadmotion]))[::-1]
[sub_list.pop(i) for i in delete_list]

# create a new subject 'sub-*' list 
subnum_list = copy.deepcopy(sub_list)
sub_template = ["sub-"] * len(sub_list)
sub_list = ["%s%s" % t for t in zip(sub_template, sub_list)]
subnum_list = list(map(int,subnum_list))

# create the mask list and bold delay list
mask_name_list= ['mask_visual', 'mask_vis_mtl', 'mask_temporal', 'mask_prefrontal', 'mask_hippocampus', 'mask_entorhinal', 'mask_mtl']
mask_index_list=[0,1,2,3]
task_bold_delay_list = [1,2,3,4,5,6,7,8,9]

# the parameter for test
# subject=sub_list[0]
# mask_index = mask_index_list[1]
# task_bold_delay = task_bold_delay_list[2]
# mask_name = mask_name_list[mask_index]
# task_peak_tr = task_bold_delay

# In[paramters]
# the number of VFL runs
n_run = 4  
# some parameters
data_list_all=[] 
pred_acc_mean_sub_all = []
long_interval = 8
# parameters for TDLM
# the list of all the sequences
uniquePerms = list(itertools.permutations([1,2,3,4],4)) # all possibilities
uniquePerms = uniquePerms[0:-1] # except the last one, because backward direction equal to the correct sequence
# the number of sequences
nShuf = len(uniquePerms)
# all possible sequences
all_sequence = np.array(uniquePerms)
# the number of timelags
maxLag = long_interval
nbins = maxLag+1
# the number of states (decoding models)
nstates = 4
# the number of subjects
nSubj = len(sub_list)
# the nan matrix for forward design matrix (2 directions * 23 shuffles * 10 timelags * 2 experimental conditions)
cross = np.full([2,maxLag+1],np.nan)
# predefine GLM data frame 
betas_1GLM = None
betas_2GLM = None
# In[for parallel function]:
def decoding(subject):
# for subject in sub_list:
    # print(subject)
    # In[CREATE PATHS TO OUTPUT DIRECTORIES]:
    path_out = opj(path_decoding, subject)
    path_out_figs = opj(path_out, "plots")
    path_out_data = opj(path_out, "data")
    path_out_logs = opj(path_out, "logs")
    path_out_masks = opj(path_out, "masks")
    path_out_all = opj(path_decoding, 'all_subject_results')
    # CREATE OUTPUT DIRECTORIES IF THE DO NOT EXIST YET:
    for path in [path_out_figs, path_out_data, path_out_logs, path_out_masks, path_out_all]:
        if not os.path.exists(path):
            os.makedirs(path)

    # In[SETUP LOGGING]:
    # remove all handlers associated with the root logger object:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # get current data and time as a string:
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    # create path for the logging file
    log = opj(path_out_logs, "%s-%s.log" % (timestr, subject))
    # start logging:
    logging.basicConfig(
        filename=log,
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    # In[LOAD BEHAVIORAL DATA (THE EVENTS.TSV FILES OF THE SUBJECT)]:
    # paths to all events files of the current subject:
    path_events = opj(path_behavior, subject,  "*tsv")
    # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
    path_events = sorted(glob.glob(path_events), key=lambda f: os.path.basename(f))

    # paths to all events files of the current subject:
    path_events_rep = opj(path_behavior, subject, "*rep*events.tsv")
    # dl.get(glob.glob(path_events)) #can't run it, and i don't know the meaning of the data
    path_events_rep = sorted(glob.glob(path_events_rep), key=lambda f: os.path.basename(f))
    # import events file and save data in dataframe:
    # df_events_rep = pd.read_csv(path_events_rep, sep="\t") 
    df_events_rep = pd.concat(
        (pd.read_csv(f, sep="\t") for f in path_events_rep), ignore_index=True
    )
    df_events_rep['stim_label'] = df_events_rep.apply(lambda x: classlabel(x.Marker), axis = 1) 
    del df_events_rep['Unnamed: 0']
    del df_events_rep['start']
    
    path_confounds_rep = glob.glob(opj(path_fmriprep, subject,'func', '*replay*confounds_timeseries.tsv'))
    df_confound_rep_list = [pd.read_csv(i, sep="\t") for i in path_confounds_rep]
    df_confound_rep = pd.concat([df_confound_rep_list[0],
                                        df_confound_rep_list[1],
                                        df_confound_rep_list[2]])

    # In[CrepTE PATHS TO THE MRI DATA]:
    # load the cue replay mri task files:
    path_func_task_rep = opj(path_level1_rep, "smooth", subject, "*", "*task*nii.gz")
    path_func_task_rep = sorted(glob.glob(path_func_task_rep), key=lambda f: os.path.basename(f))
    # load smoothed functional mri data for all task runs:
    data_task_rep = [image.load_img(i) for i in path_func_task_rep]
 
    # In[DEFINE THE TASK CONDITION]:
        
    # 1. SPLIT THE EVENTS DATAFRAME FOR EACH TASK CONDITION
    # 2. RESET THE INDICES OF THE DATAFRAMES
    # 3. SORT THE ROWS OF ALL DATAFRAMES IN CHRONOLOGICAL ORDER
    # 4. PRINT THE NUMBER OF TRIALS OF EACH TASK CONDITION
    
    test_rep_long = TaskData(
        events=df_events_rep, 
        confounds=df_confound_rep,
        task='REP', 
        bold_delay=0, 
        interval=long_interval,
        name='test-rep_long',
        num_vol_run_1 = np.size(data_task_rep[0].dataobj,axis=3),
        num_vol_run_2 = np.size(data_task_rep[1].dataobj,axis=3),
        num_vol_run_3 = np.size(data_task_rep[2].dataobj,axis=3),
        num_vol_run_4 = 0,
        )

    # In[VISUAL CORTEX AND HIPPOCAMPUS BOLD SIGNAL AUTOCORRELATION]:
    
    tr_onset = np.unique(test_rep_long.stim_trs)
    # load the hippocampus and visual task mask in cue replay task    
    def ROI_BOLD(mask,data_rep,onset,region):
        #load the path of masks:
        path_mask = sorted(glob.glob(
            opj(path_masks, mask, subject, "*", "*task*.nii.gz")
            ), key=lambda f: os.path.basename(f))
        mask_dy = [image.load_img(i)
                          for i in copy.deepcopy(path_mask[3:6])]
        masked_data = [masking.apply_mask(data, mask_dy[i]) 
                              for (i,data) in enumerate(data_rep)]
        # detrend the BOLD signal in each run
        data_detrend_test = [detrend(i)
                             for i in masked_data]
        # average the BOLD signal in voxel level
        data_detrend_test = [data_detrend_test[i].mean(axis=1) 
                             for i in np.arange(len(data_detrend_test))]
        # concat three runs' BOLD signal
        data_detrend_test = np.array(np.hstack(data_detrend_test))
        # data_detrend_test_stand = preprocessing.scale(data_detrend_test)#standardize(data_detrend_test)
        # get each trials' 10 TR
        data_trial = [data_detrend_test[onset[i]:onset[i]+10,] 
                             for i in range(len(onset))] # the least duration in a single trial
        data_trial = np.array(data_trial)
        # the relative signal strength
        data = (data_trial/np.max(data_trial))
        # visual_data_max = visual_data/visual_data.max()
        
        # get the visual data dataframes
        bold_df = pd.DataFrame(columns=['BOLDsignal','BOLD/Max','subject','region','time(TR)'])
        bold_df['BOLDsignal'] = np.reshape(data_trial,np.size(data))
        bold_df['BOLD/Max'] = np.reshape(data,np.size(data))
        bold_df['subject'] = subject
        bold_df['region'] = region
        bold_df['time(TR)'] = np.tile(np.arange(1,11),len(onset))

        return [data,bold_df]


    hippocampus_df = ROI_BOLD('mask_mtl',data_task_rep,tr_onset,'mtl')
    visual_df = ROI_BOLD('mask_visual',data_task_rep,tr_onset,'visual cortex')

    hippocampus_bold = hippocampus_df[1]['BOLDsignal']
    visual_bold = visual_df[1]['BOLDsignal']
    bold_cross = np.vstack([hippocampus_bold,visual_bold])
    
    # calculate the autocorrelation
    TDLM_cross(np.transpose(bold_cross))
    
    return cross
  
cross_TDLM = Parallel(n_jobs=60)(delayed(decoding)(subject) for subject in sub_list)
# In[]
path_out_all = opj(path_decoding, 'all_subject_results')
# cross_TDLM_seq = [all_results_list[i][0] for i in range(len(sub_list))]
# cross_TDLM_seq_list = np.array(cross_TDLM_seq)
cross_TDLM_seq_list = np.array(cross_TDLM)

# all subject mean
def cross_corr_forward(data):
    # plot 512ms probability
    s1mean = data.mean(axis=0)
    s1sem = data.std(axis=0)/np.sqrt(len(data))
    x=np.arange(0,nbins,1)
    plt.figure(dpi=400,figsize=(10,5))
    plt.xticks(range(len(x)),x,fontsize=10)
    plt.plot(x, s1mean, color='darkred', linestyle='-', linewidth=1)
    plt.fill_between(x,s1mean-s1sem,s1mean+s1sem,color='lightcoral',alpha=0.5, linewidth=1)
    plt.title('MTL to Visual Cortex.png')
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path_out_all,'autocorrelation','MTL to Visual Cortex.png'),
                dpi = 400,bbox_inches = 'tight')
    plt.show()

def cross_corr_backward(data):
    # plot 512ms probability
    s1mean = data.mean(axis=0)
    s1sem = data.std(axis=0)/np.sqrt(len(data))
    x=np.arange(0,nbins,1)
    plt.figure(dpi=400,figsize=(10,5))
    plt.xticks(range(len(x)),x,fontsize=10)
    plt.plot(x, s1mean, color='dodgerblue', linestyle='-', linewidth=1)
    plt.fill_between(x,s1mean-s1sem,s1mean+s1sem,color='lightskyblue',alpha=0.5, linewidth=1)
    plt.title('Visual Cortex to MTL')
    plt.xlabel('timelag (TRs)')
    plt.ylabel('sequenceness')
    plt.savefig(opj(path_out_all,'autocorrelation','Visual Cortex to MTL.png'),
                dpi = 400,bbox_inches = 'tight')
    plt.show()

sf_sequence = cross_TDLM_seq_list[:,0,:]

sb_sequence = cross_TDLM_seq_list[:,1,:]

cross_corr_forward(sf_sequence)
cross_corr_backward(sb_sequence)