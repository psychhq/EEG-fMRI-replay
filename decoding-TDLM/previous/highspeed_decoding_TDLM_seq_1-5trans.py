# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:22:39 2022

@author: Qi Huang
"""


'''
========================================================================
IMPORT RELEVANT PACKAGES:
========================================================================
'''
import glob
import os
import yaml
import logging
import time
from os.path import join as opj
import sys
import copy
from pprint import pformat
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn import plotting, image, masking
import pandas as pd
from numpy.matlib import repmat
from matplotlib import pyplot as plt
import itertools
import warnings
from nilearn.signal import clean
import scipy
import scipy.io as scio
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from collections import Counter
import datalad.api as dl   # can not run in the code.
import nibabel as nib
import seaborn as sns
'''
========================================================================
DEAL WITH ERRORS, WARNING ETC.
========================================================================
'''
warnings.filterwarnings('ignore', message='numpy.dtype size changed*')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed*')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
'''
========================================================================
SETUP NECESSARY PATHS ETC:
========================================================================
'''
# get start time of the script:
start = time.time()
# name of the current project:
project = 'highspeed'
# initialize empty variables:
sub = None
path_tardis = None
path_server = None
path_local = None

# define paths without depending on the operating system (OS) platform:
# for Windows
#project name     
project_name = 'highspeed-decoding'
#current working directory in Nicodata
cwd1 = os.getcwd() 
#path_root
path_root = opj(cwd1, project_name)
# define the path to the cluster:
path_tardis = path_root   
# define the path to the server:
path_server = path_tardis
#define the path to the local computer:
path_local = opj(path_tardis, 'code', 'decoding')
#define the subject id:
sub = 'sub-14'
suball = ['sub-%s' %str(i).zfill(2) for i in range(1,41)]  #40 subjects

delete_sub = [39,36,30,23]
[suball.pop(i) for i in delete_sub]
'''
========================================================================
LOAD PROJECT PARAMETERS:
========================================================================
'''
path_params = glob.glob(opj(path_local, '*parameters.yaml'))[0]
with open(path_params, 'rb') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
f.close()

'''
========================================================================
DEFINITION OF ALL FUNCTIONS
========================================================================
'''

class TaskData:
    
    def __init__(
            self, events, condition, name, trial_type, bold_delay=0,
            interval=1, t_tr=1.25, num_vol_run=530):  #?
        import pandas as pd
        # define name of the task data subset:
        self.name = name
        # define the task condition the task data subset if from:
        self.condition = condition
        # define the delay (in seconds) by which onsets are moved:
        self.bold_delay = bold_delay
        # define the number of TRs from event onset that should be selected:
        self.interval = interval
        # define the repetition time (TR) of the mri data acquisition:
        self.t_tr = t_tr
        # define the number of volumes per task run:
        self.num_vol_run = num_vol_run  
        # select events: upright stimulus, correct answer trials only:
        if trial_type == 'stimulus':
            self.events = events.loc[
                          (events['condition'] == condition) &
                          (events['trial_type'] == trial_type) &
                          (events['stim_orient'] == 0) &
                          (events['serial_position'] == 1) &
                          (events['accuracy'] != 0),
                          :]
        elif trial_type == 'cue':
            self.events = events.loc[
                          (events['condition'] == condition) &
                          (events['trial_type'] == trial_type),
                          :]
        # reset the indices of the data frame:
        self.events.reset_index()
        # sort all values by session and run:
        self.events.sort_values(by=['session', 'run_session'])
        # call further function upon initialization:
        self.define_trs()
        self.get_stats()
        if condition == 'sequence':
            self.sequence = events.loc[
                             (events['condition'] == condition) &
                             (events['trial_type'] == trial_type) &
                             (events['stim_orient'] == 0) &
                             (events['serial_position'] >= 0) &
                             (events['accuracy'] != 0),
                             :]
            self.sequences = np.array(self.sequence)[:,12]
            self.real_trials = np.unique(self.trials)
            self.real_trial = len(self.real_trials)
            self.sequences = self.sequences.reshape(self.real_trial, 5)
            self.itises = self.itis.reshape(self.real_trial, 13)




    def define_trs(self):
        # import relevant functions:
        import numpy as np
        # select all events onsets:
        self.event_onsets = self.events['onset']
        # add the selected delay to account for the peak of the hrf
        self.bold_peaks = self.events['onset'] + self.bold_delay  #delay is the time, not the TR
        # divide the expected time-point of bold peaks by the repetition time:
        self.peak_trs = self.bold_peaks / self.t_tr
        # add the number of run volumes to the tr indices:
        run_volumes = (self.events['run_study']-1) * self.num_vol_run 
        # add the number of volumes of each run:
        trs = round(self.peak_trs + run_volumes) # (run-1)*run_trs + this_run_peak_trs 
        # copy the relevant trs as often as specified by the interval:
        a = np.transpose(np.tile(trs, (self.interval, 1)))
        # create same-sized matrix with trs to be added:
        b = np.full((len(trs), self.interval), np.arange(self.interval))
        # assign the final list of trs:
        self.trs = np.array(np.add(a, b).flatten(), dtype=int)
        # save the TRs of the stimulus presentations
        self.stim_trs = round(self.event_onsets / self.t_tr + run_volumes) # continuous trs for counting

    def get_stats(self):
        import numpy as np
        self.num_trials = len(self.events)
        self.runs = np.repeat(np.array(self.events['run_study'], dtype=int), self.interval)
        self.trials = np.repeat(np.array(self.events['trial'], dtype=int), self.interval)
        self.sess = np.repeat(np.array(self.events['session'], dtype=int), self.interval)
        self.stim = np.repeat(np.array(self.events['stim_label'], dtype=object), self.interval)
        self.itis = np.repeat(np.array(self.events['interval_time'], dtype=float), self.interval)
        self.stim_trs = np.repeat(np.array(self.stim_trs, dtype=int), self.interval)

    def zscore(self, signals, run_list, t_tr=1.25):
        from nilearn.signal import clean
        import numpy as np
        # get boolean indices for all run indices in the run list:
        run_indices = np.isin(self.runs, list(run_list))
        # standardize data all runs in the run list:
        self.data_zscored = clean(
            signals=signals[self.trs[run_indices]],
            sessions=self.runs[run_indices],
            t_r=t_tr,
            detrend=False,
            standardize=True)

    def predict(self, clf, run_list):
        # import packages:
        import pandas as pd
        # get classifier class predictions:
        pred_class = clf.predict(self.data_zscored)
        # get classifier probabilistic predictions:
        pred_proba = clf.predict_proba(self.data_zscored)
        # get the classes of the classifier:
        classes_names = clf.classes_
        # get boolean indices for all run indices in the run list:
        run_indices = np.isin(self.runs, list(run_list))
        # create a dataframe with the probabilities of each class:
        df = pd.DataFrame(pred_proba, columns=classes_names)
        # get the number of predictions made:
        num_pred = len(df)
        # get the number of trials in the test set:
        num_trials = int(num_pred / self.interval)
        # add the predicted class label to the dataframe:
        df['pred_label'] = pred_class
        # add the true stimulus label to the dataframe:
        df['stim'] = self.stim[run_indices]
        # add the volume number (TR) to the dataframe:
        df['tr'] = self.trs[run_indices]
        # add the sequential TR to the dataframe:
        df['seq_tr'] = np.tile(np.arange(1, self.interval + 1), num_trials)
        # add the counter of trs on which the stimulus was presented
        df['stim_tr'] = self.stim_trs[run_indices]
        # add the trial number to the dataframe:
        df['trial'] = self.trials[run_indices] 
        # add the run number to the dataframe:
        df['run_study'] = self.runs[run_indices]
        # add the session number to the dataframe:
        df['session'] = self.sess[run_indices]
        # add the inter trial interval to the dataframe:
        df['tITI'] = self.itis[run_indices]
        # add the participant id to the dataframe:
        df['id'] = np.repeat(self.events['subject'].unique(), num_pred)
        # add the name of the classifier to the dataframe:
        df['test_set'] = np.repeat(self.name, num_pred)
        # return the dataframe:
        return df


def detrend(data, t_tr=1.25):
    from nilearn.signal import clean
    data_detrend = clean(
        signals=data, t_r=t_tr, detrend=True, standardize=False)
    return data_detrend
 

def show_weights(array):
    # https://stackoverflow.com/a/50154388
    import numpy as np
    import seaborn as sns
    n_samples = array.shape[0]
    classes, bins = np.unique(array, return_counts=True)
    n_classes = len(classes)
    weights = n_samples / (n_classes * bins)
    sns.barplot(classes, weights)
    plt.xlabel('class label')
    plt.ylabel('weight')
    plt.show()


def melt_df(df, melt_columns):
    # save the column names of the dataframe in a list:
    column_names = df.columns.tolist()
    # remove the stimulus classes from the column names;
    id_vars = [x for x in column_names if x not in melt_columns]
    # melt the dataframe creating one column with value_name and var_name:
    df_melt = pd.melt(
            df, value_name='probability', var_name='class', id_vars=id_vars)
    # return the melted dataframe:
    return df_melt

# combine all masks from the feature selection by intersection:
def multimultiply(arrays):
    import copy
    # start with the first array:
    array_union = copy.deepcopy(arrays[0].astype(np.int))
    # loop through all arrays
    for i in range(len(arrays)):
        # multiply every array with all previous array
        array_union = np.multiply(array_union, copy.deepcopy(arrays[i].astype(np.int)))
    # return the union of all arrays:
    return(array_union)

# function of producing a sequence transition matrix
def TransMO(x):
    # create an empty matrix
    transition_matrix = np.zeros([5,5])
    # transition
    for a in range(len(x)):
       transition_matrix[x[a][0]-1][x[a][1]-1] = 1
    return(transition_matrix)

def TransMI(x):
    # create an empty matrix
    transition_matrix = np.zeros([5,5])
    # transition
    for a in range(len(x)):
       transition_matrix[x[a][0]-1][x[a][1]-1] = 1
    return(transition_matrix)


def stim_index(x):
    if x =='face': 
     return 1 
    elif x =='house':
     return 2
    elif x =='cat':
     return 3
    elif x =='shoe':
     return 4
    elif x =='chair':
     return 5


def matlab_percentile(x, p):
    p = np.asarray(p, dtype=float)
    n = len(x)
    p = (p-50)*n/(n-1) + 50
    p = np.clip(p, 0, 100)
    return np.percentile(x, p)

'''
========================================================================
DEFINE THE CLASSIFIERS
========================================================================
'''
class_labels = ['cat', 'chair', 'face', 'house', 'shoe']
# create a dictionary with all values as independent instances:
# see here: https://bit.ly/2J1DvZm
clf_set = {key: LogisticRegression(
    C=1., # Inverse of regularization strength
    penalty='l2', multi_class='ovr', solver='lbfgs', # Algorithm to use in the optimization problem, Stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno
    max_iter=4000, class_weight='balanced', random_state=42  # to shuffle the data
    ) for key in class_labels}

classifiers = {
    'log_reg': LogisticRegression(
        C=1., penalty='l2', multi_class='multinomial', solver='lbfgs',
        max_iter=4000, class_weight='balanced', random_state=42)}
clf_set.update(classifiers)    


'''
========================================================================
LOAD TDLM PARAMETERS:
========================================================================
'''
# DEFINE DECODING SPECIFIC PARAMETERS:
# define the mask to be used:
mask = 'visual'  # visual or whole
# applied time-shift to account for the BOLD delay, in seconds:
bold_delay = 4  # 4, 5 or 6 secs
# define the degree of smoothing of the functional data
smooth = 4

# DEFINE RELEVANT VARIABLES:
# time of repetition (TR), in seconds:
t_tr = params['mri']['tr']
# number of volumes (TRs) for each functional task run:
n_tr_run = 530
# acquisition time window of one sequence trial, in seconds:
t_win = 16
# number of measurements that are considered per sequence time window:
n_tr_win = round(t_win / t_tr)
# number of oddball trials in the experiment:
n_tr_odd = 600
# number of sequence trials in the experiment:
n_tr_seq = 75
# number of repetition trials in the experiment:
n_tr_rep = 45
# number of scanner triggers before the experiment starts:
n_tr_wait = params['mri']['num_trigger']
# number of functional task runs in total:
n_run = params['mri']['num_runs']
# number of experimental sessions in total:
n_ses = params['mri']['num_sessions']
# number of functional task runs per session:
n_run_ses = int(n_run / n_ses)

# parameters for TDLM
# the list of all the sequences
uniquePerms = list(itertools.permutations([1,2,3,4,5],5))
# the number of sequences
nShuf = len(uniquePerms)
# all the sequences for the specific subject
real_sequence = [(1,2,3,4,5)]
# all possible sequences
all_sequence = np.array(uniquePerms)
# rest of sequences for the specific subject
set_y = set(map(tuple, real_sequence))
idx = [tuple(point) not in set_y for point in all_sequence]
rest_sequence = all_sequence[idx]
# recombine the real sequence and rest of sequences
new_all_sequence = np.vstack((real_sequence,rest_sequence))
# the number of timelags, including 32, 64, 128, 512and 2048ms
maxLag= 20
# the number of states (decoding models)
nstates = 5
# the number of subjects
# nSubj = len(suball)
nSubj = 36
nipi = 2
# the nan matrix for forward design matrix (36 subjects * 120 shuffles * 11 timelags)
sf = np.full([nSubj,len(new_all_sequence),maxLag+1],np.nan)
sb = np.full([nSubj,len(new_all_sequence),maxLag+1],np.nan)
# predefine GLM data frame 
betas_1GLM = None
betas_2GLM = None


'''
========================================================================
LOAD THE DATASET
========================================================================
'''
seq_pred_matrix_0512_list1 = scio.loadmat('F://NicoData//0512prediction//seq_pred_matrix_0512_list.mat')
seq_pred_matrix_2048_list1 = scio.loadmat('F://NicoData//2048prediction//seq_pred_matrix_2048_list.mat')

seq_pred_matrix_0512_list2 = seq_pred_matrix_0512_list1['seq_pred_matrix_0512_list']
seq_pred_matrix_2048_list2 = seq_pred_matrix_2048_list1['seq_pred_matrix_2048_list']

# classifiers prediction probability 
serial_0512 = np.reshape(seq_pred_matrix_0512_list2,(36,13,15,5),order = 'F')
serial_2048 = np.reshape(seq_pred_matrix_2048_list2,(36,13,15,5),order = 'F')

trans = 1

if trans == 4:
    rp = [np.array([1,5])]
    T = TransMO(rp)
    Tt =  np.transpose(T)
    real_sequence = [(1,5)]
    uniquePerms = list(itertools.permutations([1,2,3,4,5],2))
    all_sequence = np.array(uniquePerms)
    # rest of sequences for the specific subject
    set_y = set(map(tuple, real_sequence))
    idx = [tuple(point) not in set_y for point in all_sequence]
    rest_sequence = all_sequence[idx]
    # recombine the real sequence and rest of sequences
    new_all_sequence = np.vstack((real_sequence,rest_sequence))
    transition = '1-5'
elif trans == 3:
    rp = [np.array([1,4]),np.array([2,5])]
    new_all_sequence = [rp]
    T = TransMO(rp)
    Tt =  np.transpose(T)
    i=0
    transition = '1-4'
elif trans == 2:
    rp = [np.array([1,3]),np.array([2,4]),np.array([3,5])]
    new_all_sequence = [rp]
    T = TransMO(rp)
    Tt =  np.transpose(T)
    i=0
    transition = '1-3'
elif trans == 1:
    rp = [np.array([1,2]),np.array([2,3]),np.array([3,4]),np.array([4,5])]
    new_all_sequence = [rp]
    T = TransMO(rp)
    Tt =  np.transpose(T)
    i=0
    transition = '1-2'

# # create the transition matrix for different condition
# rp4 = [np.array([1,5])]
# T4 = TransMO(rp4)
# T4t =  np.transpose(T4)
# real_sequence4 = [(1,5)]
# uniquePerms4 = list(itertools.permutations([1,2,3,4,5],2))
# all_sequence4 = np.array(uniquePerms4)
# # rest of sequences for the specific subject
# set_y = set(map(tuple, real_sequence4))
# idx = [tuple(point) not in set_y for point in all_sequence4]
# rest_sequence4 = all_sequence4[idx]
# # recombine the real sequence and rest of sequences
# new_all_sequence4 = np.vstack((real_sequence4,rest_sequence4))

# rp3 = [np.array([1,4]),np.array([2,5])]
# T3 = TransMO(rp3)
# T3t =  np.transpose(T3)

# rp2 = [np.array([1,3]),np.array([2,4]),np.array([3,5])]
# T2 = TransMO(rp2)
# T2t =  np.transpose(T2)

# rp1 = [np.array([1,2]),np.array([2,3]),np.array([3,4]),np.array([4,5])]
# T1 = TransMO(rp1)
# T1t =  np.transpose(T1)


# TDLM for 512ms
# set the empty rp
rp = []
# the nan matrix for forward design matrix (36 subjects * 120 shuffles * 11 timelags)
sf = np.full([nSubj,len(new_all_sequence),maxLag+1],np.nan)
sb = np.full([nSubj,len(new_all_sequence),maxLag+1],np.nan)
for (subi,sub) in zip(range(len(suball[0:nSubj])),suball[0:nSubj]):
    print(subi,sub)    
    # detect for each 2048 ms trial
    # loops for real sequence and permutation trials
    # for i in range(len(new_all_sequence4)):
    #     # real sequence
    #     rp = [new_all_sequence4[i,]]
    #     # get the transition matrix
    #     Tf = TransMI(rp)
    #     Tb = np.transpose(Tf)
    nbins=maxLag+1
    # X1 = seq_pred_matrix[13*i:13*(i+1),:] #####
    X1 =seq_pred_matrix_0512_list2[subi,:,:]
    # timelag matrix 
    dm = scipy.linalg.toeplitz(X1[:,0],[np.zeros((nbins,1))]) #####
    dm = dm[:,1:]
    # 4 loops for another 4 states
    for k in range(1,5) : 
        temp = scipy.linalg.toeplitz(X1[:,k],[np.zeros((nbins,1))])
        temp = temp[:,1:]
        dm = np.hstack((dm,temp))
    # the next time point needed to be predicted 
    Y = X1
    # build a new framework for first GLM betas
    betas_1GLM = np.full([nstates*maxLag, nstates],np.nan)
    #detect for each timelag
    for l in range(maxLag): 
        temp_zinds = np.array(range(0,nstates*maxLag,maxLag)) + l
        # First GLM
        design_mat_1 = np.hstack((dm[:,temp_zinds],np.ones((len(dm[:,temp_zinds]),1))))
        temp = np.dot(np.linalg.pinv(design_mat_1),Y)
        betas_1GLM[temp_zinds,:] = temp[:-1,:]  ####?
        
    betasnbins64 = np.reshape(betas_1GLM,[maxLag,np.square(nstates)],order = "F") ######
    # Second GLM
    design_mat_2 = np.transpose(np.vstack((np.reshape(T,25,order = "F"), 
                                           np.reshape(Tt,25,order = "F"), 
                                           np.reshape(np.eye(nstates),25,order = "F"),
                                           np.reshape(np.ones([nstates,nstates]),25,order = "F"))))
    betas_2GLM = np.dot(np.linalg.pinv(design_mat_2) , np.transpose(betasnbins64))
    # four different design matrix for regressor multiple to the temporal data
    # linear regression for the backward and forward replay 
    sf[subi,i,1:] = betas_2GLM[0,:]
    sb[subi,i,1:] = betas_2GLM[1,:]
    

### PLOT the FIGURE 
decoding_ipi=0.512
# mean the real sequence and permutation sequences respectively
sf_sequence = sf[:,0,:].mean(axis=0)
# temp_per = sf[:,1:,1:].mean(axis=0)
# temp_per = abs(temp_per)
# temp_per = np.amax(temp_per,axis=1)
# sf_permutation = matlab_percentile(temp_per,95)
# sf_sequence = pd.DataFrame({
#     'For Sequenceness': sf_sequence,
#     'Timelag(TR)': np.arange(0,nbins)})

sb_sequence = sb[:,0,:].mean(axis=0)
# temp_per = sb[:,1:,1:].mean(axis=0)
# temp_per = abs(temp_per)
# temp_per = np.amax(temp_per,axis=1)
# sb_permutation = matlab_percentile(temp_per,95)

dif = sf_sequence - sb_sequence
# temp_per = sf[:,1:,1:]-sb[:,1:,1:]
# temp_per = temp_per.mean(axis=0)
# temp_per = abs(temp_per)
# temp_per = np.amax(temp_per,axis=1)
# dif_permutation = matlab_percentile(temp_per,95)

# sns.relplot(x='Timelag(TR)', y='For Sequenceness', kind="line", data=sf_sequence);
# plot the results
plt.figure(dpi=400,figsize=(10,5))
x=np.arange(0,nbins,1)
plt.xticks(range(len(x)),x,fontsize=10)
plt.plot(x,sf_sequence,color='turquoise',linestyle='-',marker = 'o')
# plt.axhline(y=sf_permutation,color='silver',linestyle='--')
# plt.axhline(y=(-sf_permutation),color='silver',linestyle='--')
plt.title('Forward replay in sequential condition in 512ms')
plt.xlabel('timelag (TRs)')
plt.ylabel('sequenceness')
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\Forward Replay-%s-%6.3f.png' % (transition, decoding_ipi)),
            dpi = 400,bbox_inches = 'tight')
plt.show()

plt.figure(dpi=400,figsize=(10,5))
x=np.arange(0,nbins,1)
plt.xticks(range(len(x)),x,fontsize=10)
l1=plt.plot(x,sb_sequence,color='turquoise',linestyle='-',marker = 'o')
# plt.axhline(y=sb_permutation,color='silver',linestyle='--')
# plt.axhline(y=(-sb_permutation),color='silver',linestyle='--')
plt.title('Backward replay in sequential condition in 512ms')
plt.xlabel('timelag (TRs)')
plt.ylabel('sequenceness')
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\Backward Replay-%s-%6.3f.png' % (transition, decoding_ipi)),
            dpi = 400,bbox_inches = 'tight')
plt.show()

plt.figure(dpi=400,figsize=(10,5))
x=np.arange(0,nbins,1)
plt.xticks(range(len(x)),x,fontsize=10)
l1=plt.plot(x,dif,color='turquoise',linestyle='-',marker = 'o')
# plt.axhline(y=dif_permutation,color='silver',linestyle='--')
# plt.axhline(y=(-dif_permutation),color='silver',linestyle='--')
plt.title('Forward-Backward replay in sequential condition in 512ms')
plt.xlabel('timelag (TRs)')
plt.ylabel('squenceness')
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\Forward-Backward replay-%s-%6.3f.png' % (transition, decoding_ipi)),
            dpi = 400,bbox_inches = 'tight')
plt.show()



# TDLM for 2048ms
# set the empty rp
rp = []
# the nan matrix for forward design matrix (36 subjects * 120 shuffles * 11 timelags)
sf = np.full([nSubj,len(new_all_sequence),maxLag+1],np.nan)
sb = np.full([nSubj,len(new_all_sequence),maxLag+1],np.nan)
for (subi,sub) in zip(range(len(suball[0:nSubj])),suball[0:nSubj]):
    print(subi,sub)    
    # detect for each 2048 ms trial
    # loops for real sequence and permutation trials
    # for i in range(len(new_all_sequence4)):
    #     # real sequence
    #     rp = [new_all_sequence4[i,]]
    #     # get the transition matrix
    #     Tf = TransMI(rp)
    #     Tb = np.transpose(Tf)
    nbins=maxLag+1
    # X1 = seq_pred_matrix[13*i:13*(i+1),:] #####
    X1 =seq_pred_matrix_2048_list2[subi,:,:]
    # timelag matrix 
    dm = scipy.linalg.toeplitz(X1[:,0],[np.zeros((nbins,1))]) #####
    dm = dm[:,1:]
    # 4 loops for another 4 states
    for k in range(1,5) : 
        temp = scipy.linalg.toeplitz(X1[:,k],[np.zeros((nbins,1))])
        temp = temp[:,1:]
        dm = np.hstack((dm,temp))
    # the next time point needed to be predicted 
    Y = X1
    # build a new framework for first GLM betas
    betas_1GLM = np.full([nstates*maxLag, nstates],np.nan)
    #detect for each timelag
    for l in range(maxLag): 
        temp_zinds = np.array(range(0,nstates*maxLag,maxLag)) + l
        # First GLM
        design_mat_1 = np.hstack((dm[:,temp_zinds],np.ones((len(dm[:,temp_zinds]),1))))
        temp = np.dot(np.linalg.pinv(design_mat_1),Y)
        betas_1GLM[temp_zinds,:] = temp[:-1,:]  ####?
        
    betasnbins64 = np.reshape(betas_1GLM,[maxLag,np.square(nstates)],order = "F") ######
    # Second GLM
    design_mat_2 = np.transpose(np.vstack((np.reshape(T,25,order = "F"), 
                                           np.reshape(Tt,25,order = "F"), 
                                           np.reshape(np.eye(nstates),25,order = "F"),
                                           np.reshape(np.ones([nstates,nstates]),25,order = "F"))))
    betas_2GLM = np.dot(np.linalg.pinv(design_mat_2) , np.transpose(betasnbins64))
    # four different design matrix for regressor multiple to the temporal data
    # linear regression for the backward and forward replay 
    sf[subi,i,1:] = betas_2GLM[0,:]
    sb[subi,i,1:] = betas_2GLM[1,:]
    

### PLOT the FIGURE 
decoding_ipi=2.048
# mean the real sequence and permutation sequences respectively
sf_sequence = sf[:,0,:].mean(axis=0)
# temp_per = sf[:,1:,1:].mean(axis=0)
# temp_per = abs(temp_per)
# temp_per = np.amax(temp_per,axis=1)
# sf_permutation = matlab_percentile(temp_per,95)
# sf_sequence = pd.DataFrame({
#     'For Sequenceness': sf_sequence,
#     'Timelag(TR)': np.arange(0,nbins)})

sb_sequence = sb[:,0,:].mean(axis=0)
# temp_per = sb[:,1:,1:].mean(axis=0)
# temp_per = abs(temp_per)
# temp_per = np.amax(temp_per,axis=1)
# sb_permutation = matlab_percentile(temp_per,95)

dif = sf_sequence - sb_sequence
# temp_per = sf[:,1:,1:]-sb[:,1:,1:]
# temp_per = temp_per.mean(axis=0)
# temp_per = abs(temp_per)
# temp_per = np.amax(temp_per,axis=1)
# dif_permutation = matlab_percentile(temp_per,95)

# sns.relplot(x='Timelag(TR)', y='For Sequenceness', kind="line", data=sf_sequence);
# plot the results
plt.figure(dpi=400,figsize=(10,5))
x=np.arange(0,nbins,1)
plt.xticks(range(len(x)),x,fontsize=10)
plt.plot(x,sf_sequence,color='turquoise',linestyle='-',marker = 'o')
# plt.axhline(y=sf_permutation,color='silver',linestyle='--')
# plt.axhline(y=(-sf_permutation),color='silver',linestyle='--')
plt.title('Forward replay in sequential condition in 2048ms')
plt.xlabel('timelag (TRs)')
plt.ylabel('sequenceness')
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\Forward Replay-%s-%6.3f.png' % (transition, decoding_ipi)),
            dpi = 400,bbox_inches = 'tight')
plt.show()

plt.figure(dpi=400,figsize=(10,5))
x=np.arange(0,nbins,1)
plt.xticks(range(len(x)),x,fontsize=10)
l1=plt.plot(x,sb_sequence,color='turquoise',linestyle='-',marker = 'o')
# plt.axhline(y=sb_permutation,color='silver',linestyle='--')
# plt.axhline(y=(-sb_permutation),color='silver',linestyle='--')
plt.title('Backward replay in sequential condition in 2048ms')
plt.xlabel('timelag (TRs)')
plt.ylabel('sequenceness')
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\Backward Replay-%s-%6.3f.png' % (transition, decoding_ipi)),
            dpi = 400,bbox_inches = 'tight')
plt.show()

plt.figure(dpi=400,figsize=(10,5))
x=np.arange(0,nbins,1)
plt.xticks(range(len(x)),x,fontsize=10)
l1=plt.plot(x,dif,color='turquoise',linestyle='-',marker = 'o')
# plt.axhline(y=dif_permutation,color='silver',linestyle='--')
# plt.axhline(y=(-dif_permutation),color='silver',linestyle='--')
plt.title('Forward-Backward replay in sequential condition in 2048ms')
plt.xlabel('timelag (TRs)')
plt.ylabel('squenceness')
plt.savefig(('C:\\Users\\NINGMEI\\Desktop\\Sequence\\Forward-Backward replay-%s-%6.3f.png' % (transition, decoding_ipi)),
            dpi = 400,bbox_inches = 'tight')
plt.show()