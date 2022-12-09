#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 21:57:36 2022

@author: huangqi
"""
from scipy.stats import pearsonr  

probability_matrix = bold_cross
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
        proj = X1[l:-l,:]#???

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
    
    return cross
        
# [subject,condition,timelag]
# plot 512ms probability
s1mean = (cross[:,0,:]-cross[:,1,:]).mean(axis=0)
s1sem = (cross[:,0,:]-cross[:,1,:]).std(axis=0)/np.sqrt(len(cross))
x = np.arange(0, nbins, 1)
plt.figure(dpi=400, figsize=(10, 5))
plt.xticks(range(len(x)), x, fontsize=10)
plt.plot(x, s1mean, color='darkred', linestyle='-', linewidth=1)
plt.fill_between(x, s1mean-s1sem, s1mean+s1sem,
                 color='lightcoral', alpha=0.5, linewidth=1)
plt.title('Correlation: fwd-bkw: MTL to Visual Cortex')
plt.xlabel('lag (TRs)')
plt.ylabel('fwd minus bkw sequenceness')
plt.savefig(opj(path_out_all, 'crosscorrelation', 'MTL to Visual Cortex.png'),
            dpi=400, bbox_inches='tight')
plt.show()
        





# def TDLM_cross(probability_matrix):
#     rp = []
#     # print(isub,sub)
#     # detect for each 2048 ms trial
#     # loops for real sequence and permutation trials
#     # real sequence
#     rp = [1, 2]
#     # get the transition matrix
#     T1 = TransM_cross(rp)
#     T2 = np.transpose(T1)
#     nbins = maxLag+1
#     # X1 = seq_pred_matrix[13*i:13*(i+1),:] #####
#     X1 = np.array(probability_matrix)
#     # timelag matrix
#     dm = scipy.linalg.toeplitz(X1[:, 0], [np.zeros((nbins, 1))])
#     dm = dm[:, 1:]
#     # 4 loops for another 4 states
#     for k in range(1, 2):
#         temp = scipy.linalg.toeplitz(X1[:, k], [np.zeros((nbins, 1))])
#         temp = temp[:, 1:]
#         dm = np.hstack((dm, temp))
#     # the next time point needed to be predicted
#     Y = X1
#     # build a new framework for first GLM betas
#     betas_1GLM = np.full([2*maxLag, 2], np.nan)
#     # detect for each timelag
#     for l in range(maxLag):
#         temp_zinds = np.array(range(0, 2*maxLag, maxLag)) + l
#         # First GLM
#         design_mat_1 = np.hstack(
#             (dm[:, temp_zinds], np.ones((len(dm[:, temp_zinds]), 1))))
#         temp = np.dot(np.linalg.pinv(design_mat_1), Y)
#         betas_1GLM[temp_zinds, :] = temp[:-1, :]  # ?

#     betasnbins64 = np.reshape(betas_1GLM, [maxLag, np.square(2)], order="F")
#     # Second GLM
#     design_mat_2 = np.transpose(np.vstack((np.reshape(T1, 4, order="F"),
#                                            np.reshape(T2, 4, order="F"),
#                                            np.reshape(np.eye(2), 4, order="F"),
#                                            np.reshape(np.ones([2, 2]), 4, order="F"))))
#     betas_2GLM = np.dot(np.linalg.pinv(design_mat_2),
#                         np.transpose(betasnbins64))
#     # four different design matrix for regressor multiple to the temporal data
#     # linear regression for the backward and forward replay
#     cross[0, 1:] = betas_2GLM[0, :]  # 51 condition forward
#     cross[1, 1:] = betas_2GLM[1, :]  # 51 condition backward