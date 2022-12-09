# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:48:50 2022

@author: Qi Huang
"""

# library the packages
import numpy as np
import scipy
import matplotlib as plt
import scipy.io as scio


# set the parameters
TF = np.mat('0,1,0,0,0,0,0,0;0,0,1,0,0,0,0,0;0,0,0,1,0,0,0,0;0,0,0,0,0,0,0,0;0,0,0,0,0,1,0,0;0,0,0,0,0,0,1,0;0,0,0,0,0,0,0,1;0,0,0,0,0,0,0,0')
# the number of timelags, including 32, 64, 128, 512and 2048ms
maxLag_rest = 60
# the number of states (decoding models)
nstates = 8 #len(betas_de_re[0,])
# set the final data frame
sf = np.full([29,maxLag_rest+1],np.nan)
sb = np.full([29,maxLag_rest+1],np.nan)
# load the data
data = scio.loadmat('C://Users//NINGMEI//Desktop//matlab.mat')
uniquePerm = scio.loadmat('C://Users//NINGMEI//Desktop//uniquePerms.mat')
uniquePerms = uniquePerm['uniquePerms']


# run the TDLM based on the existed prediction
for iShuf in range(0,29):
    T1=np.full([8,8],np.nan) #####
    # all the sequences for the specific subject
    rp = uniquePerms[iShuf]
    rp = np.array(rp, dtype=int)
    for i in range(8): #####
        for j in range(8):
            T1[i,j] = TF[rp[i]-1,rp[j]-1]
            
    # for i in range(nstates):
    T2 = np.transpose(T1) 
    X = data['preds'] #####

    nbins= maxLag_rest+1

    # timelag matrix 
    dm = scipy.linalg.toeplitz(X[:,0],[np.zeros((nbins,1))]) #####
    dm = dm[:,1:]
    
    # 4 loops
    for k in range(1,nstates) : 
        temp = scipy.linalg.toeplitz(X[:,k],[np.zeros((nbins,1))]);
        temp = temp[:,1:]
        dm = np.hstack((dm,temp))
    
    # the next time point needed to be predicted 
    Y = X
    # build a new framework for first GLM betas
    betas_1GLM = np.full([nstates*maxLag_rest, nstates],np.nan)
    
    #detect for each timelag
    for l in range(maxLag_rest): 
        temp_zinds = np.array(range(0,nstates*maxLag_rest,maxLag_rest)) + l
        # First GLM
        design_mat_1 = np.hstack((dm[:,temp_zinds],np.ones((len(dm[:,temp_zinds]),1))))
        temp = np.dot(np.linalg.pinv(design_mat_1),Y)
        betas_1GLM[temp_zinds,:] = temp[:-1,:]  ####?
    
    betasnbins64 = np.reshape(betas_1GLM,[maxLag_rest,np.square(nstates)],order = "F") ######
    # Second GLM
    design_mat_2 = np.transpose(np.vstack((np.reshape(T1,64,order = "F"), 
                                           np.reshape(T2,64,order = "F"), 
                                           np.reshape(np.eye(nstates),64,order = "F"),
                                           np.reshape(np.ones([nstates,nstates]),64,order = "F"))))
    betas_2GLM = np.dot(np.linalg.pinv(design_mat_2) , np.transpose(betasnbins64)); 
    # four different design matrix for regressor multiple to the temporal data
    # linear regression for the backward and forward replay
        
    sf[iShuf,1:] = betas_2GLM[0,:]
    sb[iShuf,1:] = betas_2GLM[1,:]




# mean the all the sequence
sf_sequence = sf[0,:]-sb[0,:]

sf_permutation = sf[1:,:]-sb[1:,:]
sf_permutation_mean = sf_permutation.mean(axis=0)


# plot the sequenceness
plt.figure(dpi=400)
plt.xticks(fontsize=10)
x=np.arange(0,nbins)
l1=plt.plot(x,sf_permutation_mean,'g-',label='permutation trial')
l2=plt.plot(x,sf_sequence,'r-',label='sequential trial')

plt.plot(x,sf_permutation_mean,'g-',x,sf_sequence,'r-')
plt.title('replay in simulated data from Matlab')
plt.xlabel('timelag (TRs)')
plt.ylabel('sequenceness')
plt.legend()
plt.show()