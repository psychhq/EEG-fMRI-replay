# -*- coding: utf-8 -*-
"""
Created on Mon May 23 23:18:53 2022

@author: Qi Huang
"""
rp = []
for (subi,sub) in zip(range(len(suball[0:nSubj])),suball[0:nSubj]):
    print(subi,sub)    
    #detect for each 2048 ms trial
    # loops for real sequence and permutation trials
    for i in range(len(new_all_sequence)):
        # real sequence
        rp = new_all_sequence[i,]
        # get the transition matrix
        T1 = TransM(rp)
        T2 = np.transpose(T1)
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
            for j in range(nstates):
                design_mat_1 = np.transpose(np.vstack((dm[:,temp_zinds[j]],np.ones((len(dm[:,temp_zinds]))))))
                temp = np.dot(np.linalg.pinv(design_mat_1),Y)
                betas_1GLM[temp_zinds[j],:] = temp[:-1,:]  ####?
            
        betasnbins64 = np.reshape(betas_1GLM,[maxLag,np.square(nstates)],order = "F") ######
        # Second GLM
        design_mat_2 = np.transpose(np.vstack((np.reshape(T1,25,order = "F"), 
                                               np.reshape(T2,25,order = "F"), 
                                               np.reshape(np.eye(nstates),25,order = "F"),
                                               np.reshape(np.ones([nstates,nstates]),25,order = "F"))))
        betas_2GLM = np.dot(np.linalg.pinv(design_mat_2) , np.transpose(betasnbins64))
        # four different design matrix for regressor multiple to the temporal data
        # linear regression for the backward and forward replay 
        sf[subi,i,1:] = betas_2GLM[0,:]
        sb[subi,i,1:] = betas_2GLM[1,:]
