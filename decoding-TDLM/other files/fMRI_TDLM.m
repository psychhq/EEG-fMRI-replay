clear;
clc;
close all;
rng('shuffle')

%% Spects
load seq_pred_matrix
nstates=5;
TF = diag(ones(nstates-1,1),1);
TR = TF';
maxLag = 10; % evaluate time lag up to 600ms
cTime = 0:1.25:maxLag*1.25; % the seconds of each cross-correlation time lag
[~, pInds] = uperms([1:nstates],29);
uniquePerms=pInds;
nShuf = size(uniquePerms,1);
preds=seq_pred_matrix;
nSubj=1;

sf = cell(nSubj,1);  sb = cell(nSubj,1);
sf2 = cell(nSubj,1);  sb2 = cell(nSubj,1);

%% Core function
for iSj = 1:nSubj
    sf{iSj} = nan(nShuf, maxLag+1);
    sb{iSj} = nan(nShuf, maxLag+1);
      
    sf2{iSj} = nan(nShuf, maxLag+1);
    sb2{iSj} = nan(nShuf, maxLag+1);    
    %% calculate sequenceness 
    for iShuf = 1:nShuf
        rp = uniquePerms(iShuf,:);  % use the 30 unique permutations (is.nShuf should be set to 29)
        T1 = TF(rp,rp); 
        T2 = T1'; % backwards is transpose of forwards
        X=preds;
        
        nbins=maxLag+1;

       warning off
       dm=[toeplitz(X(:,1),[zeros(nbins,1)])];
       dm=dm(:,2:end);
       
       for kk=2:nstates
           temp=toeplitz(X(:,kk),[zeros(nbins,1)]);
           temp=temp(:,2:end);
           dm=[dm temp]; 
       end
       
       warning on
       
       Y=X;       
       betas = nan(nstates*maxLag, nstates);
%           
      %% GLM: state regression, with other lages       
       bins=maxLag;

       for ilag=1:bins
           temp_zinds = (1:bins:nstates*maxLag) + ilag - 1ilag - 1; 
           temp = pinv([dm(:,temp_zinds) ones(length(dm(:,temp_zinds)),1)])*Y;
           betas(temp_zinds,:)=temp(1:end-1,:);           
       end  

       betasnbins64=reshape(betas,[maxLag nstates^2]);
       bbb=pinv([T1(:) T2(:) squash(eye(nstates)) squash(ones(nstates))])*(betasnbins64'); %squash(ones(nstates))

       sf{iSj}(iShuf,2:end) = bbb(1,:); 
       sb{iSj}(iShuf,2:end) = bbb(2,:); 

      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
      %% Cross-Correlation
%       for iLag=1:maxLag
%           sf2{iSj}(1,iShuf,2:end) = sequenceness_Crosscorr(preds, T1, [], iLag);
%           sb2{iSj}(1,iShuf,2:end) = sequenceness_Crosscorr(preds, T2, [], iLag);
%       end                     
    end
end

sf = cell2mat(sf);
sb = cell2mat(sb);

sf2 = cell2mat(sf2);
sb2 = cell2mat(sb2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure, 

%% GLM (fwd-bkw)
subplot(2,3,1)
a=sf(2:end,2:end)-sb(2:end,2:end);
b=abs(a);
c=max(b,[],2);
npThresh = squeeze(c);
npThreshAll = prctile(npThresh,95);  
dtp = squeeze(sf(:,1,:)-sb(:,1,:));
shadedErrorBar(cTime, dtp, dtp*0/sqrt(nSubj)), hold on,
plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
title('GLM: fwd-bkw'), xlabel('lag (ms)'), ylabel('fwd minus bkw sequenceness')

%% GLM (fwd)
subplot(2,3,2)
npThresh = squeeze(max(abs(mean(sf(:,2:end,2:end),1)),[],3));
npThreshAll = prctile(npThresh,95);  
dtp = squeeze(sf(:,1,:));
shadedErrorBar(cTime, dtp, dtp*0/sqrt(nSubj)), hold on,
plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
title('GLM: fwd'), xlabel('lag (ms)'), ylabel('fwd sequenceness')

%% GLM (bkw)
subplot(2,3,3)
npThresh = squeeze(max(abs(mean(sb(:,2:end,2:end),1)),[],3));
npThreshAll = prctile(npThresh,95);  
dtp = squeeze(sb(:,1,:));
shadedError Bar(cTime, dtp, dtp*0/sqrt(nSubj)), hold on,
plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
title('GLM: bkw'), xlabel('lag (ms)'), ylabel('bkw sequenceness')
% 
% %% Cross-Correlation (fwd-bkw)
% sf=sf2;
% sb=sb2;
% subplot(2,3,4)
% npThresh = squeeze(max(abs(mean(sf(:,2:end,2:end)-sb(:,2:end,2:end),1)),[],3));
% npThreshAll = prctile(npThresh,95);  
% dtp = squeeze(sf(:,1,:)-sb(:,1,:));
% shadedErrorBar(cTime, dtp, dtp*0/sqrt(nSubj)), hold on,
% plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
% title('Correlation: fwd-bkw'), xlabel('lag (ms)'), ylabel('fwd minus bkw sequenceness')
% 
% %% Cross-Correlation (fwd)
% subplot(2,3,5)
% npThresh = squeeze(max(abs(mean(sf(:,2:end,2:end),1)),[],3));
% npThreshAll = prctile(npThresh,95); 
% dtp = squeeze(sf(:,1,:));
% shadedErrorBar(cTime, dtp, dtp*0/sqrt(nSubj)), hold on,
% plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
% title('Correlation: fwd'), xlabel('lag (ms)'), ylabel('fwd sequenceness')
% 
% %% Cross-Correlation (bkw)
% subplot(2,3,6)
% npThresh = squeeze(max(abs(mean(sb(:,2:end,2:end),1)),[],3));
% npThreshAll = prctile(npThresh,95); 
% dtp = squeeze(sb(:,1,:));
% shadedErrorBar(cTime, dtp, dtp*0/sqrt(nSubj)), hold on,
% plot([cTime(1) cTime(end)], -npThreshAll*[1 1], 'k--'), plot([cTime(1) cTime(end)], npThreshAll*[1 1], 'k--')
% title('Correlation: bkw'), xlabel('lag (ms)'), ylabel('bkw sequenceness')
