# Nestor Cabello, Elham Naghizade, Jianzhong Qi, Lars Kulik

# Cabello N, Naghizade E, Qi J, Kulik L (2021) Fast, Accurate and Interpretable Time Series Classification Through Randomization.


import math
import time
import random
import pandas as pd

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier

from scipy.io import arff
from scipy.stats import iqr, zscore

from numba import jit

import pyfftw

from matplotlib import pyplot as plt

from statsmodels.regression.linear_model import burg

import warnings
warnings.filterwarnings("ignore")


#################### aggregation functions - functions #################################

@jit(nopython=True, fastmath=True)
def fast_mean(X):
    nrows,_ = X.shape
    _X = np.zeros((nrows,))
    for i in range(nrows):
        _X[i] = inner_mean(X[i,:])
    return _X

@jit(nopython=True, fastmath=True)
def inner_mean(X):
    ncols = len(X)
    accum = 0
    for i in range(ncols):
        accum+=X[i]
    return accum/ncols



@jit(nopython=True, fastmath=True)
def fast_std(X):
    nrows,_ = X.shape
    _X = np.zeros((nrows,))
    for i in range(nrows):
        _X[i] = inner_std(X[i,:])
    return _X

@jit(nopython=True, fastmath=True)
def inner_std(X):
    ncols = len(X)
    accum = 0
    X_mean = inner_mean(X)
    for i in range(ncols):
        accum+=(X[i]-X_mean)**2
    return (accum/ncols)**0.5



@jit(nopython=True, fastmath=True)
def fast_slope(Y):
    r,c = Y.shape
    x = np.arange(0, c)
    _X = np.zeros((r,))
    for i in range(r):
        _X[i] = inner_slope(Y[i,:],x)
    return _X

@jit(nopython=True, fastmath=True)
def inner_slope(X,indices):
    ncols = len(X)
    SUMx = 0 
    SUMy = 0
    SUMxy = 0
    SUMxx = 0
    for i in range(ncols):
        SUMx = SUMx + indices[i]
        SUMy = SUMy + X[i]
        SUMxy = SUMxy + indices[i]*X[i]
        SUMxx = SUMxx + indices[i]*indices[i]
    return ( SUMx*SUMy - ncols*SUMxy ) / ( SUMx*SUMx - ncols*SUMxx )



@jit(nopython=True, fastmath=True)
def fast_iqr(X):
    nrows,_ = X.shape
    _X = np.zeros((nrows,))
    for i in range(nrows):
        _X[i] = inner_iqr(X[i,:])
    return _X

@jit(nopython=True, fastmath=True)
def inner_iqr(a):
    n, = a.shape
    if n%2!=0:
        median_idx = n//2
        q1_idx = median_idx//2
        q3_idx = ((n-median_idx)//2)+median_idx
        return a[q3_idx]-a[q1_idx]
    median_idx_lower = (n-1)//2
    median_idx_upper = n//2
    q1_idx = median_idx_lower//2
    q3_idx = (median_idx_upper//2)+median_idx_upper
    return a[q3_idx]-a[q1_idx]


@jit(nopython=True, fastmath=True)
def count_mean_crossing(X):
    nrows = X.shape[0]
    X_ = np.zeros((nrows,))
    i = 0
    for x in X:
        X_[i] = np.count_nonzero(np.diff(x > np.mean(x)))
        i=i+1
    return X_


@jit(nopython=True, fastmath=True)
def count_values_above_mean(X):
    nrows = X.shape[0]
    X_ = np.zeros((nrows,))
    i = 0
    for x in X:
        X_[i] = np.count_nonzero(x > np.mean(x))
        i=i+1
    return X_



######################################################################################


def getXysets(train_set, test_set):

    #get X_train and y_train
    data_train = arff.loadarff(train_set)
    df_train = pd.DataFrame(data_train[0])
    df_train = df_train.fillna(0)
    raw_data = []
    for row in df_train.itertuples():
        raw_data.append(row)
    X_train = []
    y_train = []
    for data in raw_data:
        X_train.append(data[1: len(data)-1])
        y_train.append(int(data[-1]))
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    #get X_test and y_test
    data_test = arff.loadarff(test_set)
    df_test = pd.DataFrame(data_test[0])
    df_test = df_test.fillna(0)
    raw_data = []
    for row in df_test.itertuples():
        raw_data.append(row)
    X_test = []
    y_test = []
    for data in raw_data:
        X_test.append(data[1: len(data)-1])
        y_test.append(int(data[-1]))
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    return X_train, y_train, X_test, y_test




# Oversampling strategy to handle unbalanced datasets
def balanceSample(X_train,y_train):
    all_classes = np.unique(y_train)
    nclasses = len(all_classes)
    ninstances_per_class = np.zeros((nclasses,1))
    
    for c in range(0,nclasses):
        ninstances_per_class[c] = np.sum(y_train==all_classes[c])
    avg_ninstances_per_class = np.ceil(np.mean(ninstances_per_class))

    inx = ninstances_per_class<avg_ninstances_per_class
    inx = np.ndarray.flatten(inx)

    all_classes_lessthanavg = all_classes[inx]
    
    X_train_to_add = []
    y_train_to_add = []

    for p in range(0,len(all_classes_lessthanavg)):
        current_class = all_classes_lessthanavg[p]
        current_class_idx = np.where(all_classes==current_class)[0]
        toadd = avg_ninstances_per_class-ninstances_per_class[current_class_idx]
        toadd = int(toadd[0][0])

        row_idx = y_train==current_class
        X_train_c = X_train[row_idx,:]

        inx = np.random.choice(len(X_train_c),toadd,replace=True)
        X_train_to_add_c = list(X_train_c[inx,:])
        y_train_to_add_c = list(np.ndarray.flatten(np.ones((toadd,1),dtype = int))*current_class)

        X_train_to_add+=X_train_to_add_c
        y_train_to_add+=y_train_to_add_c
    
    return X_train_to_add,y_train_to_add




## this version of std uses the already computed mean
@jit(nopython=True, fastmath=True)
def inner_std2(X,Xmean):
    ncols = len(X)
    accum = 0
    for i in range(ncols):
        accum+=(X[i]-Xmean)**2
    if ncols==1:
        return 0.00001
    else:
        return (accum/(ncols-1))**0.5



@jit(nopython=True, fastmath = True)
def fisherScore(X_feat,y):
    unique_labels = np.unique(y)
    mu_feat = inner_mean(X_feat)
    accum_numerator = 0
    accum_denominator = 0
    
    for k in unique_labels:
        idx_label = np.where(y==k)[0]
        nk = len(idx_label)
        data_sub = X_feat[idx_label]

        mu_feat_label = inner_mean(data_sub)
        sigma_feat_label = max(inner_std2(data_sub, mu_feat_label),0.0001) ###to avoid div by zero in case 1 class label per instance      

        accum_numerator += nk*(mu_feat_label-mu_feat)**2
        accum_denominator +=  nk*sigma_feat_label**2
    if accum_numerator==0 or accum_denominator==0:
        return 0
    else:
        return accum_numerator/accum_denominator
    

    
# @jit(nopython=True, fastmath=True)
def supervisedSearch(X, y, ini_idx, agg_fn, repr_type, X_ori):

    candidate_agg_feats = []
    XT = np.empty((X.shape[0],1))

    while 1:

        _,len_subinterval = X.shape
        
        if (agg_fn==np.polyfit and len_subinterval < 4) or (len_subinterval < 2):
            break
        
        if agg_fn == np.polyfit:
            div_point = random.randint(2,len_subinterval-2)
        else:
            div_point = random.randint(1,len_subinterval-1)
        sub_interval_0 = X[:,:div_point]
        sub_interval_1 = X[:,div_point:]
        
        interval_feature_0 = getIntervalFeature(sub_interval_0,agg_fn)
        interval_feature_1 = getIntervalFeature(sub_interval_1,agg_fn)

        score_0 = fisherScore(interval_feature_0,y)
        score_1 = fisherScore(interval_feature_1,y)
        
        if score_0 >= score_1 and score_0!=0:     ####if true explore sub_interval_0
            w = len(sub_interval_0[0])
            ini = ini_idx+0
            ini_idx = ini
            end = ini+w
            
            candidate_agg_feats.append((w,score_0,ini,end,agg_fn,repr_type))                
            X = sub_interval_0

            interval_feature_touse = getIntervalFeature(X_ori[:,ini:end],agg_fn)
            XT = np.hstack((XT,np.reshape(interval_feature_touse,(interval_feature_touse.shape[0],1))))
            
        elif score_1 > score_0:  ####if true explore sub_interval_1
            w = len(sub_interval_1[0])
            ini = ini_idx+div_point
            ini_idx = ini
            end = ini+w
            
            candidate_agg_feats.append((w,score_1,ini,end,agg_fn,repr_type))            
            X = sub_interval_1
            
            interval_feature_touse = getIntervalFeature(X_ori[:,ini:end],agg_fn)
            XT = np.hstack((XT,np.reshape(interval_feature_touse,(interval_feature_touse.shape[0],1))))
            
        else:
            break
            
    XT = XT[:,1:]
    return candidate_agg_feats,XT



    
def getCandidateAggFeats(X,y,agg_fns,repr_type,X_ori):
    candidate_agg_feats = []
    XT = np.empty((X.shape[0],1))

    for agg_fn in agg_fns:
        random_cut_point = random.randint(1,len(X[0])-1)
        idx_ts =  int(random_cut_point)

        sub_ts_L = X[:,:idx_ts]
        ini_L = 0
        candidate_agg_feat_L,XT_L = supervisedSearch(sub_ts_L,y,ini_L,agg_fn,repr_type,X_ori)
        candidate_agg_feats.extend(candidate_agg_feat_L)
        XT = np.hstack((XT,XT_L))
        
        sub_ts_R = X[:,idx_ts:]
        ini_R = idx_ts
        candidate_agg_feat_R,XT_R = supervisedSearch(sub_ts_R,y,ini_R,agg_fn,repr_type,X_ori)
        candidate_agg_feats.extend(candidate_agg_feat_R)
        XT = np.hstack((XT,XT_R))

    XT = XT[:,1:]
    return candidate_agg_feats,XT



def getAllCandidateAggFeats(X_train, y_train, agg_fns, repr_types, 
                            per_X_train, diff_X_train, ar_X_train,
                            X_train_norm, per_X_train_norm, diff_X_train_norm, ar_X_train_norm):
    
    all_candidate_agg_feats = []
    XT = np.empty((X_train.shape[0],1))

        
    if 1 in repr_types: # raw series
        candidate_agg_feats_timeOriginal,XT_ori= getCandidateAggFeats(X_train_norm,y_train,agg_fns,1,X_train)
        all_candidate_agg_feats.extend(candidate_agg_feats_timeOriginal[0:])
        XT = np.hstack((XT,XT_ori))

    if 2 in repr_types: # periodogram
        candidate_agg_feats_freqPeriodogram,XT_per = getCandidateAggFeats(per_X_train_norm,y_train,agg_fns,2,per_X_train)
        all_candidate_agg_feats.extend(candidate_agg_feats_freqPeriodogram[0:])
        XT = np.hstack((XT,XT_per))
    
    if 3 in repr_types: # derivative
        candidate_agg_feats_timeDerivative,XT_diff = getCandidateAggFeats(diff_X_train_norm,y_train,agg_fns,3,diff_X_train)
        all_candidate_agg_feats.extend(candidate_agg_feats_timeDerivative[0:])
        XT = np.hstack((XT,XT_diff))

    if 4 in repr_types: # autoregressive
        candidate_agg_feats_AR,XT_ar = getCandidateAggFeats(ar_X_train_norm,y_train,agg_fns,4,ar_X_train)
        all_candidate_agg_feats.extend(candidate_agg_feats_AR[0:])
        XT = np.hstack((XT,XT_ar))

    
    XT = XT[:,1:]
    return all_candidate_agg_feats,XT



def getIntervalFeature(sub_interval,agg_fn):
    if agg_fn == np.polyfit:
        return fast_slope(sub_interval)
    elif agg_fn == np.mean:
        return fast_mean(sub_interval)
    elif agg_fn == np.std:
        return fast_std(sub_interval)
    elif agg_fn == iqr:
        return fast_iqr(np.sort(sub_interval,axis=1))
    elif agg_fn == np.percentile:
        return count_mean_crossing(sub_interval)
    elif agg_fn == np.quantile:
        return count_values_above_mean(sub_interval)
    else:
        return agg_fn(sub_interval,axis=1)
    

def getIntervalBasedTransform(X,per_X,diff_X,ar_X,candidate_agg_feats,relevant_candidate_agg_feats_id):
    nrows = X.shape[0]
    X_transform = np.zeros((nrows,len(candidate_agg_feats)))

    for j in relevant_candidate_agg_feats_id:

        w,score,li,ls,agg_fn,repr_type = candidate_agg_feats[j]
        if repr_type==1:
            X_temp = X
        elif repr_type==2:
            X_temp = per_X
        elif repr_type == 3:
            X_temp = diff_X
        elif repr_type == 4:
            X_temp = ar_X

        sub_interval = X_temp[:,li:ls]
        to_add = getIntervalFeature(sub_interval, agg_fn)
        X_transform[:,j] = to_add

    return X_transform

    

def getTrainTestSets(dset_name):
    path = "sampleUCRdatasets/" #replace with the path where the UCR datasets are located
    train_set = path+dset_name+"/"+dset_name+"_TRAIN.arff"
    test_set = path+dset_name+"/"+dset_name+"_TEST.arff"
    X_train, y_train, X_test, y_test = getXysets(train_set, test_set)
    return  X_train, y_train, X_test, y_test


def dataAugmented(X_train,y_train):
    X_toadd,y_toadd = balanceSample(X_train,y_train)
    if len(y_toadd)>0:
        X = np.vstack((X_train,X_toadd))
        y = np.hstack((y_train,y_toadd))
    else:
        X = X_train
        y = y_train

    per_X = getPeriodogramRepr(X)

    diff_X = np.diff(X)

    ar_X = ar_coefs(X)
    ar_X[np.isnan(ar_X)] = 0

    return X, per_X, diff_X, ar_X, y


def getPeriodogramRepr(X):
    nfeats = X.shape[1]
    fft_object = pyfftw.builders.fft(X)
    per_X = np.abs(fft_object())
    return per_X[:,:int(nfeats/2)]


def ar_coefs(X):
    X_transform = []
    lags = int(12*(X.shape[1]/100.)**(1/4.))
    for i in range(X.shape[0]):
        coefs,_ = burg(X[i,:],order=lags)
        X_transform.append(coefs)
    return np.array(X_transform)




########################## Additional functions used for interpretability ####################################

#Returns the information (i.e., starting and ending indices, aggregation function and time series representation) from each relevant interval feature
def get_lst_start_ending_indices(relevant_caf_idx_per_tree, all_candidate_agg_feats):
    all_start_idx = []
    all_end_idx = []
    all_agg_fns = []
    all_repr_types = []
    
    for relevant_caf_idx in relevant_caf_idx_per_tree:
        
        caf = np.array(all_candidate_agg_feats)[np.unique(relevant_caf_idx)]
        
        cur_start_idx = np.array(caf[:,2])
        cur_end_idx = np.array(caf[:,3])
        cur_agg_fns = np.array(caf[:,4])
        cur_repr_type = np.array(caf[:,5])
        all_start_idx.append(cur_start_idx.astype(int))
        all_end_idx.append(cur_end_idx.astype(int))
        all_agg_fns.append(cur_agg_fns)
        all_repr_types.append(cur_repr_type.astype(int))
    return all_start_idx,all_end_idx,all_agg_fns,all_repr_types


#Returns the discriminatory starting and ending indices as found on each trained tree for the given time series representation repr_type and aggregation function agg_fn
def get_all_start_end_idx_per_tree(t, agg_fn, repr_type, all_start_idx, all_end_idx, all_agg_fns, all_repr_types):
    repr_type_idx = np.where(all_repr_types[t] == repr_type)[0]
    all_agg_fn_to_use_by_repr_type = all_agg_fns[t][repr_type_idx]
    all_start_idx_to_use_by_repr_type = all_start_idx[t][repr_type_idx]
    all_end_idx_to_use_by_repr_type = all_end_idx[t][repr_type_idx]
    agg_fn_idx = np.where(all_agg_fn_to_use_by_repr_type == agg_fn)[0]
    all_start_idx_to_use_FINAL = all_start_idx_to_use_by_repr_type[agg_fn_idx]
    all_end_idx_to_use_FINAL = all_end_idx_to_use_by_repr_type[agg_fn_idx]
    return all_start_idx_to_use_FINAL,all_end_idx_to_use_FINAL



#Returns the importance of each time series representation
def getReprImportances(clf, all_candidate_agg_feats):
    ori = []
    per = []
    der = []
    reg = []
    features_importances=clf.feature_importances_
    cont = 0 
    for candAggFeats in all_candidate_agg_feats:
        if candAggFeats[5] == 1:
            ori.append(features_importances[cont])
        elif candAggFeats[5] == 2:
            per.append(features_importances[cont])
        elif candAggFeats[5] == 3:
            der.append(features_importances[cont])
        else: #candAggFeats[5] == 4:
            reg.append(features_importances[cont])
        
        cont+=1
    return np.mean(ori), np.mean(per), np.mean(der), np.mean(reg)
            

#Returns the importance of each aggregation function according to the relevant features extracted from the given time series representation _repr
def getStatsImportances(clf, all_candidate_agg_feats, _repr):
    #     agg_fns = [np.mean, np.std, np.polyfit, np.median, np.min, np.max, iqr, np.percentile, np.quantile]
    
    features_importances=clf.feature_importances_

    _mean = []
    _std = []
    _slope = []
    _median = []
    _min = []
    _max = []
    _iqr = []
    _cmc = []
    _cam = []

    cont = 0 
    for candAggFeats in all_candidate_agg_feats:
        if candAggFeats[5] == _repr:
            if candAggFeats[4] == np.mean:
                _mean.append(features_importances[cont])
            elif candAggFeats[4] == np.std:
                _std.append(features_importances[cont])
            elif candAggFeats[4] == np.polyfit:
                _slope.append(features_importances[cont])
            elif candAggFeats[4] == np.median:
                _median.append(features_importances[cont])
            elif candAggFeats[4] == np.min:
                _min.append(features_importances[cont])
            elif candAggFeats[4] == np.max:
                _max.append(features_importances[cont])
            elif candAggFeats[4] == iqr:
                _iqr.append(features_importances[cont])
            elif candAggFeats[4] == np.percentile:
                _cmc.append(features_importances[cont])
            else: #candAggFeats[4] == np.quantile:
                _cam.append(features_importances[cont])
        
        cont+=1
    return np.mean(_mean), np.mean(_std), np.mean(_slope), np.mean(_median), np.mean(_min), np.mean(_max), np.mean(_iqr), np.mean(_cmc),np.mean(_cam)



### Given a set of testing instances, and a matrix of all the predictions for each tree.
### Return a set of "intensities" (the number of times each testing instance data value is intersected by the each of discriminatory interval features)

#Considers all candidate discriminatory interval features (i.e., not only from trees reaching agreement)
def candDiscrIntFeats_per_aggfn (X_test,y_test,all_trees_predict,all_start_idx,all_end_idx,all_agg_fns,all_repr_types, repr_type, agg_fn):
    ntrees = len(all_trees_predict[0,:])
    intensity_map = np.zeros(X_test.shape)

    for i in range(1):#just need to this once..the same values apply for all the other instances
        agreements_tree_idx = np.arange(0,ntrees)
        for tree_idx in agreements_tree_idx:
            all_start_idx_to_use,all_end_idx_to_use = get_all_start_end_idx_per_tree(tree_idx, agg_fn, repr_type, all_start_idx, all_end_idx, all_agg_fns, all_repr_types)
            for j in range(len(all_start_idx_to_use)):
                intensity_map[i,all_start_idx_to_use[j]:all_end_idx_to_use[j]] += 1
    
    for i in range(len(y_test)-1):
        intensity_map[i+1,:] = intensity_map[0,:]
    return intensity_map


#Considers just the candidate discriminatory interval features from trees reaching agreement
def rois_per_aggfn (X_test,y_test,all_trees_predict,all_start_idx,all_end_idx,all_agg_fns,all_repr_types, repr_type, agg_fn):
    ntrees = len(all_trees_predict[0,:])
    intensity_map = np.zeros(X_test.shape)
    
    for i in range(len(y_test)):
        boolean_trees_agreement = all_trees_predict[i,:] == y_test[i]
        agreements_tree_idx = np.where(boolean_trees_agreement == True)[0]

        for tree_idx in agreements_tree_idx:
            all_start_idx_to_use,all_end_idx_to_use = get_all_start_end_idx_per_tree(tree_idx, agg_fn, repr_type, all_start_idx, all_end_idx, all_agg_fns, all_repr_types)

            for j in range(len(all_start_idx_to_use)):
                intensity_map[i,all_start_idx_to_use[j]:all_end_idx_to_use[j]] += 1
    return intensity_map


