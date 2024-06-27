import pandas as pd
import numpy as np


def SDMI(data,target):
    """
    Function: Calculate the State Differences Mutual Information and perform feature ranking
    Input: 
    data: Consistent with data fed into machine learning training and predictions,
    traget: List of predicted targets (e.g., accident_severity)
    Output: feature ranking based on SDMI
    """
    
    columns = list(data.columns) # Column name of the data
    features = columns.copy()
    features.remove(target)
    num_features = len(features) # Number of features
    num_data = len(data) # Number of data
    
    state_target = data[target].unique() # Target state
    num_state_target = len(state_target) # The number of states of the target
    
    sdmi = [] # Used to store calculation results
    
    for i in range(num_features):
        feature = features[i]
        state_feature = data[feature].unique()  # Feature state
        num_state_feature = len(state_feature)  # The number of states of the feature
        sum_infor = 0 # Initialize
        for j in range(num_state_target):
            information = np.zeros(num_state_feature)  # Store mutual information about each state
            for k in range(num_state_feature):
                num_AB = np.sum(data[data[target].isin([state_target[j]])][feature]==state_feature[k])
                num_A = np.sum(data[target].isin([state_target[j]]))
                num_B = np.sum(data[feature].isin([state_feature[k]]))
                P_AB = num_AB/num_data
                P_A = num_A/num_data
                P_B = num_B/num_data
                if P_AB == 0:
                    information[k] = 0
                else:
                    information[k] = P_AB*np.log(P_AB/(P_A*P_B))
            
            s_infor = 0 # The initial value of the sum of squares of the information difference
            for m in range(num_state_feature):
                for n in range(num_state_feature):
                    s_infor += (information[m]-information[n])**2
            
            s_infor_ = s_infor**(1/2)
            
            sum_infor += s_infor_/(num_state_feature*2)
        
        sdmi.append(sum_infor) # The feature state difference mutual information
    
    dd = zip(features,sdmi)
    ddd = dict(dd)
    Score=sorted(ddd.items(),key=lambda s:s[1],reverse=True)#排序
    feature_ranking = []
    for i in Score:
        feature_ranking.append(i[0])
    return feature_ranking


