import pandas as pd
import numpy as np
import PGI
import SDMI


def no_arm(data,ranking):
    """
    Function: Mining association rules
    Input: 
    data: Consistent with data fed into machine learning training and predictions,
    ranking: feature ranking based on PGI
    Output: Features not mined by PGI
    """
    no_arm=[]
    features=data.columns
    
    for f in features:
        if f not in ranking:
            no_arm.append(f)
    return no_arm


def PGI_SDMI(database,data,min_support,min_con,T,target):
    """
    Function: Features ranking based on PGI-SDMI
    Input: 
    database: Standard data form for association rule mining,
    data: Consistent with data fed into machine learning training and predictions,
    min_support: Minimum support threshold, 
    min_con: Minimum confidence threshold
    T : Maximum number of iterations,
    traget: List of predicted targets (e.g., accident_severity)
    Output: Features ranking
    """
    pgi_ranking = PGI.PGI(database,min_support,min_con,T)
    no_ranking = no_arm(data,pgi_ranking)
    data_new = data[no_ranking]
    sdmi_ranking = SDMI.SDMI(data_new,target)
    ranking = pgi_ranking+sdmi_ranking
    return ranking




