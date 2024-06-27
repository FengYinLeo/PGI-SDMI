import numpy as np
import pandas as pd
import mlxtend
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
import networkx as nx


def ARM_FP_Growth(database,min_support,min_con): 
    """
    Function: Mining association rules
    Input: 
    database: Standard data form for association rule mining,
    min_support: Minimum support threshold, 
    min_con: Minimum confidence threshold
    Output: Association rules
    """
    # Transform data structure
    tran = []
    for i in range(database.shape[0]):
        tran.append([str(database.values[i,j]) for j in range(database.shape[1])])
    tran = np.array(tran)
    te = TransactionEncoder()
    tran_te = te.fit(tran).transform(tran)
    tran_df = pd.DataFrame(tran_te, columns=te.columns_)
    tran_df = tran_df.drop(columns=['nan'])
    
    # Initiating association analysis
    frequent_items = fpgrowth(tran_df, min_support=min_support, use_colnames=True, max_len=2) # Mining frequent item sets
    rules = association_rules(frequent_items, metric='lift', min_threshold=1) # generative rules
    result = rules.sort_values("confidence", ascending=False) # Sort by confidence
    result = result[result["confidence"]>=min_con]  # Find those with a confidence level greater than 0.3
    return result


def tran_ARMtoCN(data):
    """
    Function: Map an association rule table to an adjacency table
    Input: data : association rules
    Output: an adjacency table
    """
    data = data[["antecedents","consequents","confidence"]]
    data = data.rename(columns={"antecedents":"source","consequents":"target","confidence":"weight"})
    return data



def build_cn(data):
    """
    Function: Construct complex network using adjacency table
    Input: data : an adjacency table
    Output: complex network and adjacent matrix
    """
    G = nx.DiGraph(data)
    adjacent_matrix = nx.to_numpy_array(G)
    return G, adjacent_matrix



def betweenness_centrality(G):
    """
    Function: Calculate the betweenness coefficient of nodes
    Input: G : complex network
    Output: betweenness coefficient
    """
    bc=[value for value in nx.betweenness_centrality(G).values()]
    return np.array(bc)



def out_strong(adjacent_matrix):
    """
    Function: Compute nodes out-strength
    Input: adjacent_matrix : adjacent matrix
    Output: out-strength
    """
    os = np.sum(adjacent_matrix,axis=1) # Summation of rows of the adjacency matrix
    return os


def in_degree(adjacent_matrix):
    """
    Function: Compute nodes in-degree
    Input: adjacent_matrix : adjacent matrix
    Output: in-degree
    """
    # Find the network connection matrix. The value is 1 for direct connections and 0 for non-connections.
    e_matrix = np.where(adjacent_matrix > 0, 1, adjacent_matrix) # Elements greater than 0 in the adjacency matrix are set to 1
    inde = np.sum(e_matrix,axis=0)
    return inde


def transition_probability_matrix(adjacent_matrix):
    """
    Function: Generate transition probability matrix
    Input: adjacent_matrix : adjacent matrix
    Output: transition probability matrix
    """
    N = len(adjacent_matrix)  # Number of network nodes
    # Since the Ground node is added, the transition probability matrix is a square matrix of shape (N+1)
    tpm = np.zeros((N+1,N+1)) # Initialization
    
    tpm[:N,:N] = adjacent_matrix
    
    tpm[:N,N] = 1
    
    inde = in_degree(adjacent_matrix)
    tpm[N,:N] = inde
    
    return tpm


def three_WLR(G,adjacent_matrix,T):
    """
    Function: The 3-WLR algorithm identifies key nodes in the network
    Input: 
    G : network 
    adjacent_matrix : adjacent matrix
    T : Maximum number of iterations
    Output: Node ranking
    """
    tpm = transition_probability_matrix(adjacent_matrix)
    N = len(tpm)
    
    os = out_strong(adjacent_matrix)
    bc = betweenness_centrality(G)
    
    nodes = list(G.nodes())
    
    # Initialize
    LR = np.ones(N)
    LR_save = [LR]  # Save each iteration score
    
    # denominator
    denominator = np.sum(tpm,axis=1)
     
    conver = []  # Save the convergence of each iteration
    
    # Starting iteration
    t = 0
    CI = 1 # Convergence threshold, equal to 1 is not convergent, equal to 0 convergence
    while CI != 0 and t < T:
        LR_last = LR.copy() # Copy the score from the last iteration
        LR = np.zeros(N) # Initialize
        for i in range(N):
            for j in range(N):
                LR[i] += tpm[j,i] * LR_last[j] / denominator[j]
        
        LR_save.append(LR)
        
        if len(LR_save) >= 3: # Start to determine whether convergence
            cc1 = np.linalg.norm(LR_save[-1]-LR_save[-2],ord=2)
            cc2 = np.linalg.norm(LR_save[-2]-LR_save[-3],ord=2)
            if cc1 == 0 and cc2 == 0:
                CI = 0  #收敛
        t+=1
    
    print('Number of iterations of the 3-WLR algorithm:',t)
    
    # Get extra scores from the Ground node
    u = (os/sum(os) +bc/sum(bc))/2
    node_LR = LR[:-1]
    g_LR = LR[-1]
    node_LR = node_LR + u * g_LR
    
    dd = zip(nodes,node_LR)
    ddd = dict(dd)
    Score=sorted(ddd.items(),key=lambda s:s[1],reverse=True)  # ranking
    return Score


def PGI(database,min_support,min_con,T):
    """
    Function: Algorithm PGI body code
    Input: 
    database: Standard data form for association rule mining,
    min_support: Minimum support threshold, 
    min_con: Minimum confidence threshold
    T : Maximum number of iterations
    Output: Feature ranking
    """
    arm=ARM_FP_Growth(database,min_support,min_con)
    adjacent_table = tran_ARMtoCN(arm)
    G, adjacent_matrix = build_cn(adjacent_table)
    Score = three_WLR(G,adjacent_matrix,T)
    columns = list(database.columns)
    nodes = []
    for i in columns:
        nodes += list(database[i].dropna().unique())
    feature_ranking = []
    for i in range(len(Score)):
        for j in nodes:
            if frozenset({j}) == Score[i][0]:
                feature_ranking.append(j)
    return feature_ranking            

