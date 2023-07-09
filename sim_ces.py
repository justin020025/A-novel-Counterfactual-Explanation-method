import pickle
import pandas as pd
from metrics import causal_edge_score
import numpy as np

import statsmodels.api as sm

clf_data_train = pd.read_csv('./sim_data/sim_cat_data_train.csv')

def prob_func_x4(a,b,c):
    prob1 = abs(a/100) + 0.1*(b==0) + 0.1*(b==2) * np.sqrt(abs(c)) 
    prob2 = 0.3*(b==1) + 0.4*(b==3) + c/10
    prob3 = a/100*c/10 + 0.1
    return np.array([prob1, prob2, prob3])/ sum([prob1, prob2, prob3])
SCM = {0:{'input':[], 'func':None}, 1:{'input':[], 'func':None}, 2:{'input':[], 'func':None},\
     3:{'input':[2], 'func':lambda a: (a-3)**2}, 4:{'input':[0,1,3], 'func':prob_func_x4}}

cat_feats = ['x1', 'x4']
cess = []
cols = clf_data_train.drop('y', axis=1).columns
for k in SCM.keys():
    # in feats
    in_feats = SCM[k]['input']
    if in_feats:
        cess.append(causal_edge_score(cols[in_feats].to_list(), cols[[k]].to_list(), cat_feats).fit(clf_data_train.sample(frac=1).drop('y', axis=1)))

with open('./sim_ces.pickle', 'wb') as f:
    pickle.dump(cess, f)