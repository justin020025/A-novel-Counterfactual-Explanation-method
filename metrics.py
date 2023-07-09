import numpy as np
import pandas as pd
import statsmodels.api as sm
import copy
import scipy

class causal_edge_score:
    def __init__(self, parents: list, child: list, cat_feats:list):
        self.parents = parents
        self.child = child
        self.joint = parents + child
        self.cat_feats = cat_feats

    def fit(self, train_data: pd.DataFrame):
        self.mean = train_data.mean()
        self.std = train_data.std()
        self.parent_idx = [train_data.columns.get_loc(c) for c in self.parents if c in train_data]
        self.joint_idx = [train_data.columns.get_loc(c) for c in self.joint if c in train_data]
        self.cat_idx = [train_data.columns.get_loc(c) for c in self.cat_feats if c in train_data]
        self.cat_feat_mask = np.array(['u' if i in self.cat_idx else 'c' for i in range(len(train_data.columns))])

        pa_data = (train_data[self.parents]-self.mean[self.parents])/ self.std[self.parents]
        pa_var_type = ''.join(self.cat_feat_mask[self.parent_idx])
        self.pa_model = sm.nonparametric.KDEMultivariate(data=pa_data, var_type=pa_var_type, bw='cv_ml')

        joint_data = (train_data[self.joint]-self.mean[self.joint])/ self.std[self.joint]
        joint_var_type = ''.join(self.cat_feat_mask[self.joint_idx])
        self.joint_model = sm.nonparametric.KDEMultivariate(data=joint_data, var_type=joint_var_type, bw='cv_ml')

        return self

    def evaluate(self, X, Xp):
        lower_limit = np.log(1e-8)
        norm_X = ((X-np.array(self.mean))/ np.array(self.std))
        log_p_before = np.log(self.joint_model.pdf(norm_X[:, self.joint_idx])) - np.log(self.pa_model.pdf(norm_X[:, self.parent_idx]))
        norm_Xp = ((Xp-np.array(self.mean))/ np.array(self.std))
        p_joint_after = self.joint_model.pdf(norm_Xp[:, self.joint_idx])
        p_parent_after = self.pa_model.pdf(norm_Xp[:, self.parent_idx])
        ces = []
        for i in range(len(norm_X)):
            if p_joint_after[i] < 1e-12:
                ces.append(lower_limit)
            else:
                # mathematically, p_joint > 0 implies p_parent > 0, but when the prob is small, this case doesnt hold true for the learning, so we need to handle this.
                if p_parent_after[i] < 1e-12:
                    ces.append(lower_limit)
                else:
                    log_p_after_i = np.log(p_joint_after[i]) - np.log(p_parent_after[i])
                    ces.append(max(lower_limit, log_p_after_i - log_p_before[i]))
        return np.array(ces)


# Use the counterfactual value of parent and with scm to generate value of child that perfectly match the scm.
def causal_cfe_from_parent(cfs, original, scm, cat_feats, add_bias=True):
    cat_feat_ids = [original.columns.get_loc(c) for c in cat_feats]
    endo_node = [feat for feat in scm.keys() if scm[feat]['input']]
    causal_cfs = []
    for cf, origin in zip(cfs.to_numpy(), original.to_numpy()):
        cf_ = copy.copy(cf)
        for node in endo_node:
            func = scm[node]['func']
            in_feats = scm[node]['input']
            if node in cat_feat_ids:
                scm_pred = np.argmax(func(*cf[in_feats]))
            else:
                indep_noises = origin[node] - func(*origin[in_feats]) if add_bias else 0
                scm_pred = func(*cf[in_feats]) + indep_noises
            cf_[node] = scm_pred
        causal_cfs.append(cf_)
    causal_cfs = pd.DataFrame(causal_cfs, columns=original.columns)
    return causal_cfs

def sparsity_num(cfs, original, cat_feats):
    numerical = original.columns.difference(cat_feats)
    return np.mean(np.abs(cfs[numerical].to_numpy() - original[numerical].to_numpy()) < 1e-10)

def sparsity_cat(cfs, original, cat_feats):
    return np.mean(cfs[cat_feats].to_numpy() == original[cat_feats].to_numpy())

def L1(cfs, original, cat_feats, stds):
    numerical = original.columns.difference(cat_feats)
    stds_ = stds[numerical].to_numpy()
    return np.mean(np.abs((cfs[numerical].to_numpy() - original[numerical].to_numpy()) / stds_))

def compute_log_cond_prob(child, x, scm):
    mean = np.array(scm[child]['weight'][1:]).dot(x[scm[child]['input']]) + scm[child]['weight'][0]
    std = np.array(scm[child]['std'])
    return scipy.stats.norm.logpdf(x[child], mean, std)
def cess_for_bn(before, after, scm):
    endo_nodes = [k for k in scm.keys() if scm[k]['input']]
    cess = [[compute_log_cond_prob(ch, dat_a, scm) - compute_log_cond_prob(ch, dat_b, scm) for ch in endo_nodes] for dat_a, dat_b in zip(after, before)]
    return np.array(cess).clip(min=-5)


