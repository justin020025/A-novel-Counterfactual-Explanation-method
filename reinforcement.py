import numpy as np
import math
import networkx as nx
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
import copy
torch.manual_seed(0)

class environment:
    def __init__(self, classifier, avgs, stds, lower_bound, upper_bound, cat_feats, feat_names, scm: dict, done_threshlod=0.5, reg_beta=0.5, cat_change_penalty=0.5, max_step=None, disallowd_feats=None, directions=None, order_matter=False, add_bias_to_endo=True):
        self.action_record = [len(scm)]
        self.classifier = classifier
        self.cat_feats = cat_feats
        self.num_feats = list(set(range(len(avgs)))-set(cat_feats))
        self.feat_names = np.array(feat_names)
        self.avgs = avgs
        self.stds = stds
        self.lower = lower_bound
        self.upper = upper_bound
        self.scm = scm
        self.done_threshold = done_threshlod
        self.reg_beta = reg_beta
        self.cat_change_penalty = cat_change_penalty
        self.disallowd_feats = disallowd_feats
        self.directions = directions
        if max_step:
            self.max_step = min(len(self.avgs)-len(self.disallowd_feats), max_step)
        else:
            self.max_step = len(self.avgs)-len(self.disallowd_feats)
        self.endo_node = [feat for feat in scm.keys() if scm[feat]['input']]
        self.order_matter = order_matter
        self.add_bias_to_endo = add_bias_to_endo
        self.done = False
        self.graph = nx.DiGraph()
        self.nodes = list(self.scm.keys())
        # create an nx instance from scm
        self.graph.add_nodes_from(self.nodes)
        for child in self.nodes:
            parents = self.scm[child]['input']
            for parent in parents:
                self.graph.add_weighted_edges_from([(parent, child, -1)])
        # node is a feature id 
        # since not all features corresponds to a node in graph, we should map each node to an index for FWmatrix
        self.node2id = {self.nodes[i]: i for i in range(len(self.nodes))}
        self.id2node = {i: self.nodes[i] for i in range(len(self.nodes))}
        # all pairwise longest path lenghth
        FWmatix = np.abs(nx.floyd_warshall_numpy(self.graph))
        # for each node, sort other reachable nodes by longest path length in ascending order
        longest_path_order = np.argsort(FWmatix)
        self.order_for_change = dict()
        for i in range(self.graph.number_of_nodes()):
            self.order_for_change[self.id2node[i]] = []
            for j in range(1, self.graph.number_of_nodes()):
                target_node_id = longest_path_order[i][j]
                if FWmatix[i][target_node_id] < np.inf:
                    self.order_for_change[self.id2node[i]].append(self.id2node[target_node_id])
                else:
                    break
        self.ancestors_map = copy.copy(scm)
        for k in self.ancestors_map.keys():
            self.ancestors_map[k] = nx.ancestors(self.graph, k)
        self.normalize_scalar = lambda x, which: (x-avgs[which])/stds[which]

    def normalize(self, x):
        x_ = copy.copy(x).astype('float')
        x_[self.num_feats] = (x[self.num_feats] - self.avgs[self.num_feats]) / self.stds[self.num_feats]
        return x_

    def denormalize(self, x):
        x_ = copy.copy(x)
        x_[self.num_feats] = x[self.num_feats] * self.stds[self.num_feats] + self.avgs[self.num_feats]
        return x_

    def record2ids(self, arr):
        return np.array([self.graph.number_of_nodes()]+[self.node2id[r] for r in arr[1:]])

    def calc_target_prob(self):
        prediction = self.classifier.predict_proba(self.np2df(self.denormalize(self.Xp)))[0][self.target_class]
        return prediction

    def reshape_prob(self, p):
        if p <= 0.5:
            return p
        else:
            return (1/(1+np.exp(-(p-0.5)*10))-0.5)/5 + 0.5
    
    def np2df(self, x):
        d = pd.DataFrame([x], columns=self.feat_names)
        cat_names = self.feat_names[self.cat_feats].tolist()
        d[cat_names] = d[cat_names].astype(int)
        return d

    def step(self,action,value):
        # an action corresponds to a feature to be changed
        # if the action (chosen feature) does not correspond to a node in DAG, go to else
        if action in self.graph.nodes():
            # print(action)
            # if violate rules, do not change any feature, set done and return    
            if len(self.action_record) > self.max_step or len(self.masked_actions) >= len(self.X):
                # too many steps or no remaning action are valid
                self.done = 'invalid'
                reward = -5
                return self.Xp, self.action_record, reward, self.done
            if action in self.directions.keys():
                invalid_case1 = value > self.X[action] and self.directions[action] == -1
                invalid_case2 = value < self.X[action] and self.directions[action] == 1
                if (invalid_case1 or invalid_case2):
                    self.done = 'invalid'
                    reward = -5
                    return self.Xp, self.action_record, reward, self.done    
            self.action_record.append(action)
            if self.order_matter:
                self.masked_actions = self.masked_actions.union(nx.ancestors(self.graph, action)).union({action})
            else:
                self.masked_actions = self.masked_actions.union({action})
            # print(self.masked_actions)

            
            X_last = self.Xp.copy()
            if action in self.num_feats:
                self.regularization[action] = value - self.Xp[action]
            else:
                if value != self.X[action]:
                    self.regularization[action] = self.cat_change_penalty
            self.Xp[action] = value
            
            for change_node in self.order_for_change[action]:
                input_nodes = self.scm[change_node]['input']
                func = self.scm[change_node]['func']
                if change_node in self.num_feats: # change the descendants of the chosen feature following Pearl's ablation abduction 
                    # e.g. x1, x2 to predict x3, denormalize x1, x2, preict x3, then normalize x3
                    denorm_Xp = self.denormalize(self.Xp)
                    denorm_X_last = self.denormalize(X_last) 
                    bias = denorm_X_last[change_node] - func(*denorm_X_last[input_nodes])
                    X_last = self.Xp.copy()
                    if self.add_bias_to_endo:
                        self.Xp[change_node] = self.normalize_scalar(func(*denorm_Xp[input_nodes]) + bias, change_node)
                    else:
                        self.Xp[change_node] = self.normalize_scalar(func(*denorm_Xp[input_nodes]), change_node)
                    self.Xp[change_node] = np.clip(self.Xp[change_node], self.lower[change_node], self.upper[change_node])
                else:
                    X_last = self.Xp.copy()
                    # argmax can be replaced by sampling
                    self.Xp[change_node] = np.argmax(func(*self.Xp[input_nodes]))
                    
                    
        else:
            if action in self.num_feats:
                self.regularization[action] = value - self.Xp[action]
            else:
                if value != self.X[action]:
                    self.regularization[action] = self.cat_change_penalty
            self.Xp[action] = value
            self.action_record.append(action)
            if self.order_matter:
                self.masked_actions = self.masked_actions.union(nx.ancestors(self.graph, action)).union({action})
            else:
                self.masked_actions = self.masked_actions.union({action})

        self.cur_prob = self.calc_target_prob()
        if self.cur_prob > self.best_prob:
            self.Xbest = copy.copy(self.Xp)
            self.best_prob = copy.copy(self.cur_prob)
            self.best_reg = copy.copy(self.regularization)
            self.best_record = copy.copy(self.action_record)
        # Reshape_prob  only allow prob to reward 0.5, and then increase little, so the model wouldn't change X too much
        reward = self.reshape_prob(self.cur_prob) - self.init_prob - self.reg_beta * ( np.mean(np.abs(self.regularization)[self.num_feats].tolist() + np.array(self.regularization)[self.cat_feats].tolist()).item())
        # reward = self.reshape_prob(self.cur_prob) - self.init_prob - self.reg_beta * ( 0.5*np.max(np.abs(self.regularization)[self.num_feats]).item() + np.mean(np.square(self.regularization)[self.num_feats].tolist() + np.array(self.regularization)[self.cat_feats].tolist()).item())

        if self.cur_prob >= self.done_threshold:
            self.done = 'success'
            reward += 5

        # action record for printing, a record2ids(self.action_record) for model input
        return self.Xp, self.action_record, reward, self.done

    def reset(self, start_point):
        assert isinstance(start_point, np.ndarray)
        # The first element is just for necessary input of model 
        self.action_record = [len(self.avgs)]
        self.best_record = copy.copy(self.action_record)
        self.X = self.normalize(start_point).astype('float')
        self.Xp = copy.copy(self.X)
        self.Xbest = copy.copy(self.Xp)
        original_class = int(self.classifier.predict(self.np2df(start_point)))
        self.target_class = 1 if original_class==0 else 0
        self.init_prob = self.calc_target_prob()
        self.cur_prob = copy.copy(self.init_prob)
        self.best_prob = copy.copy(self.cur_prob)
        self.regularization = [0]*len(self.X)
        self.best_reg = copy.copy(self.regularization)
        self.masked_actions = set(self.disallowd_feats) if self.disallowd_feats else set()
        self.done = False
        return self.Xp,  np.array(self.action_record)
    
class Policy(nn.Module):
    def __init__(self, n_feats, cat_feats, n_classes, disallowed_feats=[], hidden_size=32):
        super(Policy, self).__init__()
        emb_size = n_feats# - len(disallowed_feats)
        # the last embedding is the embedding of empty action record
        self.n_feats = n_feats
        self.cat_feats = cat_feats
        self.num_feats = list(set(range(n_feats))-set(self.cat_feats))
        if self.cat_feats:
            self.n_classes = n_classes
            self.cat_feat2n_classes = {cat_feat: n_class for cat_feat, n_class in zip(cat_feats, n_classes)}
            self.max_n_classes = 1
            for feat in cat_feats:
                if not(feat in disallowed_feats):
                    self.max_n_classes = max(self.max_n_classes, self.cat_feat2n_classes[feat])
            emb_sizes = [int(np.sqrt(n_class))+1 for n_class in n_classes]
            self.cat_embeddings = [nn.Embedding(n_class, emb_size) for n_class, emb_size in zip(n_classes, emb_sizes)]
            self.fc = nn.Sequential(
                nn.Linear(emb_size+len(self.num_feats)+sum(emb_sizes), hidden_size),
                nn.ReLU()
                )
        else:
            self.fc = nn.Sequential(
                nn.Linear(emb_size+len(self.num_feats), hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
                )
        self.which = nn.Sequential(
            nn.Linear(hidden_size, n_feats),
            nn.Softmax()
            )
        self.mu = nn.Sequential(
            nn.Linear(hidden_size+emb_size+n_feats, 2)
            )
        self.var = nn.Sequential(
            nn.Linear(hidden_size+emb_size+n_feats, 2),
            nn.Softplus()
            )
        self.gmm_coef = nn.Sequential(
            nn.Linear(hidden_size+emb_size+n_feats, 2),
            nn.Softmax(dim=1)
            )
        if self.cat_feats:
            self.class_prob = nn.Sequential(
                nn.Linear(hidden_size+emb_size+n_feats, self.max_n_classes)
                )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 1)
            )
        
    def forward(self, actions, state):
        num_state = state[:, self.num_feats]
        if len(actions[0]) == 1:
            action_emb = torch.zeros(1, self.n_feats)
        else:
            action_emb = torch.tensor([[1 if k in actions[0] else 0 for k in range(self.n_feats)]]) / (len(actions[0])-1)
        # print(action_emb)

        x_num = num_state
        if self.cat_feats:
            cat_state = state[:, self.cat_feats].long()
            x_cat = torch.cat([self.cat_embeddings[i](cat_state[:,i]) for i in range(cat_state.shape[1])], dim=-1)
            x = torch.cat([action_emb, x_cat, x_num], dim=-1)
        else:
            x = torch.cat([action_emb, x_num], dim=-1)
        x = F.relu(self.fc(x))
        return x
    
    def calc_logprob(self, mu_v, var_v, actions_v):
        p1 = - ((mu_v - actions_v) ** 2) / (2*var_v.clamp(min=1e-10))
        # var clamp to prevent numerical problem
        p2 = - torch.log(torch.sqrt(2 * math.pi * var_v.clamp(min=1e-10)))
        return p1 + p2

    def normal_prob(self, mu, var, actions_v):
        p1 = 1/(torch.sqrt(2*math.pi*var))
        p2 = torch.exp(-(actions_v-mu)**2/(2*var))
        return p1*p2
    
    def calc_gmm_logprob(self, mus, vars, coefs, actions_v):
        # only in forward function, we can rely on batch operation
        p1 = self.normal_prob(mus[:,0], vars[:,0], actions_v)*coefs[:,0]
        p2 = self.normal_prob(mus[:,1], vars[:,1], actions_v)*coefs[:,1]
        return torch.log((p1+p2).clamp(min=1e-5))
    
    def sample_from_gmm(self, mus, vars, coefs):
        # Only work when forward batch_size=1
        mus_ = mus.data.cpu().numpy()[0]
        vars_ = vars.data.cpu().numpy()[0]
        coefs_ = coefs.data.cpu().numpy()[0]
        # sample which normal
        chosen = np.random.choice([0,1], 1, p=coefs_)
        #sample from chosen normal
        action_value = np.random.normal(mus_[chosen], vars_[chosen]**0.5)
        return action_value

    def act(self, actions, state, env, device):
        actions_ = torch.LongTensor(actions).to(device)
        state = torch.Tensor(state).to(device)

        # choose a feature
        x1 = self.forward(actions_, state)
        probs_which = self.which(x1).cpu()

        mask = torch.tensor([0 if k in env.masked_actions else 1 for k in range(self.n_feats)])
        probs_which = probs_which * mask
        probs_which = torch.nn.functional.normalize(probs_which, p=1)
        if torch.sum(probs_which) < 1e-4:
            return 'small_remaining_prob'
        
        # print(probs_which)
        model_which = Categorical(probs_which)
        action_which = model_which.sample()
        log_prob_which = model_which.log_prob(action_which)

        # critic
        critic_value = self.critic(x1)

        # the chosen feature becomes an input of network that determines the value
        # x2 = torch.cat([x1, self.action_emb(action_which), probs_which], dim=1)
        x2 = torch.cat([x1, nn.functional.one_hot(action_which, self.n_feats), probs_which], dim=1)
        chosen_action = action_which.cpu().numpy().item()
        if chosen_action in self.cat_feats:
            class_prob = self.class_prob(x2)[:, :self.cat_feat2n_classes[chosen_action]]
            class_prob = F.softmax(class_prob, dim=1)
            # print(class_prob)
            model_class = Categorical(class_prob)
            action_class = model_class.sample()
            log_prob_class = model_class.log_prob(action_class)
            return action_which.data.cpu().numpy(), log_prob_which, action_class.cpu().numpy(), log_prob_class, critic_value, class_prob, probs_which
        else:
            # *3 becuase it's not easy for the model to output large value
            mus = self.mu(x2)*3
            vars = self.var(x2)*3
            coefs = self.gmm_coef(x2)
            # sampling from GMM
            action_value = self.sample_from_gmm(mus, vars, coefs)
            # confine the value if the featured is increase-only or decrease-only
            action_which_ = action_which.data.cpu().numpy()[0]
            # if action_which_ in env.directions.keys():
            #     if env.directions[action_which_] == 1: # increase-only
            #         action_value = max(env.X[action_which_], action_value[0])
            #     else:
            #         action_value = min(env.X[action_which_], action_value[0])
            action_value = np.clip(action_value, env.lower[action_which_], env.upper[action_which])
            action_value = np.array([action_value])
            log_prob_value = self.calc_gmm_logprob( mus, vars, coefs, torch.tensor(action_value))
            # numpy, tensor, numpy, tensor, tensor
            return action_which.data.cpu().numpy(), log_prob_which, action_value, log_prob_value, critic_value, mus, vars, coefs, probs_which

def out_of_range_loss(mus, vars, coefs, mu_d, var_d, n_allowed_std=2):
    # encourage that (model_mean +- 3 model_std) lies in (data_mean +- n_allowed_std * data_std)
    upper = torch.clip(mus[:,0] + 3*torch.sqrt(vars[:,0]) - (mu_d + n_allowed_std*torch.sqrt(var_d)), min=0) * coefs[:,0] \
        + torch.clip(mus[:,1] + 3*torch.sqrt(vars[:,1]) - (mu_d + n_allowed_std*torch.sqrt(var_d)), min=0) * coefs[:,1]
    lower = torch.clip(mu_d - n_allowed_std*torch.sqrt(var_d) - (mus[:,0] - 3*torch.sqrt(vars[:,0])), min=0) * coefs[:,0] \
        + torch.clip(mu_d - n_allowed_std*torch.sqrt(var_d) - (mus[:,1] - 3*torch.sqrt(vars[:,1])), min=0) * coefs[:,1]
    return (upper + lower) / torch.sqrt(var_d)

def normal_prob(mu, var, val):
    return 1/(torch.sqrt(2*math.pi*var))*torch.exp(-0.5*(val-mu)**2/var)

def gmm_entropy_lb(mus, vars, coefs):
    var0n1 = vars[:,0] + vars[:,1]
    return -coefs[:,0]*torch.log(coefs[:,0]*normal_prob(mus[:,0], var0n1, mus[:,0]) + coefs[:,1]*normal_prob(mus[:,1], var0n1, mus[:,0])) \
        - coefs[:,1]*torch.log(coefs[:,0]*normal_prob(mus[:,0], var0n1, mus[:,1]) + coefs[:,1]*normal_prob(mus[:,1], var0n1, mus[:,1]))

def negative_logprob_cat(logprob, one_hot_label):
    return -torch.sum(logprob*one_hot_label, axis=1)

def cross_entropy(prob, label):
    return -torch.sum(torch.log(prob)*label, axis=1)

def KL(prob, label):
     return cross_entropy(prob, label) - cross_entropy(label, label)

def causal_loss_cat(prob, label):
    return 1 - torch.exp(-KL(prob, label))

def process_input_for_condist(X, in_feats, cat_feats): # refer to ConDist.py
    in_num_feats = list(set(in_feats)-set(cat_feats))
    in_cat_feats = list(set(in_feats)-set(in_num_feats))
    if len(in_num_feats) == 0: # all cat feat
        return torch.LongTensor(X[in_feats].astype('int')).unsqueeze(0)
    elif len(in_num_feats) == len(in_feats): # all num_feats
        return torch.tensor(X[in_feats].astype('float32')).unsqueeze(0)
    else: # both cat and num
        return torch.tensor(X[in_num_feats].astype('float32')).unsqueeze(0), torch.LongTensor(X[in_cat_feats].astype('int')).unsqueeze(0)
def reinforce(data, env, policy, optimizer, checkpoint_path, n_episodes=1000, max_t=1000, gamma=0.95, print_every=100, con_dist=None, n_allowed_std=2, CON_DIST_BETA=0.5, ACTION_ENTROPY_BETA=0.0001, device='cpu'):
    scores = []
    best_score = -10000
    policy_losses = []
    advses = []
    if con_dist:
        endogenous_feats = list(con_dist.keys())
    for e in range(1, n_episodes+1):
        # shared output of numerical, categorical
        saved_action_which, saved_log_probs_which, saved_critic_value, saved_probs_which = [], [], [], []
        # output of numerical
        saved_log_probs_value, saved_mus, saved_vars, saved_coefs  = [], [], [], []
        # output of categorical
        saved_log_probs_class, saved_class_prob = [], []
        # record of cat/num feature chosen
        saved_cat_num_record = []
        # masked actions
        saved_masked_actions = []

        rewards = []
        sample_id = np.random.randint(len(data))
        if isinstance(data, pd.DataFrame):
            start = data.iloc[sample_id].to_numpy()
        elif isinstance(data, np.ndarray):
            start = data[sample_id]
        else:
            raise('data should be pandas.DataFrame or numpy.ndarray')
        X, record = env.reset(start)
        # print("***************************************************************")
        # print('start from {s}'.format(s=np.round(start,2)))
        for t in range(max_t):
            # Sample the action from current policy
            returns = policy.act(np.array([record]), np.array([X]), env, device)
            if returns == 'small_remaining_prob':
                # The remaining prob is too small, which means the previous actions are bad, so reset the last reward to negative
                try:
                    rewards[-1] = -5
                except:
                    pass
                break
            
            elif len(returns) == 9: # the chosen feature is numerical
                action_which, log_prob_which, action_value, log_prob_value, critic_value, mus, vars, coefs, probs_which = returns
                saved_log_probs_value.append(log_prob_value)
                saved_log_probs_class.append(None)
                saved_mus.append(mus)
                saved_vars.append(vars)
                saved_coefs.append(coefs)
                saved_class_prob.append(None)
                saved_cat_num_record.append('num')
                X, record, reward, done = env.step(action_which[0], action_value[0])

            else: # the chosen feature is categorical
                action_which, log_prob_which, action_class, log_prob_class, critic_value, class_prob, probs_which = returns
                saved_log_probs_value.append(None)
                saved_log_probs_class.append(log_prob_class)
                saved_mus.append(None)
                saved_vars.append(None)
                saved_coefs.append(None)
                saved_class_prob.append(class_prob)
                saved_cat_num_record.append('cat')
                X, record, reward, done = env.step(action_which[0], action_class[0])

            saved_action_which.append(action_which[0])
            saved_log_probs_which.append(log_prob_which)
            saved_critic_value.append(critic_value)
            saved_probs_which.append(probs_which)
            saved_masked_actions.append(env.masked_actions)
            rewards.append(reward)
            # if done and reward < done_threshold:
            #     print('Modify feature {feat}. Invalid action. Game over!\n'.format(feat=action_which[0]))
            # else: 
            #     print('Modify feature {feat}.'.format(feat=action_which[0]))
            #     # note that the first element of record is just for the initial record purpose
            #     print('Features becomes: {x}.\nAction record: {rec}.\nReward: {re}'.format(x=np.round(env.denormalize(X),2), rec=record[1:], re=round(reward, 2)))
            #     if reward >= done_threshold:
            #         print('sussessfully reach the goal')
            #     print('\n')
            if done:
                break
        # Calculate total expected reward
        scores.append(np.max(rewards))
        
        # Recalculate the cummulative reward applying discounted factor
        dis_cumm_rewards = [rewards[0]]*len(rewards)
        for i in reversed(range(len(rewards)-1)):
            dis_cumm_rewards[i] = dis_cumm_rewards[i+1]*gamma + rewards[i]

        # caculate advantages
        advs = [r - c for r,c in zip(dis_cumm_rewards, saved_critic_value)]
        advses.extend(advs)
        
        # Calculate the loss
        VALUE_ENTROPY_BETA = 1e-2

        for action_which, log_prob_which, log_prob_value, adv, mus, vars, coefs, log_prob_class, class_prob, cat_num, probs_which, masked_actions in zip(saved_action_which, saved_log_probs_which, saved_log_probs_value, advs, saved_mus, saved_vars, saved_coefs, saved_log_probs_class, saved_class_prob, saved_cat_num_record, saved_probs_which, saved_masked_actions) :
            # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
            if con_dist:
                if action_which in endogenous_feats:
                    in_feats = con_dist[action_which]['input']
                    X_for_con_dist = process_input_for_condist(env.Xp, in_feats, policy.cat_feats)
                    if cat_num == 'num':
                        mu_d, var_d = con_dist[action_which]['net'](X_for_con_dist)     
                        policy_losses.append(-(log_prob_value + log_prob_which) * adv + CON_DIST_BETA * out_of_range_loss(mus, vars, coefs, mu_d, var_d, n_allowed_std=n_allowed_std))
                        # print(-(log_prob_which + log_prob_value) * adv, CON_DIST_BETA * out_of_range_loss(mus, vars, coefs, mu_d, var_d))
                    else:
                        class_prob_d = con_dist[action_which]['net'](X_for_con_dist)
                        policy_losses.append(-(log_prob_class + log_prob_which) * adv + CON_DIST_BETA * causal_loss_cat(class_prob, class_prob_d))
                else:
                    if cat_num == 'num':
                        policy_losses.append(-(log_prob_value + log_prob_which) * adv - VALUE_ENTROPY_BETA * gmm_entropy_lb(mus, vars, coefs))
                        # print(-(log_prob_which + log_prob_value) * adv, VALUE_ENTROPY_BETA * gmm_entropy_lb(mus, vars, coefs))
                    else:
                        policy_losses.append(-(log_prob_class + log_prob_which) * adv - VALUE_ENTROPY_BETA * negative_logprob_cat(class_prob, class_prob))
            else:
                if cat_num == 'num': 
                    policy_losses.append(-(log_prob_which + log_prob_value) * adv - VALUE_ENTROPY_BETA * gmm_entropy_lb(mus, vars, coefs))
                else:
                    policy_losses.append(-(log_prob_which + log_prob_class) * adv - VALUE_ENTROPY_BETA * negative_logprob_cat(class_prob, class_prob))
            allowed_actions = list(set(range(len(env.X))) - masked_actions)
            if ACTION_ENTROPY_BETA:
                policy_losses[-1] += - ACTION_ENTROPY_BETA * negative_logprob_cat(probs_which[:, allowed_actions], probs_which[:, allowed_actions])
            # penalize the probability on masked actions
            policy_losses[-1] += 1e-2 * torch.sum(probs_which[:, list(masked_actions)], axis=-1)
        # sampling in experience replay, but it's not working well here, so set proportion to 1.
        sample_ids = np.random.randint(len(advses), size=int(1*len(advses)))    

    
        # After that, we concatenate whole policy loss in 0th dimension
        if e % 64 == 0:
            policy_loss = torch.cat(policy_losses)[sample_ids].mean()
            critic_loss = torch.norm(torch.cat(advses)[sample_ids], dim=1).mean()
            
            loss = policy_loss + critic_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            policy_losses = []
            advses = []          
        
        if e % print_every == 0:
            torch.save(policy, checkpoint_path + '/model_{epo}.pt'.format(epo=e))
            avg_score = np.mean(scores)
            print('Episode {}\tAverage Score: {:.2f}'.format(e, avg_score))
            if avg_score > best_score:
                torch.save(policy, checkpoint_path + '/best_model.pt')
                best_score = avg_score
            scores = []

    policy = torch.load(checkpoint_path + '/best_model.pt')
    return scores

def evaluate(test_data, env, policy, device, print_file=None):
    if print_file:
        f = open(print_file, 'w')
    CFs = []
    action_records = []
    regs = []
    L1_dists = []
    num_invalid = 0
    for i in range(len(test_data)):
        done = False
        start = test_data.iloc[i].to_numpy()
        X, record = env.reset(start)
        if print_file:   
            f.writelines("***************************************************************\n")
            f.writelines(np.array2string(start, formatter={'float_kind':lambda x: "%.4f" % x}))
            f.writelines('\nTarget class prob: {s:.{digits}f}\n'.format(s=env.init_prob, digits=4))
        with torch.no_grad():
            while not done:
                returns = policy.act(np.array([record]), np.array([X]), env, device)
                if returns == 'small_remaining_prob':
                    # The remaining prob is too small, which means the previous actions are bad
                    done = 'small_remaining_prob'

                elif len(returns) == 9: # the chosen feature is numerical
                    action_which, log_prob_which, action_value, log_prob_value, critic_value, mus, vars, coefs, probs_which = returns
                    X, record, reward, done = env.step(action_which[0], action_value[0])

                else:
                    action_which, log_prob_which, action_class, log_prob_class, critic_value, class_prob, probs_which = returns
                    X, record, reward, done = env.step(action_which[0], action_class[0])

                if print_file:
                    if done == 'small_remaining_prob':
                        f.writelines('\nNo remaining actions are qualified. Game over!\n')
                    elif done == 'invalid':
                        f.writelines('\nModify feature {feat}. Invalid action. Game over!\n'.format(feat=action_which[0]))
                    # success or not done
                    else: 
                        f.writelines('\nModify feature {feat}.\n'.format(feat=action_which[0]))
                        # note that the first element of record is just for the initial record purpose
                        f.writelines('Features becomes: {x}.\nAction record: {rec}.\nReward: {re:.{digits}f}\n'.format(x=np.array2string(env.denormalize(X), formatter={'float_kind':lambda x: "%.4f" % x}), rec=record[1:], re=reward, digits=4))
                        if done == 'success':
                            f.writelines('sussessfully reach the goal')
                if done:
                    if done == 'success':
                        CFs.append(env.denormalize(env.Xbest))                   
                        regs.append(env.best_reg)
                        action_records.append(env.best_record[1:])
                        l1_dist = np.round(np.mean(np.abs(env.best_reg)), 4).item()
                        L1_dists.append(l1_dist)
                    else:
                        # if not success, take the origin as the counterfactual
                        num_invalid += 1
                        CFs.append(test_data.iloc[i].to_numpy())                   
                        regs.append([0]*len(test_data.columns))
                        action_records.append([])
                        l1_dist = 0
                        L1_dists.append(l1_dist)                        
                    if print_file:
                        f.writelines('\nTarget class prob: {s:.{digits}f}\n'.format(s=env.cur_prob, digits=4))
                        f.writelines('L1 distance: {s:.{digits}f}\n'.format(s=l1_dist, digits=4))
                    
                    break
    if print_file:
        f.write('average path length: {:.3f}\n'.format(np.mean([len(e) for e in action_records])))
        f.write('invalid_rate: {:.3f}\n'.format(num_invalid/len(test_data)))
        f.write('average L1 distance: {:.3f}\n'.format(np.mean(L1_dists)))
        f.close()

    return {'CF': CFs, 'action_record': action_records, 'success_rate': 1-num_invalid/len(test_data), 'l1': np.mean(L1_dists), 'reg':regs}