import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import math
import matplotlib.pyplot as plt
from  torch.optim.lr_scheduler import ExponentialLR

class CsvDataset(Dataset):
    def __init__(self, data, in_feats:list, out_feat, cat_feats):
        self.data = data
        self.in_feats = in_feats
        self.out_feat = out_feat
        self.cat_feats = cat_feats
        self.in_cat_feats = list(set(in_feats).intersection(set(cat_feats)))
        self.in_num_feats = list(set(in_feats)-set(self.in_cat_feats))
        if self.in_cat_feats != [] and self.in_num_feats != []:
            self.case = 1
        elif self.in_num_feats != []:
            self.case = 2
        else:
            self.case = 3
        if isinstance(self.data, pd.DataFrame):
            if self.in_num_feats:
                self.X_num = self.data[self.data.columns[self.in_num_feats]].to_numpy()
            else:
                self.X_num = None
            if self.in_cat_feats:
                self.X_cat = self.data[self.data.columns[self.in_cat_feats]].to_numpy()
            else:
                self.X_cat = None
            self.y = self.data[self.data.columns[out_feat]].to_numpy()
        else:
            if self.in_num_feats:
                self.X_num = self.data[:, self.in_num_feats]
            else:
                self.X_num = None
            if self.in_cat_feats:
                self.X_cat = self.data[:, self.in_cat_feats]
            else:
                self.X_cat = None
            self.y = self.data[:, out_feat]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_num = torch.tensor(self.X_num[idx].astype('float32')) if self.in_num_feats else None
        x_cat = torch.LongTensor(self.X_cat[idx, :]) if self.in_cat_feats else None
        if self.out_feat in self.cat_feats:
            y = torch.LongTensor(self.y[idx].reshape([1]))
            y = torch.squeeze(F.one_hot(y, num_classes=len(np.unique(self.y))))
        else:
            y = torch.tensor(self.y[idx].astype('float32').reshape([1]))

        if self.case == 1:
            # print(x_num, x_cat, y)
            return x_num, x_cat, y
        elif self.case == 2:
            return x_num, y
        else:
            return x_cat, y

# model that predict the conditional mean and variance of a normal distribution
# class ConDist_MLE(nn.Module):
#     def __init__(self, dataset: CsvDataset, hidden_size=16, device='cpu'):
#         super(ConDist_MLE, self).__init__()
#         self.dataset = dataset
#         self.device = device
#         if dataset.case in [1,3]:
#             # has categorical input
#             x_n_classes = [len(np.unique(dataset.X_cat[:,i])) for i in range(dataset.X_cat.shape[1])]
#             emb_sizes = [int(np.sqrt(n_class))+1 for n_class in x_n_classes]
#             self.cat_embeddings = [nn.Embedding(n_class, emb_size) for n_class, emb_size in zip(x_n_classes, emb_sizes)]
#         if dataset.case == 1:
#             fc_input_size = len(dataset.in_num_feats) + sum(emb_sizes)
#         elif dataset.case == 2:
#             fc_input_size = len(dataset.in_num_feats)
#         else:
#             fc_input_size = sum(emb_sizes)

#         self.fc = nn.Sequential(
#             nn.Linear(fc_input_size, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU()
#             )
#         if dataset.out_feat in dataset.cat_feats:
#             # the endogenous feature is categorical
#             y_n_class = len(np.unique(dataset.y))
#             self.multinomial = nn.Sequential(
#                 nn.Linear(hidden_size, y_n_class),
#                 nn.Softmax(dim=1)
#             )
#         else:    
#             self.mu = nn.Sequential(
#                 nn.Linear(hidden_size, 1)
#                 )
#             self.var = nn.Sequential(
#                 nn.Linear(hidden_size, 1),
#                 nn.Softplus()
#                 )
#         self.to(device)

#     def to(self, device):
#         super().to(device)
#         if self.dataset.case in [1,3]:
#             for e in self.cat_embeddings:
#                 e.to(device)
#         self.device = device
        
#     def forward(self, cd_feats):
#         if self.dataset.case == 2:
#             x = cd_feats

#         elif self.dataset.case == 3:
#             cat_input = cd_feats
#             x = torch.cat([self.cat_embedding(cat_in) for cat_in in cat_input], dim=1)

#         else: 
#             num_input = cd_feats[0]
#             cat_input = cd_feats[1]
#             all_cat_embeddings = torch.cat([self.cat_embeddings[i](cat_input[:,i]) for i in range(cat_input.shape[-1])], dim=-1)
#             x = torch.cat([num_input, all_cat_embeddings], dim=1)

#         x = self.fc(x)        
        
#         if self.dataset.out_feat in self.dataset.cat_feats:
#             # the endogenous feature is categorical
#             return self.multinomial(x)
#         else:
#             return self.mu(x), self.var(x)

#     def negative_logprob(self, mu_v, var_v, value):
#         p1 = ((mu_v - value) ** 2) / (2*var_v.clamp(min=1e-3))
#         p2 = 0.5 * torch.log(2 * math.pi * var_v)
#         return p1 + p2
    
#     # equivalent to cross entropy
#     def negative_logprob_cat(self, prob, one_hot_label):
#         return -torch.sum(torch.log(prob)*one_hot_label, axis=1)
    
#     def fit(self, data_loader, n_epochs):
#         optimizer = optim.Adam(self.parameters(), lr=1e-5)
#         for i in range(n_epochs):
#             if self.dataset.case == 1:
#                 for x_num, x_cat, y in data_loader:
#                     x_num, x_cat, y = x_num.to(self.device), x_cat.to(self.device), y.to(self.device)
#                     if self.dataset.out_feat in self.dataset.cat_feats:
#                         probs = self.forward((x_num, x_cat))
#                         loss = torch.mean(self.negative_logprob_cat(probs, y))
#                     else:
#                         mu, var = self.forward((x_num, x_cat))
#                         # print(mu.shape, var.shape, y.shape)
#                         loss = torch.mean(self.negative_logprob(mu, var, y))
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#             elif self.dataset.case == 2:
#                 for x_num, y in data_loader:
#                     x_num, y = x_num.to(self.device), y.to(self.device)
#                     if self.dataset.out_feat in self.dataset.cat_feats:
#                         probs = self.forward(x_num)
#                         loss = torch.mean(self.negative_logprob_cat(probs, y))
#                     else:
#                         mu, var = self.forward(x_num)
#                         # print(mu.shape, var.shape, y.shape)
#                         loss = torch.mean(self.negative_logprob(mu, var, y))
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#             else:
#                 for x_cat, y in data_loader:
#                     x_cat, y = x_cat.to(self.device), y.to(self.device)
#                     if self.dataset.out_feat in self.dataset.cat_feats:
#                         probs = self.forward(x_cat)
#                         loss = torch.mean(self.negative_logprob_cat(probs, y))
#                     else:
#                         mu, var = self.forward(x_cat)
#                         # print(mu.shape, var.shape, y.shape)
#                         loss = torch.mean(self.negative_logprob(mu, var, y))
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
            
#         print('loss: ', loss.cpu().detach().numpy())
#         self.eval()

class ConDist_MO(nn.Module):
    def __init__(self, dataset: CsvDataset, hidden_size=16, device='cpu'):
        super(ConDist_MO, self).__init__()
        self.dataset = dataset
        self.device = device
        if dataset.case in [1,3]:
            # has categorical input
            x_n_classes = [len(np.unique(dataset.X_cat[:,i])) for i in range(dataset.X_cat.shape[1])]
            emb_sizes = [int(np.sqrt(n_class))+1 for n_class in x_n_classes]
            self.cat_embeddings = [nn.Embedding(n_class, emb_size) for n_class, emb_size in zip(x_n_classes, emb_sizes)]
        if dataset.case == 1:
            fc_input_size = len(dataset.in_num_feats) + sum(emb_sizes)
        elif dataset.case == 2:
            fc_input_size = len(dataset.in_num_feats)
        else:
            fc_input_size = sum(emb_sizes)

        if dataset.out_feat in dataset.cat_feats:
            # the endogenous feature is categorical
                y_n_class = len(np.unique(dataset.y))
                self.multinomial = nn.Sequential(
                nn.Linear(fc_input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, y_n_class),
                nn.Softmax(dim=1)
            )
                
        else:    
            self.mu = nn.Sequential(
                nn.Linear(fc_input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
                )

            self.var = nn.Sequential(
                nn.Linear(fc_input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Softplus()
                )
        self.to(device)

    def to(self, device):
        super().to(device)
        if self.dataset.case in [1,3]:
            for e in self.cat_embeddings:
                e.to(device)
        self.device = device
        
    def forward(self, cd_feats):
        if self.dataset.case == 2:
            x = cd_feats

        elif self.dataset.case == 3:
            cat_input = cd_feats
            x = torch.cat([self.cat_embedding(cat_in) for cat_in in cat_input], dim=1)

        else: 
            num_input = cd_feats[0]
            cat_input = cd_feats[1]
            all_cat_embeddings = torch.cat([self.cat_embeddings[i](cat_input[:,i]) for i in range(cat_input.shape[-1])], dim=-1)
            x = torch.cat([num_input, all_cat_embeddings], dim=1)
        
        if self.dataset.out_feat in self.dataset.cat_feats:
            # the endogenous feature is categorical
            return self.multinomial(x)
        else:
            return self.mu(x), self.var(x)

    def loss_func1(self, mu_v, var_v, value):
        first_moment_loss = (mu_v - value) ** 2
        return first_moment_loss

    def loss_func2(self, mu_v, var_v, value):
        second_moment_loss = (var_v + mu_v**2 - value**2)**2
        return second_moment_loss
    
    # equivalent to cross entropy
    def negative_logprob_cat(self, prob, one_hot_label):
        return -torch.sum(torch.log(prob)*one_hot_label, axis=1)
    
    def fit(self, data_loader, n_epochs):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        if self.dataset.case == 1:
            if self.dataset.out_feat in self.dataset.cat_feats:
                for i in range(n_epochs):
                    for x_num, x_cat, y in data_loader:
                        x_num, x_cat, y = x_num.to(self.device), x_cat.to(self.device), y.to(self.device)
                    
                        probs = self.forward((x_num, x_cat))
                        loss = torch.mean(self.negative_logprob_cat(probs, y))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            else:
                print('now train for mu')
                for i in range(n_epochs):    
                    for x_num, x_cat, y in data_loader:
                        x_num, x_cat, y = x_num.to(self.device), x_cat.to(self.device), y.to(self.device)       
                        mu, var = self.forward((x_num, x_cat))
                        loss = torch.mean(self.loss_func1(mu, var, y))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                print('now train for var')
                optimizer = optim.Adam(self.parameters(), lr=1e-5)
                for i in range(n_epochs*2):    
                    for x_num, x_cat, y in data_loader:
                        x_num, x_cat, y = x_num.to(self.device), x_cat.to(self.device), y.to(self.device)       
                        mu, var = self.forward((x_num, x_cat))
                        loss = torch.mean(self.loss_func2(mu, var, y))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

        elif self.dataset.case == 2:
            if self.dataset.out_feat in self.dataset.cat_feats:
                for i in range(n_epochs):
                    for x_num, y in data_loader:
                        x_num, y = x_num.to(self.device), y.to(self.device)
                    
                        probs = self.forward(x_num)
                        loss = torch.mean(self.negative_logprob_cat(probs, y))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            else:
                print('now train for mu')
                for i in range(n_epochs):    
                    for x_num, y in data_loader:
                        x_num, y = x_num.to(self.device), y.to(self.device)       
                        mu, var = self.forward(x_num)
                        loss = torch.mean(self.loss_func1(mu, var, y))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                print('now train for var')
                optimizer = optim.Adam(self.parameters(), lr=1e-5)
                for i in range(n_epochs*2):    
                    for x_num, y in data_loader:
                        x_num, y = x_num.to(self.device), y.to(self.device)       
                        mu, var = self.forward(x_num)
                        loss = torch.mean(self.loss_func2(mu, var, y))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

        else:
            if self.dataset.out_feat in self.dataset.cat_feats:
                for i in range(n_epochs):
                    for x_cat, y in data_loader:
                        x_cat, y = x_cat.to(self.device), y.to(self.device)
                    
                        probs = self.forward(x_cat)
                        loss = torch.mean(self.negative_logprob_cat(probs, y))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            else:
                print('now train for mu')
                for i in range(n_epochs):    
                    for x_cat, y in data_loader:
                        x_cat, y = x_cat.to(self.device), y.to(self.device)       
                        mu, var = self.forward(x_cat)
                        loss = torch.mean(self.loss_func1(mu, var, y))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                print('now train for var')
                optimizer = optim.Adam(self.parameters(), lr=1e-5)
                for i in range(n_epochs*2):    
                    for x_cat, y in data_loader:
                        x_cat, y = x_cat.to(self.device), y.to(self.device)       
                        mu, var = self.forward(x_cat)
                        loss = torch.mean(self.loss_func2(mu, var, y))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            
        print('loss: ', loss.cpu().detach().numpy())
        self.eval()

class Condist_for_bn:
    def __init__(self, child, scm, global_mean, global_std):
        self.child = child
        self.parents = scm[child]['input']
        self.scm = scm
        self.global_mean = global_mean
        self.global_std = global_std
    def __call__(self, x):
        x_ = self.denormalize(x.data.cpu().numpy()[0])
        mean = np.array(self.scm[self.child]['weight'][1:]).dot(x_) + self.scm[self.child]['weight'][0]
        mean = (mean - self.global_mean[self.child]) / self.global_std[self.child]
        var = np.array(scm[self.child]['std']**2)
        var = var / (self.global_std[self.child]**2)
        return torch.tensor(mean.astype('float32')).unsqueeze(0), torch.tensor(var.astype('float32')).unsqueeze(0)
    def denormalize(self, x):
        return x * self.global_std[self.parents] + self.global_mean[self.parents]