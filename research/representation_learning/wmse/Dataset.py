import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Dataset(torch.utils.data.Dataset):

    def __init__(self, trajs, hashes, L):
        super().__init__()
        self.trajs = trajs
        self.hash_trajs = hashes
        self.L = L
    
        # calculate window labels
        
        # key: md5 hash | value: dict -> key: md5 hash | value: time integer
        self.window_labels = {}
        self.imgs = {}
        self.md5s = []
        
        for i in range(len(self.hash_trajs)):
            cur_traj = self.hash_trajs[i]
            for j in range(len(cur_traj)):
            
                md5 = cur_traj[j]
                if md5 not in self.md5s:
                    self.md5s.append(md5)
                if md5 not in self.imgs.keys():
                    self.imgs[md5] = self.trajs[i][j]
        
        for i in range(len(self.hash_trajs)):
            cur_traj = self.hash_trajs[i]
            
            for parent_md5_idx in range(len(cur_traj)-1):
            
                parent_md5 = cur_traj[parent_md5_idx]
                
                # add parent md5 to dict
                if parent_md5 not in self.window_labels.keys():
                    self.window_labels[parent_md5] = []
                
                window_md5s = None
                if parent_md5_idx + self.L < len(cur_traj):
                    window_md5s = cur_traj[parent_md5_idx:parent_md5_idx+self.L]
                else:
                    window_md5s = cur_traj[parent_md5_idx::]
                
                # add childs
                for child_md5 in window_md5s:
 
                    # add md5 child to md5 parent
                    if child_md5 not in self.window_labels[parent_md5]:
                        self.window_labels[parent_md5].append(child_md5)
                
        # calc length
        self.parents = list(self.window_labels.keys())
        self.length = len(self.parents)           
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        
        # get parent md5
        
        parent_md5 = self.parents[idx]
       
        # select neighbour
        
        # md5
        neighbours = self.window_labels[parent_md5]
        idx = np.random.choice(range(len(neighbours)), 1)[0]
        neighbour_md5 = neighbours[idx]
        
        # get imgs
        x1 = self.imgs[parent_md5]
        x2 = self.imgs[neighbour_md5]
        
        return x1, x2            