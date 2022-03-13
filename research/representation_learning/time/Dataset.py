import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Dataset(torch.utils.data.Dataset):

    def __init__(self, trajs, hashes, time_window, select_from_time_window):
        super().__init__()
        self.trajs = trajs
        self.hash_trajs = hashes
        self.time_window = time_window
        self.select_from_time_window = select_from_time_window
        
        
        # calculate window labels
        
        # key: md5 hash | value: dict -> key: md5 hash | value: time integer
        self.window_labels = {}
        self.imgs = {}
        self.md5s = []
        
        # collect md5s, imgs
        for i in range(len(self.hash_trajs)):
            cur_traj = self.hash_trajs[i]
            for j in range(len(cur_traj)):
            
                md5 = cur_traj[j]
                if md5 not in self.md5s:
                    self.md5s.append(md5)
                if md5 not in self.imgs.keys():
                    self.imgs[md5] = self.trajs[i][j]
        
        # fill the windows for each observation
        for i in range(len(self.hash_trajs)):
            cur_traj = self.hash_trajs[i]
            
            for parent_md5_idx in range(len(cur_traj)-1):
            
                parent_md5 = cur_traj[parent_md5_idx]
                
                # add parent md5 to dict
                if parent_md5 not in self.window_labels.keys():
                    self.window_labels[parent_md5] = {}
                
                window_md5s = None
                if parent_md5_idx + self.time_window < len(cur_traj):
                    window_md5s = cur_traj[parent_md5_idx:parent_md5_idx+self.time_window]
                else:
                    window_md5s = cur_traj[parent_md5_idx::]
                
                # add childs
                for time_distance, child_md5 in enumerate(window_md5s):
                    
                    time_label = time_distance #+ 1
                    
                    if child_md5 == parent_md5:
                        continue
                    
                    # add md5 child to md5 parent
                    if child_md5 not in self.window_labels[parent_md5].keys():
                        self.window_labels[parent_md5][child_md5] = time_label
                    else: # take minimum time label    
                        self.window_labels[parent_md5][child_md5] = np.minimum(time_label,self.window_labels[parent_md5][child_md5])
                
        # calc length
        self.parents = list(self.window_labels.keys())
        self.length = len(self.parents)           
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        
        # get parent md5
        
        parent_md5 = self.parents[idx]
       
        # time window or random?
        
        select_from_window = np.random.rand(1)[0]
        if select_from_window < self.select_from_time_window:
            
            # take a child from parent
            child_md5s = list(self.window_labels[parent_md5].keys())
            
            if len(child_md5s) == 0:
                return self.select_random(parent_md5)
            
            # select child
            idx = np.random.choice(len(child_md5s),1)[0]
            child_md5 = child_md5s[idx]
            
            # get label
            label = self.window_labels[parent_md5][child_md5] / self.time_window
            
            # get frames
            p_frame = self.imgs[parent_md5]
            c_frame = self.imgs[child_md5]
            
            return p_frame, c_frame, torch.tensor(label)
        else:
            return self.select_random(parent_md5)
            
    def select_random(self, parent_md5):     
    
        # take random md5
        idx = np.random.choice(len(self.md5s),1)[0]
        md5 = self.md5s[idx]
        
        # md5 not allowed to be in parent window
        while md5 in self.window_labels[parent_md5].keys():
            # take random md5
            idx = np.random.choice(len(self.md5s),1)[0]
            md5 = self.md5s[idx]
        
        # get label            
        if md5 == parent_md5:
            label = 0.0 # zero time distance
        else:
            label = 1.0 # max time distance
            
        # get frames
        p_frame = self.imgs[parent_md5]
        c_frame = self.imgs[md5]

        return p_frame, c_frame, torch.tensor(label)             