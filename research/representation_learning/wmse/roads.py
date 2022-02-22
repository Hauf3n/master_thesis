import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Network import W_MSE
device = torch.device("cuda:0")
dtype = torch.float

class Roads_Node():

    def __init__(self, red_zone):
        self.red_zone = red_zone
        
        
class Roads():

    def __init__(self, root_frame, network):
        self.nodes = [Roads_Node(0.0)]
        self.frames = [root_frame]
        self.embeddings = None
        self.network = network
    
    def update_embeddings(self):
        frames = torch.tensor(self.frames).to(device).to(dtype).unsqueeze(1)
        self.embeddings = self.network.embedding(frames).detach()
        
    def update_nodes(self):
        self.nodes = [Roads_Node(0.0)]
        d = torch.cdist(self.embeddings, self.embeddings, p=2.0)
        #print("d",d)
        sorted = torch.sort(d, dim=1)
        #print("sorted ",sorted)
        for i in range(1,len(self.frames),1):
            idx = sorted[1][i][1:5]
            red_zone = torch.mean(d[i][idx])
            #print("NEW RED: ",red_zone.detach().cpu().numpy())
            self.nodes.append(Roads_Node(red_zone.detach().cpu().numpy()))
        #for 
        
        
    def add_candidate(self, candidate_embedding, candidate_frame):
        
        accepted, neighbour_distance, neighbour_idx = self.check_candidate(candidate_embedding)
        
        if accepted:
            
            red_zone = neighbour_distance
            
            self.frames.append(candidate_frame)
            self.nodes.append(Roads_Node(red_zone))
            print("red zone: ", red_zone)
            
            self.update_embeddings()
            
            return True
        
        return False
    
    def check_candidate(self, candidate_embedding):
        
        distance = torch.sum((self.embeddings - candidate_embedding)**2,dim=1)
        min_distance_idx = torch.argmin(distance)
        min_distance = distance[min_distance_idx].detach().cpu().numpy()
        
        if min_distance < self.nodes[min_distance_idx].red_zone or min_distance < 1.0:
            return (False, min_distance, min_distance_idx)
        else:
            return (True, min_distance, min_distance_idx)
    
    