import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class T_Network(nn.Module):
    # network architecture
    
    def __init__(self, in_channels, embedding_size, ff_inner):
        super().__init__()
        
        network = [
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_size),
            nn.BatchNorm1d(embedding_size)
        ] 
        
        time_network = [
            nn.Linear(embedding_size*2, ff_inner),
            nn.ReLU(),
            nn.Linear(ff_inner, 1)
        ]
        
        self.network = nn.Sequential(*network)
        self.time_network = nn.Sequential(*time_network)
    
    def forward(self, x1, x2):
        # input shape x1|x2 : batch_size, channel, x, y
        # compare images x1[i] with x2[i] index i=0..N
        
        x = self.network(torch.cat((x1, x2), dim=0))
        x1_embeddings, x2_embeddings = torch.split(x, x1.shape[0], dim=0) 
        
        time = self.time(x1_embeddings, x2_embeddings)
        return time
    
    def embedding(self, x):
        return self.network(x)
        
    def time(self, x1, x2):
        # input shape x1|x2: batch_size, embedding_size 
        # cat x1[i] and x2[i] to calc time
        
        compare = torch.cat((x1, x2), dim=1)
        time = self.time_network(compare)
        time = 1 - torch.exp( - time)
        #time = F.relu(time)
        return time
    