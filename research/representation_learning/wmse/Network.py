import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class W_MSE(nn.Module):
    # DQN nature paper architecture
    
    def __init__(self, in_channels, embedding_size):
        super().__init__()
        
        network = [
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,512),
            nn.ReLU(),
            nn.Linear(512, embedding_size)
        ]
        
        self.network = nn.Sequential(*network)
        self.whitening2d = Whitening1D(embedding_size)
    
    def forward(self, x):
        z = self.network(x)
        return self.whitening2d(z)
        
    def embedding(self, x):
        return self.network(x)

class Whitening1D(nn.Module):
    
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
    
    def forward(self, z):
        # shape: batch_size,embedding_size
        N = z.shape[0]
        mu_B = torch.mean(z, dim=0)
        z = z - mu_B
        cov_B = 1/(N - 1) * torch.mm(z.T,z)
        
        L = torch.cholesky(cov_B)
        W_B = torch.inverse(L)

        coloring = W_B @ z.T
        return coloring.T