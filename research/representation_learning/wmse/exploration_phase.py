import numpy as np
import cv2
import gym
import random
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import hashlib
from roads import *

from torch.utils.data import DataLoader
from Dataset import Dataset
from Network import W_MSE

device = torch.device("cuda:0")
dtype = torch.float

# parameter
env_name = "BreakoutDeterministic-v4"#"MontezumaRevengeDeterministic-v4"

idx_counter = 0

def preprocess_frame(frame, d_size=(84,84)):
    frame = cv2.cvtColor(frame[30::,:], cv2.COLOR_BGR2GRAY)
    return cv2.resize(frame, dsize=d_size)
      
class Cell():
    # item in archive
    
    def __init__(self, idx, restore, frame, embedding, score=-np.inf):
    
        self.visits = 0
        
        self.idx = idx
        self.restore = restore
        self.embedding = embedding
        self.score = score
        self.frame = frame

class Archive():
    def __init__(self):
        # idx | cell
        self.cells = {}
        
    def __iter__(self):
        return iter(self.cells)
    
    def init_archive(self, start_info):
        self.cells = {}
        # start cell
        self.cells[start_info[0]] = Cell(start_info[0],start_info[1],start_info[2],
                                         start_info[3], score=0)

class Embedding_Holder():

    def __init__(self, network):
        self.frames = None
        self.embeddings = None
        self.network = network
        
    def add_frame(self, frame):
        frame = torch.tensor(frame).to(device).to(dtype).unsqueeze(0).unsqueeze(1)
        if self.frames == None:
            self.frames = frame
        else:
            self.frames = torch.cat((self.frames, frame), dim=0)
            
    def compute_embeddings(self):
        self.embeddings = self.network.embedding(self.frames)

class Env_Actor():
    # suggested paper actor - random action repeating actor
    # sample from bernoulli distribution with p = 1/mean for action repetition
    
    def __init__(self, env, mean_repeat=10):
        self.num_actions = env.action_space.n
        self.mean_repeat = mean_repeat
        self.env = env
        
        self.current_action = self.env.action_space.sample()
        self.repeat_action = np.random.geometric(1 / self.mean_repeat)
        
    def get_action(self):
        
        if self.repeat_action > 0:
            self.repeat_action -= 1
            return self.current_action
            
        self.current_action = self.env.action_space.sample()
        self.repeat_action = np.random.geometric(1 / self.mean_repeat) - 1
        return self.current_action
    
class Env_Runner():
    # agent env loop
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        self.actor = Env_Actor(self.env)
        self.env.seed(0)
        
    def run(self, cell, max_steps=100):
        
        self.env.restore_full_state(cell.restore)
            
        traj_elemtents = []
        step = 0
        done = False
        while not done and step < max_steps:
            
            # collect data
            action = self.actor.get_action()
            frame, reward, d, _ = self.env.step(action)
            # resize frame
            frame = preprocess_frame(frame)
            
            restore = self.env.clone_full_state()
            
            # save data
            traj_element = (frame, action, reward, d, restore, hashlib.md5(frame).hexdigest())
            traj_elemtents.append(traj_element)
            
            if d:
                done = True
            step += 1
            
        return traj_elemtents
        
class CellSeletor():
    # select starting cells
    
    def __init__(self, archive):
        self.archive = archive
        
    def select_cells(self, amount):
        keys = []
        weights = []
        for key in self.archive.cells:
            if key == None: # done cell
                weights.append(0.0)
            else:
                weights.append(1/(np.sqrt(self.archive.cells[key].visits)+1))
            keys.append(key)
            
        indexes = np.random.choice(range(len(weights)),size=amount,p=weights/np.sum(weights))
        
        selected_cells = []
        for i in indexes:
            selected_cells.append(self.archive.cells[keys[i]])
        return selected_cells
          
# multiprocessing method
def run(start_cell): 
    env_runner = Env_Runner(env_name)
    traj = env_runner.run(start_cell)
    return traj

def main():

    global idx_counter
    # init cell archive
    env_tmp = gym.make(env_name).unwrapped
    env_tmp.seed(0)
    start_s = preprocess_frame(env_tmp.reset())
    start_restore = env_tmp.clone_full_state()
    start_cell_info = [idx_counter, start_restore, start_s, None]
    archive = Archive()
    archive.init_archive(start_cell_info)
    idx_counter += 1
    
    # init selector
    selector = CellSeletor(archive)
    
    # best score memory
    best_score = -np.inf
    # remember max score cell idx
    max_score_idx = 0
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    
    # initally collect data to train the model
    
    # always starting point: env.reset()
    start_cells = [archive.cells[0] for i in range(400)]
    result = pool.map(run, start_cells)
    frame_trajs = []
    hash_trajs = []
    for traj, start_cell in zip(result,start_cells):
    
        # collect frames
        frames = [torch.tensor(start_s)]
        hashes = [hashlib.md5(start_s).hexdigest()]
        
        for elem in traj:
            frames.append(torch.tensor(elem[0]))
            hashes.append(elem[-1])
            
        frame_trajs.append(frames)
        hash_trajs.append(hashes)
        
    # init 
    L = 2
    batch_size = 32
    spatial_shift = 4
    in_channels = 1
    embedding_size = 32
    epochs = 150

    w_mse = W_MSE(in_channels, embedding_size).to(device)
    optimizer = optim.Adam(w_mse.parameters(), lr=5e-4)
    embedding_holder = Embedding_Holder(w_mse)
    mse_loss = nn.MSELoss()
    dataset = Dataset(frame_trajs, hash_trajs, L)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    
    w_mse.train()
    optimize_model(w_mse, optimizer, dataloader, mse_loss, epochs, batch_size)  
    w_mse.eval()
    
    # set an embedding for start state
    archive.cells[0].embedding = w_mse.embedding(torch.tensor(archive.cells[0].frame).to(device).to(dtype).unsqueeze(0).unsqueeze(1)).detach().cpu()[0]
    embedding_holder.add_frame(archive.cells[0].frame)
    #print(archive.cells[0].embedding.shape)
    
    # init roads
    roads = Roads(archive.cells[0].frame, w_mse)
    roads.update_embeddings()
    
    iteration = 0 
    best_score = -np.inf
    best_score_idx = 0
    while archive.cells[max_score_idx].score < 400:
    
        # get data
        start_cells = selector.select_cells(50) 
        result = pool.map(run, start_cells)
        
        # collect frames and restores and score from data
        frame_trajs = []
        restore_trajs = []
        score_trajs = []
        hash_trajs = []
        
        for traj, start_cell in zip(result,start_cells): 
           
            frames = [start_cell.frame]
            restores = [start_cell.restore]
            scores = [start_cell.score]
            hashes = [hashlib.md5(start_cell.frame).hexdigest()]
            
            for i, traj_element in enumerate(traj):
            
                frame, action, reward, done, restore, hash = traj_element
                if reward > 0:
                    pass
                    
                frames.append(frame)
                restores.append(restore)
                scores.append(scores[-1] + reward)
                hashes.append(hash)
                
            # save collected data
            frame_trajs.append(frames)
            restore_trajs.append(restores)
            score_trajs.append(scores)
            hash_trajs.append(hashes)
        
        # OPTIMIZE MODEL
        
        dataset = Dataset(frame_trajs, hash_trajs, L)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
        
        w_mse.train()
        optimize_model(w_mse, optimizer, dataloader, mse_loss, epochs, batch_size)  
        w_mse.eval()
        
        roads.update_embeddings()
        roads.update_nodes()
        # update all holder embeddings
        embedding_holder.compute_embeddings()

        # update only cell embeddings in archive which are selected as start cells
        seen_cells_idx = []
        seen_cells = []
        cell_frames = []
        for cell in start_cells:
            if cell.idx not in seen_cells_idx:
                seen_cells_idx.append(cell.idx)
                seen_cells.append(cell)
                cell_frames.append(cell.frame)

        # recompute archive embeddings
        frame_embeddings = torch.tensor(cell_frames).to(device).to(dtype).unsqueeze(1)
        frame_embeddings = w_mse.embedding(frame_embeddings).detach().cpu()
        for i, cell in enumerate(seen_cells):
            cell.embedding = frame_embeddings[i]     

        # add a new cell to the archive if there is progress in embedding space        
        for frames, start_cell, restores, scores in zip(frame_trajs, start_cells, restore_trajs, score_trajs):    
            
            start_cell.visits += 3
            if len(frames) < 15:
                continue
            
            # transform every seen frame into an embedding
            frame_embeddings = frames[1::]
            frame_embeddings = torch.tensor(frame_embeddings).to(device).to(dtype).unsqueeze(1)
            frame_embeddings = w_mse.embedding(frame_embeddings)
            
            # get start cell embedding 
            start_cell_embedding = start_cell.embedding.to(device)
            
            # compute distance between start cell and all frames
            distance = torch.sum((start_cell_embedding - frame_embeddings)**2,dim=1)
            
            max_distance_idx = torch.argmax(distance)
            max_distance = distance[max_distance_idx].detach().cpu().numpy()
            
            mean_distance = torch.mean(distance).detach().cpu().numpy()
            
            # not far enough?
            if max_distance <= 0 or max_distance < 1*mean_distance:
                continue
            
            # get frame from max distance
            max_distance_frame = frames[max_distance_idx.detach().cpu().numpy()+1]
            
            # now check distance between best candidate and all cells in archive
            candidate_embedding = w_mse.embedding(torch.tensor(max_distance_frame).to(device).to(dtype).unsqueeze(0).unsqueeze(1))
            accepted = roads.add_candidate(candidate_embedding, max_distance_frame)
            
            if not accepted:
                continue
            
            # new cell
            new_cell_embedding = candidate_embedding.detach().cpu()
            
            new_cell = Cell(idx_counter, restores[max_distance_idx+1], frames[max_distance_idx+1], new_cell_embedding, score=scores[max_distance_idx+1])
            
            if best_score < scores[max_distance_idx+1]:
                best_score = scores[max_distance_idx+1]
                best_score_idx = idx_counter
            
            # add new cell in archive and embedding holder
            archive.cells[idx_counter] = new_cell
            embedding_holder.add_frame(new_cell.frame)
            
            print("cells: ", len(archive.cells), "score: ", archive.cells[best_score_idx].score)
            # update all holder embeddings
            embedding_holder.compute_embeddings()
            
            idx_counter += 1
            
        iteration += 1 
        if iteration%20 == 0:
            f = open(f'cell_archive{iteration}.data', 'wb')
            pickle.dump(archive,f)
            f.close()    
     
     
def optimize_model(network, optimizer, dataloader, mse_loss, epochs, batch_size):
    for epoch in range(epochs):
        print(epoch)
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            x1, x2 = batch
            x1, x2 = x1.to(device).to(dtype), x2.to(device).to(dtype)
            x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)
            
            x = torch.cat((x1,x2),dim=0)
            v = network(x)
            
            
            v1, v2 = torch.split(v, (batch_size,batch_size))
            loss = mse_loss(v1, v2)
            loss.backward()
            optimizer.step()        

if __name__ == "__main__":
    for i in range(1):
        main()

