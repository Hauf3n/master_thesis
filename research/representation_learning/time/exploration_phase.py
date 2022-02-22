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
import os
import time as t

from torch.utils.data import DataLoader
from Dataset import Dataset
from Network import T_Network

device = torch.device("cuda:0")
dtype = torch.float

# parameter
env_name = "MontezumaRevengeDeterministic-v4"
idx_counter = 0

class Cell():
    # item in archive
    
    def __init__(self, idx, restore, frame, embedding=None, score=-np.inf):
    
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
# hold images of checkpoints and their embeddings

    def __init__(self, network):
            self.frames = None
            self.embeddings = None
            self.network = network
            
    def add_frame(self, frame):
        
        frame = torch.tensor(frame).to(device).to(dtype)
        
        if self.frames == None:
            self.frames = frame
        else:
            self.frames = torch.cat((self.frames, frame),dim=0)
            
    def compute_embeddings(self):
        self.embeddings = self.network.embedding(self.frames).detach()                
        
class Env_Actor():
    # suggested paper actor - random action repeating actor
    # sample from bernoulli distribution with p = 1/mean for action repetition
    
    def __init__(self, env, mean_repeat=10):#7
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
    
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        self.actor = Env_Actor(self.env)
        self.env.seed(0)
        
    def run(self, start_cell, max_steps=50):
        
        self.env.restore_full_state(start_cell.restore)
            
        traj_elemtents = []
        step = 0
        done = False
        while not done and step < max_steps:
            
            # collect data
            action = self.actor.get_action()
            frame, reward, d, _ = self.env.step(action)
            # resize frame
            frame = cv2.resize(frame[30::,:], dsize=(84,84))/255
            
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

def main(filename='experience.data'):
    
    # create folder
    p = os.getcwd() + '/mctsarchive_'+t.asctime(t.gmtime()).replace(" ","_").replace(":","_")+'/'
    os.mkdir(p)
    logger = open(p+"/exploration.csv", "w")
    logger.write(f'step,score,cells\n')
    logger.close() 
    
    global idx_counter
    # init cell archive
    env_tmp = gym.make(env_name).unwrapped
    env_tmp.seed(0)
    start_s = cv2.resize(env_tmp.reset()[30::,:], dsize=(84,84))/255
    start_restore = env_tmp.clone_full_state()
    start_cell_info = [idx_counter, start_restore, start_s, None]
    archive = Archive()
    archive.init_archive(start_cell_info)
    idx_counter += 1
    
    # init selector
    selector = CellSeletor(archive)

    iteration = 0
    best_score = -np.inf
    # remember max score cell idx
    max_score_idx = 0
      
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    
    # initally collect data trajs to train the model
    
    # always starting point: env.reset()
    start_cells = [archive.cells[0] for i in range(100)]
    result = pool.map(run, start_cells)
    frame_trajs = []
    hash_trajs = []
    for traj, start_cell in zip(result,start_cells):
    
        # collect frames
        frames = [torch.tensor(start_cell.frame)]
        hashes = [hashlib.md5(start_cell.frame).hexdigest()]
        
        for elem in traj:
            frames.append(torch.tensor(elem[0]))
            hashes.append(elem[-1])
            
        frame_trajs.append(frames)
        hash_trajs.append(hashes)
        
    # init 
    time_window = 20
    batch_size = 64
    epochs = 15
    select_from_time_window = 0.7
    time_threshold = 0.8
    
    network = T_Network(3,32,512).to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    
    loss =  nn.MSELoss()#nn.L1Loss()#nn.MSELoss()#nn.CrossEntropyLoss()
    dataset = Dataset(frame_trajs, hash_trajs, time_window, select_from_time_window)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=12, shuffle=True, drop_last=True)
    
    network.train()
    optimize_model(network, optimizer, dataloader, loss, 20)#epochs)  
    network.eval()  

    embedding_holder = Embedding_Holder(network)
    
    # set an embedding for start state
    archive.cells[0].embedding = network.embedding(torch.tensor(archive.cells[0].frame).to(device).to(dtype).unsqueeze(0).permute(0,3,1,2)).detach().cpu()
    embedding_holder.add_frame(torch.tensor(archive.cells[0].frame).to(dtype).unsqueeze(0).permute(0,3,1,2))
    
    steps = 0                            
    while archive.cells[max_score_idx].score < 2600:#while True:
        
        # get data
        start_cells = selector.select_cells(125) 
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
            
            steps += len(traj)
            for i, traj_element in enumerate(traj):
            
                frame, action, reward, done, restore, hash = traj_element
                if reward > 0:
                    print("seen reward:", reward)
                    
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
        
        dataset = Dataset(frame_trajs, hash_trajs, time_window, select_from_time_window)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=12, shuffle=True, drop_last=True)
        
        network.train()
        optimize_model(network, optimizer, dataloader, loss, epochs)
        network.eval()
        
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
        frame_embeddings = torch.tensor(cell_frames).to(device).to(dtype).permute(0,3,1,2)
        frame_embeddings = network.embedding(frame_embeddings).detach().cpu()
        for i, cell in enumerate(seen_cells):
            cell.embedding = frame_embeddings[i]
        
        # add a new cell to the archive if there is progress in time        
        for frames, start_cell, restores, scores in zip(frame_trajs, start_cells, restore_trajs, score_trajs):
            
            start_cell.visits += 1
            
            # transform every seen frame into an embedding
            frame_embeddings = frames[1::]
            frame_embeddings = torch.tensor(frame_embeddings).to(device).to(dtype).permute(0,3,1,2)
            frame_embeddings = network.embedding(frame_embeddings)
            
            start_cell_embedding = start_cell.embedding.to(device)
            
            # look at time distance between start frame and all trajectory frames
            start_cell_embedding = start_cell_embedding.repeat(frame_embeddings.shape[0],1)
            
            time = network.time(start_cell_embedding, frame_embeddings).detach().cpu().numpy()#.squeeze(1)

            # select frame which has the greatest time distance to start frame
            # select the best candidate for a new entry in archive - max time distance
            best_idx = np.argmax(time)
            best_time = time[best_idx]
                    
            # now check time distance to every embedding in the whole archive - because we dont want loops
            cell_embeddings = embedding_holder.embeddings
            
            # extract the correct frame embedding
            frame_embedding = frame_embeddings[best_idx]
            
            # compare time elapsed best_idx with all cells in archive
            frame_embedding = frame_embedding.repeat(cell_embeddings.shape[0],1)
            times = network.time(cell_embeddings, frame_embedding).detach().cpu().numpy()#.squeeze(1)
            
            # time to every cell in archive must be larger than time threshold
            far_away = (times > time_threshold).all()
            
            if not far_away:
                # add a visit to checkpoints that are too close to candidate
                checkpoints_idx_close = (times < time_threshold).nonzero()[0]
                for c_idx in checkpoints_idx_close:
                    archive.cells[c_idx].visits += 1
            
            # accept a new checkpoint in the archive if new cell is not close to other cells in archive 
            if far_away:
                # put new cell in archive
                new_parent_embedding = network.embedding(torch.tensor(frames[best_idx+1]).to(device).to(dtype).unsqueeze(0).permute(0,3,1,2)).detach().cpu()
                add_new_cell(restores[best_idx], frames[best_idx+1], new_parent_embedding, scores[best_idx], embedding_holder, archive)
                
                if scores[best_idx] > best_score:
                    best_score = scores[best_idx]
                    max_score_idx = idx_counter-1
                    print("new best score: ", best_score)
                    torch.save(network,p+f'/network_{best_score}.pt')
            
                      
        # log    
        logger = open(p+"/exploration.csv", "a+")
        logger.write(f'{steps},{best_score},{len(archive.cells)}\n')
        logger.close()
        
        iteration += 1 
        if iteration%20 == 0:
            f = open(p+f'/cell_archive{iteration}.data', 'wb')
            pickle.dump(archive,f)
            f.close()
            
def add_new_cell(restore, frame, frame_embedding, score, embedding_holder, archive):
    global idx_counter
    # put new cell in archive
                
    new_cell = Cell(idx_counter, restore, frame, frame_embedding, score=score)          
    
    # add new cell in archive and embedding holder
    embedding_holder.add_frame(torch.tensor(frame).to(device).to(dtype).unsqueeze(0).permute(0,3,1,2))
    # update all holder embeddings
    embedding_holder.compute_embeddings()
    
    archive.cells[idx_counter] = new_cell 
    idx_counter += 1
    
def optimize_model(network, optimizer, dataloader, loss_objective, epochs):   
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            p, c, time_label = batch
            p, c, time_label = p.to(device).to(dtype), c.to(device).to(dtype), time_label.to(device).float()
            p, c = p.permute(0,3,1,2), c.permute(0,3,1,2)

            time = network(p, c)

            loss = loss_objective(time, time_label.unsqueeze(1))
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    for i in range(1):
        main(f'experience{i}.data')

