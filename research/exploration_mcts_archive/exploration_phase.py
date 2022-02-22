import numpy as np
import cv2
import gym
import random
import matplotlib.pyplot as plt
import multiprocessing
import pickle
import copy
import os
import time

from MCTS_Archive import MCTS_Node, MCTS_Archive

# parameter
env_name = "MontezumaRevengeDeterministic-v4"
downscale_features = (8,11,8)

# downscale rgb img to a key representation
def make_representation(frame):
    h, w, p = downscale_features
    greyscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(greyscale_img, (h,w))
    resized_img_pix_threshold = ((resized_img/255.0) * p).reshape(-1).astype(int)
    return tuple(resized_img_pix_threshold)     
    
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
    
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        self.actor = Env_Actor(self.env)
        self.env.seed(0)
        
    def run(self, start_cell_restore, max_steps=100):
        
        # init to selected state
        self.env.restore_full_state(start_cell_restore)
            
        traj_elemtents = []
        step = 0
        done = False
        while not done and step < max_steps:
            
            # collect data
            action = self.actor.get_action()
            frame, reward, d, _ = self.env.step(action)
            restore = self.env.clone_full_state()
            
            # save data
            traj_element = (make_representation(frame), frame, action, reward, done, restore)
            traj_elemtents.append(traj_element)
            
            if d:
                done = True
            step += 1
            
        return traj_elemtents
         
# multiprocessing methods
def run(start_cell_restore): 
    env_runner = Env_Runner(env_name)
    traj = env_runner.run(start_cell_restore)
    return traj
    
def select(archive):
    start_cell = archive.select_start_cell()
    return start_cell
    
def main(run_number, max_steps, path, num_cpu):
    
    # create folder to save information
    folder_name = f'run_{run_number}'
    
    os.mkdir(path+folder_name)
    logger = open(path+folder_name+"/exploration.csv", "w")
    logger.write(f'step,score,cells,select_leaf_ratio\n')
    logger.close()    
    
    # init mcts archive
    env_tmp = gym.make(env_name).unwrapped
    env_tmp.seed(0)
    start_s = env_tmp.reset()
    start_restore = env_tmp.clone_full_state()
    
    #setup mcts root
    
    root = MCTS_Node(make_representation(start_s), start_s, start_restore, None, score=0)
    archive = MCTS_Archive(root)
    
    pool = multiprocessing.Pool(num_cpu)
    iteration = 0
    steps = 0
    while steps < max_steps: #while True:
        
        # select start cells
        start_cells = [archive.select_start_cell() for i in range(int(num_cpu * 2.5))]
        
        # get data
        start_cells_restores = [start_cell.restore for start_cell in start_cells]
        result = pool.map(run, start_cells_restores)
        
        for traj, start_cell in zip(result,start_cells): # iterate all generated trajs
            
            # expand tree with new traj
            archive.expand_tree(start_cell, traj)
            steps += len(traj)
            
        leaf_ratio = archive.choose_leaf/archive.num_choose
        
        iteration += 1
        logger = open(path+folder_name+"/exploration.csv", "a+")
        logger.write(f'{steps},{archive.best_score},{len(archive.keys_in_tree)},{leaf_ratio}\n')
        logger.close()
    
    
if __name__ == "__main__":

    p = os.getcwd() + '/mctsarchive_'+time.asctime(time.gmtime()).replace(" ","_").replace(":","_")+'/'
    os.mkdir(p)
    num_cpu = 12
    
    for i in range(10):
        max_steps = 3000000
        main(i, max_steps, p, num_cpu)

