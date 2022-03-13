import numpy as np

class MCTS_Node():

    def __init__(self, key, frame, restore, parent, score=0, traj_len=0):
        super().__init__()
        
        # own key
        self.key = key
        self.frame = frame
        # parent node
        self.parent = parent
        
        # restore
        self.restore = restore
        # score 
        self.score = score
        # traj length
        self.traj_len = traj_len
        # child nodes 
        self.edges = {}
        
        self.sub_tree_size = 0
        self.visits = 0
        self.exp_start = 0
        self.trajectory_visits = 0

    def expanded(self):
        return len(self.edges) > 0        
        
    def add_child(self, node):
        self.edges[node.key] = node
        self.sub_tree_size += 1
        
        # inc subtree size
        cur_parent = self.parent
        while cur_parent != None:
            cur_parent.sub_tree_size += 1
            cur_parent = cur_parent.parent
     
    def update(self, restore, score, traj_len):
        # restore
        self.restore = restore
        # score 
        self.score = score
        # traj length
        self.traj_len = traj_len
        #reset visits
        self.visits = 0
        self.exp_start = 0
        
    def replaced_child_node(self, child_key, child_childs):
        
        self.sub_tree_size -= 1
        
        for child in child_childs:
            self.edges[child.key] = child
            child.parent = self
            
        # delete child
        del self.edges[child_key]
        
        # smaller subtree size
        cur_parent = self.parent
        while cur_parent != None:
            cur_parent.sub_tree_size -= 1
            cur_parent = cur_parent.parent
        

class MCTS_Archive():
    
    def __init__(self, root):
        super().__init__()
        self.root = root # root needs None as parent!
        
        # save existing keys in mcts tree
        # key: key | value: MCTS_Node
        self.keys_in_tree = {}
        self.keys_in_tree[self.root.key] = self.root
        
        # log best score
        self.best_score = - np.inf
        
        # log 
        self.choose_leaf = 0
        self.num_choose = 0
        
    def select_start_cell(self):
        
        start_node = self.root
        
        selected_start_cell = False
        while not selected_start_cell:
            
            # found leaf?
            if start_node.expanded() == False:
                selected_start_cell = True
                self.choose_leaf += 1
                continue
            
            # stop at node?
            rnd = np.random.rand(1)[0]
            if self.probability_stop(start_node) > rnd:
                selected_start_cell = True
                continue
            
            # choose branching and calculate scores
            scores = []
            childs = []
            for key in start_node.edges:
                scores.append(self.score(start_node.edges[key]))
                childs.append(key)
            
            # make decision
            idx = np.random.choice(np.arange(0,len(childs),1),1,p=scores/np.sum(scores))[0]
            branching_key = childs[idx]
            start_node = start_node.edges[branching_key]
        
        self.num_choose += 1    
        return start_node
    
    def probability_stop(self, node):
        # stop at node
        
        start_mean_total = 0
        for child in node.edges:
            mean_sub_tree_start = (node.edges[child].trajectory_visits+1)/(node.edges[child].sub_tree_size + 1)
            start_mean_total += mean_sub_tree_start
        start_mean_total = start_mean_total / len(node.edges)
        
        weight = (((node.exp_start+1) / start_mean_total)**-1)
        return  np.minimum(weight * 1/(node.sub_tree_size + 1),1.0)
    
    def score(self, node):
        # branching score
        return ((node.sub_tree_size + 1)/(node.trajectory_visits + 1))
             
    # current node is the start node but will change overtime in expand_tree 
    def expand_tree(self, current_node, traj):
        
        # inc visits of start cell
        current_node.exp_start += 1
        
        # inc trajectory visits of all nodes back to the root
        current_node.trajectory_visits += 1
        cur_parent = current_node.parent
        while cur_parent != None:
            cur_parent.trajectory_visits += 1
            cur_parent = cur_parent.parent
            
        cur_score = current_node.score
        cur_traj_length = current_node.traj_len
        
        seen = []
        for i,traj_element in enumerate(traj):
        
            key, frame, action, reward, done, restore = traj_element
            
            if done:
                break
            
            cur_score += reward
            cur_traj_length += 1
            
            if key not in seen and key in self.keys_in_tree:
                inc_visits_node = self.keys_in_tree[key]
                inc_visits_node.visits += 1
                seen.append(key)
            
            node_overwrite = False
            # key already in tree?
            if key in self.keys_in_tree.keys():
                # only overwrite cell if we get a better score or shorter traj length
                if cur_score > self.keys_in_tree[key].score or (cur_score == self.keys_in_tree[key].score and cur_traj_length < self.keys_in_tree[key].traj_len):
                    node_overwrite = True
            
            # go deeper in the tree if key in childs and there is no node overwrite
            if key in current_node.edges.keys() and node_overwrite == False:
                # go deeper in tree
                current_node = current_node.edges[key]
                continue
            
            if node_overwrite:
                # dont replace root
                if key == self.root.key:
                    continue       
            
            if node_overwrite == False and key in self.keys_in_tree.keys():
                continue
            
            # add new node to MCTS tree
            
            # construct cell 
            new_node = MCTS_Node(key, frame, restore, current_node, score=cur_score, traj_len=cur_traj_length)
            
            if self.best_score < cur_score:
                self.best_score = cur_score
            
            # add new node in the tree
            if not node_overwrite:
                current_node.add_child(new_node)
                # add node with new key to memory or overwrite node    
                self.keys_in_tree[key] = new_node
                # new current_node
                current_node = new_node 
                
            
            # if node overwrite, then clean the position where the replaced node is
            if node_overwrite:
                # get node
                replaced_node = self.keys_in_tree[key]
                replaced_node_parent = replaced_node.parent
                replaced_node_childs = [self.keys_in_tree[key] for key in replaced_node.edges]
                
                if replaced_node_parent.key == current_node.key:
                    replaced_node.update(restore, cur_score , cur_traj_length)
                    current_node = replaced_node
                    
                else:
                    replaced_node_parent.replaced_child_node(key, replaced_node_childs)
                    
                    # add new node in the tree
                    current_node.add_child(new_node)
                    # add node with new key to memory or overwrite node    
                    self.keys_in_tree[key] = new_node
                    # new current_node
                    current_node = new_node
                    
                    # delete child_node 
                    del replaced_node