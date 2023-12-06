import numpy as np
import torch as T

class MultiAgentReplayBuffer:
    def __init__(self, max_size, local_dim, global_dim, 
                 n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.batch_size = batch_size

        self.obstacle_memory = np.zeros((self.mem_size, n_agents, local_dim[0], local_dim[1]), dtype=np.float32)
        self.self_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        self.other_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        self.dirty_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        
        self.new_obstacle_memory = np.zeros((self.mem_size, n_agents, local_dim[0], local_dim[1]), dtype=np.float32)
        self.new_self_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        self.new_other_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        self.new_dirty_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        
        self.reward_memory = np.zeros((self.mem_size, n_agents), dtype=np.float32)
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        self.actor_action_memory = np.zeros((self.n_agents, self.mem_size, self.n_actions), dtype=np.float32)
        

    def store_transition(self, state, action, reward, 
                         state_, done):
        index = self.mem_cntr % self.mem_size

        for i in range(self.n_agents):
            self.obstacle_memory[index, i] = state[i].obstacle
            self.self_memory[index, i] = state[i].self_
            self.other_memory[index, i] = state[i].other
            self.dirty_memory[index, i] = state[i].dirty
            
            self.new_obstacle_memory[index, i] = state_[i].obstacle
            self.new_self_memory[index, i] = state_[i].self_
            self.new_other_memory[index, i] = state_[i].other
            self.new_dirty_memory[index, i] = state_[i].dirty
        
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.actor_action_memory[:, index, :] = action # (n_agents, mem_size, n_actions) <- (n_agents, 1(idx), n_actions)
                
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        obstacles = self.obstacle_memory[batch]
        selves = self.self_memory[batch]
        others = self.other_memory[batch]
        dirties = self.dirty_memory[batch]

        new_obstacles = self.new_obstacle_memory[batch]
        new_selves = self.new_self_memory[batch]
        new_others = self.new_other_memory[batch]
        new_dirties = self.new_dirty_memory[batch]
        
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]
        actions = self.actor_action_memory[:, batch, :]

        return obstacles, selves, others, dirties, \
            actions, rewards, \
            new_obstacles, new_selves, new_others, new_dirties, \
            terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True


class PERMA:
    def __init__(self, max_size, local_dim, global_dim, 
                 n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.priorities = np.zeros(self.mem_size)
        self.phi = 0.01 # soft mixing factor
        self.min_reward = np.inf
        self.max_reward = -np.inf

        self.obstacle_memory = np.zeros((self.mem_size, n_agents, local_dim[0], local_dim[1]), dtype=np.float32)
        self.self_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        self.other_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        self.dirty_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        
        self.new_obstacle_memory = np.zeros((self.mem_size, n_agents, local_dim[0], local_dim[1]), dtype=np.float32)
        self.new_self_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        self.new_other_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        self.new_dirty_memory = np.zeros((self.mem_size, n_agents, global_dim[0], global_dim[1]), dtype=np.float32)
        
        self.reward_memory = np.zeros((self.mem_size, n_agents), dtype=np.float32)
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        self.actor_action_memory = np.zeros((self.n_agents, self.mem_size, self.n_actions), dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        max_mem = min(self.mem_cntr, self.mem_size)
        if self.mem_cntr < self.mem_size:
            index = max_mem
        else:
            probs = self.get_probabilities()
            neg_porbs = np.ones_like(probs) - probs
            neg_porbs /= neg_porbs.sum()
            neg_porbs[-1] = 1.0 - neg_porbs[:-1].sum()
            index = np.random.choice(self.mem_size, 1, replace=False, p=neg_porbs)[0]

        for i in range(self.n_agents):
            self.obstacle_memory[index, i] = state[i].obstacle
            self.self_memory[index, i] = state[i].self_
            self.other_memory[index, i] = state[i].other
            self.dirty_memory[index, i] = state[i].dirty
            
            self.new_obstacle_memory[index, i] = state_[i].obstacle
            self.new_self_memory[index, i] = state_[i].self_
            self.new_other_memory[index, i] = state_[i].other
            self.new_dirty_memory[index, i] = state_[i].dirty
        
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.actor_action_memory[:, index, :] = action # (n_agents, mem_size, n_actions) <- (n_agents, 1(idx), n_actions)
                
        self.mem_cntr += 1

        self.priorities[index] = max(self.priorities[:max_mem], default=1.0)
        
        reward = reward.sum()
        if reward < self.min_reward:
            self.min_reward = reward
        if reward > self.max_reward:
            self.max_reward = reward
        
    def get_probabilities(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        probs = self.priorities[:max_mem] ** 0.6
        probs /= probs.sum()
        probs[-1] = 1.0 - probs[:-1].sum()
        return probs

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        sample_probs = self.get_probabilities()
        batch = np.random.choice(max_mem, self.batch_size, replace=False, p=sample_probs)
        
        obstacles = self.obstacle_memory[batch]
        selves = self.self_memory[batch]
        others = self.other_memory[batch]
        dirties = self.dirty_memory[batch]

        new_obstacles = self.new_obstacle_memory[batch]
        new_selves = self.new_self_memory[batch]
        new_others = self.new_other_memory[batch]
        new_dirties = self.new_dirty_memory[batch]
        
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]
        actions = self.actor_action_memory[:, batch, :]
        
        sum_rewards = rewards.sum(axis=1)
        new_priorities = (sum_rewards - self.min_reward) / (self.max_reward - self.min_reward)
        self.priorities[batch] = self.phi * new_priorities + (1 - self.phi) * self.priorities[batch]

        return obstacles, selves, others, dirties, \
            actions, rewards, \
            new_obstacles, new_selves, new_others, new_dirties, \
            terminal

    def ready(self):
        print('memory:', self.mem_cntr, '/', self.batch_size, '/', self.mem_size)
        if self.mem_cntr >= self.batch_size:
            return True