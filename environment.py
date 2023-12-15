import os
import numpy as np
from gym.spaces import Discrete


class Observation:
    def __init__(self, obstacle, self_, other, dirty):
        self.obstacle = obstacle
        self.self_ = self_
        self.other = other
        self.dirty = dirty


""" Rule: Never change its properties. If you need, just make a new instance """
class MAACEnv:
    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (0, 0)}

    def __init__(self, n_agent=3, n_row=10, n_col=10,
                 agent_pos=None, dirty_pos=None, obstacle_pos=None):
        self.n_row = max(5, n_row)
        self.n_col = max(5, n_col)
        self.n_agent = max(min(n_agent, 10), 1)

        self.agent_pos = agent_pos
        self.dirty_pos = dirty_pos
        self.obstacle_pos = obstacle_pos
        
        self.exported = False
        
        if obstacle_pos is None or dirty_pos is None or agent_pos is None:
            indices = np.array([(i, j) for i in range(self.n_row) for j in range(self.n_col)])
            
            if obstacle_pos is None:
                picked = np.random.choice(len(indices), self.n_row * self.n_col // 10, replace=False)
                self.obstacle_pos = list(map(tuple,  indices[picked, :]))
                indices = np.delete(indices, picked, axis=0)
                
            if dirty_pos is None:
                self.dirty_pos = list(map(tuple, indices))

            if agent_pos is None:
                self.agent_pos = list(map(tuple, indices[np.random.choice(
                    len(indices), self.n_agent, replace=False)]))

        self.reset()
        self.dirty_layer[self.dirty_layer == self.obstacle_layer] = 0
        
        """ Gym Env variable """
        self.n = self.n_agent
        self.action_space = np.array([Discrete(4) for _ in range(self.n_agent)])
        
        """ GUI control """
        self.render_callback = None

    def reset(self, keep_agent=False):
        if not hasattr(self, 'agents'):
            self.agents = {}
        self.agent_layer = -np.ones((self.n_row, self.n_col))
        self.dirty_layer = np.zeros((self.n_row, self.n_col))
        self.visited_layer = -np.ones((self.n_row, self.n_col))
        
        self.steps = 0

        for i, pos in enumerate(self.agent_pos):
            agent = {'idx': i, 'home': pos, 'pos': pos, 'new_pos': pos, 'covered': 0}
            if keep_agent:
                agent['pos'] = self.agents[i]['pos']
                agent['new_pos'] = self.agents[i]['new_pos']
            self.agents[i] = agent
            self.agent_layer[agent['pos'][0], agent['pos'][1]] = i
            self.visited_layer[agent['pos'][0], agent['pos'][1]] = i

        for pos in self.dirty_pos:
            if pos not in self.obstacle_pos:
                self.dirty_layer[pos[0], pos[1]] = 1

        # 환경 처음 만들 때만 obstacle_layer 초기화
        if not hasattr(self, 'obstacle_layer'):
            self.obstacle_layer = np.zeros((self.n_row, self.n_col))
            for pos in self.obstacle_pos:
                self.obstacle_layer[pos[0], pos[1]] = 1
                
        if not hasattr(self, 'num_dirty'):
            self.num_dirty = self.dirty_layer.sum()
        
        obs_n = []
        for agent_idx in range(self.n_agent):
            obs_n.append(self.get_observation(agent_idx))
        return obs_n

    def step(self, actions):
        rewards = np.zeros((self.n_agent,))
        for i, action_prob in enumerate(actions):
            action = np.argmax(action_prob)
            self._step_agent(self.agents[i], action)
            if action == 4:
                rewards[i] -= 0.1
            else:
                rewards[i] -= 0.05

        """ invalid action check """
        for i, a in self.agents.items():
            if any((a['new_pos'] in self.obstacle_pos,
                   a['new_pos'][0] < 0, a['new_pos'][0] >= self.n_row,
                   a['new_pos'][1] < 0,a['new_pos'][1] >= self.n_col)):
                self._rewind_agent(a)
                rewards[i] -= 0.5 # 벽, 장애물과 충돌
                     
        collided = set()
        for i, a in self.agents.items():
            for j, b in self.agents.items():
                if i >= j:
                    continue
                if a['pos'] == b['new_pos'] and a['new_pos'] == b['pos']:
                    collided.add(i)
                    collided.add(j)
                    continue
                if a['new_pos'] == b['new_pos']:
                    collided.add(i)
                    collided.add(j)

        for i in tuple(collided):
            self._rewind_agent(self.agents[i])
            rewards[i] -= 0.5 #충돌한 애들끼리 +1
        
        for i, agent in self.agents.items():
            if self.visited_layer[agent['new_pos']] == -1:
                self.visited_layer[agent['new_pos']] = i
            
            if self.dirty_layer[agent['new_pos']] == 1: # 도착한 곳이 더러운 곳이라면 reward +1
                self.dirty_layer[agent['new_pos']] = 0 # 도착한 곳은 청소 됨
                rewards[i] +=1 # 청소 했으니까 +1
          
                if 'covered' not in agent:
                    agent['covered'] = 1
                else:
                    agent['covered'] += 1 
                
        observations = [self.get_observation(i) for i in range(self.n_agent)]
        done = [False for i in range(self.n_agent)]
        info = self.get_info()
        
        if self.dirty_layer.sum() == 0:
            done = [True for i in range(self.n_agent)]   # 전부 청소되면 done
            
            for i, agent in self.agents.items():
                rewards[i] += agent['covered'] * 200 / self.num_dirty
        
        # something to do before return goes here
        self.steps += 1
                
        return observations, rewards, done, info

    def _step_agent(self, agent, action):
        if 'new_pos' in agent:
            agent['pos'] = agent['new_pos']
        agent['new_pos'] = (
            agent['pos'][0] + MAACEnv.ACTIONS[action][0],
            agent['pos'][1] + MAACEnv.ACTIONS[action][1]
        )
        agent['action'] = action
        if 'new_pos' in agent:
            self.agent_layer[agent['pos']] = -1
            try:
                self.agent_layer[agent['new_pos']] = agent['idx']
            except IndexError:
                self.agent_layer[agent['pos']] = agent['idx']
                agent['new_pos'] = agent['pos']

    def _rewind_agent(self, agent):
        if self.agent_layer[agent['pos']] not in (-1, agent['idx']):
            self._rewind_agent(self.agents[self.agent_layer[agent['pos']]])
        self.agent_layer[agent['pos']] = agent['idx']
        if 'new_pos' in agent:
            self.agent_layer[agent['new_pos']] = -1
            agent['new_pos'] = agent['pos']

    # 특정 에이전트의 local observation 반환
    def get_observation(self, agent_idx):
        pos = self.agents[agent_idx]['pos']
        
        # 장애물
        obstacle_layer = self.obstacle_layer
        
        # 에이전트 자신
        agent_self_layer = np.zeros_like(self.agent_layer)
        agent_self_layer[pos[0], pos[1]] = 1
        
        # 다른 에이전트
        other_agent_layer = np.zeros_like(self.agent_layer)
        indices = np.argwhere(self.agent_layer != -1)
        other_agent_layer[indices[:, 0], indices[:, 1]] = 1
        other_agent_layer[pos[0], pos[1]] = 0
        
        # 청소해야할 곳
        dirty_layer = self.dirty_layer
        
        return Observation(obstacle_layer, agent_self_layer, other_agent_layer, dirty_layer)
        
    def get_info(self):
        # we can use this to render GUI, do debug, and e.t.c. 
        info = {
            'steps': self.steps,
            'agents_info': self.agents,
            'visited_layer': self.visited_layer,
            'dirty_layer': self.dirty_layer,
        }
        return info
    
    def render(self, *args, **kwargs):
        self.render_callback(*args, **kwargs)

    def close(self):
        pass
    
    def export(self):
        np.array([self.n_agent, self.n_row, self.n_col]).tofile('./last_env/args.csv', sep=',')
        np.array(self.agent_pos).tofile('./last_env/agent_pos.csv', sep=',')
        np.array(self.dirty_pos).tofile('./last_env/dirty_pos.csv', sep=',')
        np.array(self.obstacle_pos).tofile('./last_env/obstacle_pos.csv', sep=',')
        
        self.exported = True
    
    def import_last_env():
        if not os.path.exists('./last_env/args.csv'):
            return None
        
        args = np.fromfile('./last_env/args.csv', sep=', ', 
                           dtype=np.int32)
        agent_pos = np.fromfile('./last_env/agent_pos.csv', sep=', ', 
                                dtype=np.int32).reshape(args[0], 2)
        dirty_pos = np.fromfile('./last_env/dirty_pos.csv', sep=', ', 
                                dtype=np.int32).reshape(-1, 2)
        obstacle_pos = np.fromfile('./last_env/obstacle_pos.csv', sep=', ', 
                                   dtype=np.int32).reshape(-1, 2)
        
        return MAACEnv(*args, agent_pos=list(map(tuple, agent_pos)), 
                       dirty_pos=list(map(tuple, dirty_pos)), 
                       obstacle_pos=list(map(tuple, obstacle_pos)))