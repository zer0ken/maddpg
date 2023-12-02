import numpy as np
from gym.spaces import Discrete

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
        
        if obstacle_pos is None or dirty_pos is None or agent_pos is None:
            indices = np.array([(i, j) for i in range(self.n_row) for j in range(self.n_col)])
            
            if obstacle_pos is None:
                picked = np.random.choice(len(indices), self.n_row * self.n_col // 10, replace=False)
                self.obstacle_pos = indices[picked, :]
                indices = np.delete(indices, picked, axis=0)
                
            if dirty_pos is None:
                self.dirty_pos = indices
            
            if agent_pos is None:
                self.agent_pos = indices[np.random.choice(len(indices), self.n_agent, replace=False)]

        self.visual_field = 3
        
        self.reset()
        self.dirty_layer[self.dirty_layer == self.obstacle_layer] = 0
        
        """Gym Env variable"""
        self.n = self.n_agent
        self.observation_space = np.array([np.zeros((self.visual_field**2 + 3*self.n_row*self.n_col,)) for _ in range(self.n_agent)])
        self.action_space = np.array([Discrete(5) for _ in range(self.n_agent)])

    def reset(self):
        self.agents = {}
        self.agent_layer = np.zeros((self.n_row, self.n_col))
        self.dirty_layer = np.zeros((self.n_row, self.n_col))

        for i, pos in enumerate(self.agent_pos):
            self.agents[i] = {'home': pos, 'pos': pos}
            self.agent_layer[pos[0], pos[1]] = 1

        for pos in self.dirty_pos:
            self.dirty_layer[pos[0], pos[1]] = 1

        # 환경 처음 만들 때만 obstacle_layer 초기화
        if not hasattr(self, 'obstacle_layer'):
            self.obstacle_layer = np.zeros((self.n_row, self.n_col))
            for pos in self.obstacle_pos:
                self.obstacle_layer[pos[0], pos[1]] = 1
        
        obs_n = []
        for agent_idx in range(self.n_agent):
            obs_n.append(self.get_observation(agent_idx))
        return obs_n

    def step(self, actions):
        rewards = [0] * self.n_agent # 에이전트 개수만큼 리워드 생성
        for i, action in enumerate(actions):
            self._step_agent(self.agents[i], action)
            rewards[i]-=1 # 1-step 마다 reward -1
            
        colision = set()
        for i, a in self.agents.items():
            for j, b in self.agents.items():
                if i == j:
                    continue
                if a['pos'] == b['new_pos'] and a['new_pos'] == b['pos']:
                    colision.add(i)
                    colision.add(j)
                if a['new_pos'] == b['new_pos']:
                    colision.add(i)
                    colision.add(j)
                    
            if self.obstacle_layer[a['new_pos']] == 1: # 벽과 충돌 시
                colision.add(i)
                
            if a['new_pos'][0]<0 or a['new_pos'][0]>self.n_row: #밖으로 
                colision.add(i)
            if a['new_pos'][1]<0 or a['new_pos'][1]>self.n_col: #나감
                colision.add(i)

        for i in tuple(colision):
            rewards[i]-=1   #충돌 시 reward -1
            self._rewind_agent(self.agents[i])

        for i, agent in self.agents.items():
            if self.dirty_layer[agent['new_pos']] == 1: # 도착한 곳이 더러운 곳이라면 reward +1
                self.dirty_layer[agent['new_pos']] = 0 # 도착한 곳은 청소 됨
                rewards[i] += 1
        
        self.done = np.all(self.dirty_layer == 0)   # 전부 청소되면 done

    def _step_agent(self, agent, action):
        if 'new_pos' in agent:
            agent['pos'] = agent['new_pos']

        agent['new_pos'] = (
            agent['pos'][0] + MAACEnv.ACTIONS[action][0],
            agent['pos'][1] + MAACEnv.ACTIONS[action][1]
        )
        agent['action'] = action
        self.agent_layer[agent['pos']] = 0
        self.agent_layer[agent['new_pos']] = 1

    def _rewind_agent(self, agent):
        self.agent_layer[agent['pos']] = 1
        self.agent_layer[agent['new_pos']] = 0
        if 'new_pos' in agent:
            agent['new_pos'] = agent['pos']

    # 특정 에이전트의 local observation 반환
    def get_observation(self, agent_idx):
        pos = self.agents[agent_idx]['pos']
        padded_pos = np.array([pos[0]+1, pos[1]+1])
        
        # 에이전트 주변 시야
        padded_obstacle_layer = np.ones((self.n_row+2, self.n_col+2))
        padded_obstacle_layer[1:-1, 1:-1] = self.obstacle_layer
        vf=self.visual_field//2
        agent_vision_obstacle = padded_obstacle_layer[padded_pos[0]-vf:padded_pos[0]+(vf+1), padded_pos[1]-vf:padded_pos[1]+(vf+1)]
        agent_vision_obstacle = agent_vision_obstacle.reshape(self.visual_field**2)
        
        # 에이전트 자신
        agent_self_layer = np.zeros((self.n_row, self.n_col))
        agent_self_layer[pos[0], pos[1]] = 1
        agent_self_layer = agent_self_layer.reshape(self.n_row*self.n_col)
        
        # 다른 에이전트
        other_agent_layer = self.agent_layer.copy()
        other_agent_layer[pos[0], pos[1]] = 0
        other_agent_layer_flatten = other_agent_layer.reshape(self.n_row*self.n_col)
        
        # 청소해야할 곳
        dirty_layer_flatten = self.dirty_layer.copy().reshape(self.n_row*self.n_col)
        
        return np.concatenate((agent_vision_obstacle, agent_self_layer, other_agent_layer_flatten, dirty_layer_flatten))
        
    def render(self):
        pass

    def close(self):
        pass