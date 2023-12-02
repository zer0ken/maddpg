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

        self.reset()
        self.dirty_layer[self.dirty_layer == self.obstacle_layer] = 0
        
        """Gym Env variable"""
        self.n = self.n_agent
        self.observation_space = np.array([np.zeros((3*3 + 3*self.n_row*self.n_col,)) for _ in range(self.n_agent)])
        self.action_space = np.array([Discrete(5) for _ in range(self.n_agent)])

    def reset(self):
        self.agents = {}
        self.agent_layer = np.zeros((self.n_row, self.n_col))
        self.dirty_layer = np.zeros((self.n_row, self.n_col))
        self.obstacle_layer = np.zeros((self.n_row, self.n_col))

        for i, pos in enumerate(self.agent_pos):
            self.agents[i] = {'home': pos, 'pos': pos}
            self.agent_layer[pos] = 1

        for pos in self.dirty_pos:
            self.dirty_layer[pos] = 1

        for pos in self.obstacle_pos:
            self.obstacle_layer[pos] = 1

        for i, pos in enumerate(self.agent_pos):
            self.agents[i] = {'home': pos, 'pos': pos}
            self.agent_layer[pos] = 1

        for pos in self.dirty_pos:
            self.dirty_layer[pos] = 1

        for pos in self.obstacle_pos:
            self.obstacle_layer[pos] = 1

    def step(self, actions):
        rewards = []
        for i, action in enumerate(actions):
            self._step_agent(self.agents[i], action)

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
        for i in tuple(colision):
            self._rewind_agent(self.agents[i])

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

    def render(self):
        pass

    def close(self):
        pass