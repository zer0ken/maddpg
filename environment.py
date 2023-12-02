import numpy as np

class MAACEnv:
    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (0, 0)}

    def __init__(self, n_agent, n_row, n_col,
                 agent_pos=None, dirty_pos=None, obstacle_pos=None):
        self.n_agent = n_agent
        self.n_row = n_row
        self.n_col = n_col

        self.agent_pos = agent_pos
        self.dirty_pos = dirty_pos
        self.obstacle_pos = obstacle_pos

        self.reset()
        self.dirty_layer[self.dirty_layer == self.obstacle_layer] = 0

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