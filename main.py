import threading
from tkinter import Tk
import numpy as np
from gui import GUI
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
# from make_env import make_env
from environment import MAACEnv
import time

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


class Main:
    PRINT_INTERVAL = 100
    N_GAMES = 50000
    MAX_STEPS = 200
    
    def __init__(self):
        self.env = MAACEnv.import_last_env()
        if self.env is None:
            self.env = MAACEnv()
    
        # configs
        self.evaluate = False
        self.load_chkpt = True
        self.force_render = False
        
        # subroutine control (GUI is main thread)
        self.force_stop = False
        self.game_progress = 0
        self.game_render_period = 100 # render 1 whole game per 100 games
        
    def prepare(self):
        scenario = '{}_agent_{}_by_{}'.format(self.env.n_agent, self.env.n_row, self.env.n_col)
        print('preparing scenario:', scenario)
        
        self.total_steps = 0
        self.score_history = []
        self.best_score = -np.inf
        self.fastest_solve = np.inf
        
        self.n_agents = self.env.n
        actor_dims = []
        for i in range(self.n_agents):
            actor_dims.append(self.env.observation_space[i].shape[0])
        critic_dims = sum(actor_dims)

        # action space is a list of arrays, assume each agent has same action space
        self.n_actions = self.env.action_space[0].n
        print(self.n_agents, actor_dims, critic_dims, self.n_actions)        
        self.maddpg_agents = MADDPG(actor_dims, critic_dims, self.n_agents, self.n_actions, 
                                    fc1=64, fc2=64,  
                                    alpha=0.01, beta=0.01, scenario=scenario,
                                    chkpt_dir='.\\tmp\\maddpg\\')

        self.memory = MultiAgentReplayBuffer(
            1000000, critic_dims, actor_dims, self.n_actions, self.n_agents, 
            batch_size=1024)
        
        print('preparation done')
    
    def run(self):
        print('thread started')
        if self.load_chkpt:
            try:
                self.maddpg_agents.load_checkpoint()
            except FileNotFoundError:
                print('no checkpoint found')
            
        for i in range(self.game_progress, Main.N_GAMES):
            """ episode loop """
            print('episode', i, 'total steps', self.total_steps)
            
            self.game_progress = i
            obs = self.env.reset()
            score = 0
            done = [False]*self.n_agents
            episode_step = 0
            
            while not any(done):
                """ step loop """
                
                if self.force_render or self.evaluate or i % self.game_render_period == 0:
                    self.env.render(visual=True, episode=self.game_progress, **self.env.get_info())
                    # time.sleep(0.1) # to slow down the action for the video
                else:
                    self.env.render(episode=self.game_progress, **self.env.get_info())
                
                # random actions
                # actions = [np.array([np.random.rand() for _ in range(self.n_actions)]) for _ in range(self.n_agents)]
                
                actions = self.maddpg_agents.choose_action(obs)
                obs_, reward, done, info = self.env.step(actions)
                
                print('@@@ step', episode_step, ', info\n', info)

                state = obs_list_to_state_vector(obs)
                state_ = obs_list_to_state_vector(obs_)

                if all(done):
                    self.fastest_solve = min(self.fastest_solve, episode_step)

                if episode_step >= Main.MAX_STEPS:
                    done = [True]*self.n_agents

                self.memory.store_transition(obs, state, actions, reward, obs_, state_, done)

                if self.total_steps % 100 == 0 and not self.evaluate:
                    self.maddpg_agents.learn(self.memory)

                obs = obs_

                score += sum(reward)
                self.total_steps += 1
                episode_step += 1
                
                if self.force_stop:
                    break
                
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])
            
            if not self.evaluate:
                if avg_score > self.best_score:
                    self.save_checkpoint()
                    self.best_score = avg_score
                    
            if i % self.PRINT_INTERVAL == 0 and i > 0:
                print('episode', i, 'average score {:.1f}'.format(avg_score))
                
            if self.force_stop:
                self.save_checkpoint()
                self.force_stop = False
                break
            
            self.force_render = False
            
        print('thread finished')

    def save_checkpoint(self):
        self.maddpg_agents.save_checkpoint()

if __name__ == '__main__':
    tk = Tk()
    gui = GUI(tk, Main())
    tk.mainloop()