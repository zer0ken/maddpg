import tkinter as tk
import numpy as np
from gui import GUI
from maddpg import MADDPG
from buffer import PERMA
import gc


class Main:
    PRINT_INTERVAL = 100
    N_GAMES = 50000
    MAX_STEPS = 600
    
    def __init__(self):
        self.env = None # need to be set by GUI
        
        # configs
        self.evaluate = False
        self.load_chkpt = True
        self.force_render = False
        self.step_per_learn = 50 # learn 1 batch per 50 steps
        self.episode_per_gc = 10 # collect garbage per 100 episodes
        
        # subroutine control (GUI is main thread)
        self.force_stop = False
        self.game_progress = 0
        self.episode_per_render = 50 # render 1 whole game per 100 games
        
    def prepare(self):
        scenario = '{}_agent_{}_by_{}'.format(self.env.n_agent, self.env.n_row, self.env.n_col)
        print('preparing scenario:', scenario)
        
        self.game_progress = 0
        
        self.total_steps = 0
        self.score_history = []
        self.best_score = -np.inf
        self.fastest_solve = np.inf
        
        self.n_agents = self.env.n
        
        input_dim = (self.env.n_row, self.env.n_col)

        # action space is a list of arrays, assume each agent has same action space
        self.n_actions = self.env.action_space[0].n
        self.maddpg_agents = MADDPG(self.n_agents, self.n_actions, input_dim=input_dim,
                                    conv1_channel=6, conv2_channel=9, fc1_dims=32, fc2_dims=64,
                                    scenario=scenario, chkpt_dir='.\\tmp\\maddpg\\')

        self.memory = PERMA(
            40000, input_dim, self.n_actions, self.n_agents, 
            batch_size=512)
        
        self.env.reset()
        
        self.force_stop = False
        
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
            
            self.game_progress = i
            # obs = self.env.reset(keep_agent=i != 0 and not self.evaluate)
            obs = self.env.reset(keep_agent=False)
            score = 0
            done = [False]*self.n_agents
            episode_step = 0
                        
            while not any(done):
                """ step loop """
                
                actions = self.maddpg_agents.choose_action(obs, noise=None if self.evaluate else 0.35)
                obs_, reward, done, info = self.env.step(actions)

                if all(done):
                    self.fastest_solve = min(self.fastest_solve, episode_step + 1)

                if episode_step >= Main.MAX_STEPS:
                    done = [True]*self.n_agents

                self.memory.store_transition(obs, actions, reward, obs_, done)

                if self.total_steps % self.step_per_learn == 0 and not self.evaluate:
                    print('\tlearning...', self.total_steps)
                    self.maddpg_agents.learn(self.memory)

                obs = obs_

                score += sum(reward)
                self.total_steps += 1
                episode_step += 1
                
                if self.force_stop:
                    print('force stop')
                    break
                
                if self.force_render or self.evaluate or i % self.episode_per_render == 0:
                    self.env.render(visual=True, episodes=self.game_progress, 
                                    fastest_solve=self.fastest_solve, **self.env.get_info())
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])
            
            if not self.evaluate:
                if avg_score > self.best_score:
                    self.save_checkpoint()
                    self.best_score = avg_score
                    
            if i % self.PRINT_INTERVAL == 0 and i > 0:
                print('episode', i, 'average score {:.1f}'.format(avg_score))
            
            if i % self.episode_per_gc == 0 and i > 0:
                print('collecting garbage...')
                gc.collect()
                
            self.env.render(episodes=self.game_progress, reset=True, 
                            fastest_solve=self.fastest_solve, **self.env.get_info())
            
            if self.force_stop:
                if not self.evaluate:
                    self.save_checkpoint()
                break
            
        print('thread finished')

    def save_checkpoint(self):
        self.maddpg_agents.save_checkpoint()

if __name__ == '__main__':
    root = tk.Tk()
    main = Main()
    gui = GUI(root, main)
    gui.mainloop()