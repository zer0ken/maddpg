from threading import Thread
import torch as T
import torch.nn.functional as F
from agent import Agent
from torchviz import make_dot

T.autograd.set_detect_anomaly(True)
T.set_default_dtype(T.float32)

class MADDPG:
    def __init__(self, n_agents, n_actions, input_dim=(10, 10), 
                 conv1_channel=16, conv2_channel=32, fc1_dims=32, fc2_dims=64,
                 scenario='simple', alpha=0.05, beta=0.02,
                 gamma=0.99, tau=0.05, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(n_actions, n_agents, agent_idx, input_dim,
                                     conv1_channel, conv2_channel, fc1_dims, fc2_dims,
                                     alpha, beta, gamma, tau, chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, obs, noise=True):
        actions = {}
        
        for agent_idx, agent in enumerate(self.agents):
            agent_obs = obs[agent_idx]
            action = agent.choose_action(
                agent_obs.obstacle, agent_obs.self_,
                agent_obs.other, agent_obs.dirty, noise=noise
            )
            actions[agent_idx] = action
            
        actions = [actions[i] for i in range(self.n_agents)]
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        obstacles, selves, others, dirties, \
            actions, rewards, \
            new_obstacles, new_selves, new_others, new_dirties, \
            dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        obstacles = T.tensor(obstacles).to(device)
        selves = T.tensor(selves).to(device)
        others = T.tensor(others).to(device)
        dirties = T.tensor(dirties).to(device)
        
        new_obstacles = T.tensor(new_obstacles).to(device)
        new_selves = T.tensor(new_selves).to(device)
        new_others = T.tensor(new_others).to(device)
        new_dirties = T.tensor(new_dirties).to(device)
        
        actions = T.tensor(actions).to(device)
        rewards = T.tensor(rewards).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = {}
        all_agents_new_mu_actions = {}
        old_agents_actions = {}
        
        for agent_idx, agent in enumerate(self.agents):
            new_obstacles_ = new_obstacles[:, agent_idx].to(device)
            new_selves_ = new_selves[:, agent_idx].to(device)
            new_others_ = new_others[:, agent_idx].to(device)
            new_dirties_ = new_dirties[:, agent_idx].to(device)
            
            new_pi = agent.target_actor.forward(new_obstacles_, new_selves_, new_others_, new_dirties_)
            all_agents_new_actions[agent_idx] = new_pi
            
            mu_obstacles_ = obstacles[:, agent_idx].to(device)
            mu_selves_ = selves[:, agent_idx].to(device)
            mu_others_ = others[:, agent_idx].to(device)
            mu_dirties_ = dirties[:, agent_idx].to(device)
            
            pi = agent.actor.forward(mu_obstacles_, mu_selves_, mu_others_, mu_dirties_)
            
            all_agents_new_mu_actions[agent_idx] = pi
            old_agents_actions[agent_idx] = actions[agent_idx]
            
        all_agents_new_actions = [all_agents_new_actions[i] for i in range(self.n_agents)]
        all_agents_new_mu_actions = [all_agents_new_mu_actions[i] for i in range(self.n_agents)]
        old_agents_actions = [old_agents_actions[i] for i in range(self.n_agents)]

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(
                new_obstacles.detach().clone(), 
                new_selves.detach().clone(), 
                new_others.detach().clone(), 
                new_dirties.detach().clone(), 
                new_actions.detach().clone()).flatten().detach().clone()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(
                obstacles.detach().clone(), 
                selves.detach().clone(), 
                others.detach().clone(), 
                dirties.detach().clone(), 
                old_actions.detach().clone()).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(
                obstacles.detach().clone(), 
                selves.detach().clone(),
                others.detach().clone(),
                dirties.detach().clone(),
                mu).flatten()
            actor_loss = -T.mean(actor_loss)
            
            agent.critic.requires_grad_(False)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            
            print(agent_idx, 'critic_loss:', critic_loss.item())
            print(agent_idx, 'actor_loss:', actor_loss.item())
            
        for agent_idx, agent in enumerate(self.agents):
            
            with T.no_grad():
                mu_obstacles_ = obstacles[:, agent_idx].to(device)
                mu_selves_ = selves[:, agent_idx].to(device)
                mu_others_ = others[:, agent_idx].to(device)
                mu_dirties_ = dirties[:, agent_idx].to(device)
                
                pi = agent.actor.forward(mu_obstacles_, mu_selves_, mu_others_, mu_dirties_)
            
            print(agent_idx, 'pi:', pi)
            agent.actor.optimizer.step()
            
            with T.no_grad():
                pi = agent.actor.forward(mu_obstacles_, mu_selves_, mu_others_, mu_dirties_)
            print(agent_idx, '>>>', pi)
            agent.update_network_parameters()
            
        for agent_idx, agent in enumerate(self.agents):
            agent.critic.requires_grad_(True)
            