import numpy as np
import torch as T
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, n_actions, n_agents, agent_idx, input_dim,
                    conv1_channel, conv2_channel, fc1_dims, fc2_dims,
                    alpha, beta, gamma, tau, chkpt_dir):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(
            alpha, input_dim, conv1_channel, conv2_channel, fc1_dims, fc2_dims, 
            n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_actor')
        self.critic = CriticNetwork(
            beta, input_dim, conv1_channel, conv2_channel, fc1_dims, fc2_dims, 
            n_agents, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(
            alpha, input_dim, conv1_channel, conv2_channel, fc1_dims, fc2_dims, 
            n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(
            beta, input_dim, conv1_channel, conv2_channel, fc1_dims, fc2_dims, 
            n_agents, n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, obstacle, self_, other, dirty, noise=0.2):
        obstacle = obstacle.reshape((1, *obstacle.shape))
        self_ = self_.reshape((1, *self_.shape))
        other = other.reshape((1, *other.shape))
        dirty = dirty.reshape((1, *dirty.shape))
        
        device = self.actor.device
        
        obstacle = T.tensor(obstacle, dtype=T.float).to(device)
        self_ = T.tensor(self_, dtype=T.float).to(device)
        other = T.tensor(other, dtype=T.float).to(device)
        dirty = T.tensor(dirty, dtype=T.float).to(device)
        
        actions = self.actor.forward(obstacle, self_, other, dirty)
        if noise is not None:
            noise = (T.rand(self.n_actions) * noise).to(device)
            actions = actions + noise
        return actions.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
