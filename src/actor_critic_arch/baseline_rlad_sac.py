# TODO reference the original repo of the implementation
import math
import random
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from src.actor_critic_arch.model.sac_model import *
from src.lib.utils.replay_buffer import *


class SAC(object):
    def __init__(self, state_dim, action_dim, params):
        self.hidden_dim = params.hidden_dim
        self.batch_size = params.batch_size
        self.gamma = params.gamma
        self.mean_lambda = params.mean_lambda
        self.std_lambda = params.std_lambda
        self.z_lambda = params.z_lambda
        self.soft_tau = params.soft_tau

        self.value_network = ValueNetwork(
            state_dim, self.hidden_dim).to(device)
        self.target_value_network = ValueNetwork(
            state_dim, self.hidden_dim).to(device)

        self.soft_q_network = SoftQNetwork(
            state_dim, action_dim, self.hidden_dim).to(device)
        self.policy_network = PolicyNetwork(
            state_dim, action_dim, self.hidden_dim).to(device)

        for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        self.value_lr = params.value_lr
        self.soft_q_lr = params.soft_q_lr
        self.policy_lr = params.policy_lr

        self.value_optimizer = optim.Adam(
            self.value_network.parameters(), lr=self.value_lr)
        self.soft_q_optimizer = optim.Adam(
            self.soft_q_network.parameters(), lr=self.soft_q_lr)
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(), lr=self.policy_lr)

        self.replay_buffer_size = params.replay_buffer_size
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def soft_q_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        expected_q_value = self.soft_q_network(state, action)
        expected_value = self.value_network(state)
        new_action, log_prob, z, mean, log_std = self.policy_network.evaluate(
            state)

        target_value = self.target_value_network(next_state)
        next_q_value = reward + (1 - done) * self.gamma * target_value
        q_value_loss = self.soft_q_criterion(
            expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_network(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = self.mean_lambda * mean.pow(2).mean()
        std_loss = self.std_lambda * log_std.pow(2).mean()
        z_loss = self.z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
    
     # Save model parameters
    def save_model(self, actor_model_name=None, critic_model_name=None, soft_model_name=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        abs_path = os.path.abspath(os.getcwd()) + "/"
        
        actor_path = abs_path + "models/sac_actor"
        if actor_model_name is not None:
            actor_path = abs_path + "models/" + actor_model_name

        critic_path = abs_path + "models/sac_critic"
        if critic_model_name is not None:
            critic_path = abs_path + "models/" + critic_model_name

        soft_path = abs_path + "models/sac_soft"
        if soft_model_name is not None:
            soft_path = abs_path + "models/" + critic_model_name
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy_network.state_dict(), actor_path)
        torch.save(self.value_network.state_dict(), critic_path)
        torch.save(self.soft_q_network.state_dict(), soft_path)
        

    # Load model parameters
    def load_model(self, actor_path=None, critic_path=None, soft_path=None):
        abs_path = os.path.abspath(os.getcwd()) + "/"
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            actor_path = abs_path + "models/" + actor_path
            self.policy_network.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            critic_path = abs_path + "models/" + critic_path
            self.value_network.load_state_dict(torch.load(critic_path))
        if soft_path is not None:
            soft_path = abs_path + "models/" + soft_path
            self.soft_q_network.load_state_dict(torch.load(soft_path))
