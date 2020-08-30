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

from src.actor_critic_arch.model.td3_model import *
from src.lib.utils.replay_buffer import *


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class TD3(object):
    def __init__(self, state_dim, action_dim, params):
        super().__init__()
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.soft_tau = params['soft_tau']
        self.noise_std = params['noise_std']
        self.noise_clip = params['noise_clip']
        self.policy_update = params['policy_update']
        self.hidden_dim = params['hidden_dim']

        self.value_network_1 = ValueNetwork(
            state_dim, action_dim, self.hidden_dim).to(device)
        self.value_network_2 = ValueNetwork(
            state_dim, action_dim, self.hidden_dim).to(device)
        self.policy_network = PolicyNetwork(
            state_dim, action_dim, self.hidden_dim).to(device)

        self.target_value_network_1 = ValueNetwork(
            state_dim, action_dim, self.hidden_dim).to(device)
        self.target_value_network_2 = ValueNetwork(
            state_dim, action_dim, self.hidden_dim).to(device)
        self.target_policy_network = PolicyNetwork(
            state_dim, action_dim, self.hidden_dim).to(device)

        self.soft_update(self.value_network_1, self.target_value_network_1, soft_tau=self.soft_tau)
        self.soft_update(self.value_network_2, self.target_value_network_2, soft_tau=self.soft_tau)
        self.soft_update(self.policy_network, self.target_policy_network, soft_tau=self.soft_tau)

        self.value_criterion = nn.MSELoss()

        self.policy_lr = params['policy_lr']
        self.value_lr = params['value_lr']

        self.value_optimizer1 = optim.Adam(self.value_network_1.parameters(), lr=self.value_lr)
        self.value_optimizer2 = optim.Adam(self.value_network_2.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.policy_lr)

        self.replay_buffer = ReplayBuffer(params['replay_buffer_size'])

    def td3_update(self, step):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        next_action = self.target_policy_network(next_state)
        noise = torch.normal(torch.zeros(
            next_action.size()), self.noise_std).to(device)
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        next_action += noise

        target_q_value1 = self.target_value_network_1(next_state, next_action)
        target_q_value2 = self.target_value_network_2(next_state, next_action)
        target_q_value = torch.min(target_q_value1, target_q_value2)
        expected_q_value = reward + (1.0 - done) * self.gamma * target_q_value

        q_value1 = self.value_network_1(state, action)
        q_value2 = self.value_network_2(state, action)

        value_loss1 = self.value_criterion(q_value1, expected_q_value.detach())
        value_loss2 = self.value_criterion(q_value2, expected_q_value.detach())

        self.value_optimizer1.zero_grad()
        value_loss1.backward()
        self.value_optimizer1.step()

        self.value_optimizer2.zero_grad()
        value_loss2.backward()
        self.value_optimizer2.step()

        if step % self.policy_update == 0:
            policy_loss = self.value_network_1(state, self.policy_network(state))
            policy_loss = -policy_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.soft_update(self.value_network_1, self.target_value_network_1, soft_tau=self.soft_tau)
            self.soft_update(self.value_network_2, self.target_value_network_2, soft_tau=self.soft_tau)
            self.soft_update(self.policy_network, self.target_policy_network, soft_tau=self.soft_tau)

    def soft_update(self, network, target_network, soft_tau=1e-2):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
    
    # Save model parameters
    def save_model(self, actor_model_name=None, critic_model_name=None):
        pass

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        pass
