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

from src.actor_critic_arch.model.ddpg_model import *
from src.lib.utils.replay_buffer import *


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class DDPG(object):
    def __init__(self, state_dim, action_dim, params):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = params['hidden_dim']
        self.gamma = params['gamma']
        self.value_lr = params['value_lr']
        self.policy_lr = params['policy_lr']
        self.soft_tau = params['soft_tau']
        self.batch_size = params['batch_size']
        self.min_value = -np.inf
        self.max_value = np.inf

        self.value_network = ValueNetwork(
            self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.policy_network = PolicyNetwork(
            self.state_dim, self.action_dim, self.hidden_dim).to(device)

        self.target_value_network = ValueNetwork(
            self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.target_policy_network = PolicyNetwork(
            self.state_dim, self.action_dim, self.hidden_dim).to(device)

        for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(
            self.value_network.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(), lr=self.policy_lr)

        self.value_criterion = nn.MSELoss()

        self.replay_buffer_size = params['replay_buffer_size']
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def ddpg_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        policy_loss = self.value_network(state, self.policy_network(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_network(next_state)
        target_value = self.target_value_network(
            next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(
            expected_value, self.min_value, self.max_value)

        value = self.value_network(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    # Save model parameters
    def save_model(self, actor_model_name=None, critic_model_name=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        abs_path = os.path.abspath(os.getcwd()) + "/"

        actor_path = abs_path + "models/ddpg_actor"
        if actor_model_name is not None:
            actor_path = abs_path + "models/" + actor_model_name

        critic_path = abs_path + "models/ddpg_critic"
        if critic_model_name is not None:
            critic_path = abs_path + "models/" + critic_model_name
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy_network.state_dict(), actor_path)
        torch.save(self.value_network.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        abs_path = os.path.abspath(os.getcwd()) + "/"
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            actor_path = abs_path + "models/" + actor_path
            self.policy_network.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            critic_path = abs_path + "models/" + critic_path
            self.value_network.load_state_dict(torch.load(critic_path))
