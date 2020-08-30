# TODO reference the original repo of the implementation
import math
import random
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from src.actor_critic_arch.model.ppo_model import *


class PPO(object):
    def __init__(self, state_dim, action_dim, params):
        super(PPO, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = params['hidden_dim']
        self.lr = params['lr']
        self.batch_size = params['batch_size']
        self.buffer_capacity = params['buffer_capacity']
        self.ppo_update_time = params['ppo_update_time']
        self.clip_param = params['clip_param']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.max_grad_norm = params['max_grad_norm']
        self.ppo_epoch = params['ppo_epoch']

        self.training_step = 0
        self.anet = ActorNet(
            self.state_dim, self.action_dim, self.hidden_dim).float()
        self.cnet = CriticNet(
            self.state_dim, self.action_dim, self.hidden_dim).float()
        self.buffer = []
        self.counter = 0

        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=self.lr)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=self.lr)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.anet(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-1.0, 1.0)
        return action.item(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            state_value = self.cnet(state)
        return state_value.item()

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1

        s = torch.tensor([t.s for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer],
                         dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in self.buffer],
                         dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in self.buffer], dtype=torch.float)

        old_action_log_probs = torch.tensor(
            [t.a_log_p for t in self.buffer], dtype=torch.float).view(-1, 1)

        r = (r - r.mean()) / (r.std() + 1e-5)
        with torch.no_grad():
            target_v = r + self.gamma * self.cnet(s_)

        adv = (target_v - self.cnet(s)).detach()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                (mu, sigma) = self.anet(s[index])
                dist = Normal(mu, sigma)
                action_log_probs = dist.log_prob(a[index])
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()

                self.optimizer_a.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.anet.parameters(), self.max_grad_norm)
                self.optimizer_a.step()

                value_loss = F.smooth_l1_loss(
                    self.cnet(s[index]), target_v[index])
                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.cnet.parameters(), self.max_grad_norm)
                self.optimizer_c.step()

        del self.buffer[:]

    # Save model parameters
    def save_model(self, actor_model_name=None, critic_model_name=None):
        pass

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        pass
