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

from src.actor_critic_arch.model.ppo_model import *


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class PPO(object):
    def __ini__(self, state_dim, action_dim, params):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = params['hidden_dim']
        self.lr = params['lr']
        self.mini_batch_size = params['mini_batch_size']
        self.ppo_epochs = params['ppo_epochs']
        self.clip_param = params['clip_param']
        self.gamma = params['gamma']
        self.tau = params['tau']

        self.model = ActorCritic(
            state_dim, action_dim, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def ppo_update(self, states, actions, log_probs, returns, advantages):
        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def compute_gae(self, next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * \
                values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    # Save model parameters
    def save_model(self, actor_model_name=None, critic_model_name=None):
        pass

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        pass
