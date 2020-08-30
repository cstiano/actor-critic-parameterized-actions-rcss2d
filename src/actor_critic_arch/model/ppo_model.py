import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class ActorNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(num_inputs, hidden_size)
        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Linear(hidden_size, 1)


    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)


class CriticNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        state_value = self.v_head(x)
        return state_value