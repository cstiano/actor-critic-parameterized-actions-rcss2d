import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.action_head = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value
