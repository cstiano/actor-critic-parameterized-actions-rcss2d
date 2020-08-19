import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ValueNetwork(nn.Module):
  def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
    super(ValueNetwork, self).__init__()

    self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, 1)

    self.linear3.weight.data.uniform_(-init_w, init_w)
    self.linear3.bias.data.uniform_(-init_w, init_w)

  def forward(self, state, action):
    # What cat do? --> torch.cat(tensors, dim=0, out=None) → Tensor
    # Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    x = torch.cat([state, action], 1)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)
    return x

class PolicyNetwork(nn.Module):
  def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
    super(PolicyNetwork, self).__init__()

    self.linear1 = nn.Linear(num_inputs, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, num_actions)

    self.linear3.weight.data.uniform_(-init_w, init_w)
    self.linear3.bias.data.uniform_(-init_w, init_w)
  
  def forward(self, state):
    x = F.relu(self.linear1(state))
    x = F.relu(self.linear2(x))
    x = F.tanh(self.linear3(x))
    return x
  
  def get_action(self, state):
    # Unsqueeze - Returns a new tensor with a dimension of size one inserted at the specified position.
    # >>> x = torch.tensor([1, 2, 3, 4])
    # >>> torch.unsqueeze(x, 0)
    # tensor([[ 1,  2,  3,  4]])
    # >>> torch.unsqueeze(x, 1)
    # tensor([[ 1],
            # [ 2],
            # [ 3],
            # [ 4]])
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = self.forward(state)
    return action.detach().cpu().numpy()[0,0]