class Params:
  def __init__(self, action_space):
    self.replay_buffer_size = 1000000
    self.hidden_dim = 256
    self.gamma = 0.99
    self.value_lr = 1e-3
    self.policy_lr = 1e-4
    self.soft_tau = 1e-2
    self.batch_size = 128
    self.action_space = action_space