PARAMS = {
  'ddpg':{
    'replay_buffer_size': 1000000,
    'hidden_dim': 256,
    'gamma': 0.99,
    'value_lr': 1e-3,
    'policy_lr': 1e-4,
    'soft_tau': 1e-2,
    'batch_size': 128,
    'saving_cycle': 200,
  },
  'sac':{
    'replay_buffer_size': 1000000,
    'hidden_dim': 256,
    'gamma': 0.99,
    'value_lr': 3e-4,
    'policy_lr': 3e-4,
    'soft_q_lr' : 3e-4,
    'soft_tau': 1e-2,
    'batch_size': 128,
    'saving_cycle': 200,
    'mean_lambda':1e-3,
    'std_lambda':1e-3,
    'z_lambda':0.0
  }
}