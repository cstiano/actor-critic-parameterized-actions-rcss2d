PARAMS = {
  'ddpg':{
    'replay_buffer_size': 1000000,
    'hidden_dim': 256,
    'gamma': 0.99,
    'value_lr': 3e-4,
    'policy_lr': 3e-4,
    'soft_tau': 1e-2,
    'batch_size': 64,
    'saving_cycle': 200,
    'noise_factor': 0.1,
    'random_steps': 1000
  },
  'sac':{
    'replay_buffer_size': 1000000,
    'hidden_dim': 256,
    'gamma': 0.99,
    'value_lr': 3e-4,
    'policy_lr': 3e-4,
    'soft_q_lr' : 3e-4,
    'soft_tau': 1e-2,
    'batch_size': 64,
    'saving_cycle': 200,
    'mean_lambda':1e-3,
    'std_lambda':1e-3,
    'z_lambda':0.0,
    'random_steps': 1000
  },
  'td3':{
    'replay_buffer_size': 1000000,
    'hidden_dim': 256,
    'gamma': 0.99,
    'value_lr': 3e-4,
    'policy_lr': 3e-4,
    'soft_tau': 1e-2,
    'batch_size': 64,
    'saving_cycle': 200,
    'noise_std': 0.2,
    'noise_clip': 0.5,
    'policy_update': 2,
    'random_steps': 1000
  },
  'ppo':{
    'hidden_dim': 256,
    'gamma': 0.99,
    'lr': 3e-4,
    'tau': 1e-2,
    'mini_batch_size': 5,
    'saving_cycle': 200,
    'ppo_epochs': 4,
    'clip_param': 0.2,
    'max_grad_norm' : 0.5,
    'ppo_update_time': 10,
    'buffer_capacity': 1000,
    'batch_size': 32,
    'ppo_epoch': 10,
    'random_steps': 1000
  }
}