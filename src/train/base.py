import datetime
import itertools
import logging
import os
import pickle

import hfo
import numpy as np
from src.lib.hfo_env import HFOEnv


team = 'helios'
port = 6000
env_actions = [hfo.DRIBBLE_TO]
env_rewards = [0]
# is_offensive=False -> is_offensive=True
hfo_env = HFOEnv(env_actions, env_rewards, is_offensive=True, strict=True, continuous=True, team=team, port=port)
test = False
gen_mem = True
unum = hfo_env.getUnum()


frame_idx = 1
goals = 0

ou_noise = OUNoise(self.hfo_env.action_space)

max_frames  = 12000
max_steps   = 500
frame_idx   = 0
rewards     = []
batch_size  = 128

writer = SummaryWriter(comment="-" + "DDPG-test")

for episode in itertools.count():
    status = hfo.IN_GAME
    done = True
    # episode_rewards = 0

    ou_noise.reset()
    episode_reward = 0
    step = 0
    state = hfo_env.get_state()
    
    while status == hfo.IN_GAME:
        action = policy_net.get_action(state)
        action = ou_noise.get_action(action, step)
        
        next_state, reward, done, status = hfo_env.step(action)

        replay_buffer.push(state, action, reward, next_state, done)
        if len(replay_buffer) > batch_size:
            ddpg_update(batch_size)

        state = next_state
        episode_reward += reward
        frame_idx += 1

        step += 1
        if status == hfo.GOAL:
            goals += 1
        if done:
            writer.add_scalar(f'Rewards/epi_reward_{unum}', episode_reward, global_step=episode)
            print("Aqui")
            break
    rewards.append(episode_reward)
writer.close()

