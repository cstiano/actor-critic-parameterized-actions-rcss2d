from actor_critic_arch.ddpg import *
from actor_critic_arch.utils.normalized_actions import NormalizedActions
from actor_critic_arch.utils.hyperparameters import Params
from torch.utils.tensorboard import SummaryWriter

import gym
import datetime

# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# %matplotlib inline

max_frames  = 12000
max_steps   = 100000
frame_idx   = 0
rewards     = []

env = NormalizedActions(gym.make("Pendulum-v0"))
params = Params(env.action_space)

ddpg = DDPG(env.observation_space.shape[0], env.action_space.shape[0], params)
writer = SummaryWriter('runs/{}_DDPG_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, args.policy, "autotune" if args.automatic_entropy_tuning else ""))

i_episode = 0

while frame_idx < max_frames:
    state = env.reset()
    ddpg.ou_noise.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = ddpg.policy_network.get_action(state)
        action = ddpg.ou_noise.get_action(action, step)
        next_state, reward, done, _ = env.step(action)

        ddpg.replay_buffer.push(state, action, reward, next_state, done)
        if len(ddpg.replay_buffer) > params.batch_size:
            ddpg.ddpg_update()

        state = next_state
        episode_reward += reward
        frame_idx += 1
        if frame_idx % max(1000, max_steps + 1) == 0:
            print("plot")

        if done:
            break
    
    episode +=1
    writer.add_scalar('reward/train', episode_reward, i_episode)
    rewards.append(episode_reward)

env.close()