import datetime
import itertools
import logging
import os
import pickle
import argparse

import hfo
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.lib.hfo_env import HFOEnv
from src.lib.utils.action_selector import *
from src.lib.utils.reward_selector import *
from src.lib.utils.action_selector import CONDITIONAL_DASH_OR_KICK_ACTION
from src.lib.utils.reward_selector import AGENT_AND_BALL_POTENCIAL_REWARD
from src.lib.utils.state_selector import AGENT_ORIENTATION_AND_BALL_POSITION_SPACE
from src.lib.utils.hyperparameters import PARAMS
from src.lib.utils.gaussian_exploration import GaussianExploration
from src.lib.utils.ounoise import OUNoise
from src.actor_critic_arch.baseline_rlad_td3 import TD3
from src.lib.utils.replay_buffer import *

parse = argparse.ArgumentParser(
    description='Agent Args', formatter_class=argparse.RawTextHelpFormatter)
parse.add_argument('--play', dest='play', action='store_true',
                   default=False, help='Agent Playing.')
args = parse.parse_args()

TEAM = 'HELIOS'
PORT = 6000
ACTOR_MODEL_NAME = "td3_actor_ball_to_goal"
CRITIC_MODEL_NAME = "td3_critic_ball_to_goal"
ENABLE_LOSS_WRITE = False

hfo_env = HFOEnv(is_offensive=True, strict=True,
                 continuous=True, team=TEAM, port=PORT,
                 selected_action=CONDITIONAL_DASH_OR_KICK_ACTION, 
                 selected_reward=AGENT_AND_BALL_POTENCIAL_REWARD,
                 selected_state=AGENT_ORIENTATION_AND_BALL_POSITION_SPACE)
unum = hfo_env.getUnum()
params = PARAMS['td3']
td3 = TD3(
    hfo_env.observation_space.shape[0], hfo_env.action_space.shape[0], params)
replay_buffer = ReplayBuffer(params['replay_buffer_size'])


def train():
    writer = SummaryWriter(
        'logs/{}_TD3_BALL_TO_GOAL'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    noise = OUNoise(hfo_env.action_space)
    frame_idx = 0

    try:
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            state = hfo_env.reset()
            episode_reward = 0
            step = 0

            while status == hfo.IN_GAME:
                action = td3.policy_network.get_action(state)
                action = noise.get_action(action, step)
                next_state, reward, done, status = hfo_env.step(action)

                replay_buffer.push(
                    state, action, reward, next_state, done)

                if len(replay_buffer) > params['batch_size']:
                    value_loss1, value_loss2, policy_loss = td3.td3_update(replay_buffer, step)

                    if ENABLE_LOSS_WRITE:
                        writer.add_scalar(f'Value_Loss1', value_loss1, frame_idx)
                        writer.add_scalar(f'Value_Loss2', value_loss2, frame_idx)
                        writer.add_scalar(f'Policy_Loss', policy_loss, frame_idx)
                        frame_idx += 1 

                state = next_state
                episode_reward += reward
                step += 1

                if done:
                    break

            if (episode % params['saving_cycle']) == 0:
                td3.save_model(ACTOR_MODEL_NAME, CRITIC_MODEL_NAME)
            writer.add_scalar(
                f'Rewards/epi_reward', episode_reward, global_step=episode)

            if status == hfo.SERVER_DOWN:
                hfo_env.act(hfo.QUIT)
                writer.close()
                exit()
    except KeyboardInterrupt:
        td3.save_model(ACTOR_MODEL_NAME, CRITIC_MODEL_NAME)
        writer.close()


def play():
    td3.load_model(ACTOR_MODEL_NAME, CRITIC_MODEL_NAME)
    for episode in itertools.count():
        status = hfo.IN_GAME
        done = True
        state = hfo_env.get_state()

        while status == hfo.IN_GAME:
            action = td3.policy_network.get_action(state)
            action = action.astype(np.float32)
            next_state, reward, done, status = hfo_env.step([action])

            state = next_state

            if done:
                break


if __name__ == '__main__':
    if args.play:
        play()
    else:
        train()
