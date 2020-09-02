import datetime
import itertools
import logging
import os
import pickle
import argparse
from collections import namedtuple

import hfo
import numpy as np
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from src.lib.hfo_env import HFOEnv
from src.lib.utils.action_selector import *
from src.lib.utils.reward_selector import *
from src.lib.utils.action_selector import DASH_ACTION
from src.lib.utils.reward_selector import GO_TO_BALL_REWARD
from src.lib.utils.state_selector import BALL_AXIS_POSITION_SPACE
from src.lib.utils.hyperparameters import PARAMS
from src.actor_critic_arch.baseline_sweetice_ppo import PPO

parse = argparse.ArgumentParser(
    description='Agent Args', formatter_class=argparse.RawTextHelpFormatter)
parse.add_argument('--play', dest='play', action='store_true',
                   default=False, help='Agent Playing.')
args = parse.parse_args()

TEAM = 'HELIOS'
PORT = 6000
ACTOR_MODEL_NAME = "ppo_actor_go_to_ball"
CRITIC_MODEL_NAME = "ppo_critic_go_to_ball"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

hfo_env = HFOEnv(is_offensive=True, strict=True,
                 continuous=True, team=TEAM, port=PORT,
                 selected_action=DASH_ACTION, selected_reward=GO_TO_BALL_REWARD,
                 selected_state=BALL_AXIS_POSITION_SPACE)
unum = hfo_env.getUnum()
params = PARAMS['ppo']
ppo = PPO(
    hfo_env.observation_space.shape[0], hfo_env.action_space.shape[0], params)


def train():
    writer = SummaryWriter(
        'logs/{}_PPO_GO_TO_BALL'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])

    try:
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            state = hfo_env.get_state()
            episode_reward = 0
            step = 0

            while status == hfo.IN_GAME:
                action, action_prob = ppo.select_action(state)
                next_state, reward, done, status = hfo_env.step([action])

                if ppo.store(Transition(state, action, action_prob, reward, next_state)):
                    ppo.update()

                state = next_state
                episode_reward += reward
                step += 1

                if done:
                    break

            if (episode % params['saving_cycle']) == 0:
                ppo.save_model(ACTOR_MODEL_NAME, CRITIC_MODEL_NAME)
            writer.add_scalar(
                f'Rewards/epi_reward_{unum}', episode_reward, global_step=episode)
        writer.close()
    except KeyboardInterrupt:
        ppo.save_model(ACTOR_MODEL_NAME, CRITIC_MODEL_NAME)
        writer.close()


def play():
    ppo.load_model(ACTOR_MODEL_NAME, CRITIC_MODEL_NAME)
    for episode in itertools.count():
        status = hfo.IN_GAME
        done = True
        state = hfo_env.get_state()

        while status == hfo.IN_GAME:
            # TODO
            next_state, reward, done, status = hfo_env.step([0.0])

            state = next_state

            if done:
                break


if __name__ == '__main__':
    if args.play:
        play()
    else:
        train()
