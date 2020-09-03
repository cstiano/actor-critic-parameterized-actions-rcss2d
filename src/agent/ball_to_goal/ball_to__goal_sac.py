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
from src.lib.utils.action_selector import DASH_OR_KICK_ACTION
from src.lib.utils.reward_selector import BALL_PROXIMITY_GOAL_REWARD
from src.lib.utils.state_selector import AGENT_ORIENTATION_AND_BALL_POSITION_SPACE
from src.lib.utils.hyperparameters import PARAMS
from src.actor_critic_arch.baseline_rlad_sac import SAC


parse = argparse.ArgumentParser(
    description='Agent Args', formatter_class=argparse.RawTextHelpFormatter)
parse.add_argument('--play', dest='play', action='store_true',
                   default=False, help='Agent Playing.')
args = parse.parse_args()

args = parse.parse_args()
TEAM = 'HELIOS'
PORT = 6000
ACTOR_MODEL_NAME = "sac_actor_ball_to_goal"
CRITIC_MODEL_NAME = "sac_critic_ball_to_goal"
SOFT_MODEL_NAME = "sac_soft_ball_to_goal"

hfo_env = HFOEnv(is_offensive=True, strict=True,
                 continuous=True, team=TEAM, port=PORT,
                 selected_action=DASH_OR_KICK_ACTION, selected_reward=BALL_PROXIMITY_GOAL_REWARD,
                 selected_state=AGENT_ORIENTATION_AND_BALL_POSITION_SPACE)
unum = hfo_env.getUnum()
params = PARAMS['sac']
sac = SAC(hfo_env.observation_space.shape[0],
          hfo_env.action_space.shape[0], params)


def train():
    writer = SummaryWriter(
        'logs/{}_SAC_BALL_TO_GOAL'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    try:
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            state = hfo_env.get_state()
            episode_reward = 0
            step = 0

            while status == hfo.IN_GAME:
                action = sac.policy_network.get_action(state)
                next_state, reward, done, status = hfo_env.step(action)

                sac.replay_buffer.push(state, action, reward, next_state, done)

                if len(sac.replay_buffer) > params['batch_size']:
                    sac.soft_q_update()

                state = next_state
                episode_reward += reward
                step += 1

                if done:
                    break

            if (episode % params['saving_cycle']) == 0:
                sac.save_model(ACTOR_MODEL_NAME,
                               CRITIC_MODEL_NAME, SOFT_MODEL_NAME)
            writer.add_scalar(
                f'Rewards/epi_reward_{unum}', episode_reward, global_step=episode)

            if status == hfo.SERVER_DOWN:
                hfo_env.act(hfo.QUIT)
                writer.close()
                exit()
    except KeyboardInterrupt:
        sac.save_model(ACTOR_MODEL_NAME, CRITIC_MODEL_NAME, SOFT_MODEL_NAME)
        writer.close()


def play():
    sac.load_model(ACTOR_MODEL_NAME, CRITIC_MODEL_NAME, SOFT_MODEL_NAME)
    for episode in itertools.count():
        status = hfo.IN_GAME
        done = True
        state = hfo_env.get_state()

        while status == hfo.IN_GAME:
            action = sac.policy_network.get_action(state)
            action = action.astype(np.float32)
            next_state, reward, done, status = hfo_env.step([action])

            state = next_state
            if done:
                break
        if status == hfo.SERVER_DOWN:
            hfo_env.act(hfo.QUIT)
            exit()


if __name__ == '__main__':
    if args.play:
        play()
    else:
        train()
