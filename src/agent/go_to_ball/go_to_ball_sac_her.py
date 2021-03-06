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
from src.lib.utils.action_selector import DASH_ACTION
from src.lib.utils.reward_selector import GO_TO_BALL_REWARD
from src.lib.utils.state_selector import BALL_AXIS_POSITION_SPACE
from src.lib.utils.hyperparameters import PARAMS
from src.actor_critic_arch.baseline_rlad_sac import SAC
from src.lib.utils.replay_buffer import *


parse = argparse.ArgumentParser(
    description='Agent Args', formatter_class=argparse.RawTextHelpFormatter)
parse.add_argument('--play', dest='play', action='store_true',
                   default=False, help='Agent Playing.')
args = parse.parse_args()

args = parse.parse_args()
TEAM = 'HELIOS'
PORT = 6000
ACTOR_MODEL_NAME = "sac_actor_go_to_ball"
CRITIC_MODEL_NAME = "sac_critic_go_to_ball"
SOFT_MODEL_NAME = "sac_soft_go_to_ball"
ENABLE_LOSS_WRITE = False

hfo_env = HFOEnv(is_offensive=True, strict=True,
                 continuous=True, team=TEAM, port=PORT,
                 selected_action=DASH_ACTION, selected_reward=GO_TO_BALL_REWARD,
                 selected_state=BALL_AXIS_POSITION_SPACE)
unum = hfo_env.getUnum()
params = PARAMS['sac']
sac = SAC(hfo_env.observation_space.shape[0],
          hfo_env.action_space.shape[0], params)
replay_buffer = ReplayGMemory(params['replay_buffer_size'])


def train():
    writer = SummaryWriter(
        'logs/{}_SAC_GO_TO_BALL'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    frame_idx = 0
    
    try:
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            state = hfo_env.reset()
            episode_reward = 0
            step = 0

            episode_acumulated = []
            goal = [0.0, 0.0]

            while status == hfo.IN_GAME:
                action = sac.policy_network.get_action(
                    np.concatenate([state, goal]))
                action = action.astype(np.float32)
                next_state, reward, done, status = hfo_env.step([action])

                replay_buffer.push(state, action, reward,
                                   next_state, float(not done), goal)

                if len(replay_buffer) > params['batch_size']:
                    q_value_loss, value_loss, policy_loss = sac.soft_q_update(replay_buffer)
                    
                    if ENABLE_LOSS_WRITE:
                        writer.add_scalar(f'Q_Value_Loss', q_value_loss, frame_idx)
                        writer.add_scalar(f'Value_Loss', value_loss, frame_idx)
                        writer.add_scalar(f'Policy_Loss', policy_loss, frame_idx)
                        frame_idx += 1 

                next_her_goal = hfo_env.get_goal()
                her_goal = next_her_goal
                state = next_state
                episode_reward += reward
                step += 1

                episode_acumulated.append((state, action, reward, float(
                    not done), next_state, goal, her_goal, next_her_goal))

                if done:
                    break

            new_goals = 5
            for i, (state, action, reward, done, next_state, goal, her_goal, next_her_goal) in enumerate(episode_acumulated):
                for t in np.random.choice(len(episode_acumulated), new_goals):
                    try:
                        episode_acumulated[t]
                    except:
                        continue
                    new_goal = episode_acumulated[t][-1]
                    reward = hfo_env.compute_reward(next_her_goal, new_goal)
                    replay_buffer.push(state, action, reward, next_state, done, new_goal)

            if (episode % params['saving_cycle']) == 0:
                sac.save_model(ACTOR_MODEL_NAME,
                               CRITIC_MODEL_NAME, SOFT_MODEL_NAME)
            writer.add_scalar(
                f'Rewards/epi_reward', episode_reward, global_step=episode)

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
