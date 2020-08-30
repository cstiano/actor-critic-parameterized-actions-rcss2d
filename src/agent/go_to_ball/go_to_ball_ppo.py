import datetime
import itertools
import logging
import os
import pickle
import argparse

import hfo
import numpy as np
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from src.lib.hfo_env import HFOEnv
from src.lib.utils.action_selector import *
from src.lib.utils.reward_selector import *
from src.lib.utils.action_selector import GO_TO_BALL_ACTION
from src.lib.utils.reward_selector import GO_TO_BALL_REWARD
from src.lib.utils.state_selector import BALL_AXIS_POSITION_SPACE
from src.lib.utils.hyperparameters import PARAMS
from src.actor_critic_arch.baseline_rlad_ppo import PPO

parse = argparse.ArgumentParser(
    description='Agent Args', formatter_class=argparse.RawTextHelpFormatter)
parse.add_argument('--play', dest='play', action='store_true',
                   default=False, help='Agent Playing.')
args = parse.parse_args()

TEAM = 'HELIOS'
PORT = 6000
ENV_ACTIONS = [hfo.DASH]
ENV_REWARDS = [0]
ACTOR_MODEL_NAME = "ppo_actor_go_to_ball"
CRITIC_MODEL_NAME = "ppo_critic_go_to_ball"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

hfo_env = HFOEnv(ENV_ACTIONS, ENV_REWARDS, is_offensive=True, strict=True,
                 continuous=True, team=TEAM, port=PORT,
                 selected_action=GO_TO_BALL_ACTION, selected_reward=GO_TO_BALL_REWARD,
                 selected_state=BALL_AXIS_POSITION_SPACE)
unum = hfo_env.getUnum()
params = PARAMS['ppo']
ppo = PPO(
    hfo_env.observation_space.shape[0], hfo_env.action_space.shape[0], params)


def train():
    writer = SummaryWriter(
        'logs/{}_PPO_GO_TO_BALL'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    
    try:
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            state = hfo_env.get_state()
            episode_reward = 0
            step = 0

            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []
            entropy = 0

            while status == hfo.IN_GAME:
                state = torch.FloatTensor(state).to(device)
                dist, value = ppo.model(state)
                action = dist.sample()
                
                next_state, reward, done, status = hfo_env.step(action.cpu().numpy())

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
                states.append(state)
                actions.append(action)

                state = next_state
                episode_reward += reward
                step += 1

                if done:
                    break
            
            next_state = torch.FloatTensor(state).to(device)
            _, next_value = ppo.model(state)
            returns = ppo.compute_gae(next_value, rewards, masks, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values
            
            ppo.ppo_update(states, actions, log_probs, returns, advantage)

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
