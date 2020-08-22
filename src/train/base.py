import datetime
import itertools
import logging
import os
import pickle

import hfo
import numpy as np
from src.lib.hfo_env import HFOEnv


team = 'HELIOS'
port = 6000
ENV_ACTIONS = [hfo.DRIBBLE_TO]
ENV_REWARDS = [0]
hfo_env = HFOEnv(ENV_ACTIONS, ENV_REWARDS, is_offensive=True, strict=True, continuous=True, team=team, port=port)

print("Started")

for episode in itertools.count():
    status = hfo.IN_GAME
    done = True
    state = hfo_env.get_state()
    
    while status == hfo.IN_GAME:
        next_state, reward, done, status = hfo_env.step([-0.6])

        if status == SERVER_DOWN:
            exit()
        if done:
            break

