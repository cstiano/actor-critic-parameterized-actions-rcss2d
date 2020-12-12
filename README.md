# Actor-Critic Architectures in Parameterized Action Space for RoboCup Soccer Simulator 2D


Deep Reinforcement Learning (DRL) is a method that combines Reinforcement Learning and Deep Learning. This method is assigned as a goal-oriented algorithm which learn how to reach a complex objective performing actions and receiving evaluated feedbacks from the environment. DRL is applied to solve problems in a large range of applications in domains such video-games, finance, robotics and many more. These domains are regularly composed by complex environments that contains a large set of features and a continuous action space such the RoboCup Soccer Simulator 2D (RCSS2D), which is an autonomous agent based simulation to mimic soccer games. For this kind of environments with continuous action space is frequently used a sub-field of the DRL: the Actor-Critic algorithms. The objective of this work is to evaluate extended Actor-Critic architectures in a parameterized continuous action space using the RCSS2D environment and also propose reward functions to perform soccer tasks in this environment. The evaluation was made by training state of art Actor-Critic models incrementally for the set of tasks with low complexity to the high complexity, which the complexity was defined by the kind of task that the robot should learning and the configuration of the action that it should perform, taking parameterization configuration as the core of the actions changes for each experiment. The evaluation of reward functions proposed in this work was made by comparing with functions proposed in related works, taking into account the convergence factor and the behavior analysis. 

Read the full paper [here](https://github.com/cstiano/actor-critic-parameterized-actions-rcss2d/blob/master/paper/Actor_Critic_Architectures_in_Parameterized_Action_Space_for_RoboCup_Soccer_Simulator_2D.pdf).

### Project

This project extends the [Half Offensive Field Framework]() (HFO) functionalities including interfaces that make easy to implement new Reinforcement Learning approaches just extending the environment components (e.g: state, action, reward).

# Install

Require *python 3.6*.

```bash
pip install -r requirements.txt
```
# Interfaces

To make a new implementation of the environment components just extend the *Selector* classes with the new method and select in the environment init.

### State Selector

The State Selector uses the original HFO state space below as dependency.

- HFO State Space
    - [0] X position
    - [1] Y position
    - [2] Orientation
    - [3] Ball X
    - [4] Ball Y
    - [5] Able to Kick
    - [6] Goal Center Proximity
    - [7] Goal Center Angle
    - [8] Goal Opening Angle
    - [9] Proximity to Opponent
    - [T] Teammate’s Goal Opening Angle
    - [T] Proximity from Teammate i to Opponent
    - [T] Pass Opening Angle
    - [3T] X, Y, and Uniform Number of Teammates
    - [3O] X, Y, and Uniform Number of Opponents
    - [+1] Last Action Success Possible

With the HFO state space you can create new state spaces by including new methods to the `state_selector` class. As an example, here the inclusion of `AGENT_BALL_SPACE_SPACE`

```python
...
AGENT_BALL_SPACE_SPACE = 2
...

...
AGENT_BALL_SPACE_SHAPE = (4,)
...


class StateSelector:

    ...
    def get_state(self, state):
        ...
        elif self.selected_state == AGENT_BALL_SPACE_SPACE:
            return self.get_agent_and_ball_position_state(state)
        ...

    def get_shape(self):
        ...
        elif self.selected_state == AGENT_BALL_SPACE_SPACE:
            return AGENT_BALL_SPACE_SHAPE
        ...

    ...

    def get_agent_and_ball_position_state(self, state):
        state_wrapper = StateWrapper(state)
        agent_position = state_wrapper.get_position()
        ball_position = state_wrapper.get_ball_position()
        return np.array([agent_position[0], agent_position[1], ball_position[0], ball_position[1]])
```

To the environment use this action is needed to select in the init, as shown below.

```python
hfo_env = HFOEnv(..., selected_state=AGENT_BALL_SPACE_SPACE)
```

To more examples, check the `lib/utils/state_selector.py` class, several state spaces are already implemented there.

### Action Selector

To include a new types of actions, check the standard HFO action in the [documentation](https://github.com/LARG/HFO/blob/master/doc/manual.pdf). Based on these actions that it can be implemented new ones. Here an example of the inclusion of `DASH_WITH_POWER_ACTION` in the `action_selector` class.

```python
...
DASH_WITH_POWER_ACTION = 2
...


class ActionSelector:
    ...

    def get_action(self, action):
        ...
        elif self.selected_action == DASH_WITH_POWER_ACTION:
            return self.get_dash_with_power(action)
        ...

    def get_action_space_info(self):
        ...
        elif self.selected_action == DASH_WITH_POWER_ACTION:
            return self.get_dict_info([hfo.DASH], [0], 2)
        ...

    ...
    def get_dash_with_power(self, action):
        angle = float(action[0] * 180.0)
        power = float(action[1] * 100)
        return ([hfo.DASH, power, angle], 3)
```

To the environment use this action is needed to select in the init, as shown below.

```python
hfo_env = HFOEnv(..., selected_action=DASH_WITH_POWER_ACTION)
```

To more examples, check the `lib/utils/action_selector.py` class, several actions are already implemented there.

### Reward Selector

To include new reward functions could be use the HFO state space to calculate the required values. Here an example of the inclusion of `GO_TO_BALL_REWARD` in the `reward_selector` class.


```python
...
BALL_PROXIMITY_GOAL_REWARD = 2
...

class RewardSelector:
    ...

    def get_reward(self, act, next_state, done, status):
        ...
        elif self.selected_reward == GO_TO_BALL_REWARD:
            return self.get_reward_go_to_ball(act, state_wrapper, done, status)
        ...

    ...
    def get_reward_go_to_ball(self, act, state_wrapper, done, status):
        distance_to_ball = state_wrapper.get_distance_to_ball()
        reward = (MAX_DISTANCE - distance_to_ball) / MAX_DISTANCE
        if distance_to_ball > self.last_distance_to_ball:
            reward = (-1.0) * (1.0 - reward)
        self.last_distance_to_ball = distance_to_ball
        if distance_to_ball <= 2.0:
            return 1.0
        return reward
```

To the environment use this action is needed to select in the init, as shown below.

```python
hfo_env = HFOEnv(..., selected_reward=GO_TO_BALL_REWARD)
```

To more examples, check the `lib/utils/reward_selector.py` class, several reward functions are already implemented there.

### Agent

As the paper shows, this project has some agent implemented to due the task of `go_to_ball` and `ball_to_goal`. But to implement new agent it's provided these mentioned implementations as baseline and the `base.py` agent, all these implementation can be finded in the `src/agent`.
The `src/agent/base.py` agent it's pure agent without deep reinforcement learning models, just the environment workflow, below you can see the code:

```python
import itertools
import hfo
from src.lib.hfo_env import HFOEnv
from src.lib.utils.action_selector import TEST_ACTION

team = 'HELIOS'
port = 6000
hfo_env = HFOEnv(is_offensive=True, strict=True,
                 continuous=True, team=team, port=port, selected_action=TEST_ACTION)

for episode in itertools.count():
    status = hfo.IN_GAME
    done = True
    state = hfo_env.reset()

    while status == hfo.IN_GAME:
        next_state, reward, done, status = hfo_env.step([-0.6])
        if done:
            break
        
    if status == hfo.SERVER_DOWN:
            hfo_env.act(hfo.QUIT)
            exit()
```

# Running

As part of the project the rcssserver was extended to give a more flexible setup, you can see the fork [here](https://github.com/cstiano/rcssserver). The rcssserver setup options define where the offensive agent and the ball will begin for every episode.

- Rcssserver Setup Options
    - DEFAULT_ENV = 0
    - DEFAULT_DYNAMIC_ENV = 1
    - GO_TO_BALL_RANDOM_POS_ENV = 2
    - ALL_RANDOM_ENV = 3
    - START_WITH_BALL_ENV = 4
    - START_WITH_BALL_RANDOM_ENV = 5
    - START_MEDIUM_BALL_RANDOM_ENV = 6
    - START_HIGH_BALL_RANDOM_ENV = 7
    - PENALTY_ENV = 8
    - PENALTY_MEDIUM_ENV = 9
    - PENALTY_HIGH_ENV = 10
    - PENALTY_MEDIUM_STATIC_ENV = 11
    - PENALTY_HIGH_STATIC_ENV = 12


There are two modes of running the agent, the training mode and the test mode.

**Training**:

```sh
./shell/run_agent_train.sh <agent relative path> <number of episodes> <number of maximum cycles untouching the ball> <number of maximum cycles per episode> <extended rcssserver option>
```

**Test**:

```sh
./shell/run_agent_play.sh <agent relative path> <number of episodes> <number of maximum cycles untouching the ball> <number of maximum cycles per episode> <extended rcssserver option>
```

Example:
```sh
./shell/run_agent_train.sh go_to_ball/go_to_ball_ddpg 1000 200 400 0
```

# References
- [RoboCup 2D Half Field Offense](https://github.com/LARG/HFO)
- [RL-Adventure-2: Policy Gradients](https://github.com/higgsfield/RL-Adventure-2)
- [Comparing DQN, Dueling Double DQN and Deep Deterministic Policy Gradient applied to Robocup Soccer Simulation 2D](https://github.com/goncamateus/graduationMgm)
- [rcssserver](https://github.com/mhauskn/rcssserver)