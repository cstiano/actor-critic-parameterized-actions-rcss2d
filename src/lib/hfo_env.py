import math

import hfo
import numpy as np
from scipy.spatial import distance
from gym import spaces
from src.lib.utils.action_selector import ActionSelector
from src.lib.utils.reward_selector import RewardSelector
from src.lib.utils.state_selector import StateSelector


class ObservationSpace():
    def __init__(self, env, rewards, shape=None):
        self.state = env.getState()
        self.rewards = rewards
        self.nmr_out = 0
        self.taken = 0
        self.shape = shape if shape else self.state.shape
        self.goals_taken = 0


class ActionSpace():
    def __init__(self, actions):
        self.actions = actions
        self.n = len(actions)


class ActionSpaceContinuous(spaces.Box):
    def __init__(self, low, high, actions, shape=None, dtype=np.float32):
        super().__init__(low, high, shape=shape, dtype=dtype)
        self.actions = actions


class HFOEnv(hfo.HFOEnvironment):
    pitchHalfLength = 52.5
    pitchHalfWidth = 34
    tolerance_x = pitchHalfLength * 0.1
    tolerance_y = pitchHalfWidth * 0.1
    FEAT_MIN = -1
    FEAT_MAX = 1
    pi = 3.14159265358979323846
    max_R = np.sqrt(pitchHalfLength * pitchHalfLength +
                    pitchHalfWidth * pitchHalfWidth)
    stamina_max = 8000
    w_ball_grad = 0.8
    prev_ball_potential = None

    def __init__(self, actions=None, rewards=None,
                 is_offensive=True, play_goalie=False,
                 strict=False, port=6000, continuous=False, team='base',
                 selected_action=0, selected_reward=0, selected_state=0,
                 continuous_action_dim=None):
        super(HFOEnv, self).__init__()
        self.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET, './formations-dt',
                             port, 'localhost',
                             team + '_left' if is_offensive else team + '_right',
                             play_goalie=play_goalie)
        self.num_teammates = self.getNumTeammates()
        self.num_opponents = self.getNumOpponents()
        self.choosed_mates = min(1, self.num_teammates)
        self.choosed_ops = min(1, self.num_opponents)
        self.play_goalie = play_goalie
        self.continuous = continuous

        self.action_selector = ActionSelector(selected_action)
        self.reward_selector = RewardSelector(selected_reward)
        self.state_selector = StateSelector(selected_state)
        self.selected_state = selected_state
        self.last_state = None
        self.last_strict_state = None

        action_space_info = self.action_selector.get_action_space_info()
        self.actions = actions
        if self.actions == None:
            self.actions = action_space_info['actions']
        self.rewards = rewards
        if self.rewards is None:
            self.rewards = action_space_info['env_rewards']
        self.continuous_action_dim = continuous_action_dim
        if self.continuous_action_dim == None:
            self.continuous_action_dim = action_space_info['action_dim']
        
        # TODO: check the usability of this code
        if not strict:
            self.observation_space = ObservationSpace(self, self.rewards)
        else:
            shape = self.state_selector.get_shape()
            if self.selected_state == 0:
                shape = 11 + 3 * self.choosed_mates + 2 * self.choosed_ops
                shape = (shape,)
            self.observation_space = ObservationSpace(self,
                                                      self.rewards,
                                                      shape=shape)
        if self.continuous:
            self.action_space = ActionSpaceContinuous(
                -1, 1, self.actions, shape=(self.continuous_action_dim,))
        else:
            self.action_space = ActionSpace(self.actions)

    def step(self, action, is_offensive=True):
        # Update the action selector for conditional actions.
        self.action_selector.update_if_necessary(self.strict_state(self.getState()))

        # Action is the foward from the neural network (array of actions).
        # Get the selected parameterized action using the forward nn action.
        (result_action, params_action) = self.action_selector.get_action(action)
        action = result_action[0]
        if params_action == 1:
            self.act(result_action[0])
        elif params_action == 2:
            self.act(result_action[0], result_action[1])
        elif params_action == 3:
            self.act(result_action[0], result_action[1], result_action[2])
        elif params_action == 4:
            self.act(result_action[0],
                     result_action[1],
                     result_action[2],
                     result_action[3])
        act = self.action_space.actions.index(action)
        status = super(HFOEnv, self).step()
        done = True
        if status == hfo.IN_GAME:
            done = False

        next_state = self.get_state()
        next_strict_state = self.strict_state(self.getState())
        if done:
            next_state = self.last_state
            next_strict_state = self.last_strict_state

        # Getting Reward
        self.update_observation_space(done, status)
        reward = self.reward_selector.get_reward(
            act, next_strict_state, done, status)

        self.last_state = next_state
        self.last_strict_state = next_strict_state

        return next_state, reward, done, status

    def get_state(self):
        if self.selected_state == 0:
            return self.strict_state(self.getState())
        return self.state_selector.get_state(self.strict_state(self.getState()))

    def reset(self):
        state = self.get_state()
        self.reward_selector.reset(self.strict_state(self.getState()))
        return state
    
    # Goal is to be used in HER architecture
    def get_goal(self):
        return self.state_selector.get_state(self.strict_state(self.getState()))
    
    def compute_reward(self, achieved_goal, desired_goal):
        # TODO the reward calculation for the goal.
        pass

    def update_observation_space(self, done, status):
        if done:
            if status == hfo.GOAL:
                self.observation_space.goals_taken += 1
            else:
                self.observation_space.taken += 1

    # Reward for the offensive - include the implementation here
    def get_reward_off(self, act, next_state, done, status):
        reward = 0
        if done:
            if status == hfo.GOAL:
                reward = self.observation_space.rewards[act]
                self.observation_space.goals_taken += 1
                if self.observation_space.goals_taken % 5 == 0:
                    reward *= 10000
            else:
                self.observation_space.taken += 1
                reward -= 100000000
                if self.observation_space.taken % 5 == 0:
                    reward *= 5
        return reward

    def get_reward_def(self, act, next_state, done, status):
        reward = 0
        # ball_potential = self.ball_potential(next_state[3], next_state[4])

        # if self.prev_ball_potential is not None:
        #     grad_ball_potential = self.clip(
        #         (ball_potential - self.prev_ball_potential), -1.0,
        #         1.0)
        # else:
        #     grad_ball_potential = 0

        if status == hfo.GOAL:
            reward = -10
            self.observation_space.goals_taken += 1
            if self.observation_space.goals_taken % 5 == 0:
                reward = -100
            if '-{}'.format(self.getUnum()) in self.statusToString(status):
                reward = -200
        else:
            if done:
                self.observation_space.taken += 1
                reward = 10
                if self.observation_space.taken % 5 == 0:
                    reward = 100
            else:
                if abs(next_state[10]) <= 1:
                    reward = -1
        # self.prev_ball_potential = grad_ball_potential
        return reward

    def get_reward_goalie(self, act, next_state, done, status):
        reward = 0
        if done:
            if status == hfo.GOAL:
                reward = -200000
                self.observation_space.goals_taken += 1
                if self.observation_space.goals_taken % 5 == 0:
                    reward -= 1000000
                if '-{}'.format(self.getUnum()) in self.statusToString(status):
                    reward -= 1000000
            else:
                self.observation_space.taken += 1
                reward = self.observation_space.rewards[act] * 5
                if self.observation_space.taken % 5 == 0:
                    reward = reward * 1000

        else:
            reward = self.observation_space.rewards[act]\
                - next_state[3] * 3
        return reward

    def unnormalize(self, val, min_val, max_val):
        return ((val - self.FEAT_MIN) / (self.FEAT_MAX - self.FEAT_MIN))\
            * (max_val - min_val) + min_val

    def abs_x(self, normalized_x_pos, playing_offense):
        if playing_offense:
            return self.unnormalize(normalized_x_pos, -self.tolerance_x,
                                    self.pitchHalfLength + self.tolerance_x)
        else:
            return self.unnormalize(normalized_x_pos,
                                    -self.pitchHalfLength - self.tolerance_x,
                                    self.tolerance_x)

    def abs_y(self, normalized_y_pos):
        return self.unnormalize(normalized_y_pos,
                                -self.pitchHalfWidth - self.tolerance_y,
                                self.pitchHalfWidth + self.tolerance_y)

    def remake_state(self, state, is_offensive=True):
        num_mates, num_ops = self.num_teammates, self.num_opponents
        state[0] = self.abs_x(state[0], is_offensive)
        state[1] = self.abs_y(state[1])
        state[2] = self.unnormalize(state[2], -self.pi, self.pi)
        state[3] = self.abs_x(state[3], is_offensive)
        state[4] = self.abs_y(state[4])
        state[5] = self.unnormalize(state[5], 0, 1)
        state[6] = self.unnormalize(state[6], 0, self.max_R)
        state[7] = self.unnormalize(state[7], -self.pi, self.pi)
        state[8] = self.unnormalize(state[8], 0, self.pi)
        if num_ops > 0:
            state[9] = self.unnormalize(state[9], 0, self.max_R)
        else:
            state[9] = -1000
        for i in range(10, 10 + num_mates):
            if state[i] != -2:
                state[i] = self.unnormalize(state[i], 0, self.pi)
            else:
                state[i] = -1000
        for i in range(10 + num_mates, 10 + 2 * num_mates):
            if state[i] != -2:
                state[i] = self.unnormalize(state[i], 0, self.max_R)
            else:
                state[i] = -1000
        for i in range(10 + 2 * num_mates, 10 + 3 * num_mates):
            if state[i] != -2:
                state[i] = self.unnormalize(state[i], 0, self.pi)
            else:
                state[i] = -1000
        index = 10 + 3 * num_mates
        for i in range(num_mates):
            if state[index] != -2:
                state[index] = self.abs_x(state[index], is_offensive)
            else:
                state[index] = -1000
            index += 1
            if state[index] != -2:
                state[index] = self.abs_y(state[index])
            else:
                state[index] = -1000
            index += 2
        index = 10 + 6 * num_mates
        for i in range(num_ops):
            if state[index] != -2:
                state[index] = self.abs_x(state[index], is_offensive)
            else:
                state[index] = -1000
            index += 1
            if state[index] != -2:
                state[index] = self.abs_y(state[index])
            else:
                state[index] = -1000
            index += 2
        state[-1] = self.unnormalize(state[-1], 0, self.stamina_max)
        return state

    def get_dist(self, init, end):
        return distance.euclidean(init, end)

    def get_ball_dist(self, state):
        agent = (state[0], state[1])
        ball = (state[3], state[4])
        return distance.euclidean(agent, ball)

    def strict_state(self, state):
        num_mates = self.num_teammates
        new_state = state[:10].tolist()
        for i in range(10 + num_mates, 10 + num_mates + self.choosed_mates):
            new_state.append(state[i])
        index = 10 + 3 * num_mates
        for i in range(self.choosed_mates):
            new_state.append(state[index])
            index += 1
            new_state.append(state[index])
            index += 2
        index = 10 + 6 * num_mates
        for i in range(self.choosed_ops):
            new_state.append(state[index])
            index += 1
            new_state.append(state[index])
            index += 2
        new_state.append(state[-2])
        new_state = np.array(new_state)
        return new_state

    def ball_potential(self, ball_x, ball_y):
        """
            Calculate ball potential according to this formula:
            pot = ((-sqrt((52.5-x)^2 + 2*(0-y)^2) +
                    sqrt((-52.5- x)^2 + 2*(0-y)^2))/52.5 - 1)/2
            the potential is zero (maximum) at the center of attack goal
            (52.5,0) and -1 at the defense goal (-52.5,0)
            it changes twice as fast on y coordinate than on the x coordinate
        """

        dx_d = -self.pitchHalfLength - ball_x  # distance to defence
        dx_a = self.pitchHalfLength - ball_x  # distance to attack
        dy = 0 - ball_y
        potential = ((-math.sqrt(dx_a ** 2 + 2 * dy ** 2) +
                      math.sqrt(dx_d ** 2 + 2 * dy ** 2)) / self.pitchHalfLength - 1) / 2  # noqa

        return potential

    def clip(self, val, vmin, vmax):
        return min(max(val, vmin), vmax)
