from src.lib.utils.state_wrapper import StateWrapper
import hfo
from math import *

TEST_REWARD = 0
GO_TO_BALL_REWARD = 1
BALL_PROXIMITY_GOAL_REWARD = 2
PAPER_REWARD = 3
BALL_POTENCIAL_DIFF_REWARD = 4
AGENT_AND_BALL_POTENCIAL_REWARD = 5
AGENT_POTENCIAL_TO_BALL_REWARD = 6
PAPER_SKILL_GO_TO_BALL_REWARD = 7
AGENT_AND_BALL_POTENCIAL_WITH_OPPONENT_REWARD = 8

MAX_DISTANCE = 50.0
MIN_DISTANCE_TO_BALL = 2.0
THRESHOLD_DISTANCE = 10.0
MAX_BALL_DISTANCE_TO_GOAL = 30.0
HALF_GOAL_SIZE = 7.2
HALF_X_AXIS_SIZE = 52.5
HALF_Y_AXIS_SIZE = 34.0

GOAL_FACTOR = 1000.0
POTENCIAL_BALL_FACTOR = 100.0
AGENT_POTENCIAL_FACTOR = 10.0


class RewardSelector:
    def __init__(self, selected_reward=0):
        super().__init__()
        self.selected_reward = selected_reward
        self.last_distance_to_ball = MAX_DISTANCE
        self.last_ball_distance_to_goal = MAX_BALL_DISTANCE_TO_GOAL

    # To include new rewards create the reward function and include the selection here
    def get_reward(self, act, next_state, done, status):
        state_wrapper = StateWrapper(next_state)
        if self.selected_reward == TEST_REWARD:
            return 1000.0
        elif self.selected_reward == GO_TO_BALL_REWARD:
            return self.get_reward_go_to_ball(act, state_wrapper, done, status)
        elif self.selected_reward == BALL_PROXIMITY_GOAL_REWARD:
            return self.get_reward_ball_proximity_goal(act, state_wrapper, done, status)
        elif self.selected_reward == PAPER_REWARD:
            return self.get_reward_paper(act, state_wrapper, done, status)
        elif self.selected_reward == BALL_POTENCIAL_DIFF_REWARD:
            return self.get_reward_ball_potencial(act, state_wrapper, done, status)
        elif self.selected_reward == AGENT_AND_BALL_POTENCIAL_REWARD:
            return self.get_reward_agent_and_ball_potencial(act, state_wrapper, done, status)
        elif self.selected_reward == AGENT_POTENCIAL_TO_BALL_REWARD:
            return self.get_reward_agent_potencial_to_ball(act, state_wrapper, done, status)
        elif self.selected_reward == PAPER_SKILL_GO_TO_BALL_REWARD:
            return self.get_reward_paper_skill(act, state_wrapper, done, status)
        elif self.selected_reward == AGENT_AND_BALL_POTENCIAL_WITH_OPPONENT_REWARD:
            return self.get_reward_agent_and_ball_potencial_with_opponent(act, state_wrapper, done, status)
        return 0.0

    def get_reward_go_to_ball(self, act, state_wrapper, done, status):
        distance_to_ball = state_wrapper.get_distance_to_ball()
        reward = (MAX_DISTANCE - distance_to_ball) / MAX_DISTANCE
        if distance_to_ball > self.last_distance_to_ball:
            reward = (-1.0) * (1.0 - reward)
        self.last_distance_to_ball = distance_to_ball
        if distance_to_ball <= 2.0:
            return 1.0
        return reward

    def get_reward_ball_proximity_goal(self, act, state_wrapper, done, status):
        ball_distance_to_goal = state_wrapper.get_ball_distance_to_goal()
        reward = (MAX_BALL_DISTANCE_TO_GOAL -
                  ball_distance_to_goal) / MAX_BALL_DISTANCE_TO_GOAL
        if ball_distance_to_goal > self.last_ball_distance_to_goal:
            reward = (-1.0) * (1.0 - reward)
        self.last_ball_distance_to_goal = ball_distance_to_goal
        if ball_distance_to_goal <= 2.0:
            return 1.0
        return reward
    
    def get_reward_agent_potencial_to_ball(self, act, state_wrapper, done, status):
        distance_to_ball = state_wrapper.get_distance_to_ball()
        agent_potencial_difference_to_ball = self.last_distance_to_ball - distance_to_ball
        self.last_distance_to_ball = distance_to_ball

        if bool(state_wrapper.is_able_to_kick):
            return 1.0

        return agent_potencial_difference_to_ball
    
    def get_reward_paper_skill(self, act, state_wrapper, done, status):
        distance_to_ball = state_wrapper.get_distance_to_ball()
        if bool(state_wrapper.is_able_to_kick):
            return 100.0
        return (5 / pow(2 * math.pi, 1 / 2)) * math.exp(-((distance_to_ball*0.001)**2) / 2) - 2 

    def get_reward_paper(self, act, state_wrapper, done, status):
        distance_to_ball = state_wrapper.get_distance_to_ball()
        ball_distance_to_goal = state_wrapper.get_ball_distance_to_goal()
        i_kick = 0.0
        i_goal = 0.0

        if state_wrapper.is_able_to_kick():
            i_kick = 1.0
        if status == hfo.GOAL:
            i_goal = 5.0

        r_dist_ball = self.last_distance_to_ball - distance_to_ball
        r_dist_goal = self.last_ball_distance_to_goal - ball_distance_to_goal

        reward = r_dist_ball + i_kick + (3.0 * r_dist_goal) + i_goal

        self.last_distance_to_ball = distance_to_ball
        self.last_ball_distance_to_goal = ball_distance_to_goal
        return reward

    def get_reward_ball_potencial(self, act, state_wrapper, done, status):
        ball_position = state_wrapper.get_ball_position()
        ball_distance_to_goal = state_wrapper.get_ball_distance_to_goal()

        potencial_difference = self.last_ball_distance_to_goal - ball_distance_to_goal
        self.last_ball_distance_to_goal = ball_distance_to_goal

        if status == hfo.GOAL:
            return GOAL_FACTOR

        return potencial_difference

    def get_reward_agent_and_ball_potencial(self, act, state_wrapper, done, status):
        ball_position = state_wrapper.get_ball_position()
        ball_distance_to_goal = state_wrapper.get_ball_distance_to_goal()
        distance_to_ball = state_wrapper.get_distance_to_ball()

        agent_potencial_difference_to_ball = self.last_distance_to_ball - distance_to_ball
        self.last_distance_to_ball = distance_to_ball
        potencial_difference = self.last_ball_distance_to_goal - ball_distance_to_goal
        self.last_ball_distance_to_goal = ball_distance_to_goal

        if status == hfo.GOAL:
            return GOAL_FACTOR

        return (AGENT_POTENCIAL_FACTOR * agent_potencial_difference_to_ball) + (POTENCIAL_BALL_FACTOR * potencial_difference)
    
    def get_reward_agent_and_ball_potencial_with_opponent(self, act, state_wrapper, done, status):
        ball_position = state_wrapper.get_ball_position()
        ball_distance_to_goal = state_wrapper.get_ball_distance_to_goal()
        distance_to_ball = state_wrapper.get_distance_to_ball()

        agent_potencial_difference_to_ball = self.last_distance_to_ball - distance_to_ball
        self.last_distance_to_ball = distance_to_ball
        potencial_difference = self.last_ball_distance_to_goal - ball_distance_to_goal
        self.last_ball_distance_to_goal = ball_distance_to_goal

        if status == hfo.GOAL:
            return GOAL_FACTOR
        if status == hfo.CAPTURED_BY_DEFENSE:
            return ((-1) * GOAL_FACTOR)

        return (AGENT_POTENCIAL_FACTOR * agent_potencial_difference_to_ball) + (POTENCIAL_BALL_FACTOR * potencial_difference)

    def reset(self, state):
        state_wrapper = StateWrapper(state)
        self.last_distance_to_ball = state_wrapper.get_distance_to_ball()
        self.last_ball_distance_to_goal = state_wrapper.get_ball_distance_to_goal()
