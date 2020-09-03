from src.lib.utils.state_wrapper import StateWrapper

TEST_REWARD = 0
GO_TO_BALL_REWARD = 1
BALL_PROXIMITY_GOAL_REWARD = 2

MAX_DISTANCE = 50.0
MIN_DISTANCE_TO_BALL = 2.0
THRESHOLD_DISTANCE = 10.0
MAX_BALL_DISTANCE_TO_GOAL = 30.0

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
        return 0.0

    def get_reward_go_to_ball(self, act, state_wrapper, done, status):
        distance_to_ball = state_wrapper.get_distance_to_ball()
        reward = (MAX_DISTANCE - distance_to_ball) / MAX_DISTANCE
        if distance_to_ball <= 2.0:
            return 1.0
        if distance_to_ball > self.last_distance_to_ball:
            reward = (-1.0) * (1.0 - reward)
        self.last_distance_to_ball = distance_to_ball
        return reward
    
    def get_reward_ball_proximity_goal(self, act, state_wrapper, done, status):
        ball_distance_to_goal = state_wrapper.get_ball_distance_to_goal()
        reward = (MAX_BALL_DISTANCE_TO_GOAL - ball_distance_to_goal) / MAX_BALL_DISTANCE_TO_GOAL
        if ball_distance_to_goal > 30.0:
            return -1
        if ball_distance_to_goal <= 2.0:
            return 1.0
        if ball_distance_to_goal > self.last_distance_to_ball:
            reward = (-1.0) * (1.0 - reward)
        self.last_ball_distance_to_goal = ball_distance_to_goal
        return reward
