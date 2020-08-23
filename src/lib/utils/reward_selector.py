from src.lib.utils.state_wrapper import StateWrapper

TEST_REWARD = 0
GO_TO_BALL_REWARD = 1

MAX_DISTANCE = 130.0
MIN_DISTANCE_TO_BALL = 2.0
THRESHOLD_DISTANCE = 10.0

class RewardSelector:
    def __init__(self, selected_reward=0):
        super().__init__()
        self.selected_reward = selected_reward

    def get_reward(self, act, next_state, done, status):
        state_wrapper = StateWrapper(next_state)
        if self.selected_reward == TEST_REWARD:
            return 1000.0
        elif self.selected_reward == GO_TO_BALL_REWARD:
            return self.get_reward_go_to_ball(act, state_wrapper, done, status)
        return 0.0

    def get_reward_go_to_ball(self, act, state_wrapper, done, status):
        distance_to_ball = state_wrapper.get_distance_to_ball()
        if distance_to_ball <= MIN_DISTANCE_TO_BALL:
            return 100000.0
        elif distance_to_ball >= THRESHOLD_DISTANCE:
            return float(-100.0 * distance_to_ball)
        return float(MAX_DISTANCE / (float(distance_to_ball) * 0.1))
