TEST_REWARD = 0


class RewardSelector:
    def __init__(self, selected_reward=0):
        super().__init__()
        self.selected_reward = selected_reward

    def get_reward(self, act, next_state, done, status):
        if self.selected_reward == TEST_REWARD:
            return 1000.0
        return 0.0
