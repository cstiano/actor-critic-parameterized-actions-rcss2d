class RewardSelector:
    def __init__(self, selected_reward=0):
        super().__init__()
        self.selected_reward = selected_reward

    def get_reward(self, act, next_state, done, status):
        return 0.0
