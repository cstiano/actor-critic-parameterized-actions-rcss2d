import hfo

TEST_ACTION = 0
GO_TO_BALL_ACTION = 1


MAX_DASH = 100


class ActionSelector:
    def __init__(self, selected_action=0, actions=[]):
        super().__init__()
        self.selected_action = selected_action
        self.actions = actions

    # actions and the quantity - maximum is 4
    def get_action(self, action):
        if self.selected_action == TEST_ACTION:
            return ([hfo.DRIBBLE_TO, 0.0, 0.0], 3)
        elif self.selected_action == GO_TO_BALL_ACTION:
            return self.get_go_to_ball_action(action)
        return ([], 0)

    def get_go_to_ball_action(self, action):
        return ([hfo.DASH, MAX_DASH, action[0]], 3)