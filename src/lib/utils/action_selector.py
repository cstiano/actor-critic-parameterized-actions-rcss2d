import hfo

TEST_ACTION = 0


class ActionSelector:
    def __init__(self, selected_action=0, actions=[]):
        super().__init__()
        self.selected_action = selected_action
        self.actions = actions

    # actions and the quantity - maximum is 4
    def get_action(self, action):
        if self.selected_action == TEST_ACTION:
            return ([hfo.DRIBBLE_TO, 0.0, 0.0], 3)
        return ([], 0)
