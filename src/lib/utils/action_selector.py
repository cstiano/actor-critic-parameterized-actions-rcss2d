import hfo

TEST_ACTION = 0
DASH_ACTION = 1
DASH_WITH_POWER_ACTION = 2
DASH_OR_KICK_ACTION = 3
DASH_OR_KICK_OR_TURN_ACTION = 3
ALL_LOW_ACTIONS = 4
ALL_MID_ACTIONS = 5

MAX_DASH = 100


class ActionSelector:
    def __init__(self, selected_action=0, actions=[]):
        super().__init__()
        self.selected_action = selected_action
        self.actions = actions

    def get_action(self, action):
        if self.selected_action == TEST_ACTION:
            return ([hfo.DRIBBLE_TO, 0.0, 0.0], 3)
        elif self.selected_action == DASH_ACTION:
            return self.get_dash(action)
        elif self.selected_action == DASH_WITH_POWER_ACTION:
            return self.get_dash_with_power(action)
        elif self.selected_action == DASH_OR_KICK_ACTION:
            return self.get_dash_or_kick(action)
        elif self.selected_action == DASH_OR_KICK_OR_TURN_ACTION:
            return self.get_dash_or_kick_or_turn(action)
        elif self.selected_action == ALL_LOW_ACTIONS:
            return self.get_all(action)
        elif self.selected_action == ALL_MID_ACTIONS:
            return ([], 0)
        return ([], 0)

    def get_dash(self, action):
        angle = float(action[0] * 180.0)
        return ([hfo.DASH, MAX_DASH, angle], 3)
    
    def get_dash_with_power(self, action):
        angle = float(action[0] * 180.0)
        power = float(action[1] * 100)
        return ([hfo.DASH, power, angle], 3)
    
    def get_dash_or_kick(self, action):
        angle = float(action[0] * 180.0)
        if action[0] < 0:
            power = float(action[1] * 100)
            return ([hfo.DASH, power, angle], 3)
        else:
            power = float(((action[1] + 1.0) * 0.5) * 100)
            return ([hfo.KICK, power, angle], 3)
    
    def get_dash_or_kick_or_turn(self, action):
        angle = float(action[1] * 180.0)
        if action[0] < -0.333:
            power = float(action[2] * 100)
            return ([hfo.DASH, power, angle], 3)
        elif action[0] < 0.333:
            power = float(((action[2] + 1.0) * 0.5) * 100)
            return ([hfo.KICK, power, angle], 3)
        else: 
            return ([hfo.TURN, angle], 2)

    def get_all(self, action):
        angle = float(action[1] * 180.0)
        if action[0] < -0.5:
            power = float(action[2] * 100)
            return ([hfo.DASH, power, angle], 3)
        elif action[0] < 0.0:
            power = float(((action[2] + 1.0) * 0.5) * 100)
            return ([hfo.KICK, power, angle], 3)
        elif action[0] < 0.5:
            return ([hfo.TACKLE, angle], 2)
        else: 
            return ([hfo.TURN, angle], 2)
    
    
