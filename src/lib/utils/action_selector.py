import hfo

TEST_ACTION = 0
DASH_ACTION = 1
DASH_WITH_POWER_ACTION = 2
DASH_OR_KICK_ACTION = 3
DASH_OR_KICK_OR_TURN_ACTION = 3
ALL_LOW_ACTIONS = 4
ALL_MID_ACTIONS = 5
DRIBBLE_ACTION = 6
CONDITIONAL_DASH_OR_KICK_ACTION = 7
MID_LEVEL_AND_SHOOT_ACTION = 8

MAX_DASH = 100


class ActionSelector:
    def __init__(self, selected_action=0):
        super().__init__()
        self.selected_action = selected_action
        self.kickable = None
    
    def update_if_necessary(self, state):
        # Update the kickable with the current state
        self.kickable = bool(state[5])

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
            return self.get_low_level_action(action)
        elif self.selected_action == ALL_MID_ACTIONS:
            return self.get_mid_level_action(action)
        elif self.selected_action == DRIBBLE_ACTION:
            return self.get_dribble_action(action)
        elif self.selected_action == CONDITIONAL_DASH_OR_KICK_ACTION:
            return self.get_conditional_dash_or_kick(action)
        elif self.selected_action == MID_LEVEL_AND_SHOOT_ACTION:
            return self.get_mid_level_and_shoot_action(action)
        return ([], 0)

    def get_action_space_info(self):
        if self.selected_action == TEST_ACTION:
            return self.get_dict_info([hfo.DRIBBLE_TO], [0], 1)
        elif self.selected_action == DASH_ACTION:
            return self.get_dict_info([hfo.DASH], [0], 1)
        elif self.selected_action == DASH_WITH_POWER_ACTION:
            return self.get_dict_info([hfo.DASH], [0], 2)
        elif self.selected_action == DASH_OR_KICK_ACTION:
            return self.get_dict_info([hfo.DASH, hfo.KICK], [0, 0], 3)
        elif self.selected_action == DASH_OR_KICK_OR_TURN_ACTION:
            return self.get_dict_info([hfo.DASH, hfo.KICK, hfo.TURN], [0, 0, 0], 3)
        elif self.selected_action == ALL_LOW_ACTIONS:
            return self.get_dict_info([hfo.DASH, hfo.KICK, hfo.TURN, hfo.TACKLE],
                                      [0, 0, 0, 0], 3)
        elif self.selected_action == ALL_MID_ACTIONS:
            return self.get_dict_info([hfo.KICK_TO, hfo.MOVE_TO, hfo.DRIBBLE_TO, hfo.INTERCEPT],
                                      [0, 0, 0, 0], 4)
        elif self.selected_action == DRIBBLE_ACTION:
            return self.get_dict_info([hfo.DRIBBLE_TO], [0], 2)
        elif self.selected_action == CONDITIONAL_DASH_OR_KICK_ACTION:
            return self.get_dict_info([hfo.DASH, hfo.KICK], [0, 0], 3)
        elif self.selected_action == MID_LEVEL_AND_SHOOT_ACTION:
            return self.get_dict_info([hfo.KICK_TO, hfo.MOVE_TO, hfo.DRIBBLE_TO, hfo.SHOOT],
                                      [0, 0, 0, 0], 4)
        return ([], [], 0)

    def get_dict_info(self, actions, env_rewards_config_action, action_dim):
        return {
            'actions': actions,
            'env_rewards': env_rewards_config_action,
            'action_dim': action_dim
        }

    def get_dash(self, action):
        angle = float(action[0] * 180.0)
        return ([hfo.DASH, MAX_DASH, angle], 3)

    def get_dash_with_power(self, action):
        angle = float(action[0] * 180.0)
        power = float(action[1] * 100)
        return ([hfo.DASH, power, angle], 3)

    def get_dash_or_kick(self, action):
        angle = float(action[1] * 180.0)
        if action[0] < 0:
            power = float(action[2] * 100)
            return ([hfo.DASH, power, angle], 3)
        else:
            power = float(((action[2] + 1.0) * 0.5) * 100)
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

    def get_low_level_action(self, action):
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

    def get_mid_level_action(self, action):
        if self.kickable:
            if action[0] < -0.5:
                speed = float((action[3] + 1.0) * 3.0)
                return ([hfo.KICK_TO, action[1], action[2], speed], 4)
            elif action[0] < 0.0:
                return ([hfo.MOVE_TO, action[1], action[2]], 3)
            elif action[0] < 0.5:
                return ([hfo.DRIBBLE_TO, action[1], action[2]], 3)
            else:
                return ([hfo.INTERCEPT], 1)
        else:
            return ([hfo.MOVE_TO, action[1], action[2]], 3)
    
    def get_mid_level_and_shoot_action(self, action):
        if self.kickable:
            if action[0] < -0.5:
                speed = float((action[3] + 1.0) * 3.0)
                return ([hfo.KICK_TO, action[1], action[2], speed], 4)
            elif action[0] < 0.0:
                return ([hfo.MOVE_TO, action[1], action[2]], 3)
            elif action[0] < 0.5:
                return ([hfo.DRIBBLE_TO, action[1], action[2]], 3)
            else:
                return ([hfo.SHOOT], 1)
        else:
            return ([hfo.MOVE_TO, action[1], action[2]], 3)

    def get_dribble_action(self, action):
        return ([hfo.DRIBBLE_TO, action[0], action[1]], 3)
    
    def get_conditional_dash_or_kick(self, action):
        if self.kickable:
            angle = float(action[1] * 180.0)
            if action[0] < 0:
                power = float(action[2] * 100)
                return ([hfo.DASH, power, angle], 3)
            else:
                power = float(((action[2] + 1.0) * 0.5) * 100)
                return ([hfo.KICK, power, angle], 3)
        else:    
            angle = float(action[1] * 180.0)
            power = float(action[2] * 100)
            return ([hfo.DASH, power, angle], 3)
