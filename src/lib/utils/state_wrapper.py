class StateWrapper:
    def __init__(self, state=None):
        super().__init__()
        self.state = state
    
    def get_position(self):
        return (self.state[0], self.state[1])

    def get_orientation(self):
        return self.state[2]
    
    def get_ball_position(self):
        return (self.state[3], self.state[4])
    
    def is_able_to_kick(self):
        return self.state[5]
    
    def get_goal_center_proximity(self):
        return self.state[6]

    def get_goal_center_angle(self):
        return self.state[7]
    
    def get_goal_opening_angle(self):
        return self.state[8]
    
    def get_proximity_to_opponent(self):
        return self.state[9]
    
    # TODO
    def get_distance_to_ball(self):
        return float(0.0)
    
    # TODO
    def get_distance_to_goal(self):
        return float(0.0)
    
    # TODO
    def get_ball_distance_to_goal(self):
        return float(0.0)