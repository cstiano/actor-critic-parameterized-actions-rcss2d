from scipy.spatial import distance

GOAL_CENTER_POSITION = (52.5, 0)


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

    def get_distance_to_ball(self):
        agent = self.get_position()
        ball = self.get_ball_position()
        return distance.euclidean(agent, ball)
    
    def get_distance_to_goal(self):
        agent = self.get_position()
        return distance.euclidean(agent, GOAL_CENTER_POSITION)
    
    def get_ball_distance_to_goal(self):
        ball = self.get_ball_position()
        return distance.euclidean(ball, GOAL_CENTER_POSITION)
    
    # TODO
    # def getRelativeRobotToBallAngle(self):
    #     agent = self.get_position()
    #     ball = self.get_ball_position()
    #     dist_left = [ball[0] - agent[0], ball[1] - agent[1]]
    #     angle_left = self.toPiRange(angle(dist_left[0], dist_left[1]) - frame.robotsBlue[0].theta)
    #     #print(angle_left)
    #     return angle_left
    
    # def toPiRange(self, angle):
    #     angle = math.fmod(angle, 2 * math.pi)
    #     if angle < -math.pi:
    #         angle = angle + 2 * math.pi
    #     elif angle > math.pi:
    #         angle = angle - 2 * math.pi

    #     return angle
