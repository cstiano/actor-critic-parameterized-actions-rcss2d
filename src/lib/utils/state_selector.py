import numpy as np
from src.lib.utils.state_wrapper import *

DEFAULT_STATE = 0
BALL_AXIS_POSITION_SPACE = 1

BALL_AXIS_POSITION_SPACE_SHAPE = (2,)


class StateSelector:
    def __init__(self, selected_state=DEFAULT_STATE):
        super().__init__()
        self.selected_state = selected_state

    def get_state(self, state):
        if self.selected_state == BALL_AXIS_POSITION_SPACE:
            return self.get_ball_axis_position_state(state)
        return np.array([0.0])

    def get_shape(self):
        if self.selected_state == BALL_AXIS_POSITION_SPACE:
            return BALL_AXIS_POSITION_SPACE_SHAPE
        return (0,)

    # This state delivers the positon of the agent relative to the ball
    # considering the ball as the (0,0) of the axis
    def get_ball_axis_position_state(self, state):
        state_wrapper = StateWrapper(state)
        agent_position = state_wrapper.get_position()
        ball_position = state_wrapper.get_ball_position()

        agent_x_relative = agent_position[0] - ball_position[0]
        agent_y_relative = agent_position[1] - ball_position[1]

        return np.array([agent_x_relative, agent_y_relative])
