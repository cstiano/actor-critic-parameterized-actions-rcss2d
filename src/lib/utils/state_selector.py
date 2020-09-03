import numpy as np
from src.lib.utils.state_wrapper import *

DEFAULT_STATE = 0
BALL_AXIS_POSITION_SPACE = 1
AGENT_BALL_SPACE_SPACE = 2
AGENT_ORIENTATION_AND_BALL_POSITION_SPACE = 3
WITHOUT_OPPONENT_INFO_SPACE = 4

BALL_AXIS_POSITION_SPACE_SHAPE = (2,)
AGENT_BALL_SPACE_SHAPE = (4,)
AGENT_ORIENTATION_AND_BALL_POSITION_SHAPE = (5,)
WITHOUT_OPPONENT_INFO_SHAPE = (9,)


class StateSelector:
    def __init__(self, selected_state=DEFAULT_STATE):
        super().__init__()
        self.selected_state = selected_state

    def get_state(self, state):
        if self.selected_state == BALL_AXIS_POSITION_SPACE:
            return self.get_ball_axis_position_state(state)
        elif self.selected_state == AGENT_BALL_SPACE_SPACE:
            return self.get_agent_and_ball_position_state(state)
        elif self.selected_state == AGENT_ORIENTATION_AND_BALL_POSITION_SPACE:
            return self.get_anget_orientation_ball_state(state)
        elif self.selected_state == WITHOUT_OPPONENT_INFO_SPACE:
            return self.get_state_without_opponent_info(state)
        return np.array([0.0])

    def get_shape(self):
        if self.selected_state == BALL_AXIS_POSITION_SPACE:
            return BALL_AXIS_POSITION_SPACE_SHAPE
        elif self.selected_state == AGENT_BALL_SPACE_SPACE:
            return AGENT_BALL_SPACE_SHAPE
        elif self.selected_state == AGENT_ORIENTATION_AND_BALL_POSITION_SPACE:
            return AGENT_ORIENTATION_AND_BALL_POSITION_SHAPE
        elif self.selected_state == WITHOUT_OPPONENT_INFO_SPACE:
            return WITHOUT_OPPONENT_INFO_SHAPE
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

    def get_agent_and_ball_position_state(self, state):
        state_wrapper = StateWrapper(state)
        agent_position = state_wrapper.get_position()
        ball_position = state_wrapper.get_ball_position()
        return np.array([agent_position[0], agent_position[1], ball_position[0], ball_position[1]])

    def get_anget_orientation_ball_state(self, state):
        return np.array([state[0], state[1], state[2], state[3], state[4]])

    def get_state_without_opponent_info(self, state):
        return np.array([state[0], state[1], state[2],
                        state[3], state[4], state[5],
                        state[6], state[7], state[8]])
