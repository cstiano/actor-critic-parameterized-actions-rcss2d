import numpy as np
from src.lib.utils.state_wrapper import *

HFO_STATE = -1
DEFAULT_STATE = 0
BALL_AXIS_POSITION_SPACE = 1
AGENT_BALL_SPACE_SPACE = 2
AGENT_ORIENTATION_AND_BALL_POSITION_SPACE = 3
WITHOUT_OPPONENT_INFO_SPACE = 4
AGENT_ORIENTATION_AND_BALL_POSITION_KICKABLE_SPACE = 5
AGENT_AND_BALL_POSITION_KICKABLE_SPACE = 6
BALL_AXIS_AND_BALL_POSITION_KICKABLE_SPACE = 7

BALL_AXIS_POSITION_SPACE_SHAPE = (2,)
AGENT_BALL_SPACE_SHAPE = (4,)
AGENT_ORIENTATION_AND_BALL_POSITION_SHAPE = (5,)
WITHOUT_OPPONENT_INFO_SHAPE = (9,)
AGENT_ORIENTATION_AND_BALL_POSITION_KICKABLE_SHAPE = (6,)
AGENT_AND_BALL_POSITION_KICKABLE_SHAPE = (5,)
BALL_AXIS_AND_BALL_POSITION_KICKABLE_SHAPE = (5,)
HFO_SHAPE = (9,)


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
            return self.get_agent_orientation_ball_state(state)
        elif self.selected_state == WITHOUT_OPPONENT_INFO_SPACE:
            return self.get_without_opponent_info_state(state)
        elif self.selected_state == AGENT_ORIENTATION_AND_BALL_POSITION_KICKABLE_SPACE:
            return self.get_agent_orientation_ball_kickable_state(state)
        elif self.selected_state == AGENT_AND_BALL_POSITION_KICKABLE_SPACE:
            return self.get_agent_and_ball_kickable_state(state)
        elif self.selected_state == BALL_AXIS_AND_BALL_POSITION_KICKABLE_SPACE:
            return self.get_ball_axis_and_ball_position_kickable_state(state)
        elif self.selected_state == HFO_STATE:
            return self.get_hfo_state(state)
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
        elif self.selected_state == AGENT_ORIENTATION_AND_BALL_POSITION_KICKABLE_SPACE:
            return AGENT_ORIENTATION_AND_BALL_POSITION_KICKABLE_SHAPE
        elif self.selected_state == AGENT_AND_BALL_POSITION_KICKABLE_SPACE:
            return AGENT_AND_BALL_POSITION_KICKABLE_SHAPE
        elif self.selected_state == BALL_AXIS_AND_BALL_POSITION_KICKABLE_SPACE:
            return BALL_AXIS_AND_BALL_POSITION_KICKABLE_SHAPE
        elif self.selected_state == HFO_STATE:
            return HFO_SHAPE
        return (0,)

    def get_hfo_state(self, state):
        return np.array([state[0], state[1], state[2],
                         state[3], state[4], state[5],
                         state[12], state[13], state[14]])

    # This state delivers the positon of the agent relative to the ball
    # considering the ball as the (0,0) of the axis
    def get_ball_axis_position_state(self, state):
        ball_axis_position = self.get_ball_axis_position(state)
        return np.array([ball_axis_position[0], ball_axis_position[1]])

    def get_agent_and_ball_position_state(self, state):
        state_wrapper = StateWrapper(state)
        agent_position = state_wrapper.get_position()
        ball_position = state_wrapper.get_ball_position()
        return np.array([agent_position[0], agent_position[1], ball_position[0], ball_position[1]])

    def get_agent_orientation_ball_state(self, state):
        return np.array([state[0], state[1], state[2], state[3], state[4]])

    def get_without_opponent_info_state(self, state):
        return np.array([state[0], state[1], state[2],
                         state[3], state[4], state[5],
                         state[6], state[7], state[8]])

    def get_agent_orientation_ball_kickable_state(self, state):
        return np.array([state[0], state[1], state[2], state[3], state[4], state[5]])

    def get_agent_and_ball_kickable_state(self, state):
        return np.array([state[0], state[1], state[3], state[4], state[5]])

    def get_ball_axis_and_ball_position_kickable_state(self, state):
        ball_axis_position = self.get_ball_axis_position(state)
        return np.array([ball_axis_position[0], ball_axis_position[1], state[3], state[4], state[5]])

    def get_ball_axis_position(self, state):
        state_wrapper = StateWrapper(state)
        agent_position = state_wrapper.get_position()
        ball_position = state_wrapper.get_ball_position()

        agent_x_relative = agent_position[0] - ball_position[0]
        agent_y_relative = agent_position[1] - ball_position[1]

        return (agent_x_relative, agent_y_relative)
