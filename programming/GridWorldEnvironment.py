"""
This class represents the GridWorldEnvironment for RL 687 class.
It essentially represents an MDP for the environment
with states, action, rewards, transitions
"""
import numpy as np
import math

class GridWorldEnvironment(object):
    """
    Represents the GridWorld for CS687
    All the information about this GridWorld is provided in the notes.
    """

    def __init__(self, num_states, num_actions, initial_state=1, terminal_state=23, discount_factor = .9):
        self.num_states = num_states
        self.num_actions = num_actions
        self.actions = np.array(["AU", "AR", "AD", "AL"])
        self.env_dynamics = {
            "S": .80,
            "VR": .05,
            "VL": .05,
            "B": .10
        }

        self.rewards = {
            "W": -10,
            "G": 10,
            "O": 0
        }

        # discount reward factor
        self.discount = discount_factor

        # initial state for the agent
        self.initial_state = initial_state

        # goal state
        self.terminal_state = terminal_state

        # current state of the agent
        self.current_state = self.initial_state

    def get_initial_state(self):
        """
        Return the inital state of the episode. This is like sampling from d_0.
        The initial state is always State 1 in this gridworld
        """
        return self.initial_state

    def get_action(self, state, agent_policy):
        """ Sample an action from the agent's policy given the current state """
        action_prob_dist = agent_policy[state-1].reshape(-1,)
        action = np.random.choice(self.actions.reshape(-1,), p=action_prob_dist)
        return action

    def get_new_state(self, current_state, current_action):
        """ Return a sampled new state, s_{t+1} given the current state, s_t, and current action, a_t"""
        # the terminal state always goes to terminal absorbing state with prob 1
        if current_state == self.terminal_state:
            return "s_inf"

        env_dynamics_keys = np.array(self.env_dynamics.keys()).reshape(-1,)
        env_dynamics_probs = np.array(self.env_dynamics.values()).reshape(-1,)
        transition = np.random.choice(env_dynamics_keys, p=env_dynamics_probs)

        if transition == "S":
            return self.take_action(current_action)
        elif transition == "B":
            return self.current_state
        elif transition == "VL" and current_action == "AL":
            return self.take_action("AD")
        elif transition == "VL" and current_action == "AD":
            return self.take_action("AR")
        elif transition == "VL" and current_action == "AR":
            return self.take_action("AU")
        elif transition == "VL" and current_action == "AU":
            return self.take_action("AL")
        elif transition == "VR" and current_action == "AL":
            return self.take_action("AU")
        elif transition == "VR" and current_action == "AD":
            return self.take_action("AL")
        elif transition == "VR" and current_action == "AR":
            return self.take_action("AD")
        elif transition == "VR" and current_action == "AU":
            return self.take_action("AR")
        return None

    def get_reward(self, current_state, current_action, new_state):
        """
        Return a sampled reward from d_R given current state, s_t, current action, a_t
        and new state, s_{t+1}
        """
        if new_state == 21:
            return self.rewards["W"]
        elif new_state == 23:
            return self.rewards["G"]
        return self.rewards["O"]

    def take_action(self, action):
        """
        Takes the current action and updates the internal current state
        representation and returns the new state
        """
        can_take_action = self.check_valid_action(action, self.current_state)
        if can_take_action:
            if action == "AL":
                self.current_state = self.current_state - 1
            elif action == "AU":
                if self.current_state < 13 or self.current_state > 21:
                    self.current_state = self.current_state - 5
                else:
                    self.current_state = self.current_state - 4
            elif action == "AD":
                if self.current_state < 8 or self.current_state > 16:
                    self.current_state = self.current_state + 5
                else:
                    self.current_state = self.current_state + 4
            elif action == "AR":
                self.current_state = self.current_state + 1
        return self.current_state

    def check_valid_action(self, action, current_state):
        """
        Checks to see if the action taken results in a valid state transition
        """
        if (action == "AL"  and current_state == 13) or (action == "AR" and current_state == 12):
            return False
        elif (action == "AR" and current_state ==16) or (action == "AL" and current_state == 17):
            return False
        elif action == "AU" and current_state in [1, 2, 3, 4, 5]:
            return False
        elif action == "AL" and current_state in [1, 6, 11, 15, 19]:
            return False
        elif action == "AR" and current_state in [5, 10, 14, 18, 23]:
            return False
        elif action == "AD" and current_state in [8, 19, 20, 21, 22, 23]:
            return False
        elif action == "AU" and current_state == 21:
            return False
        return True

    def reset(self):
        """ Reset the environment after each episode """
        self.current_state = self.initial_state
