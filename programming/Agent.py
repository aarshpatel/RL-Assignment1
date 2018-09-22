"""
Class that represents the Agent in the environment. The agent has a policy, pi,
that is either uniform or optimal. The goal is find the optimal policy
for the agent
"""
import numpy as np

class Agent(object):
    """
    Represents an Agent in an environment. Has a policy
    associated with it
    """
    def __init__(self, num_states, num_actions, policy_type):
        self.num_states = num_states
        self.num_actions = num_actions
        if policy_type == "uniform":
            self.policy = self.uniform_policy()
        else:
            self.policy = self.optimal_policy()

    def uniform_policy(self):
        """ Create a uniform policy"""
        policy = np.zeros((self.num_states, self.num_actions))
        policy.fill(1/float(self.num_actions))
        return policy

    def optimal_policy(self):
        """ Create an optimal policy for this gridworld """
        policy = np.zeros((self.num_states, self.num_actions))
        policy[0][1] = 1.0
        policy[1][1] = 1.0
        policy[2][1] = 1.0
        policy[3][1] = 1.0
        policy[4][2] = 1.0
        policy[5][1] = 1.0
        policy[6][1] = 1.0
        policy[7][1] = 1.0
        policy[8][2] = 1.0
        policy[9][2] = 1.0
        policy[10][0] = 1.0
        policy[11][0] = 1.0
        policy[12][2] = 1.0
        policy[13][2] = 1.0
        policy[14][0] = 1.0
        policy[15][0] = 1.0
        policy[16][2] = 1.0
        policy[17][2] = 1.0
        policy[18][0] = 1.0
        policy[19][0] = 1.0
        policy[20][1] = 1.0
        policy[21][1] = 1.0
        policy[22][0] = 1.0
        return policy

    def train(self, current_state, current_action, current_reward, new_state):
        """ Alerts the agent that the transition has occurred and to update the agent's current policy """
        pass

    def get_current_policy(self):
        """ Returns the current policy of the agent """
        return self.policy

