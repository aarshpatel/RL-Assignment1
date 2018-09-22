import argparse
import numpy as np
import math
import logging
from GridWorldEnvironment import GridWorldEnvironment
from Agent import Agent


if __name__ == "__main__":
    # setup ArgumentParser
    parser = argparse.ArgumentParser(description='Interact with the 687 GridWorld')
    parser.add_argument("-e", "--num_episodes",
                        required=True,
                        help="Number of episodes to run the agent-environment loop")
    parser.add_argument('-p', "--policy_type",
                        required=True,
                        help='The type of policy for the agent (uniform, optimal)')
    args = parser.parse_args()

    # setup the number the parameters of this gridworld (num states/actions)
    num_states = 23
    num_actions = 4
    num_episodes = int(args.num_episodes)
    policy_type = args.policy_type

    # setup logging
    logging.basicConfig(filename="grid_world_episodes_{}_policy_{}.log".format(num_episodes, policy_type), level=logging.INFO, filemode="w")

    # create the Environment and Agent
    grid_world = GridWorldEnvironment(num_states, num_actions)
    agent = Agent(num_states, num_actions, policy_type=policy_type)
    agent_policy = agent.get_current_policy()

    print("Running the Agent-Env Loop for {} episodes".format(num_episodes))

    all_rewards = []
    state_21_given_18_count = 0
    for episode in range(num_episodes):

        # sample initial state
        state = grid_world.get_initial_state()

        #  print "Initial State: {}".format(state)
        #  print "Current Grid World State: {}".format(grid_world.current_state)

        state_timestep_8 = None
        state_timestep_19 = None

        t = 0
        discounted_sum_of_rewards = 0
        while True:

            #  agent ==> get action from grid_world env pi(s)
            action = grid_world.get_action(state, agent_policy)
            #  sample a new state based on the transition function (s, a)
            new_state = grid_world.get_new_state(state, action)
            #  sample new reward based on dR (s, a, s')
            reward = grid_world.get_reward(state, action, new_state)
            # update the agent's policy
            agent.train(state, action, reward, new_state)

            # calculated the discounted reward at this timestep
            discounted_sum_of_rewards += ((grid_world.discount**t) * reward)

            if new_state == 23: # terminal state:
                break

            #  if t == 7:
                #  state_timestep_8 = new_state
            #  if t == 18:
                #  state_timestep_19 = new_state
                #  break

            state = new_state
            t += 1

        if state_timestep_8 == 18 and state_timestep_19 == 21:
            state_21_given_18_count += 1

        # reset the environment for the next episode
        grid_world.reset()

        if episode % 1000 == 0:
            logging.info("State 21 Given State 18: {}".format(state_21_given_18_count))
            logging.info("Episode: {}, Discounted Sum of Rewards: {}".format(episode, discounted_sum_of_rewards))

        all_rewards.append(discounted_sum_of_rewards)

    logging.info("Simulation: {}".format(state_21_given_18_count / float(num_episodes)))
    logging.info("Mean: {}".format(np.mean(np.array(all_rewards))))
    logging.info("Std: {}".format(np.std(np.array(all_rewards))))
    logging.info("Max: {}".format(np.max(np.array(all_rewards))))
    logging.info("Min: {}".format(np.min(np.array(all_rewards))))
