import argparse
import numpy as np
import math
from GridWorldEnvironment import GridWorldEnvironment
from Agent import Agent


if __name__ == "__main__":
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

    # create the Environment and Agent
    grid_world = GridWorldEnvironment(num_states, num_actions, 1, 23, .9)
    agent = Agent(num_states, num_actions, policy_type=policy_type)
    agent_policy = agent.get_current_policy()

    print("Running the Agent-Env Loop for {} episodes".format(num_episodes))

    all_rewards = []
    state_21_given_18_count = 0
    for episode in range(num_episodes):

        states = []

        # sample initial state
        state = grid_world.get_initial_state()

        states.append(state)

        t = 0
        discounted_sum_of_rewards = 0
        while True:

            #  agent ==> get action from grid_world env pi(s)
            action = grid_world.get_action(state, agent_policy)
            #  sample a new state based on the transition function (s, a)
            new_state = grid_world.get_new_state(state, action)

            states.append(new_state)

            #  sample new reward based on dR (s, a, s')
            reward = grid_world.get_reward(state, action, new_state)
            # update the agent's policy
            agent.train(state, action, reward, new_state)

            #  print "Transition at time {}: ({}, {}, {}, {})".format(t, state, action, reward, new_state)

            # calculated the discounted reward at this timestep
            discounted_sum_of_rewards += ((grid_world.discount**t) * reward)

            if new_state == 23: # terminal state:
                break

            if t == 18:
                break

            state = new_state
            t += 1

        if len(states) > 18:
            if states[7] == 18 and states[18] == 21:
                state_21_given_18_count += 1

        # reset the environment for the next episode
        grid_world.reset()

        if episode % 1000 == 0:
            print("State 21 Given State 18: {}".format(state_21_given_18_count))
            print "Episode: {}, Discounted Sum of Rewards: {}".format(episode, discounted_sum_of_rewards)

        all_rewards.append(discounted_sum_of_rewards)

    print("Simulation: {}".format(state_21_given_18_count / float(num_episodes)))
    print "Mean: {}".format(np.mean(np.array(all_rewards)))
    print "Std: {}".format(np.std(np.array(all_rewards)))
    print "Max: {}".format(np.max(np.array(all_rewards)))
    print "Min: {}".format(np.min(np.array(all_rewards)))
