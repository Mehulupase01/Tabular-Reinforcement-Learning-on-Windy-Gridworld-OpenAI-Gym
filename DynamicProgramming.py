#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax


class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        # Initialize QValueIterationAgent with specified parameters
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Sa_q = np.zeros((n_states, n_actions))  # (7x10) x 4
        self.V = np.zeros(n_states)

    def prepare_action(self, ns):
        ''' Returns the greedy best action in state s '''
        dpaction = np.argwhere(self.Sa_q[ns,] == np.amax(self.Sa_q[ns,])).flatten()

        if dpaction.shape[0] > 1:
            dpaction = np.random.choice(dpaction)  # If there are multiple maxima

        return dpaction[0] if dpaction.shape[0] == 1 else dpaction

    def update(self, env, gamma=1.0):
        ''' Function updates Q(s,a) using p_sas and r_sas '''

        delta = 0
        for sd in range(self.n_states):
            for ay in range(self.n_actions):
                t_sa, ra_m = env.model(sd, ay)
                q_sum = np.sum(t_sa[next_ns] * (ra_m[next_ns] + gamma * np.max(self.Sa_q[next_ns,])) for next_ns in
                               np.where(t_sa != 0)[0])
                current_Q_sa = self.Sa_q[sd, ay]  # Store current estimate
                self.Sa_q[sd, ay] = q_sum
                delta = max(np.abs((current_Q_sa - q_sum)), delta)
                self.V[sd] = np.mean(self.Sa_q[sd, :])

        return delta, self.Sa_q, self.V


def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''

    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    Sa_q_list = []
    i = 0

    while True:
        delta, Q_sa, V = QIagent.update(env, gamma)
        Sa_q_list.append(np.array(Q_sa))  # for analysis only
        i += 1
        print(f"Q-value iteration, iteration {i}, max error {delta}")

        if delta < threshold:
            break

    # for analysis only: beginning, midway and convergence.
    env.render(Q_sa=Sa_q_list[0], plot_optimal_policy=True, step_pause=5)
    env.render(Q_sa=np.array(Sa_q_list[len(Sa_q_list) // 2]), plot_optimal_policy=True, step_pause=5)
    env.render(Q_sa=np.array(Sa_q_list[-1]), plot_optimal_policy=True, step_pause=5)

    return QIagent


def experiment():
    # Value iteration using action-value function (i.e., Q(s,a))
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)  # initialize environment
    env.render()
    QIagent = Q_value_iteration(env, gamma, threshold)  # update

    # View optimal policy
    done = False
    s = env.reset()
    rewards = []
    while not done:
        a = QIagent.prepare_action(s)
        s_next, r, done = env.step(a)
        rewards.append(r)
        env.render(Q_sa=QIagent.Sa_q, plot_optimal_policy=True, step_pause=0.05)
        s = s_next

    # Open a file for writing the output
    with open("experiment_output.txt", "w") as output_file:
        mean_reward_per_timestep = np.mean(rewards)
        output_file.write("Mean reward per timestep under optimal policy: {}\n".format(mean_reward_per_timestep))
        print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

        final_V_values = QIagent.V
        state_of_interest = 3  # Replace with the desired state index
        V_s = final_V_values[state_of_interest]
        output_file.write("Final V-value for state {}: {}\n".format(state_of_interest, V_s))
        print("Final V-value for state {}: {}".format(state_of_interest, V_s))


if __name__ == '__main__':
    experiment()
