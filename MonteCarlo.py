#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax
from Agent import BaseAgent

class MonteCarloAgent(BaseAgent):

    def update(self, sg, ao, rr, gs):
        # Update Q-values using Monte Carlo method
        self.Q_sa[sg, ao] += gs * (rr - self.Q_sa[sg, ao])


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, policy='egreedy', eps=None, temp=None, plot=True):
    ''' Runs a single repetition of a Monte Carlo RL agent.
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)

    monte_rewards = []

    while n_timesteps > 0:
        NSL = env.reset()
        S = [NSL]
        A = []
        Final_rewards = []

        for j in range(max_episode_length):
            n_timesteps -= 1
            vh = pi.prepare_action(S[j], policy, eps, temp)
            nxts, l, done = env.step(vh)
            A.append(vh)
            S.append(nxts)
            Final_rewards.append(l)
            monte_rewards.append(l)

            if done or n_timesteps == 0:
                break

        Return = 0
        for i in range(j, -1, -1):  # inverse: i = T-1, T-2 ,..., 0
            Return = Final_rewards[i] + gamma * Return
            pi.update(S[i], A[i], Return, learning_rate)

    return monte_rewards


def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    monterewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma,
                          policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(monterewards))


if __name__ == '__main__':
    test()
