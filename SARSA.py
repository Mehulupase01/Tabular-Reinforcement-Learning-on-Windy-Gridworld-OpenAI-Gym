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

class SarsaAgent(BaseAgent):

    def update(self, sn, ao, rn, nxts, nxty, xx, gamma):
        # Update Q-values using SARSA algorithm
        self.Q_sa[sn, ao] += xx * (rn + gamma * self.Q_sa[nxts, nxty] - self.Q_sa[sn, ao])


def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' Runs a single repetition of SARSA.
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    sar_rewards = []

    ns = env.reset()
    ay = pi.prepare_action(ns, policy, epsilon, temp)

    for _ in range(n_timesteps):
        next_ns, re, done = env.step(ay)
        sar_rewards.append(re)
        next_ay = pi.prepare_action(next_ns, policy, epsilon, temp)
        pi.update(ns, ay, re, next_ns, next_ay, learning_rate, gamma)

        if done:
            ns = env.reset()
            ay = pi.prepare_action(ns, policy, epsilon, temp)
        else:
            ns, ay = next_ns, next_ay

    return sar_rewards


def test():
    n_timesteps = 100000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    sar_rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(sar_rewards))


if __name__ == '__main__':
    test()
