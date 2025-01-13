#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

from pydoc import Helper
import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax
from Agent import BaseAgent

class QLearningAgent(BaseAgent):

    def update(self, sn, ao, re, nxts, xx, gamma):
        # Update Q-values using Q-learning algorithm
        self.Q_sa[sn, ao] += xx * (re + gamma * max(self.Q_sa[nxts]) - self.Q_sa[sn, ao])


def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' Runs a single repetition of Q-learning.
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    qrewards = []

    ns = env.reset()

    for _ in range(n_timesteps):
        ay = pi.prepare_action(ns, policy, epsilon, temp)
        next_ns, re, done = env.step(ay)
        qrewards.append(re)
        pi.update(ns, ay, re, next_ns, learning_rate, gamma)

        ns = env.reset() if done else next_ns

    return qrewards


def test():
    n_timesteps = 50000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    qrewards = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(qrewards))


if __name__ == '__main__':
    test()
