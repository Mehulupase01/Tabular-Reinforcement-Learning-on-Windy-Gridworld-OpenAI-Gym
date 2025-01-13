#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax

class BaseAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        # Initialize the BaseAgent with the specified parameters
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))  # Initialize Q-values matrix

    def prepare_action(self, ns, policy='egreedy', epsilon=None, temp=None):
        # Choose action based on the specified policy and parameters
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            else:
                # Select action using epsilon-greedy strategy
                a = argmax(self.Q_sa[ns,]) if np.random.random() >= epsilon else np.random.choice(
                    range(len(self.Q_sa[ns,])))
                return a
        elif policy == 'boltzmann':
            if temp is None:
                raise KeyError("Provide a temperature")
            # Select action using Boltzmann exploration
            p = softmax(self.Q_sa[ns,], temp)
            return np.random.choice(range(self.n_actions), 1, p=p)[0]

    def update(self):
        # Abstract method to be implemented in subclasses
        raise NotImplementedError('For each agent you need to implement its specific back-up method') # Leave this and overwrite in subclasses in other files

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        # Evaluate the agent's performance on the given evaluation environment
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')  # Select action using greedy policy during evaluation
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
