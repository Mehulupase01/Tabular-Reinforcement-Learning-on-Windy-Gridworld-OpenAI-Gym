#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from sympy import re
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax
from Agent import BaseAgent

class NstepQLearningAgent(BaseAgent):

    def update(self, sl, ao, rr, nx):
        # Update Q-values using n-step Q-learning algorithm
        self.Q_sa[sl, ao] += nx * (rr - self.Q_sa[sl, ao])


def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
             policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    """
      Runs a single episode of n-step Q-learning.

      Args:
          n_timesteps (int): Total number of timesteps to simulate.
          max_episode_length (int): Maximum length of an episode.
          learning_rate (float): Learning rate for updating Q-values.
          gamma (float): Discount factor for future rewards.
          policy (str, optional): Exploration policy ('egreedy' or 'softmax'). Defaults to 'egreedy'.
          epsilon (float, optional): Epsilon value for epsilon-greedy exploration. Defaults to None.
          temp (float, optional): Temperature value for softmax exploration. Defaults to None.
          plot (bool, optional): Whether to visualize the environment during learning. Defaults to True.
          n (int, optional): Number of steps to consider in the target calculation. Defaults to 5.

      Returns:
          list: List of rewards received at each timestep.
      """
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)

    n_rewards = []

    while n_timesteps > 0:
        ns = env.reset()
        S, A, final_rewards = [ns], [], []

        for _ in range(max_episode_length):
            n_timesteps -= 1
            af = pi.prepare_action(S[-1], policy, epsilon, temp)
            next_fs, r, done = env.step(af)
            A.append(af)
            S.append(next_fs)
            final_rewards.append(r)
            n_rewards.append(r)

            if done or n_timesteps == 0:
                break

        f_ep = len(final_rewards)

        for sd in range(f_ep):
            m = min(n, f_ep - sd)
            Return = 0

            for i in range(m):
                Return += (gamma ** i) * final_rewards[sd + i]

            if not np.all(env._state_to_location(S[sd + m]) == (7, 3)):
                Return += (gamma ** m) * max(pi.Q_sa[S[sd + m],])

            pi.update(S[sd], A[sd], Return, learning_rate)

    return n_rewards


def test():
    n_timesteps = 50000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5

    # Exploration
    policy = 'egreedy'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    # Plotting parameters
    plot = True

    nrewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma,
                       policy, epsilon, temp, plot=True, n=n)
    print("Obtained rewards: {}".format(nrewards))


if __name__ == '__main__':
    test()
