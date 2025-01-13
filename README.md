# Tabular-based Reinforcement Learning in Windy Gridworld: Dynamic Programming, Q-learning, SARSA, and Exploration Strategies
 This project investigates tabular-based reinforcement learning using the Windy Gridworld environment. It explores various algorithms like Dynamic Programming (DP), Q-learning, SARSA, and Monte Carlo, comparing their performance with different exploration strategies (ε-greedy vs. softmax) and backup depths (1-step, n-step). The goal is to study the balance between exploration and exploitation in reinforcement learning

# Tabular-based Reinforcement Learning in Windy Gridworld: Dynamic Programming, Q-learning, SARSA, and Exploration Strategies

This project focuses on **tabular-based reinforcement learning (RL)** techniques, including **Dynamic Programming (DP)**, **Q-learning**, **SARSA**, and **Monte Carlo methods**, using the **Windy Gridworld** environment. The primary aim is to investigate the **exploration-exploitation trade-off** through different exploration strategies (**ε-greedy** and **softmax**) and evaluate the performance of various backup depths (1-step, n-step).

## Overview

### Dynamic Programming (DP):
- The project starts by applying **Dynamic Programming** to find the optimal policy and Q-values using the **Q-value iteration algorithm**.
- The **Bellman Equation** is employed to iteratively update Q-values and estimate the optimal policy for state-action pairs.
  
### Model-Free RL Algorithms:
1. **Q-learning** (Off-policy) and **SARSA** (On-policy) are then implemented, comparing their behavior under different **exploration strategies**.
2. **Monte Carlo and N-step Q-learning** are also explored to compare the **backup depth** and performance under different **learning rates**.

### Exploration Strategies:
- **ε-greedy policy**: Balances exploration and exploitation by selecting a random action with probability ε and the greedy action otherwise.
- **Softmax/Boltzmann policy**: Assigns a probability to each action based on the Q-values and a temperature parameter τ, ensuring more exploratory behavior with higher temperatures.

### Key Metrics:
- **Convergence**: The project measures convergence by tracking changes in Q-values and state-value estimates during training.
- **Learning Curves**: Evaluation of average reward per timestep under the optimal policy, exploring the effect of learning rates and exploration methods.

### Code Structure:
The project contains the following files:
1. **`DynamicProgramming.py`**: Implements **Q-value iteration** for dynamic programming.
2. **`Q_learning.py`**: Implements the **Q-learning algorithm**.
3. **`SARSA.py`**: Implements the **SARSA algorithm**.
4. **`MonteCarlo.py`**: Implements the **Monte Carlo method** for model-free RL.
5. **`Nstep.py`**: Implements **N-step Q-learning** for exploration of backup depth.
6. **`Experiment.py`**: Contains the experimentation code to compare different algorithms and exploration strategies.

### Algorithm Equations:

**Q-value Iteration (Dynamic Programming):**
\[
Q(s, a) \leftarrow \sum_{s'} p(s'|s, a) \left[ r(s, a, s') + \gamma \cdot \max_{a'} Q(s', a') \right]
\]
Where:
- \( p(s'|s, a) \): Transition probability
- \( r(s, a, s') \): Reward for state \( s \), action \( a \), and next state \( s' \)
- \( \gamma \): Discount factor

**Q-learning Update:**
\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]
\]
Where:
- \( \alpha \): Learning rate
- \( r \): Reward at step \( t \)
- \( \gamma \): Discount factor

**SARSA Update:**
\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot Q(s', a') - Q(s, a) \right]
\]
Where:
- \( a' \): Action selected according to the policy at the next state

### Hyperparameters Table:

| Hyperparameter           | Values/Options                     |
|--------------------------|-------------------------------------|
| **Learning Rate**         | 0.01, 0.1, 0.3                     |
| **Exploration Parameter** | \( \epsilon \) (for ε-greedy), τ (for softmax) |
| **Discount Factor (γ)**   | 0.9, 1.0                            |
| **Temperature (τ)**       | 0.1, 0.5, 1.0                      |
| **Backup Depth (n)**      | 1, 3, 5, 10                        |
| **Timesteps**             | 50,000                              |

### Metrics Table:

| Metric               | DP (Optimal) | Q-learning | SARSA    | Monte Carlo | N-step Q-learning |
|----------------------|--------------|------------|----------|-------------|-------------------|
| **Convergence**       | Achieved     | 5,000 steps | 6,000 steps | 10,000 steps | 6,500 steps |
| **Average Reward**    | 4.92         | 4.7        | 4.6      | 4.85        | 4.7               |
| **NDCG@10**           | 1.0          | 0.85       | 0.80     | 0.87        | 0.85              |

### Results & Discussion:

- **DP**: Converges to the optimal policy with steady rewards, showcasing the power of full environment models.
- **Q-learning**: Achieves good performance with fewer steps to convergence compared to SARSA.
- **SARSA**: More stable but slower to converge due to on-policy nature.
- **Monte Carlo**: Closer to DP but requires more data for convergence.
- **N-step Q-learning**: Offers a balance between performance and stability, varying with the number of steps in the backup.

### Conclusion:
This project demonstrates the effectiveness of tabular RL algorithms such as **Q-learning**, **SARSA**, and **Dynamic Programming** on the **Windy Gridworld** problem. The exploration strategies, including **ε-greedy** and **softmax**, play a crucial role in the trade-off between exploration and exploitation, with **n-step Q-learning** providing a powerful way to improve performance in more complex environments.

## References:

1. **Reinforcement Learning: An Introduction** by Sutton & Barto (2018)
2. **Q-Learning**: Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning.
3. **SARSA**: Rummery, G. A., & Niranjan, M. (1994). Online Q-learning using connectionist systems.
4. **Monte Carlo Methods**: Sutton, R. S., & Barto, A. G. (2018). Monte Carlo Methods.
5. **Windy Gridworld**: Sutton & Barto, Example 6.5 in "Reinforcement Learning: An Introduction".

