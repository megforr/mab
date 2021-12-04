'''
Recreate MAB experiment described in Intro to RL (pg 28)

10-arm bandit
Stationary = rewards are not changing over time
Non-associative = have no states to associate action/reward with
Q* selected according to a gaussian normal distribution with mean 0 and variance 1
Actual reward was selected from a normal distribution with mean q*(At) and variance 1
N(mean, variance) = sigma * np.random.randn() + mu
    https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html

TODO: Need to run 2000 experiments and average results by timesteps to truly understand performance
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import deque
import numpy as np

import time
import datetime


def epsilon_greedy(epsilon, q_values):
    '''
    With probability < epsilon then explore, else select greedily
    :return: chosen action
    '''
    if np.random.uniform(0,1) <= epsilon:
        #print('explore')
        return random.randrange(action_space)
    else:
        #print('exploit')
        #return np.argmax(q_values) # if max ties then argmax only chooses first
        return np.random.choice(np.flatnonzero(q_values == q_values.max())) # if max ties break ties randomly

def update(old_estimate, target, step_size):
    '''
    In place update of values
    :return: new_estimate
    '''
    return old_estimate + step_size * (target - old_estimate)


action_space = 10
sigma = 1 # variance
action_mu = 0 # mean
q_star_values = action_mu + sigma * np.random.randn(1, action_space)

print('-------------------------------------')
print('Actual q-star values: ', q_star_values)
print('-------------------------------------')

no_steps = 1000
no_experiments = 2000

# value estimates by amount of exploration (epsilon)
q_t_values = {
    0: np.zeros(action_space),
    0.1: np.zeros(action_space),
    0.01: np.zeros(action_space)
}

q_t_rolling_avg_window = {
    0: deque(maxlen=50),
    0.1: deque(maxlen=50),
    0.01: deque(maxlen=50)
}

q_t_rolling_avg_values = {
    0: [],
    0.1: [],
    0.01: []
}


for t in np.arange(1, no_steps+1):

    print('Timestep: ', t)
    print('Current value estimates: ', q_t_values)

    for eps in q_t_values.keys():
        #print('Epsilon: ', eps)

        # select action according to epsilon value
        action = epsilon_greedy(eps, q_t_values[eps])
        #print('Action: ', action)

        # receive reward with some variance around q*-value
        reward = q_star_values[0][action] + sigma * np.random.randn()
        #print('Reward: ', reward)

        # update q-value estimates
        new_q_value = update(q_t_values[eps][action], reward, 1/t)
        q_t_values[eps][action] = new_q_value
        q_t_rolling_avg_window[eps].append(reward)
        q_t_rolling_avg_values[eps].append(np.mean(q_t_rolling_avg_window[eps]))
        #print('New estimated value: ', new_q_value)


time_now = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M')

q_t_values_df = pd.DataFrame(q_t_rolling_avg_values)
print(q_t_values_df.head())
sns.lineplot(np.arange(1,no_steps+1), y=q_t_values_df[0.00], color='green', label='0.0')
sns.lineplot(np.arange(1,no_steps+1), y=q_t_values_df[0.01], color='red', label='0.01')
sns.lineplot(np.arange(1,no_steps+1), y=q_t_values_df[0.10], color='blue', label='0.1')
plt.xlabel('Timesteps')
plt.ylabel('Rolling average reward')
plt.savefig(f'experiment_results/mab_10_arm_{time_now}.png')
plt.show()



