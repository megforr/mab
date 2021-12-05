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
    Update values in place
    :return: new_estimate
    '''
    return old_estimate + step_size * (target - old_estimate)

time_now = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M')

action_space = 10
sigma = 1 # variance
action_mu = 0 # mean
no_steps = 1000
no_experiments = 2000

# store experiment results
# 1. store the rolling average at each timestep for each experiment
exp_q_t_rolling_average_values = {}
# 2. store the loss between q_* and q_t values at each timestep
#exp_q_t_loss = {}

for exp in np.arange(1, no_experiments+1):

    print('Experiment: ', exp)
    exp_q_t_rolling_average_values[exp] = {}
    #exp_q_t_loss[exp] = {}

    q_star_values = action_mu + sigma * np.random.randn(1, action_space)

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
        0: {},
        0.1: {},
        0.01: {}
    }

    for t in np.arange(1, no_steps+1):

        # if t % 250 == 0:
        #     print('Timestep: ', t)
        #     print('Current value estimates: ', q_t_values)

        for eps in q_t_values.keys():
            # select action according to epsilon value
            action = epsilon_greedy(eps, q_t_values[eps])

            # receive reward with some variance around q*-value
            reward = q_star_values[0][action] + sigma * np.random.randn()

            # update q-value estimates
            new_q_value = update(q_t_values[eps][action], reward, 1/t)
            q_t_values[eps][action] = new_q_value

            # store
            q_t_rolling_avg_window[eps].append(reward)
            q_t_rolling_avg_values[eps][t] = np.mean(q_t_rolling_avg_window[eps])

    exp_q_t_rolling_average_values[exp] = q_t_rolling_avg_values
    #exp_q_t_loss[exp] = {}

data_rows = []
for exp in exp_q_t_rolling_average_values:
    for eps in exp_q_t_rolling_average_values[exp]:
        for t in exp_q_t_rolling_average_values[exp][eps]:
            data_rows.append([exp, eps, t, exp_q_t_rolling_average_values[exp][eps][t]])

df = pd.DataFrame(data_rows, columns=['experiment','epsilon','timestep','q_t_values'])
avg_df = df.groupby(['epsilon','timestep']).agg(avg_q_values=('q_t_values','mean'),
                                                std_q_values=('q_t_values','std')
                                                ).reset_index()

palette = {
    0: 'tab:green',
    0.01: 'tab:red',
    0.1: 'tab:blue',
}

sns.lineplot(x='timestep', y='avg_q_values', data=avg_df, hue='epsilon', palette=palette)
plt.xlabel('Timesteps')
plt.ylabel('Average reward')
plt.savefig(f'experiment_results/mab_10_arm_2000experiments_{time_now}.png')
plt.show()

sns.lineplot(x='timestep', y='q_t_values', data=df, hue='epsilon', palette=palette)
plt.xlabel('Timesteps')
plt.ylabel('Average reward')
plt.savefig(f'experiment_results/mab_10_arm_2000experiments_boxplot_{time_now}.png')
plt.show()






