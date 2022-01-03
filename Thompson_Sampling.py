
import numpy as np
# import math
from matplotlib import pyplot as plt

m = 10 # No. of bandits
T = 10000 # No. of rounds
num_runs = 20 # No. of independent runs (with T rounds each)
probs = np.linspace(0.01, 1, m, endpoint=False) ## equally spaced in (0,1)

bst_arm = np.argmax(probs)

## Thompson Sampling

cum_reg = np.zeros((num_runs, T)) ## cumulative regret v/s time


for run in range(num_runs):

    S = np.zeros((m, )) ## Success per arm
    params = np.zeros((m, ))

    pull_count = np.zeros((m, )) ## Keep track of number of pulls per arms

    cum_reward = 0.
    cum_reward_best = 0.

    for t in range(T):

        for i in range(m):
            F = pull_count[i] - S[i] ## No. of failures for arm-i at round-t
            params[i] = np.random.beta(1 + S[i], 1 + F)

        I_t = np.argmax(params)

        # the pulled arm reveals the reward
        X_ti = np.random.binomial(1,probs[I_t])

        # Update S
        S[I_t] += X_ti

        # update pull counts
        pull_count[I_t] += 1

        cum_reward += X_ti
        cum_reward_best += np.random.binomial(1,probs[9]) ## count reward if best arm was pulled at round-t

        cum_reg[run, t] = - cum_reward + cum_reward_best 

    
mean_cum_reg = np.mean(cum_reg, axis=0)
std_cum_reg = np.std(cum_reg, axis=0)
plt.figure()
plt.plot(np.arange(len(mean_cum_reg)), mean_cum_reg)
plt.fill_between(np.arange(len(cum_reg[0, :])), mean_cum_reg - std_cum_reg, mean_cum_reg + std_cum_reg, alpha=.1)
plt.xlabel('# Rounds')
plt.ylabel('Cumulative Regret')
plt.title('Thompson Sampling')