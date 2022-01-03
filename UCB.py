
import numpy as np
import math
from matplotlib import pyplot as plt

eps = 1 # Choose epsilon: 0.1 or 1
m = 10 # No. of bandits
T = 10000 # No. of rounds
num_runs = 20 # No. of independent runs (with T rounds each)
probs = np.linspace(0.01, 1, m, endpoint=False) ## equally spaced in (0,1)

bst_arm = np.argmax(probs)

## Explore-Then-Commit (ETC)

cum_reg = np.zeros((num_runs, T)) ## cumulative regret v/s time

for run in range(num_runs):

    samp_mean = np.zeros((m, 1)) ## Keep track of Sample Mean per arm
    reward = np.zeros((m, 1)) ## Keep track of rewards per arm
    pull_count = np.zeros((m, 1)) ## Keep track of number of pulls per arms
    # optim_bon = np.zeros((m, 1)) ## Optimism bonus per arm

    cum_reward = 0.
    cum_reward_best = 0.

    for t in range(T):
        if t < m:
            # pull arms in round-robin fashion
            I_t = t
        else:
            # be optimistic while pulling arm
            optim_bon = math.sqrt(2*math.log(t))*(1./np.sqrt(pull_count))
            I_t = np.argmax(samp_mean + optim_bon)

        # the pulled arm reveals the reward
        X_ti = np.random.binomial(1,probs[I_t])

        # update rewards, pull counts and sample means
        reward[I_t] = reward[I_t] + X_ti
        pull_count[I_t] = pull_count[I_t] + 1
        samp_mean[I_t] = reward[I_t]/pull_count[I_t]

        cum_reward += X_ti
        cum_reward_best += np.random.binomial(1,probs[9]) ## count rewards if best arm was pulled at round-t

        cum_reg[run, t] = - cum_reward + cum_reward_best
    
mean_cum_reg = np.mean(cum_reg, axis=0)
std_cum_reg = np.std(cum_reg, axis=0)
plt.figure()
plt.plot(np.arange(len(mean_cum_reg)), mean_cum_reg)
plt.fill_between(np.arange(len(cum_reg[0, :])), mean_cum_reg - std_cum_reg, mean_cum_reg + std_cum_reg, alpha=.1)
plt.xlabel('# Rounds')
plt.ylabel('Cumulative Regret')
plt.title(f'UCB')