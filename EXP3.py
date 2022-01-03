
import numpy as np
import math
from matplotlib import pyplot as plt

m = 10 # No. of bandits
T = 10000 # No. of rounds
num_runs = 20 # No. of independent runs (with T rounds each)
p_list = np.linspace(0.01, 1, m, endpoint=False) ## equally spaced in (0,1)

gamma = 0.05 # parameter for EXP3

bst_arm = np.argmax(p_list)

## Thompson Sampling

cum_reg = np.zeros((num_runs, T)) ## cumulative regret v/s time


# def select(weights):
#     return np.random.choice(np.arange(0,10).tolist(),
#                             p = weights)

for run in range(num_runs):

    wts = np.ones((m,))
    prob = np.zeros((m,))

    pull_count = np.zeros((m, )) ## Keep track of number of pulls per arms

    cum_reward = 0.
    cum_reward_best = 0.

    for t in range(T):

        prob = (((1-gamma)/np.sum(wts))*wts) + (gamma/m) # current probability vector at round-t
        w_t = np.random.multinomial(1, prob, size=1) #  choose an arm
        I_t = np.argmax(w_t)
        
        # the pulled arm reveals the reward
        X_ti = np.random.binomial(1,p_list[I_t])

        # unbiased estimate of reward
        est_reward = X_ti/prob[I_t]

        # update prob. vector
        wts[I_t] *= math.exp((gamma/m)*est_reward)

        # update pull counts
        pull_count[I_t] = pull_count[I_t] + 1

        cum_reward += X_ti
        cum_reward_best += (np.random.binomial(1,p_list[9]))

        cum_reg[run, t] = -cum_reward + cum_reward_best

    
mean_cum_reg = np.mean(cum_reg, axis=0)
std_cum_reg = np.std(cum_reg, axis=0)
plt.figure()
plt.plot(np.arange(len(mean_cum_reg)), mean_cum_reg)
plt.fill_between(np.arange(len(cum_reg[0, :])), mean_cum_reg - std_cum_reg, mean_cum_reg + std_cum_reg, alpha=.1)
plt.xlabel('# Rounds')
plt.ylabel('Cumulative Regret')
plt.title('EXP3')


