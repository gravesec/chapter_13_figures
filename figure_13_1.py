'''
This script requires Python 3.7.0 and the following packages:
numpy==1.16.3
matplotlib==3.0.3 (for plotting results)
joblib==0.13.2 (for running experiments in parallel)
scipy==1.3.1 (for computing error bars)
'''


import argparse
import os.path
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


class ShortCorridor:
    start_state = 0
    goal_state = 3
    num_states = 4
    num_actions = 2
    left = 0
    right = 1

    @staticmethod
    def init():
        return ShortCorridor.start_state

    @staticmethod
    def reset():
        return ShortCorridor.start_state

    @staticmethod
    def step(state, action):
        assert ShortCorridor.start_state <= state < ShortCorridor.goal_state
        assert action == ShortCorridor.left or action == ShortCorridor.right

        if action == ShortCorridor.left:
            if state == 1:
                state += 1
            elif ShortCorridor.start_state < state:
                state -= 1
        elif action == ShortCorridor.right:
            if state == 1:
                state -= 1
            elif state < ShortCorridor.goal_state:
                state += 1
        else:
            raise ValueError('Invalid Action!')

        if state == ShortCorridor.goal_state:
            return -1, None
        else:
            return -1, state


class ReinforceAgent:
    """
    A REINFORCE agent with a discrete policy parameterization and linear function approximation.
    """

    def __init__(self, num_actions, alpha):
        self.num_actions = num_actions
        self.alpha = alpha
        # Initialize the policy parameters:
        self.theta = np.log([[19], [1]]) # 5% chance of taking action 'right'

    def pi(self, x_s):
        """
        Compute action probabilities from action preferences:
        :param x_s: state feature vector
        :return: an array of action probabilities
        """
        # Compute action preferences for the given feature vector:
        preferences = self.theta.dot(x_s)
        # Convert overflows to underflows:
        preferences = preferences - preferences.max()
        # Convert the preferences into probabilities:
        exp_prefs = np.exp(preferences)
        return exp_prefs / np.sum(exp_prefs)

    def select_action(self, x_s):
        return np.random.choice(2, p=self.pi(x_s).squeeze())

    def eligibility_vector(self, a, s):
        return self.x(s, a) - self.pi(self.x(s)) * (self.x(s, ShortCorridor.left) + self.x(s, ShortCorridor.right))

    def x(self, s, a=None):
        """
        Function approximator that computes state or state-action features.
        """
        if a is None:
            return np.array([[1]])
        elif a == ShortCorridor.right:
            return np.array([[0], [1]])
        elif a == ShortCorridor.left:
            return np.array([[1], [0]])
        else:
            raise ValueError('Invalid Action!')

    def learn(self, s_t, a_t, g_t):
        # Get state features:
        x_s = self.x(s_t)

        # Update policy weights:
        self.theta += self.alpha * g_t * self.eligibility_vector(a_t, s_t)


def experiment(returns, alpha_index, alpha, run_num, random_seed, num_episodes, max_timesteps):
    np.random.seed(random_seed)
    agent = ReinforceAgent(num_actions=ShortCorridor.num_actions, alpha=alpha)

    for episode_num in range(num_episodes):
        episode = []
        g = 0.0
        t = 0

        # Start an episode:
        s = ShortCorridor.init()
        x_s = agent.x(s)

        # Play out the episode:
        while (s is not None) and (t < max_timesteps):
            # Select action to take:
            a = agent.select_action(x_s)

            # Take action a, observe reward r' and next state s':
            r_prime, s_prime = ShortCorridor.step(s, a)

            # Save sequence for later:
            episode.append((s, a, r_prime))

            # Update counters:
            s = s_prime
            g = g + r_prime
            t = t + 1

        # Store returns:
        returns[alpha_index, run_num, episode_num] = g

        # Episode finished, so update the agent:
        gt = g
        for t in range(len(episode)):
            # Unpack timestep:
            s, a, r_prime = episode[t]

            agent.learn(s, a, gt)

            # Compute return from t until end of episode for next timestep:
            gt = gt - r_prime


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script to generate figure 13.1 from Sutton and Barto (2nd Ed.)')
    parser.add_argument('--alphas', type=float, nargs='*', default=[2**-12, 2**-13, 2**-14], help='Policy step sizes')
    parser.add_argument('--num_runs', type=int, default=100, help='The number of runs to average over')
    parser.add_argument('--num_episodes', type=int, default=1000, help='The number of episodes per run')
    parser.add_argument('--max_timesteps', type=int, default=1000, help='The maximum number of timesteps allowed per episode')
    parser.add_argument('--random_seed', type=int, default=2565, help='The random seed to use')
    parser.add_argument('--num_cpus', type=int, default=-1, help='The number of cpus to use')
    parser.add_argument('--confidence_intervals', action='store_true', help='Plot confidence intervals')
    args = parser.parse_args()

    # Set the random seed:
    np.random.seed(args.random_seed)
    # Generate a random seed for each run:
    random_seeds = [np.random.randint(low=0, high=2**32) for run in range(args.num_runs)]

    # If the data file already exists, use it instead of re-generating the data:
    if os.path.exists('returns_13_1.npy'):
        # Create memmapped arrays to be populated in parallel:
        returns = np.memmap('returns_13_1.npy', shape=(len(args.alphas), args.num_runs, args.num_episodes), dtype=np.int16, mode='r')
    else:
        # Create memmapped arrays to be populated in parallel:
        returns = np.memmap('returns_13_1.npy', shape=(len(args.alphas), args.num_runs, args.num_episodes), dtype=np.int16, mode='w+')

        # Run experiments in parallel:
        Parallel(n_jobs=args.num_cpus, verbose=10)(delayed(experiment)(returns, alpha_index, alpha, run_num, random_seed, args.num_episodes, args.max_timesteps) for run_num, random_seed in enumerate(random_seeds) for alpha_index, alpha in enumerate(args.alphas))


    # Plot the results:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for alpha_index, alpha in enumerate(args.alphas):
            # Average over runs:
            means = np.mean(returns[alpha_index], axis=0)
            p = plt.plot(np.arange(args.num_episodes), means, label='2^{}'.format(int(np.log2(alpha))))  # keep reference for colour-matching with errorbars.

            if args.confidence_intervals:
                # Plot 95% confidence intervals:
                sems = st.sem(returns[alpha_index], axis=0)
                confs = sems * st.t.ppf((1.0 + 0.95) / 2, args.num_runs - 1)
                ax.errorbar(np.arange(args.num_episodes), means, yerr=[confs, confs], color=p[0].get_color(), alpha=.15)

    ax.legend(title='Step size $\\alpha$:')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total reward on episode')
    ax.set_ylim(-90,-10)
    ax.set_title('Performance of REINFORCE (averaged over {} runs)'.format(args.num_runs))
    plt.savefig('figure_13_1.png')