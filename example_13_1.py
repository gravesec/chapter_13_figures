'''
This script requires Python 3.7.0 and the following packages:
numpy==1.16.3
matplotlib==3.0.3 (for plotting results)
'''


import numpy as np
import matplotlib.pyplot as plt


# Calculate the value of the start state for a given 'right' probability:
def v_s(pi):
    return (4 - 2 * pi)/(pi * pi - pi)


# For each right probability, plot the value of the policy:
right_probabilities = np.linspace(0.01, 0.99, 99)
values = np.array([v_s(pi) for pi in right_probabilities])
plt.plot(right_probabilities, values, color='black')

# Plot the value of e-greedy left policy:
pi_e_greedy_left = .05
v_e_greedy_left = v_s(pi_e_greedy_left)
plt.plot(pi_e_greedy_left, v_e_greedy_left, color='black', marker='o')
plt.annotate('$\\epsilon$-greedy \'left\'', (pi_e_greedy_left, v_e_greedy_left), xycoords='data', xytext=(10,-3), textcoords='offset points')

# Plot the value of e-greedy right policy:
pi_e_greedy_right = .95
v_e_greedy_right = v_s(pi_e_greedy_right)
plt.plot(pi_e_greedy_right, v_e_greedy_right, color='black', marker='o')
plt.annotate('$\\epsilon$-greedy \'right\'', (pi_e_greedy_right, v_e_greedy_right), xycoords='data', xytext=(-85,-3), textcoords='offset points')

# Plot the value of the optimal stochastic policy:
pi_opt = 2 - np.sqrt(2)
v_opt = v_s(pi_opt)
plt.plot(pi_opt, v_opt, color='black', marker='o')
plt.annotate('optimal stochastic policy', (pi_opt, v_opt), xycoords='data', xytext=(0,10), textcoords='offset points')

# Configure the figure:
plt.xlabel('Probability of action \'right\'')
plt.ylabel('$J(\\mathbf{\\theta}) = v_{\\pi_{\\mathbf{\\theta}}}(S)$')
plt.title('Short corridor with switched actions')
plt.ylim([-100,0])
plt.yticks(list(plt.yticks()[0]) + [-11])
plt.savefig('example_13_1.png')