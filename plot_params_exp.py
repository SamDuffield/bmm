import matplotlib.pyplot as plt
import numpy as np

# Simulated data
sim = True
params = {'distance_params': {'zero_dist_prob_neg_exponent': np.array([0.10729586, 0.10438631, 0.10999758, 0.10726037, 0.11018791,
       0.10924558, 0.10916854, 0.10840754]), 'lambda_speed': np.array([0.1       , 0.10225549, 0.09050505, 0.09575637, 0.08954178,
       0.09031274, 0.08954296, 0.09082274])}, 'deviation_beta': np.array([0.01      , 0.05780789, 0.00957608, 0.04706977, 0.01597669,
       0.02011883, 0.01461327, 0.02256415]), 'gps_sd': np.array([7.        , 5.16131411, 4.14023521, 3.50565839, 3.19484447,
       3.05357698, 3.01227152, 2.97323328])}


n_iter = len(params['gps_sd'])

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(7, 10))

axes[0].plot(np.arange(n_iter), np.exp(-15 * params['distance_params']['zero_dist_prob_neg_exponent']))
axes[1].plot(np.arange(n_iter), params['distance_params']['lambda_speed'])
axes[2].plot(np.arange(n_iter), params['deviation_beta'])
axes[3].plot(np.arange(n_iter), params['gps_sd'])

axes[0].set_ylabel(r'$p^0$')
axes[1].set_ylabel(r'$\lambda$')
axes[2].set_ylabel(r'$\beta$')
axes[3].set_ylabel(r'$\sigma_{GPS}$')


if sim:
       line_colour = 'purple'
       axes[0].hlines(0.15, 0, n_iter, colors=line_colour)
       axes[1].hlines(1/15, 0, n_iter, colors=line_colour)
       axes[2].hlines(0.03, 0, n_iter, colors=line_colour)
       axes[3].hlines(3, 0, n_iter, colors=line_colour)

plt.tight_layout()
plt.show()
