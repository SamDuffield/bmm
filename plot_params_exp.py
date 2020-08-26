import matplotlib.pyplot as plt
import numpy as np
import pickle

sim = False
params = {'distance_params': {'zero_dist_prob_neg_exponent': np.array([0.12647467, 0.13069739, 0.13299723, 0.13402539, 0.13438583,
       0.13398806, 0.13373651, 0.13349616, 0.133643  , 0.13377113,              
       0.13382329, 0.13331461, 0.13346924, 0.1335121 , 0.13360461,              
       0.13312256, 0.13300212, 0.13308532, 0.13319074, 0.13327864,              
       0.13299082, 0.13318676, 0.13285037, 0.13231158, 0.13223606,              
       0.13198401, 0.13178399, 0.13144418, 0.13138585, 0.13170469,              
       0.13148925, 0.13179542, 0.13180117, 0.13218665, 0.13225388,              
       0.13258568, 0.13246797, 0.13213118, 0.13190626, 0.13157687]), 'lambda_speed': np.array([0.1       , 0.08878175, 0.08221098, 0.07951065, 0.0786044 ,
       0.07976388, 0.08084575, 0.08199646, 0.08131367, 0.0808153 ,              
       0.08009263, 0.08130629, 0.08084219, 0.08045435, 0.07983895,              
       0.08075183, 0.08194811, 0.08122353, 0.08076125, 0.08002065,              
       0.08142254, 0.08050642, 0.08128949, 0.0816716 , 0.08231711,              
       0.08253037, 0.0827606 , 0.08275678, 0.08338643, 0.08180134,              
       0.08217859, 0.08236855, 0.08188342, 0.08121958, 0.08103344,              
       0.0803914 , 0.08130038, 0.08173423, 0.0821147 , 0.08237913])}, 'deviation_beta': np.array([0.1       , 0.08092358, 0.0669153 , 0.05973189, 0.05155055,    
       0.05053078, 0.05000102, 0.05567248, 0.0560041 , 0.05607126,              
       0.05013621, 0.05592796, 0.05593496, 0.05614488, 0.05024485,              
       0.04992194, 0.05566986, 0.05573958, 0.05598133, 0.05005928,              
       0.05602814, 0.05007765, 0.04976902, 0.04932198, 0.04948861,              
       0.04909261, 0.04928343, 0.0489397 , 0.0553265 , 0.04950773,              
       0.04923668, 0.05353535, 0.05533361, 0.05428786, 0.05566435,              
       0.05007742, 0.04981148, 0.04957155, 0.04929595, 0.04955819]), 'gps_sd': np.array([7.        , 6.3825907 , 6.04282961, 5.84881813, 5.73590889,
       5.66401117, 5.64934122, 5.65569945, 5.67516552, 5.69273626,              
       5.65956429, 5.66320077, 5.6605009 , 5.65821173, 5.63140864,              
       5.62291219, 5.6385823 , 5.62568927, 5.65367725, 5.61298126,              
       5.66263109, 5.6220483 , 5.61406479, 5.61811794, 5.61169722,              
       5.59284126, 5.59946025, 5.61611878, 5.65537518, 5.62643844,              
       5.62410186, 5.68317898, 5.69740579, 5.67455591, 5.68857257,              
       5.68120703, 5.62070397, 5.60014144, 5.5989139, 5.61316532])}

# Simulated data
sim = True
params = pickle.load(open('/Users/samddd/Main/bayesian-map-matching/simulations/cambridge/tuned_sim_params.pickle', 'rb'))
# params = pickle.load(open('/Users/samddd/Main/bayesian-map-matching/simulations/cambridge/tuned_sim_params._0.1_0.05_0.05_3_prcap500_final.pickle', 'rb'))


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
       axes[0].hlines(0.10, 0, n_iter, colors=line_colour)
       axes[1].hlines(1/20, 0, n_iter,  colors=line_colour)
       axes[2].hlines(0.05, 0, n_iter, colors=line_colour)
       axes[3].hlines(3.0, 0, n_iter, colors=line_colour)

plt.tight_layout()
plt.show()
