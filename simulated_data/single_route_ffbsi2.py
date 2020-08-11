import os
import json

import numpy as np
import matplotlib.pyplot as plt

import bmm

from simulated_data.utils import sample_route, clear_cache, cambridge_graph, random_positions

########################################################################################################################
# Setup
seed = 3
np.random.seed(seed)

# Model parameters
time_interval = 100
route_length = 4
gps_sd = 12

# Inference parameters
n_samps = 1000

max_rejections = 10

proposal_dict = {'proposal': 'optimal'}

ind = 1

run_indicator = f'{seed}_{time_interval}_{gps_sd}_{ind}'

save_dir = '/Users/samddd/Main/Data/bayesian-map-matching/simulations/single_route/' \
           + str(run_indicator) + '/ffbsi/'

########################################################################################################################

if __name__ == '__main__':

    # Initiate map-matching probabilistic model
    mm_model = bmm.GammaMapMatchingModel()
    mm_model.gps_sd = gps_sd

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + 'proposal_dict', 'w+') as f:
        json.dump(proposal_dict, f)

    setup_dict = {'seed': seed,
                  'time_interval': time_interval,
                  'max_route_length': route_length,
                  'zero_dist_prob_neg_exponent': mm_model.zero_dist_prob_neg_exponent,
                  'b_speed': mm_model.distance_params['b_speed'],
                  'deviation_beta': mm_model.deviation_beta,
                  'gps_sd': mm_model.gps_sd,
                  'n_samps': n_samps,
                  'max_rejections': max_rejections}

    with open(save_dir + 'setup_dict', 'w+') as f:
        json.dump(setup_dict, f)

    graph = cambridge_graph()

    # Add noise to generate observations
    observations_ll = [[0.12188, 52.198387],
                       [0.125389, 52.197771],
                       [0.128354, 52.199379],
                       [0.130296, 52.201701],
                       [0.127742, 52.20407],
                       [0.126433, 52.205753],
                       [0.127536, 52.207831],
                       [0.126082, 52.212281]]

    observations = bmm.long_lat_to_utm(observations_ll, graph)

    fig, ax = bmm.plot(graph, polyline=observations)

    ffbsi_route = bmm.offline_map_match(graph, observations, n_samps, time_interval, mm_model,
                                        max_rejections=max_rejections, num_inter_cut_off=7)

    ffbsi_route_arr = np.empty(1, dtype='object')
    ffbsi_route_arr[0] = ffbsi_route
    np.save(save_dir + 'ffbsi_route', ffbsi_route_arr)

    ffbsi_route = np.load(save_dir + 'ffbsi_route.npy', allow_pickle=True)[0]
    fig2, ax2 = bmm.plot(graph, ffbsi_route, observations)
    fig2.savefig(save_dir + 'ffbsi_fig')


