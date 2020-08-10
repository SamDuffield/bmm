import os
import json

import numpy as np
import osmnx as ox

import bmm

from simulated_data.utils import sample_route, clear_cache, cambridge_graph
from bmm.src.data.preprocess import longlat_polys_to_utm

########################################################################################################################
# Setup
seed = 3
np.random.seed(seed)

# Model parameters
time_interval = 60
route_length = 4

# Inference parameters
n_samps = 200

max_rejections = 3

initial_truncation = 100

zero_dist_prob_neg_exponent = 0.07
# b_speed = 0.067
b_speed = 0.05
# deviation_beta = 0.056
deviation_beta = 0.01
gps_sd = 3

proposal_dict = {'proposal': 'optimal'}

run_indicator = f'{seed}_{time_interval}'

save_dir = '/Users/samddd/Main/Data/bayesian-map-matching/simulations/single_route/' \
           + str(run_indicator) + '/ffbsi/'

########################################################################################################################

if __name__ == '__main__':

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + 'proposal_dict', 'w+') as f:
        json.dump(proposal_dict, f)

    setup_dict = {'seed': seed,
                  'time_interval': time_interval,
                  'max_route_length': route_length,
                  'zero_dist_prob_neg_exponent': zero_dist_prob_neg_exponent,
                  'b_speed': b_speed,
                  'deviation_beta': deviation_beta,
                  'gps_sd': gps_sd,
                  'n_samps': n_samps,
                  'max_rejections': max_rejections,
                  'initial_truncation': initial_truncation}

    with open(save_dir + 'setup_dict', 'w+') as f:
        json.dump(setup_dict, f)

    graph = cambridge_graph()

    cued_long_lat = [52.198640, 0.121680]
    cued_utm = bmm.long_lat_to_utm(cued_long_lat, graph)
    cued_edge = ox.get_nearest_edge(graph, cued_utm)
    # cued_edge = [cued_edge[1], cued_edge[0], cued_edge[2]]
    cued_start_pos = np.array([*cued_edge, 0.1])
    # cued_start_pos = np.array([1950, 1966, 0., 0.9])

    # Initiate map-matching probabilistic model
    mm_model = bmm.GammaMapMatchingModel()
    mm_model.zero_dist_prob_neg_exponent = zero_dist_prob_neg_exponent
    mm_model.distance_params['b_speed'] = b_speed
    mm_model.deviation_beta = deviation_beta
    mm_model.gps_sd = gps_sd

    end_time = time_interval * (route_length - 1)

    sample_int_cut_off = 3

    # Generate a full route
    sampled_route = sample_route(graph, mm_model, time_interval, route_length,
                                 start_position=cued_start_pos[None], num_inter_cut_off=sample_int_cut_off)
    while sampled_route[-1, 0] != end_time:
        sampled_route = sample_route(graph, mm_model, time_interval, route_length,
                                     start_position=cued_start_pos[None], num_inter_cut_off=sample_int_cut_off)

    # Extract true positions at observation times
    cartesianised_route = bmm.cartesianise_path(graph, sampled_route, t_column=True, observation_time_only=True)

    # Add noise to generate observations
    observations = cartesianised_route + mm_model.gps_sd * np.random.normal(size=cartesianised_route.shape)

    # np.save(save_dir + 'route', sampled_route)
    # np.save(save_dir + 'observations', observations)

    fig, ax = bmm.plot(graph, sampled_route, observations)
