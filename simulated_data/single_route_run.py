import os
import json

import numpy as np
import osmnx as ox

import bmm

from simulated_data.utils import sample_route, clear_cache

########################################################################################################################
# Setup
seed = 0
np.random.seed(seed)

# Model parameters
time_interval = 15
gps_sd = 20
route_length = 50

# Inference parameters
n_samps = np.array([50, 75, 100])
# n_samps = np.array([500, 1000])

lags = np.array([0, 1, 2])

max_rejections = 3

initial_truncation = 100

proposal_dict = {'proposal': 'optimal'}
# proposal_dict = {'proposal': 'aux_dist', 'dist_expand': 30, 'var': 100}
# proposal_dict = {'proposal': 'dist_then_edge', 'var': 100}


run_indicator = f'{seed}_{time_interval}_{gps_sd}'

save_dir = '/Users/samddd/Main/Data/bayesian-map-matching/simulations/single_route/'\
           + str(run_indicator) + '/' + proposal_dict['proposal'] + '/'

########################################################################################################################


if __name__ == '__main__':

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + 'proposal_dict', 'w+') as f:
        json.dump(proposal_dict, f)

    setup_dict = {'time_interval': time_interval,
                  'route_length': route_length,
                  'gps_sd': gps_sd,
                  'n_samps': n_samps.tolist(),
                  'lags': lags.tolist(),
                  'max_rejections': max_rejections,
                  'initial_truncation': initial_truncation}

    with open(save_dir + 'setup_dict', 'w+') as f:
        json.dump(setup_dict, f)

    # Where to load graph
    graph_dir = '/Users/samddd/Main/Data/bayesian-map-matching/graphs/Cambridge/'
    graph_name = 'cambridge_latest_utm_simplified_clean_int_rde'
    graph_path = graph_dir + graph_name + '.graphml'

    graph = ox.load_graphml(graph_path)

    # Initiate map-matching probabilistic model
    mm_model = bmm.GammaMapMatchingModel()
    mm_model.gps_sd = gps_sd

    end_time = time_interval * (route_length - 1)

    # Generate a full route
    sampled_route = sample_route(graph, mm_model, time_interval, route_length)
    while sampled_route[-1, 0] != end_time:
        sampled_route = sample_route(graph, mm_model, time_interval, route_length)

    np.save(save_dir + 'route', sampled_route)

    # Extract true positions at observation times
    cartesianised_route = bmm.cartesianise_path(graph, sampled_route, t_column=True, observation_time_only=True)

    # Add noise to generate observations
    observations = cartesianised_route + mm_model.gps_sd * np.random.normal(size=cartesianised_route.shape)
    np.save(save_dir + 'observations', observations)

    fl_pf_routes = np.empty((len(lags), len(n_samps)), dtype=object)
    fl_bsi_routes = np.empty((len(lags), len(n_samps)), dtype=object)
    ffbsi_routes = np.empty(len(n_samps), dtype=object)

    for j, n in enumerate(n_samps):
        for k, lag in enumerate(lags):
            print(j, k)

            fl_pf_routes[k, j] = bmm._offline_map_match_fl(graph,
                                                           observations,
                                                           n,
                                                           time_interval=time_interval,
                                                           mm_model=mm_model,
                                                           lag=lag,
                                                           update='PF',
                                                           max_rejections=max_rejections,
                                                           initial_d_truncate=initial_truncation,
                                                           **proposal_dict)
            print(fl_pf_routes[k, j].time)
            clear_cache()

            fl_bsi_routes[k, j] = bmm._offline_map_match_fl(graph,
                                                            observations,
                                                            n,
                                                            time_interval=time_interval,
                                                            mm_model=mm_model,
                                                            lag=lag,
                                                            update='BSi',
                                                            max_rejections=max_rejections,
                                                            initial_d_truncate=initial_truncation,
                                                            **proposal_dict)
            print(fl_bsi_routes[k, j].time)
            clear_cache()

        ffbsi_routes[j] = bmm.offline_map_match(graph,
                                                observations,
                                                n,
                                                time_interval=time_interval,
                                                mm_model=mm_model,
                                                max_rejections=max_rejections,
                                                initial_d_truncate=initial_truncation,
                                                **proposal_dict)

        print(ffbsi_routes[j].time)
        clear_cache()

    np.save(save_dir + 'fl_pf', fl_pf_routes)
    np.save(save_dir + 'fl_bsi', fl_bsi_routes)
    np.save(save_dir + 'ffbsi', ffbsi_routes)




