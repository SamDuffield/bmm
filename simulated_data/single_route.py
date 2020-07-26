import os
import json

import numpy as np

import bmm

from simulated_data.utils import sample_route, clear_cache, cambridge_graph, plot_rmse, plot_pei

########################################################################################################################
# Setup
seed = 9
np.random.seed(seed)

# Model parameters
time_interval = 10
gps_sd = 7.5

# Route parameters
route_length = 50

# Number of repeated runs
num_repeats = 50

# Varying inference parameters
n_samps = np.array([50, 100, 200])
lags = np.array([0, 2, 4])

# Fixed inference parameters
max_rejections = 0
initial_truncation = None

# Proposal
proposal_dict = {'proposal': 'optimal'}

# Save location
run_indicator = f'{seed}_{time_interval}_{gps_sd}_{num_repeats}'

save_dir = '/Users/samddd/Main/Data/bayesian-map-matching/simulations/single_route/' \
           + str(run_indicator) + '/' + proposal_dict['proposal'] + '/'

########################################################################################################################

if __name__ == '__main__':

    ####################################################################################################################
    # Setup

    # Load graph
    graph = cambridge_graph()

    # Initiate map-matching probabilistic model
    mm_model = bmm.GammaMapMatchingModel()
    mm_model.gps_sd = gps_sd

    end_time = time_interval * (route_length - 1)

    # Generate a full route
    sampled_route, cartesianised_route, observations = sample_route(graph, mm_model, time_interval, route_length,
                                                                    cart_route=True, observations=True)
    while sampled_route[-1, 0] != end_time:
        print('route fail')
        sampled_route, cartesianised_route, observations = \
            sample_route(graph, mm_model, time_interval, route_length, cart_route=True, observations=True)

    fig_route, ax_route = bmm.plot(graph, sampled_route, observations)

    # Save simulation setup
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig_route.savefig(save_dir + 'route.png', dpi=350)

    with open(save_dir + 'proposal_dict', 'w+') as f:
        json.dump(proposal_dict, f)

    setup_dict = {'time_interval': time_interval,
                  'max_route_length': route_length,
                  'gps_sd': gps_sd,
                  'n_samps': n_samps.tolist(),
                  'lags': lags.tolist(),
                  'max_rejections': max_rejections,
                  'initial_truncation': initial_truncation,
                  'num_repeats': num_repeats}

    with open(save_dir + 'setup_dict', 'w+') as f:
        json.dump(setup_dict, f)

    np.save(save_dir + 'route', sampled_route)
    np.save(save_dir + 'observations', observations)

    ####################################################################################################################
    # Map-matching

    fl_pf_routes = np.empty((num_repeats, len(lags), len(n_samps)), dtype=object)
    fl_bsi_routes = np.empty((num_repeats, len(lags), len(n_samps)), dtype=object)
    ffbsi_routes = np.empty((num_repeats, len(n_samps)), dtype=object)

    n_pf_failures = 0
    n_bsi_failures = 0
    n_ffbsi_failures = 0

    print(setup_dict)

    for i in range(num_repeats):
        for j, n in enumerate(n_samps):
            for k, lag in enumerate(lags):
                print(i, j, k)
                try:
                    fl_pf_routes[i, k, j] = bmm._offline_map_match_fl(graph,
                                                                      observations,
                                                                      n,
                                                                      timestamps=time_interval,
                                                                      mm_model=mm_model,
                                                                      lag=lag,
                                                                      update='PF',
                                                                      max_rejections=max_rejections,
                                                                      initial_d_truncate=initial_truncation,
                                                                      **proposal_dict)
                    print(f'FL PF {i} {j} {k}: {fl_pf_routes[i, k, j].time}')
                except:
                    n_pf_failures += 1
                print(f'FL PF failures: {n_pf_failures}')
                clear_cache()

                try:
                    fl_bsi_routes[i, k, j] = bmm._offline_map_match_fl(graph,
                                                                       observations,
                                                                       n,
                                                                       timestamps=time_interval,
                                                                       mm_model=mm_model,
                                                                       lag=lag,
                                                                       update='BSi',
                                                                       max_rejections=max_rejections,
                                                                       initial_d_truncate=initial_truncation,
                                                                       **proposal_dict)
                    print(f'FL BSi {i} {j} {k}:', fl_bsi_routes[i, k, j].time)
                except:
                    n_bsi_failures += 1
                print(f'FL BSi failures: {n_bsi_failures}')

                clear_cache()

            try:
                ffbsi_routes[i, j] = bmm.offline_map_match(graph,
                                                           observations,
                                                           n,
                                                           timestamps=time_interval,
                                                           mm_model=mm_model,
                                                           max_rejections=max_rejections,
                                                           initial_d_truncate=initial_truncation,
                                                           **proposal_dict)

                print(f'FFBSi {i} {j}: {ffbsi_routes[i, j].time}')
            except:
                n_ffbsi_failures += 1
            print(f'FFBSi failures: {n_ffbsi_failures}')

            clear_cache()

    print(f'FL PF failures: {n_pf_failures}')
    print(f'FL BSi failures: {n_bsi_failures}')
    print(f'FFBSi failures: {n_ffbsi_failures}')

    np.save(save_dir + 'fl_pf', fl_pf_routes)
    np.save(save_dir + 'fl_bsi', fl_bsi_routes)
    np.save(save_dir + 'ffbsi', ffbsi_routes)

    plot_rmse(graph, setup_dict, cartesianised_route, fl_pf_routes, fl_bsi_routes, ffbsi_routes, save_dir)
    plot_pei(setup_dict, sampled_route, fl_pf_routes, fl_bsi_routes, ffbsi_routes, save_dir)

