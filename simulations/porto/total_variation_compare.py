import os
import json

import numpy as np
import osmnx as ox
import pandas as pd

import bmm

from . import utils

seed = 0
np.random.seed(seed)

timestamps = 15
ffbsi_n_samps = int(1e3)
fl_n_samps = np.array([50, 100, 150, 200])
lags = np.array([0, 3, 10])
max_rejections = 30
initial_truncation = None
num_repeats = 20
max_speed = 35
proposal_dict = {'proposal': 'optimal',
                 'num_inter_cut_off': 10,
                 'resample_fails': False,
                 'd_max_fail_multiplier': 2.}

setup_dict = {'seed': seed,
              'ffbsi_n_samps': ffbsi_n_samps,
              'fl_n_samps': fl_n_samps.tolist(),
              'lags': lags.tolist(),
              'max_rejections': max_rejections,
              'initial_truncation': initial_truncation,
              'num_repeats': num_repeats,
              'num_inter_cut_off': proposal_dict['num_inter_cut_off'],
              'max_speed': max_speed,
              'resample_fails': proposal_dict['resample_fails'],
              'd_max_fail_multiplier': proposal_dict['d_max_fail_multiplier']}

print(setup_dict)

porto_sim_dir = os.getcwd()
graph_path = porto_sim_dir + '/portotaxi_graph_portugal-140101.osm._simple.graphml'
graph = ox.load_graphml(graph_path)

test_route_data_path = porto_sim_dir + '/test_route.csv'

# Load long-lat polylines
polyline_ll = np.array(json.loads(pd.read_csv(test_route_data_path)['POLYLINE'][0]))
# Convert to utm
polyline = bmm.long_lat_to_utm(polyline_ll, graph)

save_dir = porto_sim_dir + '/tv_output/'

# Create save_dir if not found
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save simulation parameters
with open(save_dir + 'setup_dict', 'w+') as f:
    json.dump(setup_dict, f)

# Setup map-matching model
mm_model = bmm.ExponentialMapMatchingModel()
mm_model.max_speed = max_speed

# Run FFBSi
ffbsi_route = bmm.offline_map_match(graph,
                                    polyline,
                                    ffbsi_n_samps,
                                    timestamps=timestamps,
                                    mm_model=mm_model,
                                    max_rejections=max_rejections,
                                    initial_d_truncate=initial_truncation,
                                    **proposal_dict)
utils.clear_cache()

fl_pf_routes = np.empty((num_repeats, len(fl_n_samps), len(lags)), dtype=object)
fl_bsi_routes = np.empty((num_repeats, len(fl_n_samps), len(lags)), dtype=object)

n_pf_failures = 0
n_bsi_failures = 0

for i in range(num_repeats):
    for j, n in enumerate(fl_n_samps):
        for k, lag in enumerate(lags):
            print(i, j, k)
            # try:
            fl_pf_routes[i, j, k] = bmm._offline_map_match_fl(graph,
                                                              polyline,
                                                              n,
                                                              timestamps=timestamps,
                                                              mm_model=mm_model,
                                                              lag=lag,
                                                              update='PF',
                                                              max_rejections=max_rejections,
                                                              initial_d_truncate=initial_truncation,
                                                              **proposal_dict)
            print(f'FL PF {i} {j} {k}: {fl_pf_routes[i, j, k].time}')
            # except:
            #     n_pf_failures += 1
            print(f'FL PF failures: {n_pf_failures}')
            utils.clear_cache()

            if lag == 0 and fl_pf_routes[i, j, k] is not None:
                fl_bsi_routes[i, j, k] = fl_pf_routes[i, j, k].copy()
                print(f'FL BSi {i} {j} {k}:', fl_bsi_routes[i, j, k].time)
            else:
                # try:
                fl_bsi_routes[i, j, k] = bmm._offline_map_match_fl(graph,
                                                                   polyline,
                                                                   n,
                                                                   timestamps=timestamps,
                                                                   mm_model=mm_model,
                                                                   lag=lag,
                                                                   update='BSi',
                                                                   max_rejections=max_rejections,
                                                                   initial_d_truncate=initial_truncation,
                                                                   **proposal_dict)
                print(f'FL BSi {i} {j} {k}:', fl_bsi_routes[i, j, k].time)
                # except:
                #     n_bsi_failures += 1
                print(f'FL BSi failures: {n_bsi_failures}')

                utils.clear_cache()

print(f'FL PF failures: {n_pf_failures}')
print(f'FL BSi failures: {n_bsi_failures}')

np.save(save_dir + 'fl_pf', fl_pf_routes)
np.save(save_dir + 'fl_bsi', fl_bsi_routes)
ffbsi_route_arr = np.empty(1, dtype=object)
ffbsi_route_arr[0] = ffbsi_route
np.save(save_dir + 'ffbsi', ffbsi_route_arr)
#
# fl_pf_routes = np.load(save_dir + 'fl_pf.npy', allow_pickle=True)
# fl_bsi_routes = np.load(save_dir + 'fl_bsi.npy', allow_pickle=True)
# ffbsi_route = np.load(save_dir + 'ffbsi.npy', allow_pickle=True)[0]
# with open(save_dir + 'setup_dict') as f:
#     setup_dict = json.load(f)

observation_times = ffbsi_route.observation_times

fl_pf_tvs = np.empty(
    (setup_dict['num_repeats'], len(setup_dict['fl_n_samps']), len(setup_dict['lags']), len(observation_times)))
fl_bsi_tvs = np.empty_like(fl_pf_tvs)
fl_pf_times = np.empty((setup_dict['num_repeats'], len(setup_dict['fl_n_samps']), len(setup_dict['lags'])))
fl_bsi_times = np.empty_like(fl_pf_times)

inc_alpha = True
round_alpha = None

# Calculate TV distances from FFBSi for each observations time
for i in range(setup_dict['num_repeats']):
    for j, n in enumerate(setup_dict['fl_n_samps']):
        for k, lag in enumerate(setup_dict['lags']):
            print(i, j, k)
            if fl_pf_routes[i, j, k] is not None:
                fl_pf_tvs[i, j, k] = utils.each_edge_route_total_variation(ffbsi_route.particles,
                                                                           fl_pf_routes[i, j, k].particles,
                                                                           observation_times,
                                                                           include_alpha=inc_alpha,
                                                                           round_alpha=round_alpha)
                fl_pf_times[i, j, k] = fl_pf_routes[i, j, k].time
            else:
                fl_pf_tvs[i, j, k] = 1.
                fl_pf_times[i, j, k] = 0.
            if fl_bsi_routes[i, j, k] is not None:
                fl_bsi_tvs[i, j, k] = utils.each_edge_route_total_variation(ffbsi_route.particles,
                                                                            fl_bsi_routes[i, j, k].particles,
                                                                            observation_times,
                                                                            include_alpha=inc_alpha,
                                                                            round_alpha=round_alpha)
                fl_bsi_times[i, j, k] = fl_bsi_routes[i, j, k].time
            else:
                fl_bsi_tvs[i, j, k] = 1.
                fl_bsi_times[i, j, k] = 0.

np.save(save_dir + f'fl_pf_tv_alpha{inc_alpha * 1}_round{round_alpha}', fl_pf_tvs)
np.save(save_dir + f'fl_bsi_tv_alpha{inc_alpha * 1}_round{round_alpha}', fl_bsi_tvs)
np.save(save_dir + 'fl_pf_times', fl_pf_times)
np.save(save_dir + 'fl_bsi_times', fl_bsi_times)
#
# fl_pf_tvs = np.load(save_dir + f'fl_pf_tv_alpha{inc_alpha*1}_round{round_alpha}.npy', allow_pickle=True)
# fl_bsi_tvs = np.load(save_dir + f'fl_bsi_tv_alpha{inc_alpha*1}_round{round_alpha}.npy', allow_pickle=True)
# fl_pf_times = np.load(save_dir + 'fl_pf_times.npy', allow_pickle=True)
# fl_bsi_times = np.load(save_dir + 'fl_bsi_times.npy', allow_pickle=True)

utils.plot_metric_over_time(setup_dict,
                            np.mean(fl_pf_tvs, axis=0),
                            np.mean(fl_bsi_tvs, axis=0),
                            np.sum(fl_pf_times, axis=0) / np.sum(fl_pf_times > 0, axis=0),
                            np.sum(fl_bsi_times, axis=0) / np.sum(fl_bsi_times > 0, axis=0),
                            save_dir=save_dir + f'each_tv_compare_alpha{inc_alpha * 1}_round{round_alpha}')

speeds = False
bins = 5
interval = 60
num_ints = int(observation_times[-1] / interval)

fl_pf_dist_tvs = np.empty(
    (setup_dict['num_repeats'], len(setup_dict['fl_n_samps']), len(setup_dict['lags']), num_ints))
fl_bsi_dist_tvs = np.empty_like(fl_pf_dist_tvs)
# Calculate TV distance distances from FFBSi for each observations time
for i in range(setup_dict['num_repeats']):
    for j, n in enumerate(setup_dict['fl_n_samps']):
        for k, lag in enumerate(setup_dict['lags']):
            print(i, j, k)
            if fl_pf_routes[i, j, k] is not None:
                fl_pf_dist_tvs[i, j, k] = utils.interval_tv_dists(ffbsi_route,
                                                                  fl_pf_routes[i, j, k],
                                                                  interval=interval,
                                                                  speeds=speeds,
                                                                  bins=bins)
            else:
                fl_pf_dist_tvs[i, j, k] = 1.
            if fl_bsi_routes[i, j, k] is not None:
                fl_bsi_dist_tvs[i, j, k] = utils.interval_tv_dists(ffbsi_route,
                                                                   fl_bsi_routes[i, j, k],
                                                                   interval=interval,
                                                                   speeds=speeds,
                                                                   bins=bins)
            else:
                fl_bsi_dist_tvs[i, j, k] = 1.

np.save(save_dir + f'fl_pf_tv_dist_speeds{speeds}_bins{bins}_interval{interval}', fl_pf_dist_tvs)
np.save(save_dir + f'fl_bsi_tv_dist_speeds{speeds}_bins{bins}_interval{interval}', fl_bsi_dist_tvs)

utils.plot_metric_over_time(setup_dict,
                            np.mean(fl_pf_dist_tvs, axis=0),
                            np.mean(fl_bsi_dist_tvs, axis=0),
                            np.sum(fl_pf_times, axis=0) / np.sum(fl_pf_times > 0, axis=0),
                            np.sum(fl_bsi_times, axis=0) / np.sum(fl_bsi_times > 0, axis=0),
                            save_dir=save_dir + f'each_tv_compare_dist_speeds{speeds}_bins{bins}_interval{interval}',
                            t_linspace=np.arange(1, num_ints + 1),
                            x_lab='Minute',
                            x_ticks=np.arange(num_ints + 1, step=int(num_ints/8)))

fl_pf_all_tvs = np.empty(
    (setup_dict['num_repeats'], len(setup_dict['fl_n_samps']), len(setup_dict['lags'])))
fl_bsi_all_tvs = np.empty_like(fl_pf_all_tvs)

# Calculate TV distances from FFBSi for overall series of edges
for i in range(setup_dict['num_repeats']):
    for j, n in enumerate(setup_dict['fl_n_samps']):
        for k, lag in enumerate(setup_dict['lags']):
            print(i, j, k)
            if fl_pf_routes[i, j, k] is not None:
                fl_pf_all_tvs[i, j, k] = utils.all_edges_total_variation(ffbsi_route,
                                                                         fl_pf_routes[i, j, k])
            else:
                fl_pf_all_tvs[i, j, k] = 1.
            if fl_bsi_routes[i, j, k] is not None:
                fl_bsi_all_tvs[i, j, k] = utils.all_edges_total_variation(ffbsi_route,
                                                                          fl_bsi_routes[i, j, k])
            else:
                fl_bsi_all_tvs[i, j, k] = 1.

np.save(save_dir + 'fl_pf_all_tv', fl_pf_all_tvs)
np.save(save_dir + 'fl_bsi_all_tv', fl_bsi_all_tvs)

# fl_pf_all_tvs = np.load(save_dir + 'fl_pf_all_tv.npy', allow_pickle=True)
# fl_bsi_all_tvs = np.load(save_dir + 'fl_bsi_all_tv.npy', allow_pickle=True)

utils.plot_conv_metric(np.median(fl_pf_all_tvs, axis=0),
                       fl_n_samps, lags,
                       np.quantile(fl_pf_all_tvs, 0.25, axis=0),
                       np.quantile(fl_pf_all_tvs, 0.75, axis=0),
                       save_dir=save_dir + 'fl_pf_all_tv_conv_quantiles')

utils.plot_conv_metric(np.median(fl_bsi_all_tvs, axis=0),
                       fl_n_samps, lags,
                       np.quantile(fl_bsi_all_tvs, 0.25, axis=0),
                       np.quantile(fl_bsi_all_tvs, 0.75, axis=0),
                       save_dir=save_dir + 'fl_bsi_all_tv_conv_quantiles',
                       leg=True)

