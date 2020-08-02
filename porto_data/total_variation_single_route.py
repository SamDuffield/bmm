########################################################################################################################
# Module: total_variation_single_route.py
# Description: Run simulations on real data from Porto taxi dataset to assess fixed-lag algorithms on a single route.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import json
import os

import numpy as np
import pandas as pd

import os
import sys

sim_dat_path = os.getcwd()
repo_path = os.path.dirname(sim_dat_path)
sys.path.append(sim_dat_path)
sys.path.append(repo_path)

import bmm

from bmm.src.tools.graph import load_graph
from bmm.src.data.utils import source_data, read_data

from porto_data.utils import clear_cache, each_edge_route_total_variation, plot_metric_over_time

_, process_data_path = source_data()

graph = load_graph()

run_indicator = 10

# Load taxi data
# data_path = data.utils.choose_data()
data_path = process_data_path + "/data/portotaxi_06052014_06052014_utm_1730_1745.csv"
raw_data = read_data(data_path, 100).get_chunk()

# Select single route
route_index = 0
route_polyline = np.asarray(raw_data['POLYLINE_UTM'][route_index])

# Save directory
save_dir = f'{process_data_path}/simulations/porto/{route_index}/{run_indicator}/'

# Plot route
# fig_route, ax_route = bmm.plot(graph, polyline=route_polyline)

# Setup
seed = 0
np.random.seed(seed)

# Model parameters
time_interval = 15

# Inference parameters
ffbsi_n_samps = int(1e3)
fl_n_samps = np.array([50, 100, 200])
lags = np.array([0, 3, 10])
max_rejections = 5
initial_truncation = None
num_repeats = 30
proposal_dict = {'proposal': 'optimal'}

setup_dict = {'seed': seed,
              'time_interval': time_interval,
              # 'gps_sd': gps_sd,
              # 'intersection_penalisation': intersection_penalisation,
              'ffbsi_n_samps': ffbsi_n_samps,
              'fl_n_samps': fl_n_samps.tolist(),
              'lags': lags.tolist(),
              'max_rejections': max_rejections,
              'initial_truncation': initial_truncation,
              'num_repeats': num_repeats}

print(setup_dict)

# Create save_dir if not found
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save route figure
# fig_route.savefig(save_dir + 'route.png', dpi=350)

# Save simulation parameters
with open(save_dir + 'setup_dict', 'w+') as f:
    json.dump(setup_dict, f)

# Save proposal parameters
with open(save_dir + 'proposal_dict', 'w+') as f:
    json.dump(proposal_dict, f)

# Setup map-matching model
mm_model = bmm.GammaMapMatchingModel()
# mm_model.gps_sd = gps_sd
# mm_model.intersection_penalisation = intersection_penalisation

# Run FFBSi
ffbsi_route = bmm.offline_map_match(graph,
                                    route_polyline,
                                    ffbsi_n_samps,
                                    timestamps=time_interval,
                                    mm_model=mm_model,
                                    max_rejections=max_rejections,
                                    initial_d_truncate=initial_truncation,
                                    **proposal_dict)
clear_cache()

fl_pf_routes = np.empty((num_repeats, len(lags), len(fl_n_samps)), dtype=object)
fl_bsi_routes = np.empty((num_repeats, len(lags), len(fl_n_samps)), dtype=object)

n_pf_failures = 0
n_bsi_failures = 0

for i in range(num_repeats):
    for j, n in enumerate(fl_n_samps):
        for k, lag in enumerate(lags):
            print(i, j, k)
            try:
                fl_pf_routes[i, k, j] = bmm._offline_map_match_fl(graph,
                                                                  route_polyline,
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

            if lag == 0 and fl_bsi_routes[i, k, j] is not None:
                fl_bsi_routes[i, k, j] = fl_pf_routes[i, k, j].copy()
            else:
                try:
                    fl_bsi_routes[i, k, j] = bmm._offline_map_match_fl(graph,
                                                                       route_polyline,
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

fl_pf_tvs = np.empty((setup_dict['num_repeats'], len(setup_dict['fl_n_samps']), len(setup_dict['lags']),  len(observation_times)))
fl_bsi_tvs = np.empty_like(fl_pf_tvs)
fl_pf_times = np.empty((setup_dict['num_repeats'],  len(setup_dict['fl_n_samps']), len(setup_dict['lags'])))
fl_bsi_times = np.empty_like(fl_pf_times)

# Calculate TV distances from FFBSi
for i in range(setup_dict['num_repeats']):
    for j, n in enumerate(setup_dict['fl_n_samps']):
        for k, lag in enumerate(setup_dict['lags']):
            print(i, j, k)
            fl_pf_tvs[i, j, k] = each_edge_route_total_variation(ffbsi_route.particles,
                                                                 fl_pf_routes[i, j, k].particles,
                                                                 observation_times)
            fl_pf_times[i, j, k] = fl_pf_routes[i, j, k].time

            fl_bsi_tvs[i, j, k] = each_edge_route_total_variation(ffbsi_route.particles,
                                                                  fl_bsi_routes[i, j, k].particles,
                                                                  observation_times)
            fl_bsi_times[i, j, k] = fl_bsi_routes[i, j, k].time

np.save(save_dir + 'fl_pf_tv', fl_pf_tvs)
np.save(save_dir + 'fl_bsi_tv', fl_bsi_tvs)
np.save(save_dir + 'fl_pf_times', fl_pf_times)
np.save(save_dir + 'fl_bsi_times', fl_bsi_times)


# fl_pf_tvs = np.load(save_dir + 'fl_pf_tv.npy', allow_pickle=True)
# fl_bsi_tvs = np.load(save_dir + 'fl_bsi_tv.npy', allow_pickle=True)
# fl_pf_times = np.load(save_dir + 'fl_pf_times.npy', allow_pickle=True)
# fl_bsi_times = np.load(save_dir + 'fl_bsi_times.npy', allow_pickle=True)


# fig, axes = plot_metric_over_time(setup_dict,
#                       save_dir,
#                       np.mean(fl_pf_tvs, axis=0),
#                       np.mean(fl_pf_times, axis=0),
#                       np.mean(fl_bsi_tvs, axis=0),
#                       np.mean(fl_bsi_times, axis=0))
#
# sub_route_plot_xlim = (532399.6154033017, 532860.7133234204)
# sub_route_plot_ylim = (4556923.901931656, 4557388.957773835)

