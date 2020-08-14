################################################################################################################

import json

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

sim_dat_path = os.getcwd()
repo_path = os.path.dirname(sim_dat_path)
sys.path.append(sim_dat_path)
sys.path.append(repo_path)

import bmm

from bmm.src.tools.graph import load_graph
from bmm.src.data.utils import source_data, read_data

from porto_data.utils import clear_cache, all_edges_total_variation

_, process_data_path = source_data()

graph = load_graph()

run_indicator = 0

# Load taxi data
# data_path = data.utils.choose_data()
data_path = process_data_path + "/data/portotaxi_06052014_06052014_utm_1730_1745.csv"
raw_data = read_data(data_path, 100).get_chunk()

# Select single route
route_indices = np.array([0])
route_polylines = [np.array(raw_data['POLYLINE_UTM'][i]) for i in route_indices]

# Save directory
save_dir = f'{process_data_path}/simulations/porto/all_edges_tv/{run_indicator}/'

# Setup
seed = 0
np.random.seed(seed)

# Model parameters
time_interval = 15

# Inference parameters
ffbsi_n_samps = int(1e3)
fl_n_samps = np.array([50, 100, 150, 200])
lags = np.array([0, 3, 10])
max_rejections = 0
initial_truncation = None
num_repeats = 1
proposal_dict = {'proposal': 'optimal'}

setup_dict = {'seed': seed,
              'route_indices': route_indices.tolist(),
              'ffbsi_n_samps': ffbsi_n_samps,
              'fl_n_samps': fl_n_samps.tolist(),
              'lags': lags.tolist(),
              'max_rejections': max_rejections,
              'initial_truncation': initial_truncation,
              'num_repeats': num_repeats}

print(setup_dict)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(save_dir + 'setup_dict', 'w+') as f:
    json.dump(setup_dict, f)

mm_model = bmm.GammaMapMatchingModel()

# Run MM
ffbsi_routes = np.empty(len(route_polylines), dtype='object')
fl_pf_routes = np.empty((len(route_polylines), num_repeats, len(fl_n_samps), len(lags)), dtype='object')
fl_bsi_routes = np.empty((len(route_polylines), num_repeats, len(fl_n_samps), len(lags)), dtype='object')

n_pf_failures = 0
n_bsi_failures = 0

for i in range(len(route_polylines)):
    ffbsi_routes[i] = bmm.offline_map_match(graph,
                                            route_polylines[i],
                                            ffbsi_n_samps,
                                            timestamps=time_interval,
                                            mm_model=mm_model,
                                            max_rejections=max_rejections,
                                            initial_d_truncate=initial_truncation,
                                            **proposal_dict)
    clear_cache()

    for j in range(num_repeats):
        for k in range(len(fl_n_samps)):
            for l in range(len(lags)):
                print(i, j, k, l)
                try:
                    fl_pf_routes[i, j, k, l] = bmm._offline_map_match_fl(graph,
                                                                         route_polylines[i],
                                                                         fl_n_samps[k],
                                                                         timestamps=time_interval,
                                                                         mm_model=mm_model,
                                                                         lag=lags[l],
                                                                         update='PF',
                                                                         max_rejections=max_rejections,
                                                                         initial_d_truncate=initial_truncation,
                                                                         **proposal_dict)
                except:
                    n_pf_failures += 1
                print(f'FL PF failures: {n_pf_failures}')
                print(f'FL PF {i} {j} {k} {l}: {fl_pf_routes[i, j, k, l].time}')
                clear_cache()

                if lags[l] == 0:
                    if fl_pf_routes[i, j, k, l] is not None:
                        fl_bsi_routes[i, j, k, l] = fl_pf_routes[i, j, k, l].copy()
                else:
                    try:
                        fl_bsi_routes[i, j, k, l] = bmm._offline_map_match_fl(graph,
                                                                              route_polylines[i],
                                                                              fl_n_samps[k],
                                                                              timestamps=time_interval,
                                                                              mm_model=mm_model,
                                                                              lag=lags[l],
                                                                              update='BSi',
                                                                              max_rejections=max_rejections,
                                                                              initial_d_truncate=initial_truncation,
                                                                              **proposal_dict)
                    except:
                        n_bsi_failures += 1
                print(f'FL BSi failures: {n_bsi_failures}')
                print(f'FL BSi {i} {j} {k} {l}: {fl_bsi_routes[i, j, k, l].time}')
                clear_cache()

np.save(save_dir + 'fl_pf', fl_pf_routes)
np.save(save_dir + 'fl_bsi', fl_bsi_routes)
np.save(save_dir + 'ffbsi', ffbsi_routes)

fl_pf_tvs = np.empty(fl_pf_routes.shape)
fl_bsi_tvs = np.empty_like(fl_pf_tvs)
for i in range(len(route_polylines)):
    for j in range(num_repeats):
        for k in range(len(fl_n_samps)):
            for l in range(len(lags)):
                fl_pf_tvs[i, j, k, l] = all_edges_total_variation(ffbsi_routes[i], fl_pf_routes[i, j, k, l])
                fl_bsi_tvs[i, j, k, l] = all_edges_total_variation(ffbsi_routes[i], fl_bsi_routes[i, j, k, l])

np.save(save_dir + 'fl_pf_tv', fl_pf_tvs)
np.save(save_dir + 'fl_bsi_tv', fl_bsi_tvs)


#
# fl_pf_tvs = np.load(save_dir + 'fl_pf_tv.npy', allow_pickle=True)
# fl_bsi_tvs = np.load(save_dir + 'fl_bsi_tv.npy', allow_pickle=True)


def plot_conv_tv(tv_mat, leg=False):
    fig, ax = plt.subplots()
    for i in range(tv_mat.shape[1]):
        ax.plot(fl_n_samps, tv_mat[:, i], label=f'Lag: {lags[i]}')

    plt.tight_layout()
    if leg:
        plt.legend()
    return fig, ax


fig_pf, ax_pf = plot_conv_tv(np.mean(fl_pf_tvs, axis=(0, 1)))
plt.savefig(save_dir + 'pf_comp.png', dpi=400)
fig_bsi, ax_bsi = plot_conv_tv(np.mean(fl_bsi_tvs, axis=(0, 1)), leg=True)
plt.savefig(save_dir + 'bsi_comp.png', dpi=400)

