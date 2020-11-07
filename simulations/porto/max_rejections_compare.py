import os
import json

import numpy as np
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt

import bmm

from . import utils

seed = 0
np.random.seed(seed)

timestamps = 15
n_samps = np.array([50, 100, 150, 200])
lag = 3
mr_max = 20
# max_rejections = np.arange(0, mr_max + 1, step=int(mr_max/5))
max_rejections = np.array([0, 1, 2, 4, 8, 16, 32])
initial_truncation = None
num_repeats = 1
max_speed = 35
proposal_dict = {'proposal': 'optimal',
                 'num_inter_cut_off': 10,
                 'resample_fails': False,
                 'd_max_fail_multiplier': 2.}

setup_dict = {'seed': seed,
              'n_samps': n_samps.tolist(),
              'lag': lag,
              'max_rejections': max_rejections.tolist(),
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

save_dir = porto_sim_dir + '/mr_output/'

# Create save_dir if not found
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save simulation parameters
with open(save_dir + 'setup_dict', 'w+') as f:
    json.dump(setup_dict, f)

# Setup map-matching model
mm_model = bmm.ExponentialMapMatchingModel()
mm_model.max_speed = max_speed

fl_pf_routes = np.empty((num_repeats, len(n_samps), len(max_rejections)), dtype=object)
fl_bsi_routes = np.empty((num_repeats, len(n_samps), len(max_rejections)), dtype=object)
fl_pf_times = np.zeros((num_repeats, len(n_samps), len(max_rejections)))
fl_bsi_times = np.zeros((num_repeats, len(n_samps), len(max_rejections)))

n_pf_failures = 0
n_bsi_failures = 0

for i in range(num_repeats):
    for j, n in enumerate(n_samps):
        for k, max_reject_int in enumerate(max_rejections):
            print(i, j, k)
            try:
                fl_pf_routes[i, j, k] = bmm._offline_map_match_fl(graph,
                                                                  polyline,
                                                                  n,
                                                                  timestamps=timestamps,
                                                                  mm_model=mm_model,
                                                                  lag=lag,
                                                                  update='PF',
                                                                  max_rejections=max_reject_int,
                                                                  initial_d_truncate=initial_truncation,
                                                                  **proposal_dict)
                fl_pf_times[i, j, k] = fl_pf_routes[i, j, k].time
                print(f'FL PF {i} {j} {k}: {fl_pf_routes[i, j, k].time}')
            except:
                n_pf_failures += 1
            print(f'FL PF failures: {n_pf_failures}')
            utils.clear_cache()

            try:
                fl_bsi_routes[i, j, k] = bmm._offline_map_match_fl(graph,
                                                                   polyline,
                                                                   n,
                                                                   timestamps=timestamps,
                                                                   mm_model=mm_model,
                                                                   lag=lag,
                                                                   update='BSi',
                                                                   max_rejections=max_reject_int,
                                                                   initial_d_truncate=initial_truncation,
                                                                   **proposal_dict)
                fl_bsi_times[i, j, k] = fl_bsi_routes[i, j, k].time
                print(f'FL BSi {i} {j} {k}: {fl_bsi_routes[i, j, k].time}')
            except:
                n_bsi_failures += 1
            print(f'FL BSi failures: {n_bsi_failures}')
            utils.clear_cache()

print(f'FL PF failures: {n_pf_failures}')
print(f'FL BSi failures: {n_bsi_failures}')

np.save(save_dir + 'fl_pf_routes', fl_pf_routes)
np.save(save_dir + 'fl_pf_times', fl_pf_times)
np.save(save_dir + 'fl_bsi_routes', fl_bsi_routes)
np.save(save_dir + 'fl_bsi_times', fl_bsi_times)

#
# fl_pf_routes = np.load(save_dir + 'fl_pf_routes.npy', allow_pickle=True)
# fl_pf_times = np.load(save_dir + 'fl_pf_times.npy')
# fl_bsi_routes = np.load(save_dir + 'fl_bsi_routes.npy', allow_pickle=True)
# fl_bsi_times = np.load(save_dir + 'fl_bsi_times.npy')
# with open(save_dir + 'setup_dict') as f:
#     setup_dict = json.load(f)


n_obs = len(polyline)
fl_pf_times_per_obs = fl_pf_times / n_obs
fl_bsi_times_per_obs = fl_bsi_times / n_obs


line_styles = ['-', '--', ':', '-.']


def comp_plot(n_samps,
              max_rejects,
              times,
              leg=False,
              **kwargs):
    fig, ax = plt.subplots()
    for i, n in reversed(list(enumerate(n_samps))):
        ax.plot(max_rejects, times[i], label=str(n), linestyle=line_styles[i], **kwargs)
        # ax.plot(max_rejects, times[i], label=str(n))
    ax.set_xlabel('Maximum rejections')
    ax.set_ylabel('Runtime per observation, s')
    if leg:
        ax.legend(loc='upper right', title='N')
    fig.tight_layout()
    return fig, ax


pf_fig, pf_ax = comp_plot(n_samps, max_rejections, np.mean(fl_pf_times_per_obs, axis=0), color='red', leg=True)

bsi_fig, bsi_ax = comp_plot(n_samps, max_rejections, np.mean(fl_bsi_times_per_obs, axis=0), color='blue', leg=True)
pf_ax.set_ylim(bsi_ax.get_ylim())

pf_fig.savefig(save_dir + 'pf_mr_compare', dpi=400)
bsi_fig.savefig(save_dir + 'bsi_mr_compare', dpi=400)

# pf_ax.set_xticks(xt)
# bsi_ax.set_xticks(xt)