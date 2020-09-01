import os
import json
import sys

sim_dat_path = os.getcwd()
repo_path = os.path.dirname(os.path.dirname(sim_dat_path))
sys.path.append(sim_dat_path)
sys.path.append(repo_path)

import numpy as np
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt

import bmm

import utils

seed = 0
np.random.seed(seed)

timestamps = 15
n_samps = 200
lag = 3
max_rejections = np.array([0, 10, 20, 30, 40, 50])
initial_truncation = None
num_repeats = 1
max_speed = 35
proposal_dict = {'proposal': 'optimal',
                 'num_inter_cut_off': 10,
                 'resample_fails': False,
                 'd_max_fail_multiplier': 2.}
update = 'BSi'

setup_dict = {'seed': seed,
              'n_samps': n_samps,
              'lag': lag,
              'update': update,
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

# Run FFBSi
fl_routes = np.empty((num_repeats, len(max_rejections)), dtype=object)
fl_times = np.zeros((num_repeats, len(max_rejections)))

n_failures = 0

for i in range(num_repeats):
    for j, max_reject_int in enumerate(max_rejections):
        print(i, j)
        try:
            fl_routes[i, j] = bmm._offline_map_match_fl(graph,
                                                        polyline,
                                                        n_samps,
                                                        timestamps=timestamps,
                                                        mm_model=mm_model,
                                                        lag=lag,
                                                        update=update,
                                                        max_rejections=max_reject_int,
                                                        initial_d_truncate=initial_truncation,
                                                        **proposal_dict)
            fl_times[i, j] = fl_routes[i, j].time
            print(f'FL PF {i} {j}: {fl_routes[i, j].time}')
        except:
            n_failures += 1
        print(f'FL PF failures: {n_failures}')
        utils.clear_cache()

print(f'FL PF failures: {n_failures}')

np.save(save_dir + 'fl_routes', fl_routes)
np.save(save_dir + 'fl_times', fl_times)

#
# fl_routes = np.load(save_dir + 'fl_routes.npy', allow_pickle=True)
# fl_times = np.load(save_dir + 'fl_times.npy', allow_pickle=True)
# with open(save_dir + 'setup_dict') as f:
#     setup_dict = json.load(f)


n_obs = len(polyline)
fl_times_per_obs = fl_times/n_obs

plt.plot(max_rejections, np.mean(fl_times_per_obs, axis=0))
plt.savefig(save_dir + 'max_reject_compare', dpi=400)

