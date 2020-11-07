import json

import numpy as np
import matplotlib.pyplot as plt

import os

from .utils import download_cambridge_graph, load_graph

import bmm

# Setup
seed = 0
np.random.seed(seed)

# Model parameters
time_interval = 100
route_length = 4
gps_sd = 10
num_inter_cut_off = 10

# Inference parameters
n_samps = 1000

max_rejections = 30

proposal_dict = {'proposal': 'optimal'}

save_dir = os.getcwd() + '/single_ffbsi/'

# Initiate map-matching probabilistic model
mm_model = bmm.ExponentialMapMatchingModel()
mm_model.gps_sd = gps_sd

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

setup_dict = {'seed': seed,
              'time_interval': time_interval,
              'max_route_length': route_length,
              'zero_dist_prob_neg_exponent': mm_model.distance_params['zero_dist_prob_neg_exponent'],
              'lambda_speed': mm_model.distance_params['lambda_speed'],
              'deviation_beta': mm_model.deviation_beta,
              'gps_sd': mm_model.gps_sd,
              'num_inter_cut_off': num_inter_cut_off,
              'n_samps': n_samps,
              'max_rejections': max_rejections}

with open(save_dir + 'setup_dict', 'w+') as f:
    json.dump(setup_dict, f)

# Load cam_graph
graph_path = os.getcwd() + '/cambridge_projected_simple.graphml'

if not os.path.exists(graph_path):
    download_cambridge_graph(graph_path)

cam_graph = load_graph(graph_path)

# Add noise to generate observations
# observations_ll = [[0.12188, 52.198387],
#                    [0.125389, 52.197771],
#                    [0.128354, 52.199379],
#                    [0.130296, 52.201701],
#                    [0.127742, 52.20407],
#                    [0.126433, 52.205753],
#                    [0.127536, 52.207831],
#                    [0.126082, 52.212281]]

observations_ll = [[0.12188, 52.198387],
                   [0.125389, 52.197771],
                   [0.128354, 52.199379],
                   [0.130296, 52.201701],
                   [0.127742, 52.20407],
                   [0.126433, 52.205753],
                   [0.127836, 52.207831],
                   [0.126082, 52.212281]]

observations = bmm.long_lat_to_utm(observations_ll, cam_graph)

fig, ax = bmm.plot(cam_graph, polyline=observations)
fig.savefig(save_dir + 'observations', dpi=400, bbox_inches='tight')

ffbsi_route = bmm.offline_map_match(cam_graph, observations, n_samps, time_interval, mm_model,
                                    max_rejections=max_rejections, num_inter_cut_off=num_inter_cut_off, d_max=700)

ffbsi_route_arr = np.empty(1, dtype='object')
ffbsi_route_arr[0] = ffbsi_route
np.save(save_dir + 'ffbsi_route', ffbsi_route_arr)

# ffbsi_route = np.load(save_dir + 'ffbsi_route.npy', allow_pickle=True)[0]
fig2, ax2 = bmm.plot(cam_graph, ffbsi_route, observations)
fig2.savefig(save_dir + 'ffbsi', dpi=400, bbox_inches='tight')


def dist_hist(particle_distances, viterbi_distances=None):
    fig, axes = plt.subplots(len(particle_distances), sharex=True)
    axes[-1].set_xlabel('Metres')
    # axes[0].xlim = (0, 165)
    for i, d in enumerate(particle_distances):
        axes[i].hist(d, bins=40, color='purple', alpha=0.5, zorder=0, density=True)
        axes[i].set_yticklabels([])
        if viterbi_distances is not None:
            axes[i].scatter(viterbi_distances[i], 0, s=100, zorder=1, color='blue')
        axes[i].set_ylabel(f'$d_{i + 1}$')
    plt.tight_layout()
    return fig, axes

ffbsi_dists = np.array([bmm.observation_time_rows(p)[1:, -1] for p in ffbsi_route])

optim_route = np.load(save_dir + 'optim_route.npy', allow_pickle=True)
optim_dists = bmm.observation_time_rows(optim_route)[1:, -1]

fig_optim, ax_optim = bmm.plot(cam_graph, optim_route, observations)
fig_optim.savefig(save_dir + 'optim', dpi=400, bbox_inches='tight')

fig_hists, axes_hists = dist_hist(ffbsi_dists.T, optim_dists)
fig_hists.savefig(save_dir + 'ffbsi_hists', dpi=400)
