import json

import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt

import bmm
from simulated_data.utils import rmse

# Load graph
graph_dir = '/Users/samddd/Main/Data/bayesian-map-matching/graphs/Cambridge/'
graph_name = 'cambridge_latest_utm_simplified_clean_int_rde'
graph_path = graph_dir + graph_name + '.graphml'

graph = ox.load_graphml(graph_path)

# Simulation data directory
run_indicator = '0_15_20'

# Proposal directory
proposal = 'optimal'
load_dir = '/Users/samddd/Main/Data/bayesian-map-matching/simulations/single_route/' \
           + str(run_indicator) + '/' + str(proposal) + '/'

# Simulation params
with open(load_dir + 'setup_dict') as f:
    setup_dict = json.load(f)

# Load true route and convert
route = np.load(load_dir + 'route.npy', allow_pickle=True)
cartesianised_route = bmm.cartesianise_path(graph, route, t_column=True, observation_time_only=True)

# Load noisy observations
observations = np.load(load_dir + 'observations.npy', allow_pickle=True)

# Load simulation data
fl_pf_routes = np.load(load_dir + 'fl_pf.npy', allow_pickle=True)
fl_bsi_routes = np.load(load_dir + 'fl_bsi.npy', allow_pickle=True)
ffbsi_routes = np.load(load_dir + 'ffbsi.npy', allow_pickle=True)

lags = setup_dict['lags']
t_linspace = np.linspace(0, (setup_dict['route_length'] - 1) * setup_dict['time_interval'], setup_dict['route_length'])

fontsize = 5
shift = 0.08

l_start = 0.01
u_start = 0.85

lines = [None] * (len(lags) + 1)

fig, axes = plt.subplots(3, 2, sharex='all', sharey='all', figsize=(8, 6))
for j, n in enumerate(setup_dict['n_samps']):
    for k, lag in enumerate(lags):
        axes[j, 0].plot(t_linspace, rmse(graph, fl_pf_routes[k, j], cartesianised_route, each_time=True),
                        label=f'Lag: {lag}')
        lines[k], = axes[j, 1].plot(t_linspace, rmse(graph, fl_bsi_routes[k, j], cartesianised_route, each_time=True),
                                    label=f'Lag: {lag}')

    lines[len(lags)], = axes[j, 0].plot(t_linspace, rmse(graph, ffbsi_routes[j], cartesianised_route, each_time=True),
                                        label='FFBSi')

    axes[j, 1].plot(t_linspace, rmse(graph, ffbsi_routes[j], cartesianised_route, each_time=True),
                    label='FFBSi')

for j, n in enumerate(setup_dict['n_samps']):
    for k, lag in enumerate(lags):
        axes[j, 0].text(l_start, u_start - k * shift, "{:.1f}".format(fl_pf_routes[k, j].time),
                        color=lines[k].get_color(),
                        fontsize=fontsize, transform=axes[j, 0].transAxes)
        axes[j, 1].text(l_start, u_start - k * shift, "{:.1f}".format(fl_bsi_routes[k, j].time),
                        color=lines[k].get_color(),
                        fontsize=fontsize, transform=axes[j, 1].transAxes)
    axes[j, 0].text(l_start, u_start - len(lags) * shift, "{:.1f}".format(ffbsi_routes[j].time),
                    color=lines[len(lags)].get_color(),
                    fontsize=fontsize, transform=axes[j, 0].transAxes)
    axes[j, 1].text(l_start, u_start - len(lags) * shift, "{:.1f}".format(ffbsi_routes[j].time),
                    color=lines[len(lags)].get_color(),
                    fontsize=fontsize, transform=axes[j, 1].transAxes)

    axes[j, 0].text(l_start, u_start + shift, "Av Runtime (s)",
                    fontsize=fontsize, transform=axes[j, 0].transAxes)

    axes[j, 1].text(l_start, u_start + shift, "Av Runtime (s)",
                    fontsize=fontsize, transform=axes[j, 1].transAxes)

    axes[j, 0].set_ylabel(f'RMSE   N={n}')

axes[-1, 0].set_xlabel('t')
axes[-1, 1].set_xlabel('t')

axes[0, 0].set_title('FL Particle Filter')
axes[0, 1].set_title('FL (Partial) Backward Simulation')

plt.legend(loc='upper right')

plt.tight_layout()

plt.savefig(load_dir + 'route_rmse_compare.png', dpi=350)
