

import numpy as np
import matplotlib.pyplot as plt

import bmm

from bmm.src.tools.graph import load_graph
from bmm.src.data.utils import source_data, read_data

_, process_data_path = source_data()

graph = load_graph()

data_path = process_data_path + "/data/portotaxi_06052014_06052014_utm_1730_1745.csv"
raw_data = read_data(data_path, 100).get_chunk()

route_indices = np.arange(20, 40)
polylines = [np.asarray(raw_data['POLYLINE_UTM'][i]) for i in route_indices]
# for i in route_indices:
#     bmm.plot(graph, polyline=np.asarray(raw_data['POLYLINE_UTM'][i]))
#
# ffbsi_routes_list = []
# for po in polylines:
#     ffbsi_routes_list.append(bmm.offline_map_match(graph, po, 200, 15, max_rejections=0, initial_d_truncate=100))

ffbsi_routes_list = list(np.load('/Users/samddd/Desktop/ffbsi_20_34.npy', allow_pickle=True))


# Box plot
def dist_box_plot(particle_distances, viterbi_distances=None):
    flierprops = dict(marker='.')
    medianprops = dict(color='black')

    fig, ax = plt.subplots()
    ax.boxplot(particle_distances.T, vert=False, showfliers=False,
               flierprops=flierprops, medianprops=medianprops,
               zorder=0)
    ax.invert_yaxis()
    ax.set_ylabel('t')
    if viterbi_distances is not None:
        ax.scatter(viterbi_distances[::-1], np.arange(len(particle_distances), 0, -1), s=15, zorder=1)
    plt.tight_layout()
    return fig, ax


# Histograms
def dist_hist(particle_distances, viterbi_distances=None):
    fig, axes = plt.subplots(len(particle_distances), sharex=True)
    axes[-1].set_xlabel('m')
    # axes[0].xlim = (0, 165)
    for i, d in enumerate(particle_distances):
        axes[i].hist(d, bins=50, color='purple', alpha=0.5, zorder=0, density=True)
        axes[i].set_yticklabels([])
        if viterbi_distances is not None:
            axes[i].scatter(viterbi_distances[i], 0, s=100, zorder=1, color='blue')
        axes[i].set_ylabel(f'$d_{i+1}$')
    plt.tight_layout()
    return fig, axes


for parts in ffbsi_routes_list:
    bmm_distances = np.empty((parts.m-1, parts.n))
    for i in range(parts.n):
        bmm_distances[:, i] = bmm.observation_time_rows(parts[i])[1:, -1]
    # dist_box_plot(bmm_distances)
    dist_hist(bmm_distances)
