########################################################################################################################
# Module: map_match.py
# Description: Script that loads graph and polyline before running map-matching algorithm.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from data.utils import source_data, read_data
from tools.graph import load_graph
from tools.edges import plot_particles
from inference.smc import offline_map_match_fl, offline_map_match
from inference import proposal

# Source data paths
_, process_data_path = source_data()

# Load networkx graph
graph = load_graph()

# Load taxi data
# data_path = data.utils.choose_data()
data_path = process_data_path + "/data/portotaxi_06052014_06052014_utm_1730_1745.csv"
raw_data = read_data(data_path, 100).get_chunk()

# Select single polyline
single_index = np.random.choice(100, 1)[0]
single_index = 0
# single_index = 44         # map incorrect?
# single_index = 76           # dead end with too high probability issue
# single_index = 86         # times out
# sinlge_index = 11         # can't initialise
# single_index = 32
poly_single_list = raw_data['POLYLINE_UTM'][single_index]
poly_single = np.asarray(poly_single_list)

print(single_index)

# Run offline map-matching
n_samps = 100

# particles = offline_map_match_fl(graph, poly_single[:13], n_samps, time_interval=15,
#                                  lag=3, gps_sd=7,
#                                  d_refine=1,
#                                  max_rejections=20,
#                                  d_max=None)

particles = offline_map_match_fl(graph, poly_single[:20], n_samps, time_interval=15,
                                 lag=10, gps_sd=7,
                                 d_refine=1,
                                 max_rejections=20,
                                 proposal=proposal.auxiliary_distance_proposal,
                                 update='BSi',
                                 dist_expand=50,
                                 var=5)


# particles = offline_map_match(graph, poly_single[:13], n_samps, time_interval=15,
#                               gps_sd=7,
#                               d_refine=1,
#                               max_rejections=10,
#                               proposal=proposal.auxiliary_distance_proposal,
#                               ess_threshold=0.5,
#                               dist_expand=50,
#                               var=5)

# particles = offline_map_match(graph, poly_single[:], n_samps, time_interval=15,
#                               gps_sd=7,
#                               d_refine=1,
#                               max_rejections=10,
#                               proposal=proposal.optimal_proposal,
#                               ess_threshold=0.5,
#                               d_max=None)

print(particles.time)
print(particles.time / len(poly_single))

# Plot
plot_particles(graph, particles, poly_single)
plt.show()
