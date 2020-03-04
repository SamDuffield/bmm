########################################################################################################################
# Module: map_match.py
# Description: Script that loads graph and polyline before running map-matching algorithm.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import numpy as np

from data.utils import source_data, read_data
from tools.graph import load_graph
from tools.edges import plot_particles
from inference.smc import offline_map_match
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
# single_index = np.random.choice(100, 1)[0]
single_index = 0
poly_single_list = raw_data['POLYLINE_UTM'][single_index]
poly_single = np.asarray(poly_single_list)


# Run offline map-matching
n_samps = 100
# particles = offline_map_match(graph, poly_single[:5], n_samps, time_interval=15, lag=4, gps_sd=7, d_init_refine=1,
#                               d_max=None, d_refine=1)
particles = offline_map_match(graph, poly_single, n_samps, time_interval=15, lag=4, gps_sd=7, d_init_refine=1,
                              proposal=proposal.dist_then_edge_proposal, var=10)

# Plot
plot_particles(graph, particles, poly_single)
