########################################################################################################################
# Module: map_match.py
# Description: Script that loads cam_graph and polyline before running map-matching algorithm.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

import numpy as np

import matplotlib.pyplot as plt

from bmm.src.data.utils import source_data, read_data
from bmm.src.tools.graph import load_graph

from bmm import offline_map_match, _offline_map_match_fl, plot, ExponentialMapMatchingModel

np.random.seed(0)

# Source data paths
_, process_data_path = source_data()

# Load networkx cam_graph
graph = load_graph()

# Load taxi data
# data_path = data.utils.choose_data()
data_path = process_data_path + "/data/portotaxi_06052014_06052014_utm_1730_1745.csv"
raw_data = read_data(data_path, 100).get_chunk()

# Select single polyline
# single_index = np.random.choice(100, 1)[0]
single_index = 0
# single_index = 68         # deviation to rule out
# single_index = 44         # map incorrect? + 54
# single_index = 76         # dead end with too high probability issue
# single_index = 86         # times out
# single_index = 11         # can't initialise
# single_index = 46
poly_single_list = raw_data['POLYLINE_UTM'][single_index]
poly_single = np.asarray(poly_single_list)

print(single_index)

# Number of particles
n_samps = 100

polyline_truncation = None

max_rejects = 30

num_inter_cut_off = None

lag = 3

mm_model = ExponentialMapMatchingModel()
# mm_model.max_speed = 35

resample_fails = False

# Run offline map-matching
particles = _offline_map_match_fl(graph, poly_single[:polyline_truncation], n_samps, timestamps=15,
                                  lag=lag, mm_model=mm_model,
                                  d_refine=1, num_inter_cut_off=num_inter_cut_off,
                                  max_rejections=max_rejects,
                                  update='PF',
                                  resample_fails=resample_fails)

particles = _offline_map_match_fl(graph, poly_single[:polyline_truncation], n_samps, timestamps=15,
                                  lag=lag, mm_model=mm_model,
                                  d_refine=1,
                                  max_rejections=max_rejects,
                                  update='BSi',
                                  num_inter_cut_off=num_inter_cut_off,
                                  resample_fails=resample_fails)

particles = offline_map_match(graph, poly_single[:polyline_truncation], n_samps, timestamps=15,
                              mm_model=mm_model,
                              d_refine=1, num_inter_cut_off=num_inter_cut_off,
                              max_rejections=max_rejects,
                              ess_threshold=0.8)

print(particles.time)
print(particles.time / len(poly_single))

# Plot
# plot(cam_graph, particles, poly_single)
# plt.show()


