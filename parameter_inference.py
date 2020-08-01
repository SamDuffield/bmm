########################################################################################################################
# Module: parameter_inference.py
# Description: Tune hyperparameters using some Porto taxi data.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

import numpy as np

import os
import sys

repo_path = os.getcwd()
sys.path.append(repo_path)


from bmm.src.data.utils import source_data, read_data, choose_data
from bmm.src.tools.graph import load_graph

import bmm

np.random.seed(0)

timestamps = 15
n_iter = 50
n_particles = 100

# Source data paths
_, process_data_path = source_data()

# Load networkx graph
graph = load_graph()

# Load taxi data
data_path = process_data_path + "/data/portotaxi_05052014_12052014_utm_bbox.csv"
raw_data = read_data(data_path)
polyline_indices = np.array([0, 1652,  3300,  4951,  6601,  8250,  9900, 11550,
                             13200, 14850, 16502., 18152, 19800, 21450, 23101, 24750,
                             26400, 28050, 29700, 31350])

polylines = [np.asarray(raw_data['POLYLINE_UTM'][single_index]) for single_index in polyline_indices]
lens = [len(po) for po in polylines]
del raw_data

# Initiate model
mm_model = bmm.GammaMapMatchingModel()
mm_model.zero_dist_prob_neg_exponent = -np.log(0.2) / timestamps
mm_model.distance_params['a_speed'] = 1.
mm_model.distance_params_bounds['a_speed'] = (1., 1.)
mm_model.distance_params['b_speed'] = 0.1
mm_model.deviation_beta = 0.01
mm_model.gps_sd = 7.

# mm_model.deviation_beta = 0
# mm_model.deviation_beta_bounds = (0, 0)


params_track = bmm.offline_em(graph, mm_model, timestamps, polylines, n_iter=n_iter, max_rejections=0,
                              n_ffbsi=n_particles, initial_d_truncate=50,
                              gradient_stepsize_scale=1e-5, gradient_stepsize_neg_exp=0.5)

