########################################################################################################################
# Module: parameter_inference.py
# Description: Tune hyperparameters using some Porto taxi data.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################


import os
import json

import numpy as np
import osmnx as ox
import pandas as pd

import bmm

np.random.seed(0)

timestamps = 15
n_iter = 200
n_particles = 100

sim_dir = os.getcwd()
graph_path = sim_dir + '/portotaxi_graph_portugal-140101.osm._simple.graphml'
graph = ox.load_graphml(graph_path)

train_data_path = sim_dir + '/training_data.csv'

# Load long-lat polylines
polylines_ll = [np.array(json.loads(poly)) for poly in pd.read_csv(train_data_path)['POLYLINE']]
# Convert to utm
polylines = [bmm.long_lat_to_utm(poly, graph) for poly in polylines_ll]

# Initiate model
mm_model = bmm.ExponentialMapMatchingModel()
mm_model.distance_params['zero_dist_prob_neg_exponent'] = -np.log(0.15) / timestamps
mm_model.distance_params['lambda_speed'] = 1 / 10
mm_model.deviation_beta = 0.1
mm_model.gps_sd = 7.

params_track = bmm.offline_em(graph, mm_model, timestamps, polylines,
                              save_path=os.getcwd() + '/tuned_params.pickle',
                              n_iter=n_iter, max_rejections=30,
                              n_ffbsi=n_particles, initial_d_truncate=50,
                              gradient_stepsize_scale=1e-5, gradient_stepsize_neg_exp=0.5,
                              num_inter_cut_off=10, ess_threshold=1.)
