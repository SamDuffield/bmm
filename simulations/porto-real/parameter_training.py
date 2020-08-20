########################################################################################################################
# Module: parameter_inference.py
# Description: Tune hyperparameters using some Porto taxi data.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################


import os
import json
import sys

repo_path = os.getcwd()
sys.path.append(repo_path)

import numpy as np
import osmnx as ox
import pandas as pd

import bmm

np.random.seed(0)

timestamps = 15
n_iter = 50
n_particles = 100

graph_path = os.getcwd() + '/portotaxi_graph_portugal-140101.osm._simple.graphml'
# graph_path = repo_path + '/simulations/porto/portotaxi_graph_portugal-140101.osm._simple.graphml'
graph = ox.load_graphml(graph_path)

train_data_path = os.getcwd() + '/training_data.csv'
# train_data_path = repo_path + '/simulations/porto//training_data.csv'
# Load long-lat polylines
polylines_ll = [np.array(json.loads(poly)) for poly in pd.read_csv(train_data_path)['POLYLINE']]
# Convert to utm
polylines = [bmm.long_lat_to_utm(poly, graph) for poly in polylines_ll]

# Initiate model
mm_model = bmm.ExponentialMapMatchingModel()
# mm_model.zero_dist_prob_neg_exponent = -np.log(0.2) / timestamps
mm_model.distance_params['zero_dist_prob_neg_exponent'] = -np.log(0.2) / timestamps
mm_model.distance_params['lambda_speed'] = 1/10
mm_model.deviation_beta = 0.01
mm_model.gps_sd = 7.


params_track = bmm.offline_em(graph, mm_model, timestamps, polylines, n_iter=n_iter, max_rejections=0,
                              n_ffbsi=n_particles, initial_d_truncate=50,
                              gradient_stepsize_scale=1e-5, gradient_stepsize_neg_exp=0.1,
                              num_inter_cut_off=10, ess_threshold=0.8)
