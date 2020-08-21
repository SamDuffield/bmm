import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import sys

sim_dat_path = os.getcwd()
repo_path = os.path.dirname(sim_dat_path)
sys.path.append(sim_dat_path)
sys.path.append(repo_path)

from simulations.cambridge.utils import sample_route, download_cambridge_graph, load_graph

import bmm

np.random.seed(1)

# Load graph
graph_path = os.getcwd() + '/cambridge_projected_simple.graphml'

if not os.path.exists(graph_path):
    download_cambridge_graph(graph_path)

# Load networkx graph
cam_graph = load_graph(graph_path)

timestamps = 15

gen_model = bmm.ExponentialMapMatchingModel()
gen_model.max_speed = 40
gen_model.distance_params['zero_dist_prob_neg_exponent'] = -np.log(0.05) / timestamps # 0.1535056728662697
gen_model.distance_params['lambda_speed'] = 1/15
gen_model.deviation_beta = 0.02
gen_model.gps_sd = 3.0

num_inter_cut_off = None
num_pos_routes_cap = 100

# Generate simulated routes
num_routes = 50
min_route_length = 20
max_route_length = 50
sample_d_refine = 1
n_iter = 200

params_track = []

routes = [sample_route(cam_graph, gen_model, timestamps, max_route_length, d_refine=sample_d_refine,
          num_inter_cut_off=num_inter_cut_off, num_pos_route_cap=num_pos_routes_cap) for _ in range(num_routes)]
true_polylines = [bmm.observation_time_rows(rou)[:, 5:7] for rou in routes]
routes_obs_rows = [bmm.observation_time_rows(rou) for rou in routes]
len_routes = [len(rou) for rou in routes]
len_obs = np.array([len(po) for po in true_polylines])

while np.any(len_obs < min_route_length):
    for i in range(num_routes):
        if len_obs[i] < min_route_length:
            routes[i] = sample_route(cam_graph, gen_model, timestamps, max_route_length, d_refine=sample_d_refine,
                                     num_inter_cut_off=num_inter_cut_off, num_pos_route_cap=num_pos_routes_cap)
    true_polylines = [bmm.observation_time_rows(rou)[:, 5:7] for rou in routes]
    routes_obs_rows = [bmm.observation_time_rows(rou) for rou in routes]
    len_routes = [len(rou) for rou in routes]
    len_obs = np.array([len(po) for po in true_polylines])
    print(np.sum(len_obs < min_route_length))

observations = [po + gen_model.gps_sd * np.random.normal(size=po.shape) for po in true_polylines]

###
distances = np.concatenate([a[1:, -1] for a in routes_obs_rows])
print(np.mean(distances < 1e-5))
print(-np.log(np.mean(distances < 1e-5)) / 15)
print(np.sum(distances < 1e-5))

# Run EM
tune_model = bmm.ExponentialMapMatchingModel()
tune_model.distance_params['zero_dist_prob_neg_exponent'] = -np.log(0.2) / timestamps
tune_model.distance_params['lambda_speed'] = 1/10
tune_model.deviation_beta = 0.05
tune_model.gps_sd = 5.

# tune_model = bmm.ExponentialMapMatchingModel()
# tune_model.distance_params['zero_dist_prob_neg_exponent'] = gen_model.distance_params['zero_dist_prob_neg_exponent']
# tune_model.distance_params['lambda_speed'] = gen_model.distance_params['lambda_speed']
# tune_model.deviation_beta = gen_model.deviation_beta
# tune_model.gps_sd = gen_model.gps_sd

# tune_model.deviation_beta_bounds = (0, 0)

params_track_single = bmm.offline_em(cam_graph, tune_model, timestamps, observations,
                                     save_path=os.getcwd() + '/tuned_sim_params.pickle',
                                     n_iter=n_iter,
                                     max_rejections=0,
                                     initial_d_truncate=50, num_inter_cut_off=num_inter_cut_off,
                                     gradient_stepsize_scale=1e-6, gradient_stepsize_neg_exp=0.5)

