import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
import sys

sim_dat_path = os.getcwd()
repo_path = os.path.dirname(sim_dat_path)
sys.path.append(sim_dat_path)
sys.path.append(repo_path)

from simulated_data.utils import sample_route, cambridge_graph

import bmm

np.random.seed(0)

# Load networkx graph
cam_graph = cambridge_graph()

timestamps = 15

gen_model = bmm.GammaMapMatchingModel()
# gen_model.max_speed = 30
# gen_model.zero_dist_prob_neg_exponent = -np.log(0.25) / timestamps
gen_model.distance_params['zero_dist_prob_neg_exponent'] = -np.log(0.25) / timestamps
gen_model.distance_params['a_speed'] = 1.
gen_model.distance_params['b_speed'] = 1/15
gen_model.deviation_beta = 0.05
gen_model.gps_sd = 3.
num_inter_cut_off = None

# Generate simulated routes
num_repeats = 1
num_routes = 20
min_route_length = 10
max_route_length = 50
sample_d_refine = 1
n_iter = 50

params_track = []

for repeat_int in range(num_repeats):

    print(repeat_int, '\n\n')

    routes = [sample_route(cam_graph, gen_model, timestamps, max_route_length, d_refine=sample_d_refine) for _ in
              range(num_routes)]
    true_polylines = [bmm.observation_time_rows(rou)[:, 5:7] for rou in routes]
    routes_obs_rows = [bmm.observation_time_rows(rou) for rou in routes]
    len_routes = [len(rou) for rou in routes]
    len_obs = np.array([len(po) for po in true_polylines])

    while np.any(len_obs < min_route_length):
        for i in range(num_routes):
            if len_obs[i] < min_route_length:
                routes[i] = sample_route(cam_graph, gen_model, timestamps, max_route_length, d_refine=sample_d_refine,
                                         num_inter_cut_off=num_inter_cut_off, num_pos_route_cap=10)
        true_polylines = [bmm.observation_time_rows(rou)[:, 5:7] for rou in routes]
        routes_obs_rows = [bmm.observation_time_rows(rou) for rou in routes]
        len_routes = [len(rou) for rou in routes]
        len_obs = np.array([len(po) for po in true_polylines])
        # print(np.sum(len_obs < 10))

    observations = [po + gen_model.gps_sd * np.random.normal(size=po.shape) for po in true_polylines]

    ###
    distances = np.concatenate([a[1:, -1] for a in routes_obs_rows])
    print(np.mean(distances < 1e-5))
    print(-np.log(np.mean(distances < 1e-5)) / 15)
    print(np.sum(distances < 1e-5))
    # for route, obs in zip(routes, observations):
    #     bmm.plot(cam_graph, route, obs)
    # plt.hist(distances[distances > 1e-5]/timestamps, bins=50, density=True)
    # linsp = np.linspace(0.01, 25, 100)
    # plt.plot(linsp, gen_model.distance_params['b_speed'] * np.exp(-gen_model.distance_params['b_speed']*linsp))

    # Run EM
    tune_model = bmm.GammaMapMatchingModel()
    tune_model.distance_params['a_speed'] = 1.
    tune_model.distance_params_bounds['a_speed'] = (1., 1.)
    tune_model.distance_params['b_speed'] = 0.1
    # tune_model.zero_dist_prob_neg_exponent = -np.log(0.2) / timestamps
    tune_model.distance_params['zero_dist_prob_neg_exponent'] = -np.log(0.2) / timestamps
    tune_model.deviation_beta = 0.01
    tune_model.gps_sd = 7.

    # tune_model.zero_dist_prob_neg_exponent = gen_model.zero_dist_prob_neg_exponent
    # tune_model.distance_params['zero_dist_prob_neg_exponent'] = gen_model.distance_params['zero_dist_prob_neg_exponent']
    # tune_model.distance_params['a_speed'] = gen_model.distance_params['a_speed']
    # tune_model.distance_params_bounds['a_speed'] = (1., 1.)
    # tune_model.distance_params['b_speed'] = gen_model.distance_params['b_speed']
    # tune_model.deviation_beta = gen_model.deviation_beta
    # tune_model.gps_sd = gen_model.gps_sd

    # tune_model.deviation_beta_bounds = (0, 0)

    params_track_single = bmm.offline_em(cam_graph, tune_model, timestamps, observations, n_iter=n_iter,
                                         max_rejections=0,
                                         initial_d_truncate=50, num_inter_cut_off=num_inter_cut_off,
                                         gradient_stepsize_scale=1e-4, gradient_stepsize_neg_exp=0.75)

    params_track.append(params_track_single)

    pickle.dump(params_track, open('param_track_repeats.pickle', 'wb'))
