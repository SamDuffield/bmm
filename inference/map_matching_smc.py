################################################################################
# Module: map_matching_mcmc.py
# Description: Infer route taken by vehicle given sparse observations.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import numpy as np
import data
from tools.graph import load_graph
import tools.edges
import tools.sampling
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd

# Relative probability of u-turn at intersection
u_turn_downscale = 0.02

# Speed prior hyperparameters (all links a priori iid)
v_mean = 8.58
v_std = 8.34
v_max = 40

# Convert to underlying normal parameters (for lognormal distribution)
v_log_mean = np.log(v_mean / np.sqrt(1 + (v_std / v_mean) ** 2))
v_log_std = np.log(1 + (v_std / v_mean) ** 2)

# Conditional speed proposal variance
prop_v_std = 3


def sample_speed_given_x_t_y_t1(x_t, y_t1, delta_y, size=None):
    """
    Sample proposal speed conditional on current and next observed positions. q(v_t_t1| x_t, y_t1)
    Speed is assumed constant between observations.
    :param x_t: current position pd.df or pd.Series with geometry and alpha columns
    :param y_t1: next observed position, cartesian.
    :param delta_y: time between observations (seconds)
    :param size: number of samples to return, default is 1 (returns float or np.array if size != None)
    :return: sampled speed(s) (float or np.array if size != None)
    """
    # Convert x to cartesian
    x_t_cartesian = tools.edges.edge_interpolate(x_t['geometry'], x_t['alpha'])

    # Distance between x and y
    x_t_y_t_distance = ox.euclidean_dist_vec(x_t_cartesian[0], x_t_cartesian[1], y_t1[0], y_t1[1])

    # Distance as mean of lognormal distribution (variance defined outside function)
    prop_v_mean = x_t_y_t_distance / delta_y

    # Convert to log parameters
    phi = (prop_v_std ** 2 + prop_v_mean ** 2) ** 0.5
    prop_v_log_mean = np.log(prop_v_mean ** 2 / phi)
    prop_v_log_std = (np.log(phi ** 2 / prop_v_mean ** 2)) ** 0.5

    # Return sample (capped at v_max from prior))
    return np.minimum(np.random.lognormal(prop_v_log_mean, prop_v_log_std, size), v_max)


def sample_next_edge(graph_edges, prev_edge):
    """
    Given an edge sample an edge for a vehicle to travel on next (after intersection).
    Assumes equally likely to pass onto all adjacent edges at intersection
    (except u_turn which is downscaled but not necessarily 0)
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param prev_edge: edge just traversed, pandas.Series
    :return: sampled new edge, pandas.Series
    """
    adj_v_edges = graph_edges[graph_edges['u'] == prev_edge['v']].reset_index(drop=True)

    n_adj = len(adj_v_edges)

    weights = np.ones(n_adj) / n_adj

    u_turn_possible = prev_edge['u'] in adj_v_edges['v'].to_list()

    if u_turn_possible:
        weights[adj_v_edges['v'] == prev_edge['u']] = 0.2 / n_adj
        weights /= sum(weights)

    sampled_index = np.random.choice(n_adj, p=weights)

    return adj_v_edges.iloc[sampled_index]


def propagate_x(graph_edges, edge, delta_x, speed=None):
    """
    Increment vehicle forward one time step (discretisation time step not observation)
    If propagation reaches the end of the edge (alpha = 1) sample a new edge to traverse to and new speed for that edge.
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param edge: pandas.Series with u, v, key, geometry, alpha, distance_to_obs, speed (if speed is None,
    otherwise speed column will be replaced by speed input in new returned series)
    :param delta_x: time discretisation (not observation time interval!)
    :param speed: inputted speed
    :return: propagated edge_and_speed
    """
    # Initiate output
    out_edge_and_speed = edge.copy()
    out_edge_and_speed['t'] = edge['t'] + delta_x
    out_edge_and_speed['distance_to_obs'] = None

    if speed is not None:
        out_edge_and_speed['speed'] = speed

    # Check if reached intersection
    if edge['alpha'] == 1:
        alpha_dash = 0
        out_edge_and_speed[['u', 'v', 'key', 'geometry']]\
            = sample_next_edge(graph_edges, edge[['u', 'v', 'key', 'geometry']])
    else:
        alpha_dash = edge['alpha']

    # Propagate alpha
    out_edge_and_speed['alpha'] = min(1,
                                      alpha_dash
                                      + delta_x * out_edge_and_speed['speed'] / edge['geometry'].length)

    return out_edge_and_speed


def sample_x0_n_lookahead_y0_n_lookahead(graph_edges, y, n_lookahead, delta_x, delta_y, n_propose_max=100):

    for iter_count in range(n_propose_max):
        # Sample x_0 from p(x_0|y_0)
        xv_df = tools.sampling.sample_x0(graph_edges, y[0], 1)
        xv_df.insert(loc=0, column='t', value=[0])

        # Speed 0 at time 0
        xv_df['speed'] = 0

        # Sample v_(0,1] from q(v_(0,1] | x_0, y_1)
        speed = sample_speed_given_x_t_y_t1(xv_df.iloc[-1][['geometry', 'alpha']], y[1], delta_y)

        # Sample x_(0,1] from p(x_(0,1] | x_0, v_(0,1])
        i = 0
        while xv_df['t'][i] < delta_y:
            xv_df = xv_df.append(propagate_x(graph_edges, xv_df.iloc[i], delta_x, speed)).reset_index(drop=True)
            i += 1

        # Measure distance between x_1 and y_1
        xv_obs_time = xv_df.iloc[-1]
        xv_xy = tools.edges.edge_interpolate(xv_obs_time['geometry'], xv_obs_time['alpha'])
        xv_dist_obs = ox.euclidean_dist_vec(y[1][1], y[1][0], xv_xy[1], xv_xy[0])
        xv_df.at[xv_df.shape[0] - 1, 'distance_to_obs'] = xv_dist_obs

        # Start again if x_1 outside truncation of y_1
        if xv_dist_obs > tools.edges.dist_retain:
            continue

        # Sample v_(1,2] from q(v_(1,2] | x_1, y_2)
        speed = sample_speed_given_x_t_y_t1(xv_df.iloc[-1][['geometry', 'alpha']], y[2], delta_y)

        # Sample x_(1,2] from p(x_(1,2] | x_1, v_(1,2])
        while xv_df['t'][i] < delta_y*2:
            xv_df = xv_df.append(propagate_x(graph_edges, xv_df.iloc[i], delta_x, speed)).reset_index(drop=True)
            i += 1

        # Measure distance between x_2 and y_2
        xv_obs_time = xv_df.iloc[-1]
        xv_xy = tools.edges.edge_interpolate(xv_obs_time['geometry'], xv_obs_time['alpha'])
        xv_dist_obs = ox.euclidean_dist_vec(y[2][1], y[2][0], xv_xy[1], xv_xy[0])
        xv_df.at[xv_df.shape[0] - 1, 'distance_to_obs'] = xv_dist_obs

        # Start again if x_2 outside truncation of y_2
        if xv_dist_obs > tools.edges.dist_retain:
            continue
        else:
            return xv_df, iter_count + 1

    return None, iter_count + 1


if __name__ == '__main__':
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Load networkx graph and edges gdf
    graph = load_graph()
    edges_gdf = tools.edges.graph_edges_gdf(graph)

    # Load taxi data
    data_path = data.utils.choose_data()
    raw_data = data.utils.read_data(data_path, 100).get_chunk()

    # Select single polyline
    single_index = 0
    poly_single = raw_data['POLYLINE_UTM'][single_index]

    # Number of observations
    M_obs = len(poly_single)

    # Sample size
    N_samps = 5

    # Observation time increment (s)
    delta_obs = 15

    # Between observation discretisation
    # Number of inference times per observation
    N_time_dis = 5
    delta_x = delta_obs/N_time_dis

    # Lookahead size (sample from p(x_n | y_0:(n+N_lookahead))
    N_lookahead = 2

    # Single sample from p(x_0:N_lookahead, y_0:N_lookahead)
    xv_single, n_iters = sample_x0_n_lookahead_y0_n_lookahead(edges_gdf, poly_single, N_lookahead, delta_x, delta_obs)

    # Print df and samples required
    print(xv_single)
    print(n_iters)

    # Plot sample
    tools.edges.plot_graph_with_weighted_points(graph, poly_single, xv_single)

    plt.show(block=True)
