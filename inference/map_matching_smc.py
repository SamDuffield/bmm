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
prop_v_std = 1


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

    # Distance as mean of lognormal distribution
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


def propagate_x(graph_edges, edges_df, delta_x, speed=None):
    """
    Increment vehicle forward one time step (discretisation time step not observation).
    If propagation reaches the end of the edge (alpha = 1) sample a new edge to traverse to and new speed for that edge.
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param edges_df: gdf with u, v, key, geometry, alpha, distance_to_obs, speed
    :param delta_x: time discretisation (not observation time interval!)
    :param speed: inputted speed (if speed is None, takes speed of last row in edges_df)
    :return: edges_df with propagation appended to bottom of dataframe
    """
    # Extract last edge from dataframe
    prev_edge = edges_df.iloc[-1].copy()
    prev_edge['distance_to_obs'] = None

    if speed is not None:
        prev_edge['speed'] = speed

    # Total distance travelled between observations (constant speed)
    distance_left_to_travel = delta_x * speed

    # Distance tracker
    while distance_left_to_travel > 0:
        distance_left_on_edge = (1 - prev_edge['alpha']) * prev_edge['geometry'].length

        if distance_left_on_edge < distance_left_to_travel:
            time_to_reach_end_of_edge = distance_left_on_edge / speed
            prev_edge['alpha'] = 1
            prev_edge['t'] += time_to_reach_end_of_edge
            edges_df = edges_df.append(prev_edge)
            prev_edge[['u', 'v', 'key', 'geometry']] =\
                sample_next_edge(graph_edges, prev_edge[['u', 'v', 'key', 'geometry']])
            prev_edge['alpha'] = 0

        else:
            prev_edge['t'] += distance_left_to_travel / speed
            prev_edge['alpha'] += distance_left_to_travel / prev_edge['geometry'].length
            edges_df = edges_df.append(prev_edge)

        distance_left_to_travel -= distance_left_on_edge

    return edges_df.reset_index(drop=True)


def sample_xv(graph_edges, xv0_df, y1, delta_x, delta_y):
    """
    Propagates forward vehicle for one observation interval.
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param xv0_df: gdf of previous edges and speeds
    :param y1: next observation
    :param delta_x: time discretisation of route
    :param delta_y: observation interval
    :return: xv0_df with newly sampled edges and speed appended
    """
    # Sample v_(0,1] from q(v_(0,1] | x_0, y_1)
    speed = sample_speed_given_x_t_y_t1(xv0_df.iloc[-1][['geometry', 'alpha']], y1, delta_y)

    # Sample x_(0,1] from p(x_(0,1] | x_0, v_(0,1])
    t = 0
    while t < delta_y:
        xv0_df = propagate_x(graph_edges, xv0_df, delta_x, speed)
        t += delta_x

    # Measure distance between x_1 and y_1
    xv_obs_time = xv0_df.iloc[-1]
    xv_xy = tools.edges.edge_interpolate(xv_obs_time['geometry'], xv_obs_time['alpha'])
    xv_dist_obs = ox.euclidean_dist_vec(y1[1], y1[0], xv_xy[1], xv_xy[0])
    xv0_df.at[xv0_df.shape[0] - 1, 'distance_to_obs'] = xv_dist_obs

    return xv0_df.copy()


def sample_xv_0_n_lookahead(graph_edges, y, n_lookahead, delta_x, delta_y, n_propose_max=100):
    """
    Sample from p(x_0:n_lookahead, v_0:n_lookahead|y_0:n_lookahead).
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param y: observed polyline
    :param n_lookahead: number of observation times to look forward
    :param delta_x: time discretisation
    :param delta_y: observation interval
    :param n_propose_max: maximum possible number of samples to generate before breaking
    :return: gdf with edges and speeds that pass through n_lookahead observation truncations
    """
    for iter_count in range(n_propose_max):
        # Sample x_0 from p(x_0|y_0)
        xv_df = tools.sampling.sample_x0(graph_edges, y[0], 1)
        xv_df.insert(loc=0, column='t', value=[0])

        # Speed 0 at time 0
        xv_df['speed'] = 0

        # Initiate variable checking route falls within observation truncation
        hitball = True

        for n_forward in range(n_lookahead):
            # Sample v_01 from p(v_01|x_0, y_1)
            # and x_01 from p(x_01|x_0, v_01)
            xv_df = sample_xv(graph_edges, xv_df, y[n_forward + 1], delta_x, delta_y)

            # Start again if x_1 outside truncation of y_1
            if xv_df.iloc[-1]['distance_to_obs'] > tools.edges.dist_retain:
                hitball = False
                break

        if hitball is True:
            return xv_df, iter_count + 1

    return None, n_propose_max


def sample_xv_n_lookahead_given_xv_prev(graph_edges, xv_df, yn_onwards, n_lookahead, delta_x, delta_y, n_propose_max=100):
    """
    Sample from p(x_n:n+n_lookahead, v_n:n+n_lookahead|x_n-1, y_n:n_lookahead).
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param xv_df: previous path (x_0:n-1)
    :param yn_onwards: end of observed polyline - must start one observation step after last entry in xv_df
    :param n_lookahead: number of observation times to look forward
    :param delta_x: time discretisation
    :param delta_y: observation interval
    :param n_propose_max: maximum possible number of samples to generate before breaking
    :return: gdf with edges and speeds of propagated xv_df through n_lookahead observation truncations
    """
    for iter_count in range(n_propose_max):
        # Initiate output
        xv_df_out = xv_df.copy()

        # Initiate variable checking route falls within observation truncation
        hitball = True

        for n_forward in range(n_lookahead + 1):
            # Sample v_01 from p(v_01|x_0, y_1)
            # and x_01 from p(x_01|x_0, v_01)
            xv_df_out = sample_xv(graph_edges, xv_df_out, yn_onwards[n_forward], delta_x, delta_y)

            # Start again if x_1 outside truncation of y_1
            if xv_df_out.iloc[-1]['distance_to_obs'] > tools.edges.dist_retain:
                hitball = False
                break

        if hitball is True:
            return xv_df_out, iter_count + 1

    return xv_df, n_propose_max


def sample_full_path(graph_edges, y, n_lookahead, delta_x, delta_y, n_propose_max=100):
    """
    Sample from p(x_0:M, v_0:M|y_0:M).
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param y: observed polyline
    :param n_lookahead: number of observation times to look forward
    :param delta_x: time discretisation
    :param delta_y: observation interval
    :param n_propose_max: maximum possible number of samples to generate before breaking
    :return: gdf with edges and speeds of propagated xv_df through n_lookahead observation truncations
    """
    # Number of observations
    M = len(y)

    # Sample x0|y0:2
    xv_df, iterations_n = sample_xv_0_n_lookahead(graph_edges, y,  min(M, n_lookahead),
                                                  delta_x, delta_y, n_propose_max)

    if iterations_n == n_propose_max:
        raise ValueError("Reached max iterations")

    for m in range(1, max(1, M-n_lookahead)):

        ind_keep = int(np.where(abs(xv_df['t'] - (m - 1) * delta_y) < 0.01)[0]) + 1

        xv_df = xv_df.iloc[0:ind_keep].copy()

        xv_df, iterations_n = sample_xv_n_lookahead_given_xv_prev(graph_edges, xv_df, y[m:], min(M-m, n_lookahead),
                                                                  delta_x, delta_y, n_propose_max)

        if iterations_n == n_propose_max:
            # raise ValueError("Reached max iterations")
            print("Reached max iterations")
            print(m, M)
            return xv_df

    return xv_df


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
    single_index = 4
    poly_single = raw_data['POLYLINE_UTM'][single_index]

    # Number of observations
    M_obs = len(poly_single)

    # Sample size
    N_samps = 5

    # Observation time increment (s)
    delta_obs = 15

    # Between observation discretisation
    # Number of inference times per observation
    N_time_dis = 1
    delta_x_dis = delta_obs/N_time_dis

    # Lookahead size (sample from p(x_n | y_0:(n+N_lookahead))
    N_lookahead = 2

    # # Single sample from p(x_0:N_lookahead, y_0:N_lookahead)
    # xv_single, n_iters = sample_xv_0_n_lookahead(edges_gdf, poly_single, N_lookahead, delta_x_dis, delta_obs)
    #
    # # Print df and samples required
    # print(xv_single)
    # print(n_iters)
    #
    # # Plot sample
    # tools.edges.plot_graph_with_weighted_points(graph, poly_single, xv_single)

    # Single sample of full path
    xv_full = sample_full_path(edges_gdf, poly_single, N_lookahead, delta_x_dis, delta_obs)

    # Plot sampled full path
    tools.edges.plot_graph_with_weighted_points(graph, poly_single, xv_full)
    plt.show(block=True)
