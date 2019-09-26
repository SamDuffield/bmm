################################################################################
# Module: map_matching.py
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
from scipy.stats import truncnorm

# Relative probability of u-turn at intersection
u_turn_downscale = 0.2

# Speed prior hyperparameters (all links a priori iid)
v_mean = 8.58
v_std = 8.34
v_max = 40

# Weighting for conditional speed
v_rho = 0.5


def sample_marginal_speed():
    """
    Samples a vehicle speed from prior. Assumes all speeds iid across links.
    Assumes truncated Gaussian distribution.
    Hyperparameters defined at top of file (outside function).
    :return: speed (float)
    """
    return truncnorm.rvs(a=-v_mean/v_std, b=(v_max-v_mean)/v_std, loc=v_mean, scale=v_std, size=1).item()


def sample_conditional_speed(previous_link_speed):
    """
    Samples a vehicle speed given it's speed on previous link.
    Weighted average of previous speed and sample from marginal on new link
    Hyperparameters defined at top of file (outside function).
    :return: speed (float)
    """
    return previous_link_speed * v_rho + (1-v_rho) * sample_marginal_speed()


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
        weights[adj_v_edges['v']==prev_edge['u']] = 0.2 / n_adj
        weights /= sum(weights)

    sampled_index = np.random.choice(n_adj, p=weights)

    return adj_v_edges.iloc[sampled_index]


def propagate_x(graph_edges, edge_and_speed, delta_x):
    """
    Increment vehicle forward one time step (discretisation time step not observation)
    If propagation reaches the end of the edge (alpha = 1) sample a new edge to traverse to and new speed for that edge.
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param edge_and_speed: pandas.Series with u, v, key, geometry, alpha, distance_to_obs, speed
    :param delta_x: time discretisation (not observation time interval!)
    :return: propagated edge_and_speed
    """
    # Initiate output
    out_edge_and_speed = edge_and_speed.copy()
    out_edge_and_speed['t'] = edge_and_speed['t'] + delta_x
    out_edge_and_speed['distance_to_obs'] = None

    # Check if reached intersection
    if edge_and_speed['alpha'] == 1:
        alpha_dash = 0
        out_edge_and_speed[['u', 'v', 'key', 'geometry']]\
            = sample_next_edge(graph_edges, edge_and_speed[['u', 'v', 'key', 'geometry']])
        out_edge_and_speed['speed'] = sample_conditional_speed(edge_and_speed['speed'])
    else:
        alpha_dash = edge_and_speed['alpha']

    # Propagate alpha
    out_edge_and_speed['alpha'] = min(1,
                                      alpha_dash
                                      + delta_x * edge_and_speed['speed'] / edge_and_speed['geometry'].length)

    return out_edge_and_speed


def sample_x_y0y1(graph_edges, y_0_y_1, delta_x, delta_y):
    """
    Sample x_[t0:t1] given first two observations.
    Samples x_t0, then propagates to t1 and returns path only if x_t1 is within dist.retain of y_t1.
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param y_0_y_1: first two observations (list of arrays)
    :param delta_x: x time discretisation
    :param delta_y: observation time intervals
    :return: gdf for a single sample of x[t0:t1] (truncated to ball around y_t1)
    """
    hit_ball = False
    while not hit_ball:
        # Sample x_0
        xv_df = tools.sampling.sample_x0(graph_edges, y_0_y_1[0], 1)
        xv_df.insert(loc=0, column='t', value=[0])
        xv_df['speed'] = sample_marginal_speed()

        i = 0
        while xv_df['t'][i] < delta_y:
            xv_df = xv_df.append(propagate_x(graph_edges, xv_df.iloc[i], delta_x)).reset_index(drop=True)
            i += 1

        xv_obs_time = xv_df.iloc[-1]
        xv_xy = tools.edges.edge_interpolate(xv_obs_time['geometry'], xv_obs_time['alpha'])
        xv_dist_obs = ox.euclidean_dist_vec(y_0_y_1[1][1], y_0_y_1[1][0], xv_xy[1], xv_xy[0])

        if xv_dist_obs < tools.edges.dist_retain:
            xv_df.at[xv_df.shape[0]-1, 'distance_to_obs'] = xv_dist_obs
            hit_ball = True

    return xv_df


def sample_xt_xtmin1_single(graph_edges, xv_particles, weights, y_ti, delta_x, delta_y):
    """
    Sample a particle from previous observation time according to given weights, extend path to new observation.
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param xv_particles: list of gdf particles from previous obsrvation time
    :param weights: weights at previous observation time
    :param y_ti: new observation
    :param delta_x: x time discretisation
    :param delta_y: observation time intervals
    :return: single gdf sampled extended path to new observation (within ball of new observation)
    """
    N_particles = len(weights)

    hit_ball = False
    while not hit_ball:
        # Sample x_[t0:ti-1]
        sampled_index = np.random.choice(N_particles, p=weights)

        xv_df = xv_particles[sampled_index]

        i = i_stat = xv_df.shape[0] - 1
        while xv_df['t'][i] < (xv_df['t'][i_stat] + delta_y):
            xv_df = xv_df.append(propagate_x(graph_edges, xv_df.iloc[i], delta_x)).reset_index(drop=True)
            i += 1

        xv_obs_time = xv_df.iloc[-1]
        xv_xy = tools.edges.edge_interpolate(xv_obs_time['geometry'], xv_obs_time['alpha'])
        xv_dist_obs = ox.euclidean_dist_vec(y_ti[1], y_ti[0], xv_xy[1], xv_xy[0])

        if xv_dist_obs < tools.edges.dist_retain:
            xv_df.at[xv_df.shape[0]-1, 'distance_to_obs'] = xv_dist_obs
            hit_ball = True

    return xv_df


def sample_xt_xtmin1_multi(graph_edges, xv_particles, weights, y_ti, delta_x, delta_y):
    """
    Iterate sample_xt_xtmin1_single to produce N samples
    :param graph_edges:
    :param xv_particles:
    :param weights:
    :param y_ti:
    :param delta_x:
    :param delta_y:
    :return: list of N gdfs all appended with extended path
    """
    return [sample_xt_xtmin1_single(graph_edges, xv_particles, weights, y_ti, delta_x, delta_y)
            for _ in range(len(weights))]


def weight_particles(gdf_list):
    """
    Given list of particles return Gaussian weight on how close they are to most recent observation.
    Assumes bottom entry of distance_to_obs column is non-empty for each particles gdf.
    :param gdf_list: list of gdfs - each one a particle
    :return: np.array weights
    """
    weights = np.array([np.exp(-0.5 / tools.edges.sigma2_GPS * df['distance_to_obs'].to_list()[-1])**2
                          for df in gdf_list])
    weights /= np.sum(weights)
    return weights


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
    N_samps = 10

    # Observation time increment (s)
    delta_obs = 15

    # Between observation discretisation
    # Number of inference times per observation
    N_time_dis = 5
    delta_x = delta_obs/N_time_dis

    # Sample particles from q((x_t0:t1)|(y_t0, y_t1))
    xv_particles = [sample_x_y0y1(edges_gdf, poly_single[:2], delta_x, delta_obs) for i in range(N_samps)]

    # Weight particles
    weights = [np.ones(N_samps) / N_samps]
    weights += [weight_particles(xv_particles)]

    for m in range(2, M_obs):
        xv_particles = sample_xt_xtmin1_multi(edges_gdf, xv_particles, weights[-1], poly_single[m], delta_x, delta_obs)
        weights += [weight_particles(xv_particles)]


    # Plot
    tools.edges.plot_graph_with_weighted_points(graph, poly_single, pd.concat(xv_particles))
    # tools.edges.plot_graph_with_weighted_points(graph, poly_single, xv_particles[0])

    plt.show(block=True)
