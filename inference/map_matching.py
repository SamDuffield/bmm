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
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd

# Relative probability of u-turn at intersection
u_turn_downscale = 0.2

# Speed prior hyperparameters (all links a priori iid)
v_mu = 9
v_sigma = 5
v_max = v_mu * 2.5



def sample_TGaus_speed(mu, sigma, vmax, size=1):
    """
    Sample a speed from univariate Gaussian truncated to [0, vmax].
    All quantities in m/s.
    :param mu: mean of (pre-truncated) Gaussian
    :param sigma: standard deviation of (pre-truncated Gaussian)
    :param vmax: maximum speed (upper truncation)
    :return: float (or array if size>1), speed in m/s.
    """
    sample = truncnorm.rvs(a=-mu/sigma, b=(vmax-mu)/sigma, loc=mu, scale=sigma, size=size)

    if len(sample) == 1:
        return sample.item()
    else:
        return sample


def sample_speed():
    """
    Samples a vehicle speed from prior. Assumes all speeds iid across links.
    Hyperparameters defined inside function.
    :return: speed, float
    """
    return sample_TGaus_speed(v_mu, v_sigma, v_max)


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
    edge_length = edge_and_speed['geometry'].length

    alpha_metre = edge_length * edge_and_speed['alpha']

    alpha_metre_prop = alpha_metre + edge_and_speed['speed']*delta_x

    alpha_prop = alpha_metre_prop / edge_length

    out_edge_and_speed = edge_and_speed.copy()
    out_edge_and_speed['t'] = edge_and_speed['t'] + delta_x

    if alpha_prop < 1:
        out_edge_and_speed['alpha'] = alpha_prop
        out_edge_and_speed['distance_to_obs'] = None
    else:
        out_edge_and_speed[['u', 'v', 'key', 'geometry']]\
            = sample_next_edge(graph_edges, edge_and_speed[['u', 'v', 'key', 'geometry']])
        out_edge_and_speed['alpha'] = 0
        out_edge_and_speed['distance_to_obs'] = None
        out_edge_and_speed['speed'] = sample_speed()

    return out_edge_and_speed


def sample_x_y0y1(graph_edges, y_0_y_1, delta_x, delta_y):

    hit_ball = False
    while not hit_ball:
        # Sample x_0
        xv_df = tools.sampling.sample_x0(graph_edges, y_0_y_1[0], 1)
        xv_df.insert(loc=0, column='t', value=[0])
        xv_df['speed'] = sample_speed()

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

    # Plot
    tools.edges.plot_graph_with_weighted_points(graph, poly_single, pd.concat(xv_particles))
    # tools.edges.plot_graph_with_weighted_points(graph, poly_single, xv_particles[1])

    # Weight particles
    weights = [np.ones(N_samps) / N_samps]
    weights += [np.array([np.exp(-0.5 / tools.edges.sigma2_GPS * df['distance_to_obs'].to_list()[-1])**2
                          for df in xv_particles])]
    weights[1] /= sum(weights[1])




    plt.show(block=True)
