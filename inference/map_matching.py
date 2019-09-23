################################################################################
# Module: map_matching.py
# Description: Infer route taken by vehicle given sparse observations.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import numpy as np
import data
from tools.graph import load_graph, plot_graph
import tools.edges
import tools.sampling
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import osmnx as ox


def sample_TGaus_speed(mu, sigma, vmax, size=1):
    """
    Sample a speed from univariate Gaussian truncated to [0,vmax].
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
    # Speed prior hyperparameters (all links a priori iid)
    v_mu = 9
    v_sigma = 5
    v_max = v_mu*2.5
    return sample_TGaus_speed(v_mu, v_sigma, v_max)



def plot_TGaus(mu, sigma, vmax):
    """
    Plots Gaussian truncated to [0, vmax]
    :param mu: mean of (pre-truncated) Gaussian
    :param sigma: standard deviation of (pre-truncated Gaussian)
    :param vmax: upper truncation
    :return: fig, ax
    """
    fig, ax = plt.subplots(1, 1)

    rv = truncnorm(-mu/sigma, (vmax-mu)/sigma, scale=sigma)

    x = np.linspace(-mu/sigma, (vmax-mu)/sigma, 100)
    ax.plot(x*sigma + mu, rv.pdf(x))

    return fig, ax


def sample_next_edge(graph_edges, prev_edge):
    """
    Given an edge sample an edge for a vehicle to travel on next (after intersection).
    Assumes equally likely to pass onto all adjacent edges at intersection
    (except u_turn which is downscaled but not necessarily 0)
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param prev_edge: edge just traversed, [u, v, k, geometry]
    :return: sampled new edge, [u, v, k, geometry]
    """
    adj_v_edges = [edge for edge in graph_edges if edge[0] == prev_edge[1]]

    n_adj = len(adj_v_edges)

    u_turn_downscale = 0.2

    weights = np.ones(n_adj) / n_adj

    u_turn_possible = False
    for i, edge in enumerate(adj_v_edges):
        if edge[1] == prev_edge[0]:
            u_turn_possible = True
            weights[i] *= u_turn_downscale

    if u_turn_possible:
        weights /= sum(weights)

    sampled_index = np.random.choice(n_adj, p=weights)

    return adj_v_edges[sampled_index]


def propagate_x(graph_edges, edge_and_speed, delta_x):
    """
    Increment vehicle forward one time step (discretisation time step not observation)
    If propagation reaches the end of the edge (alpha = 1) sample a new edge to traverse to and new speed for that edge.
    :param graph_edges: simplified graph converted to edge list (with tools.edges.graph_edges_extract)
    :param edge_and_speed: [edge, alpha, speed]
    :param delta_x: time discretisation (not observation time interval!)
    :return: [propagated edge, propagated_alpha, propagated_speed]
    """
    edge, alpha, speed = edge_and_speed

    edge_length = edge[3].length

    alpha_metre = edge_length * alpha

    alpha_metre_prop = alpha_metre + speed*delta_x

    alpha_prop = alpha_metre_prop / edge_length

    if alpha_prop < 1:
        return [edge, alpha_prop, speed]
    else:
        next_edge = sample_next_edge(graph_edges, edge)
        new_speed = sample_speed()
        return [next_edge, 0., new_speed]


def propagate_particles(graph_edges, particles_xv, obs_y, delta_x, delta_y):

    prop_particles = []

    n = len(particles_xv)

    old_weights = [particle[1] for particle in particles_xv]

    sum_weights = 0
    for i in range(n):
        hit_ball = False
        while not hit_ball:
            sampled_old_xv_ind = np.random.choice(n, p=old_weights)
            xv = particles_xv[sampled_old_xv_ind][0]
            t = 0
            while t < delta_y:
                xv = propagate_x(graph_edges, xv, delta_x)
                t += delta_x

            xv_xy = tools.edges.edge_interpolate(xv[0], xv[1])

            xv_dist_obs = ox.euclidean_dist_vec(obs_y[1], obs_y[0], xv_xy[1], xv_xy[0])

            if xv_dist_obs < tools.edges.dist_retain:
                weight = np.exp(-0.5 / tools.edges.sigma2_GPS * xv_dist_obs ** 2)
                sum_weights += weight
                prop_particles += [[xv, weight]]
                hit_ball = True

    prop_particles = [[e, w/sum_weights] for e, w in prop_particles]

    return prop_particles


def plot_graph_with_weighted_samples(graph, polyline=None, samples=None, speeds=True):
    """
    Wrapper for plot_graph. Adds weighted sampled points to graph.
    :param graph: road network
    :param polyline: observed coordinates
    :param samples: sampled points
    :return: fig, ax of plotted road network (plus polyline and samples)
    """

    # Initiate graph
    fig, ax = plot_graph(graph, polyline)

    if samples is not None:
        if speeds:
            samples = [[xv[:-1], w] for xv, w in samples]

        weights = np.array([w for xv, w in samples])
        samples = [xv for xv, w in samples]

        # Extract xy coordinates of samples
        points_xy = np.asarray([tools.edges.edge_interpolate(edge, alpha) for edge, alpha in samples])

        # Min opacity
        opa_min = 0.4
        alphas = opa_min + (1-opa_min)*weights
        rgba_colors = np.zeros((len(weights), 4))
        rgba_colors[:, 0] = 1.0
        rgba_colors[:, 1] = 0.6
        rgba_colors[:, 3] = alphas



        ax.scatter(points_xy[:, 0], points_xy[:, 1], c=rgba_colors)

    return fig, ax


if __name__ == '__main__':
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Load networkx graph
    graph = load_graph()
    graph_edges = tools.edges.graph_edges_extract(graph)

    # Load small taxi data set (i.e. only 15 minutes)
    data_path = data.utils.choose_data()
    raw_data = data.utils.read_data(data_path)

    # Select single polyline
    single_index = 0
    poly_single = raw_data['POLYLINE_UTM'][single_index]

    # Number of observations
    M_obs = len(poly_single)

    # Sample size
    N_samps = 100

    # Observation time increment (s)
    delta_obs = 15

    # Between observation discretisation
    # Number of inference times per observation
    N_time_dis = 5
    delta_x = delta_obs/N_time_dis

    # Initiate sample storage, preallocate?
    xv_samples = []

    # Sample x_t0|y_t0
    xv_samples += [tools.sampling.sample_x0(graph_edges, poly_single[0], N_samps)]

    # Sample initial speeds and set initial weights
    xv_samples[0] = [[[edge, alpha, sample_speed()], 1/N_samps]
                     for edge, alpha in xv_samples[0]]

    # Plot initial
    plot_graph_with_weighted_samples(graph, poly_single, xv_samples[0])

    # Propagate edges and reweight
    xv_samples += [propagate_particles(graph_edges, xv_samples[0], poly_single[1], delta_x, delta_obs)]

    # Plot first propagation
    plot_graph_with_weighted_samples(graph, poly_single, xv_samples[1])
    plt.show(block=True)




