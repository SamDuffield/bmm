################################################################################
# Module: sampling.py
# Description: Sample initial coordinates.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import data.utils
import tools.edges
from tools.graph import load_graph
import data.preprocess


def gaussian_weights(points, y_obs):
    """
    Calculates (normalised) weights, p(y|x).
    :param points: list of [edge, alpha] points (edge = [u,v,k,geom])
    :param y_obs: [x,y] observed coordinate
    :return: np.array, normalised weights
    """
    points_xy = tools.edges.cartesianise(points)

    un_weights = np.exp(-0.5 / tools.edges.sigma2_GPS * np.sum((points_xy - y_obs)**2, axis=1))

    return un_weights / sum(un_weights)


def sample_x0(graph_edges, y_0, N_sample):
    """
    Samples from Gaussian centred around y0, constrained to the road network.
    :param graph_edges: simplified graph edges, gdf
    :param y_0: observation point
    :param N_sample: number of samples
    :return: gdf of edges, some may well be duplicates
    """

    # Discretize nearby edges
    dis_points = tools.edges.get_truncated_discrete_edges(graph_edges, y_0)

    # Calculate likelihood weights
    weights = gaussian_weights(dis_points, y_0)

    # Sample indices according to weights
    sampled_indices = np.random.choice(len(weights), N_sample, True, weights)

    # Sampled points
    sampled_points = dis_points.iloc[sampled_indices].copy()

    return sampled_points.reset_index(drop=True)


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

    # Sample size
    N_samps = 100

    # Sample x0|y0
    x0_samples = sample_x0(edges_gdf, poly_single[0], N_samps)

    # Plot initial samples
    tools.edges.plot_graph_with_weighted_points(graph, poly_single, x0_samples)
    plt.show(block=True)
