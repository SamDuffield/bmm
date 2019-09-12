################################################################################
# Module: sampling.py
# Description: Sample initial coordinate. Given average speeds across all links
#              sample trajectories.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import data.utils
import tools.edges
from tools.graph import load_graph, plot_graph
import data.preprocess


def gaussian_weights(points, obs_point):
    """
    Calculates (normalised) weights, p(y|x).
    :param points: list of [edge, alpha] points (edge = [u,v,k,geom])
    :param obs_point: [x,y] observed coordinate
    :return: np.array, normalised weights
    """
    points_xy = [tools.edges.edge_interpolate(edge, alpha) for edge, alpha in points]

    un_weights = np.array([np.exp(-0.5 / tools.edges.sigma2_GPS * sum((coords - obs_point) ** 2))
                           for coords in points_xy])

    return un_weights / sum(un_weights)


def sample_x0(graph, y0, N_sample):
    """
    Samples from Gaussian centred around y0, constrained to the road network.
    :param graph: defines road network
    :param y0: observation point
    :param N_sample: number of samples
    :return: list of [edge, alpha] (edge = [u,v,k,geom]). some may well be duplicates
    """

    # Discretize nearby edges
    dis_points = tools.edges.get_truncated_discrete_edges(graph, y0, tools.edges.sigma2_GPS)

    # Calculate likelihood weights
    weights = gaussian_weights(dis_points, y0)

    # Sample indices according to weights
    sampled_indices = np.random.choice(len(weights), N_sample, True, weights)

    # Sampled points
    sampled_points = [dis_points[i] for i in sampled_indices]

    return sampled_points


def propagate_particle(edge, alpha, av_speeds, t_end):

    t_current = 0











def plot_graph_with_samples(graph, polyline=None, samples=None):
    """
    Wrapper for plot_graph. Adds sampled points to graph.
    :param graph: road network
    :param polyline: observed coordinates
    :param samples: sampled points
    :return: fig, ax of plotted road network (plus polyline and samples)
    """
    # Initiate graph
    fig, ax = plot_graph(graph, polyline)

    if samples is not None:
        # Extract xy coordinates of samples
        points_xy = np.asarray([tools.edges.edge_interpolate(edge, alpha) for edge, alpha in samples])

        # Min opacity
        opa_min = 0.4

        ax.scatter(points_xy[:, 0], points_xy[:, 1], c='orange', alpha=(opa_min + (1-opa_min)*1/len(samples)))

    return fig, ax


if __name__ == '__main__':
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Load networkx graph
    graph = load_graph()

    # Load small taxi data set (i.e. only 15 minutes)
    data_path = data.utils.choose_data()
    raw_data = data.utils.read_data(data_path)

    # Select single polyline
    single_index = 0
    poly_single = raw_data['POLYLINE_UTM'][single_index]

    # Sample size
    N_samps = 100

    # Sample x0|y0
    x0_samples = sample_x0(graph, poly_single[0], N_samps)

    # Plot initial samples
    plot_graph_with_samples(graph, poly_single, x0_samples)
    plt.show(block=True)





