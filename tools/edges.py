################################################################################
# Module: edges.py
# Description: Some tools including interpolation along a proportion of a
#              given edge, selecting edges within a distance of a point and
#              discretisation of an edge for sampling.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
import data.utils
from tools.graph import load_graph, plot_graph, get_bbox_from_graph

# Set road discretisation distance in metres
increment_dist = 3

# GPS noise variance (isotropic)
sigma2_GPS = 20 ** 2

# Cut off distance from a point
dist_retain = np.sqrt(sigma2_GPS) * 3


def edge_interpolate(edge, alpha):
    """
    Given edge and proportion travelled, return lat-long.
    :param edge: [u, v, k, geometry]
    :param alpha: proportion of edge travelled
    :return: [lat, long]
    """
    length_arb = edge[3].length
    coord = np.asarray(edge[3].interpolate(alpha * length_arb))
    return coord


def get_edges_within_dist(graph, coord, dist):
    """
    Given a point returns all edges that fall within a radius of dist.
    :param graph: simplified graph
    :param coord: central point
    :param dist: radius
    :return: list of edges, each element a [u,v,k, geometry] object
    """
    gdf = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    graph_edges = gdf[["geometry", "u", "v", "key"]].values.tolist()

    edges_with_distances = [
        (
            graph_edge,
            ox.Point(tuple(coord)).distance(graph_edge[0])
        )
        for graph_edge in graph_edges
    ]

    edges_with_distances = sorted(edges_with_distances, key=lambda x: x[1])

    edges_within_dist = [[u, v, k, geometry] for [[geometry, u, v, k], d] in edges_with_distances if d < dist]

    return edges_within_dist


def discretise_edge(edge):
    """
    Given edge return, series of [long, lat] points
    at determined discretisation increments along edge
    :param edge: [u, v, k, geometry]
    :return: list of [edge, alpha] at each discretisation point
    """

    ds = np.arange(increment_dist/2, edge[3].length, increment_dist)
    alphas = ds / edge[3].length

    return [[edge, a] for a in alphas]


def get_truncated_discrete_edges(graph, coord, sigma2_GPS):
    """
    Samples N possible positions on graph for a given GPS ping
    :param graph: simplified graph
    :param coord: conformal with graph (i.e. UTM)
    :param sigma2_GPS: isotropic variance of GPS noise (in metres if UTM)
    :return: list [edge, alpha] of unique edge [u, v, k, geometry] (order of u,v dictates direction)
                and alpha in [0,1] indicating proportion along edge from u to v
    """

    close_edges = get_edges_within_dist(graph, coord, dist_retain)

    discritised_edges = []

    for edge in close_edges:
        discrete_edge = discretise_edge(edge)
        truncate_discrete_edge = [edge_a for edge_a in discrete_edge
                                  if np.linalg.norm(edge_interpolate(edge_a[0], edge_a[1]) - coord) < dist_retain]
        discritised_edges += truncate_discrete_edge

    return discritised_edges


if __name__ == '__main__':
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Load networkx graph
    graph = load_graph()

    # Load taxi data
    data_path = data.utils.choose_data()
    raw_data = data.utils.read_data(data_path, 100).get_chunk()

    # Get UTM bbox
    bbox_utm = get_bbox_from_graph(graph)

    # Select single polyline
    single_index = 4
    poly_single = raw_data['POLYLINE_UTM'][single_index]

    # Discretise edges close to start point of polyline
    dis_edges = get_truncated_discrete_edges(graph, poly_single[0], sigma2_GPS)

    # Coords of discretised edges
    dis_edge_coords = np.asarray([edge_interpolate(edge, alpha) for edge, alpha in dis_edges])

    # Plot
    fig, ax = plot_graph(graph, poly_single)
    ax.scatter(dis_edge_coords[:, 0], dis_edge_coords[:, 1], c='blue')
    truncate_circle = plt.Circle(tuple(poly_single[0]), np.sqrt(sigma2_GPS) * 3, color='orange', fill=False)
    ax.add_patch(truncate_circle)
    plt.show(block=True)
