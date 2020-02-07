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
from tools.graph import load_graph, plot_graph
import geopandas as gpd

# Set road discretisation distance in metres
increment_dist = 3

# GPS noise variance (isotropic)
sigma2_GPS = 7 ** 2


def edge_interpolate(geometry, alpha):
    """
    Given edge and proportion travelled, return lat-long.
    :param edge: edge geometry
    :param alpha: proportion of edge travelled
    :return: [lat, long]
    """
    length_arb = geometry.length
    coord = np.asarray(geometry.interpolate(alpha * length_arb))
    return coord


def graph_edges_gdf(graph):
    """
    Converts networkx graph to geopandas data frame and then returns geopandas dataframe. (Fast!)
    :param graph: networkx object
    :return: list of edges, [u, v, k, geometry]
    """
    gdf = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    edge_gdf = gdf[["u", "v", "key", "geometry"]]
    return edge_gdf


def get_edges_within_dist(graph_edges, coord, dist):
    """
    Given a point returns all edges that fall within a radius of dist.
    :param graph_edges: simplified graph edges (either gdf or list)
    :param coord: central point
    :param dist: radius
    :return: geopandas dataframe (gdf) of edges, columns: u, v, k, geometry, dist from coord
    """

    graph_edges_dist = graph_edges.copy()

    graph_edges_dist['distance_to_obs'] = graph_edges['geometry'].apply(
        lambda geom: ox.Point(tuple(coord)).distance(geom))

    edges_within_dist = graph_edges_dist[graph_edges_dist['distance_to_obs'] < dist]

    return edges_within_dist


def discretise_edge(geom, edge_refinement):
    """
    Given edge return, series of [edge, alpha] points at determined discretisation increments along edge.
    alpha is proportion of edge traversed.
    :param edge: [u, v, k, geometry]
    :param edge_refinement: float, discretisation increment of edges (metres)
    :return: list of [edge, alpha] at each discretisation point
    """
    ds = np.arange(increment_dist/2, geom.length, edge_refinement)
    alphas = ds / geom.length
    return alphas


def get_truncated_discrete_edges(graph, coord, edge_refinement, dist_retain):
    """
    Discretises edges within dist_retain of coord
    :param graph_edges: simplified graph edges, gdf
    :param coord: conformal with graph (i.e. UTM)
    :param edge_refinement: float, discretisation increment of edges (metres)
    :return: numpy.ndarray, shape = (number of points within truncation, 5)
        columns: u, v, k, alpha, distance_to_coord
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
            alpha: in [0,1], position along edge
            distance_to_coord: metres, distance to input coord
    """
    # Extract geodataframe
    graph_edges = graph_edges_gdf(graph)

    # Remove edges with closest point outside truncation
    close_edges = get_edges_within_dist(graph_edges, coord, dist_retain)

    # Discretise edges
    close_edges['alpha'] = close_edges['geometry'].apply(discretise_edge, edge_refinement=edge_refinement)

    # Remove distance from closest point on edge column
    close_edges = close_edges.drop(columns='distance_to_obs')

    # Elongate, remove points outside truncation and store in list of lists
    discretised_edges = []
    for _, row in close_edges.iterrows():
        for a in row['alpha']:
            xy = edge_interpolate(row['geometry'], a)
            dist = ox.euclidean_dist_vec(coord[1], coord[0], xy[1], xy[0])
            if dist < dist_retain:
                add_row = row.copy()
                add_row['alpha'] = a
                add_row['distance_to_obs'] = dist
                discretised_edges += [[row['u'], row['v'], row['key'], a, dist]]

    # Convert to numpy.ndarray
    discretised_edges = np.array(discretised_edges)

    return discretised_edges


def cartesianise(points):
    """
    Converts a gdf of edge, alpha coordinates into a numpy array of xy coordinates
    :param points: points to plot (gdf with geometry and alpha columns)
    :return: numpy array, of xy coordinates
    """
    points_xy = points.apply(lambda row: edge_interpolate(row['geometry'], row['alpha']), axis=1)
    points_xy = np.asarray(points_xy.to_list())
    return points_xy


def gaussian_weights(points, y_obs):
    """
    Calculates (normalised) weights, p(y|x).
    :param points: list of [edge, alpha] points (edge = [u,v,k,geom])
    :param y_obs: [x,y] observed coordinate
    :return: np.array, normalised weights
    """
    points_xy = cartesianise(points)

    un_weights = np.exp(-0.5 / sigma2_GPS * np.sum((points_xy - y_obs)**2, axis=1))

    return un_weights / sum(un_weights)


def plot_graph_with_weighted_points(graph, polyline=None, points=None, weights=None):
    """
    Wrapper for plot_graph. Adds weighted sampled points to graph.
    :param graph: road network
    :param polyline: observed coordinates
    :param points: points to plot (gdf with geometry and alpha columns)
    :param weights: weights for points
    :return: fig, ax of plotted road network (plus polyline and samples)
    """

    # Initiate graph
    fig, ax = plot_graph(graph, polyline)

    if points is not None:
        # Extract xy coordinates of samples
        points_xy = cartesianise(points)

        # Set orange colour
        rgba_colors = np.zeros((len(points), 4))
        rgba_colors[:, 0] = 1.0
        rgba_colors[:, 1] = 0.6

        # Weighted opacity (if inputted)
        if weights is None:
            n = points_xy.shape[0]
            weights = np.ones(n) / n

        # Min opacity
        opa_min = 0.2

        alphas = opa_min + (1 - opa_min) * weights
        rgba_colors[:, 3] = alphas

        # Highlight points at observation time
        not_at_obs = points['alpha'] == 1

        # Add points to plot
        ax.scatter(points_xy[:, 0], points_xy[:, 1], c=rgba_colors, linewidths=[0.5 if a else 3 for a in not_at_obs],
                   zorder=2)

    return fig, ax


if __name__ == '__main__':
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Load networkx graph
    graph = load_graph()

    # Load taxi data
    data_path = data.utils.choose_data()
    raw_data = data.utils.read_data(data_path, 100).get_chunk()

    # Select single polyline
    single_index = 0
    poly_single = raw_data['POLYLINE_UTM'][single_index]

    # Discretise edges close to start point of polyline
    dis_edges = get_truncated_discrete_edges(graph, poly_single[0], 1)

    # Plot
    fig, ax = plot_graph_with_weighted_points(graph, poly_single, points=dis_edges)
    truncate_circle = plt.Circle(tuple(poly_single[0]), dist_retain, color='orange', fill=False)
    ax.add_patch(truncate_circle)
    plt.show()
