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

# Cut off distance from a point
dist_retain = np.sqrt(sigma2_GPS) * 3


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


def get_edges_within_dist(graph_edges, coord, dist=dist_retain):
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


def discretise_edge(geom):
    """
    Given edge return, series of [edge, alpha] points at determined discretisation increments along edge.
    alpha is proportion of edge traversed.
    :param edge: [u, v, k, geometry]
    :return: list of [edge, alpha] at each discretisation point
    """
    ds = np.arange(increment_dist/2, geom.length, increment_dist)
    alphas = ds / geom.length
    return alphas


def get_truncated_discrete_edges(graph_edges, coord):
    """
    Discretises edges within dist_retain of coord
    :param graph_edges: simplified graph edges, gdf
    :param coord: conformal with graph (i.e. UTM)
    :return: list [edge, alpha] of edge [u, v, k, geometry] (order of u,v dictates direction)
                and alpha in [0,1] indicating proportion along edge from u to v
    """

    close_edges = get_edges_within_dist(graph_edges, coord, dist_retain)
    close_edges['alpha'] = close_edges['geometry'].apply(discretise_edge)
    close_edges = close_edges.drop(columns='distance_to_obs')

    # Elongate dataframe and remove points outside truncation
    discretised_edges = []
    for _, row in close_edges.iterrows():
        for a in row['alpha']:
            xy = edge_interpolate(row['geometry'], a)
            dist = ox.euclidean_dist_vec(coord[1], coord[0], xy[1], xy[0])
            if dist < dist_retain:
                add_row = row.copy()
                add_row['alpha'] = a
                add_row['distance_to_obs'] = dist
                discretised_edges.append(add_row)

    discretised_edges = gpd.GeoDataFrame(discretised_edges, crs=close_edges.crs)

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

        ax.scatter(points_xy[:, 0], points_xy[:, 1], c=rgba_colors)

    return fig, ax


if __name__ == '__main__':
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Load networkx graph and edges gdf
    graph = load_graph()
    edges_gdf = graph_edges_gdf(graph)

    # Load taxi data
    data_path = data.utils.choose_data()
    raw_data = data.utils.read_data(data_path, 100).get_chunk()

    # Select single polyline
    single_index = 0
    poly_single = raw_data['POLYLINE_UTM'][single_index]

    # Discretise edges close to start point of polyline
    dis_edges = get_truncated_discrete_edges(edges_gdf, poly_single[0])

    # Plot
    fig, ax = plot_graph_with_weighted_points(graph, poly_single, points=dis_edges)
    truncate_circle = plt.Circle(tuple(poly_single[3]), dist_retain, color='orange', fill=False)
    ax.add_patch(truncate_circle)
    plt.show(block=True)
