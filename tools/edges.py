################################################################################
# Module: edges.py
# Description: Some tools including interpolation along a proportion of a
#              given edge, selecting edges within a distance of a point and
#              discretisation of an edge for sampling.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import numpy as np
import osmnx as ox
from tools.graph import plot_graph
from shapely.geometry import Point
from shapely.geometry import LineString

# Set road discretisation distance in metres
increment_dist = 3

# GPS noise variance (isotropic)
sigma2_GPS = 7 ** 2


def edge_interpolate(geometry, alpha):
    """
    Given edge and proportion travelled, return (x,y) coordinate.
    :param geometry: edge geometry
    :param alpha: proportion of edge travelled
    :return: coordinate
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


def get_geometry(graph, edge_array):
    """
    Extract geometry of an edge from global graph object. If geometry doesn't exist set to straight line.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param edge_array: list-like, length = 3
        elements u, v, k
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
    :return: NetowrkX geometry object
    """

    # Extract edge data, in particular the geometry
    edge_data = graph.get_edge_data(edge_array[0], edge_array[1], edge_array[2])

    # If no geometry attribute, manually add straight line
    if 'geometry' in edge_data:
        edge_geom = edge_data['geometry']
    else:
        point_u = Point((graph.nodes[edge_array[0]]['x'], graph.nodes[edge_array[0]]['y']))
        point_v = Point((graph.nodes[edge_array[1]]['x'], graph.nodes[edge_array[1]]['y']))
        edge_geom = LineString([point_u, point_v])

    return edge_geom


def interpolate_path(graph, path, d_refine=1, t_column=False):
    """
    Turns path into a discrete collection of positions to be plotted
    :param path: numpy.ndarray, shape = (_, 4)
    :param d_refine: float
        metres
        resolution of distance discretisation
    :param t_column: boolean
        boolean describing if input has a first column for the time variable
    :return: numpy.ndarray, shape = (_, 6)
        elongated array for plotting path
    """
    start_col = 1 * t_column
    out_arr = path[:1].copy()
    prev_point = out_arr[0]
    for point in path[1:]:
        edge_geom = get_geometry(graph, point[start_col:(start_col + 3)])
        edge_length = edge_geom.length
        if np.array_equal(point[start_col:(start_col + 3)], prev_point[start_col:(start_col + 3)]):
            edge_metres = np.arange(prev_point[start_col + 3] * edge_length
                                    + d_refine, point[start_col + 3]*edge_length, d_refine)
        else:
            edge_metres = np.arange(0, point[start_col + 3]*edge_length, d_refine)
        edge_alphas = edge_metres / edge_length
        append_arr = np.zeros((len(edge_alphas), out_arr.shape[1]))
        append_arr[:, start_col:(start_col + 3)] = point[start_col:(start_col + 3)]
        append_arr[:, start_col + 3] = edge_alphas
        out_arr = np.append(out_arr, append_arr, axis=0)
        prev_point = point
    return out_arr


def cartesianise_path(graph, path, t_column=False):
    """
    Converts particle or array of edges and alphas into cartesian points.
    :param path: numpy.ndarray, shape=(_, 5+)
        columns - (t), u, v, k, alpha, ...
    :param t_column: boolean
        boolean describing if input has a first column for the time variable
    :return: numpy.ndarray, shape = (_, 2)
        cartesian points
    """
    start_col = 1*t_column

    cart_points = np.zeros(shape=(path.shape[0], 2))

    for i, point in enumerate(path):
        edge_geom = get_geometry(graph, point[start_col:(3+start_col)])
        cart_points[i, :] = edge_interpolate(edge_geom, point[3+start_col])

    return cart_points


def plot_particles(graph, particles, polyline=None, weights=None):

    fig, ax = plot_graph(graph, polyline=polyline)

    xlim = [None, None]
    ylim = [None, None]

    for i, particle in enumerate(particles):
        path = interpolate_path(graph, particle, t_column=True)

        cart_path = cartesianise_path(graph, path, t_column=True)

        xlim[0] = np.min(cart_path[:, 0]) if xlim[0] is None else min(np.min(cart_path[:, 0]), xlim[0])
        xlim[1] = np.max(cart_path[:, 0]) if xlim[1] is None else max(np.max(cart_path[:, 0]), xlim[1])
        ylim[0] = np.min(cart_path[:, 1]) if ylim[0] is None else min(np.min(cart_path[:, 1]), ylim[0])
        ylim[1] = np.min(cart_path[:, 1]) if ylim[1] is None else max(np.max(cart_path[:, 1]), ylim[1])

        ax.plot(cart_path[:, 0], cart_path[:, 1], color='orange', linewidth=5,
                alpha=1 if weights is None else weights[i])

    expand_coef = 0.1

    x_range = xlim[1] - xlim[0]
    xlim[0] -= x_range * expand_coef
    xlim[1] += x_range * expand_coef

    y_range = ylim[1] - ylim[0]
    ylim[0] -= y_range * expand_coef
    ylim[1] += y_range * expand_coef

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    return fig, ax









#
#
# if __name__ == '__main__':
#     # Source data paths
#     _, process_data_path = data.utils.source_data()
#
#     # Load networkx graph
#     graph = load_graph()
#
#     # Load taxi data
#     data_path = data.utils.choose_data()
#     raw_data = data.utils.read_data(data_path, 100).get_chunk()
#
#     # Select single polyline
#     single_index = 0
#     poly_single = raw_data['POLYLINE_UTM'][single_index]
#
#     # Discretise edges close to start point of polyline
#     dis_edges = get_truncated_discrete_edges(graph, poly_single[0], 1)
#
#     # Plot
#     fig, ax = plot_graph_with_weighted_points(graph, poly_single, points=dis_edges)
#     truncate_circle = plt.Circle(tuple(poly_single[0]), dist_retain, color='orange', fill=False)
#     ax.add_patch(truncate_circle)
#     plt.show()
