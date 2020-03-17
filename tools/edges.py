################################################################################
# Module: edges.py
# Description: Some tools including interpolation along a proportion of a
#              given edge, selecting edges within a distance of a point and
#              discretisation of an edge for sampling.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

from functools import lru_cache

import numpy as np
import osmnx as ox
from shapely.geometry import Point
from shapely.geometry import LineString

from tools.graph import plot_graph


def edge_interpolate(geometry, alpha):
    """
    Given edge and proportion travelled, return (x,y) coordinate.
    :param geometry: edge geometry
    :param alpha: proportion of edge travelled
    :return: coordinate
    """
    length_arb = geometry.length
    coord = np.array(geometry.interpolate(alpha * length_arb))
    return coord


def get_geometry(graph, edge):
    """
    Extract geometry of an edge from global graph object. If geometry doesn't exist set to straight line.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param edge: list_like, length = 3
        elements u, v, k
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
    :return: NetowrkX geometry object
    """
    edge_tuple = tuple(int(e) for e in edge)

    out_geom = get_geometry_cached(graph, edge_tuple)

    return out_geom


@lru_cache(maxsize=2**8)
def get_geometry_cached(graph, edge_tuple):
    """
    Cacheable
    Extract geometry of an edge from global graph object. If geometry doesn't exist set to straight line.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param edge_tuple: tuple (hashable for lru_cache), length = 3
        elements u, v, k
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
    :return: NetowrkX geometry object
    """

    # Extract edge data, in particular the geometry
    edge_data = graph.get_edge_data(edge_tuple[0], edge_tuple[1], edge_tuple[2])

    # If no geometry attribute, manually add straight line
    if 'geometry' in edge_data:
        edge_geom = edge_data['geometry']
    else:
        point_u = Point((graph.nodes[edge_tuple[0]]['x'], graph.nodes[edge_tuple[0]]['y']))
        point_v = Point((graph.nodes[edge_tuple[1]]['x'], graph.nodes[edge_tuple[1]]['y']))
        edge_geom = LineString([point_u, point_v])

    return edge_geom


def discretise_edge(graph, edge, d_refine, observation=None, gps_sd=None):
    """
    Discretises edge to given edge refinement parameter.
    Will also return observation likelihood if received.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param edge_tuple: list-like, length = 3
        elements u, v, k
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
    :param d_refine: float
        metres
        resolution of distance discretisation
        increase of speed, decrease for accuracy
    :param observation: numpy.ndarray, shape = (2,)
        UTM projection
        coordinate of first observation
    :param gps_sd: float
        metres
        standard deviation of GPS noise
    :return: numpy.ndarray, shape = (_, 2 or 3)
        columns
            alpha: float in (0,1], position along edge
            distance: float, distance from start of edge
            likelihood: float, unnormalised Gaussian likelihood of observation (only if observeration and gps_sd given)
    """
    edge_tuple = tuple(int(e) for e in edge)
    if observation is not None:
        observation = tuple(float(o) for o in observation)
    return discretise_edge_cached(graph, edge_tuple, d_refine, observation, gps_sd).copy()


@lru_cache(maxsize=2**8)
def discretise_edge_cached(graph, edge_tuple, d_refine, observation=None, gps_sd=None):
    """
    Cacheable
    Discretises edge to given edge refinement parameter.
    Will also return observation likelihood if received.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param edge_tuple: tuple (hashable for lru_cache), length = 3
        elements u, v, k
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
    :param d_refine: float
        metres
        resolution of distance discretisation
        increase of speed, decrease for accuracy
    :param observation: tuple, shape = (2,)
        UTM projection
        coordinate of first observation
    :param gps_sd: float
        metres
        standard deviation of GPS noise
    :return: numpy.ndarray, shape = (_, 2 or 3)
        columns
            alpha: float in (0,1], position along edge
            distance: float, distance from start of edge
            likelihood: float, unnormalised Gaussian likelihood of observation (only if observeration and gps_sd given)
    """
    if observation is None and gps_sd is not None:
        raise ValueError("Received gps_sd but not observation")

    if observation is not None and gps_sd is None:
        raise ValueError("Received observation but not gps_sd")

    edge_geom = get_geometry(graph, edge_tuple)

    edge_length = edge_geom.length

    distances = np.arange(edge_length, d_refine/10, -d_refine)

    n_distances = len(distances)

    out_mat = np.zeros((n_distances, 2)) if observation is None else np.zeros((len(distances), 3))

    out_mat[:, 0] = distances / edge_length
    out_mat[:, 1] = distances

    if observation is not None:
        cart_coords = np.zeros((n_distances, 2))
        for i in range(n_distances):
            cart_coords[i] = edge_geom.interpolate(distances[i])

        out_mat[:, 2] = np.exp(-0.5 / gps_sd**2 * np.sum((observation - cart_coords)**2, axis=1))

    return out_mat


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


def discretise_edge_alphas(geom, edge_refinement, return_dists=False):
    """
    Given edge return, series of [edge, alpha] points at determined discretisation increments along edge.
    alpha is proportion of edge traversed.
    :param geom: edge geometry
    :param return_dists: bool
    :return: list of alphas at each discretisation point
    """
    ds = np.arange(geom.length, edge_refinement/10, -edge_refinement)
    alphas = ds / geom.length
    if return_dists:
        return alphas, ds
    else:
        return alphas


def get_truncated_discrete_edges(graph, coord, edge_refinement, dist_retain):
    """
    Discretises edges within dist_retain of coord
    :param graph: simplified graph
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
    close_edges['alpha'] = close_edges['geometry'].apply(discretise_edge_alphas, edge_refinement=edge_refinement)

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


def observation_time_rows(path):
    """
    Returns rows of path only at observation times (not intersections)
    :param path: numpy.ndarray, shape=(_, 5+)
        columns - (t), u, v, k, alpha, ...
    :return: trimmed path
        numpy.ndarray, shape like path
    """
    return path[path[:, 4] != 1]


def plot_particles(graph, particles, polyline=None, alpha=None):
    """
    Plots particle approximation of trajectory
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param particles: MMParticles object (from inference.particles)
        particle approximation
    :param polyline: list-like, each element length 2
        UTM - metres
        series of GPS coordinate observations
    :param alpha: float in [0, 1]
        plotting parameter
        opacity of routes
    :return: fig, ax
    """
    fig, ax = plot_graph(graph, polyline=polyline)

    if isinstance(particles, np.ndarray):
        particles = [particles]

    alpha_min = 0.3

    if alpha is None:
        alpha = 1 / len(particles) * (1 - alpha_min) + alpha_min

    xlim = [None, None]
    ylim = [None, None]

    for i, particle in enumerate(particles):

        if len(particle) > 1:
            int_path = interpolate_path(graph, particle, t_column=True)

            cart_int_path = cartesianise_path(graph, int_path, t_column=True)
            ax.plot(cart_int_path[:, 0], cart_int_path[:, 1], color='orange', linewidth=1.5,
                    alpha=alpha)

            cart_path = cartesianise_path(graph, observation_time_rows(particle), t_column=True)
            ax.scatter(cart_path[:, 0], cart_path[:, 1], color='orange', alpha=alpha, zorder=2)
        else:
            cart_path = cartesianise_path(graph, particle, t_column=True)
            ax.scatter(cart_path[:, 0], cart_path[:, 1], color='orange', alpha=alpha)

        xlim[0] = np.min(cart_path[:, 0]) if xlim[0] is None else min(np.min(cart_path[:, 0]), xlim[0])
        xlim[1] = np.max(cart_path[:, 0]) if xlim[1] is None else max(np.max(cart_path[:, 0]), xlim[1])
        ylim[0] = np.min(cart_path[:, 1]) if ylim[0] is None else min(np.min(cart_path[:, 1]), ylim[0])
        ylim[1] = np.max(cart_path[:, 1]) if ylim[1] is None else max(np.max(cart_path[:, 1]), ylim[1])

    expand_coef = 0.1

    x_range = max(xlim[1] - xlim[0], 200)
    xlim[0] -= x_range * expand_coef
    xlim[1] += x_range * expand_coef

    y_range = max(ylim[1] - ylim[0], 200)
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
