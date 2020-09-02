########################################################################################################################
# Module: edges.py
# Description: Some tools including interpolation along a proportion of a given edge, selecting edges within a distance
#              of a point and discretisation of an edge for sampling.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

from functools import lru_cache
from typing import Union, Tuple

import numpy as np
import osmnx as ox
from shapely.geometry import Point
from shapely.geometry import LineString
from networkx.classes import MultiDiGraph
from geopandas import GeoDataFrame


def edge_interpolate(geometry: LineString,
                     alpha: float) -> np.ndarray:
    """
    Given edge and proportion travelled, return (x,y) coordinate.
    :param geometry: edge geometry
    :param alpha: in (0,1] proportion of edge travelled
    :return: cartesian coordinate
    """
    return np.array(geometry.interpolate(alpha, normalized=True))


def get_geometry(graph: MultiDiGraph,
                 edge: np.ndarray) -> LineString:
    """
    Extract geometry of an edge from global cam_graph object. If geometry doesn't exist set to straight line.
    :param graph: encodes road network, simplified and projected to UTM
    :param edge: length = 3
        elements u, v, k
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
    :return: edge geometry
    """
    edge_tuple = tuple(int(e) for e in edge)
    return get_geometry_cached(graph, edge_tuple)


@lru_cache(maxsize=2 ** 8)
def get_geometry_cached(graph: MultiDiGraph,
                        edge_tuple: tuple) -> LineString:
    """
    Cacheable
    Extract geometry of an edge from global cam_graph object. If geometry doesn't exist set to straight line.
    :param graph: encodes road network, simplified and projected to UTM
    :param edge_tuple: (hashable for lru_cache), length = 3
        elements u, v, k
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
    :return: edge geometry
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


def discretise_geometry(geom: LineString,
                        d_refine: float,
                        return_dists: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Given edge, return series of [edge, alpha] points at determined discretisation increments along edge.
    alpha is proportion of edge traversed.
    :param geom: edge geometry
    :param d_refine: metres, resolution of distance discretisation
    :param return_dists: if true return distance along edge as well as alpha (proportion)
    :return: list of alphas at each discretisation point
    """
    ds = np.arange(geom.length, d_refine / 10, -d_refine)
    alphas = ds / geom.length
    return (alphas, ds) if return_dists else alphas


def discretise_edge(graph: MultiDiGraph,
                    edge: np.ndarray,
                    d_refine: float) -> np.ndarray:
    """
    Discretises edge to given edge refinement parameter.
    Returns array of proportions along edge, xy cartesian coordinates and distances along edge
    :param graph: encodes road network, simplified and projected to UTM
    :param edge: list-like, length = 3
        elements u, v, k
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
    :param d_refine: metres, resolution of distance discretisation
    :return: shape = (_, 4)
        columns
            alpha: float in (0,1], position along edge
            x: float, metres, cartesian x coordinate
            y: float, metres, cartesian y coordinate
            distance: float, distance from start of edge
    """
    edge_tuple = tuple(int(e) for e in edge)
    return discretise_edge_cached(graph, edge_tuple, d_refine).copy()


@lru_cache(maxsize=2 ** 8)
def discretise_edge_cached(graph: MultiDiGraph,
                           edge_tuple: tuple,
                           d_refine: float) -> np.ndarray:
    """
    Cacheable
    Discretises edge to given edge refinement parameter.
    Returns array of proportions along edge, xy cartesian coordinates and distances along edge
    :param graph: encodes road network, simplified and projected to UTM
    :param edge_tuple: tuple (hashable for lru_cache), length = 3
        elements u, v, k
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
    :param d_refine: metres, resolution of distance discretisation
    :return: shape = (_, 4)
        columns
            alpha: float in (0,1], position along edge
            x: float, metres, cartesian x coordinate
            y: float, metres, cartesian y coordinate
            distance: float, distance from start of edge
    """

    edge_geom = get_geometry_cached(graph, edge_tuple)

    alphas, distances = discretise_geometry(edge_geom, d_refine, True)

    n_distances = len(distances)

    out_mat = np.zeros((n_distances, 4))

    out_mat[:, 0] = alphas
    out_mat[:, 3] = distances

    for i in range(n_distances):
        out_mat[i, 1:3] = edge_geom.interpolate(distances[i])

    return out_mat


def graph_edges_gdf(graph: MultiDiGraph) -> GeoDataFrame:
    """
    Converts networkx cam_graph to geopandas data frame and then returns geopandas dataframe. (Fast!)
    :param graph: encodes road network, simplified and projected to UTM
    :return: gdf of edges with columns [u, v, k, geometry]
    """
    gdf = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
    edge_gdf = gdf[["u", "v", "key", "geometry"]]
    return edge_gdf


def get_edges_within_dist(graph_edges: GeoDataFrame,
                          coord: np.ndarray,
                          dist_retain: float) -> GeoDataFrame:
    """
    Given a point returns all edges that fall within a radius of dist.
    :param graph_edges: gdf of edges with columns [u, v, k, geometry]
    :param coord: central point
    :param dist_retain: metres, retain radius
    :return: gdf of edges with columns [u, v, k, geometry, distance_to_obs]
        all with distance_to_obs < dist_retain
    """

    graph_edges_dist = graph_edges.copy()

    graph_edges_dist['distance_to_obs'] = graph_edges['geometry'].apply(
        lambda geom: Point(tuple(coord)).distance(geom))

    edges_within_dist = graph_edges_dist[graph_edges_dist['distance_to_obs'] < dist_retain]

    return edges_within_dist


def get_truncated_discrete_edges(graph: MultiDiGraph,
                                 coord: np.ndarray,
                                 d_refine: float,
                                 d_truncate: float,
                                 return_dists_to_coord: bool = False) \
        -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Discretises edges within dist_retain of coord
    :param graph: encodes road network, simplified and projected to UTM
    :param coord: conformal with cam_graph (i.e. UTM)
    :param d_refine: metres, resolution of distance discretisation
    :param d_truncate: metres, distance within which of coord to retain points
    :param return_dists_to_coord: if true additionally return array of distances to coord
    :return: numpy.ndarray, shape = (number of points within truncation, 6)
        columns: u, v, k, alpha, distance_to_coord
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
            alpha: in [0,1], position along edge
            x: float, metres, cartesian x coordinate
            y: float, metres, cartesian y coordinate
        if return_dists_to_coord also return np.ndarray, shape = (number of points within truncation,)
            with distance of each point to coord
    """

    # Extract geodataframe
    graph_edges = graph_edges_gdf(graph)

    # Remove edges with closest point outside truncation
    close_edges = get_edges_within_dist(graph_edges, coord, d_truncate)

    # Discretise edges
    close_edges['alpha'] = close_edges['geometry'].apply(discretise_geometry, d_refine=d_refine)

    # Remove distance from closest point on edge column
    # (as this refers to closest point of edge and now we want specific point on edge)
    close_edges = close_edges.drop(columns='distance_to_obs')

    # Elongate, remove points outside truncation and store in list of lists
    points = []
    dists = []
    for _, row in close_edges.iterrows():
        for a in row['alpha']:
            xy = edge_interpolate(row['geometry'], a)
            dist = np.sqrt(np.square(coord - xy).sum())
            if dist < d_truncate:
                add_row = row.copy()
                add_row['alpha'] = a
                add_row['distance_to_obs'] = dist
                points += [[row['u'], row['v'], row['key'], a, *xy]]
                dists += [dist]

    # Convert to numpy.ndarray
    points_arr = np.array(points)
    dists_arr = np.array(dists)

    return (points_arr, dists_arr) if return_dists_to_coord else points_arr


def observation_time_indices(times: np.ndarray) -> np.ndarray:
    """
    Remove zeros (other than the initial zero) from a series
    :param times: series of timestamps
    :return: bool array of timestamps that are either non-zero or the first timestamp
    """
    return np.logical_or(times != 0, np.arange(len(times)) == 0.)


def observation_time_rows(path: np.ndarray) -> np.ndarray:
    """
    Returns rows of path only at observation times (not intersections)
    :param path: numpy.ndarray, shape=(_, 5+)
        columns - t, u, v, k, alpha, ...
    :return: trimmed path
        numpy.ndarray, shape like path
    """
    return path[observation_time_indices(path[:, 0])]


def long_lat_to_utm(points: Union[list, np.ndarray], graph=None) -> np.ndarray:
    """
    Converts a collection of long-lat points to UTM
    :param points: points to be projected, shape = (N, 2)
    :param graph: optional cam_graph containing desired crs in cam_graph.gra['crs']
    :return: array of projected points
    """
    points = np.atleast_2d(points)
    points_gdf = GeoDataFrame({'index': np.arange(len(points)),
                               'x': points[:, 0],
                               'y': points[:, 1]})
    points_gdf['geometry'] = points_gdf.apply(lambda row: Point(row['x'], row['y']), axis=1)
    points_gdf.crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' # long lat crs
    points_gdf_utm = ox.projection.project_gdf(points_gdf, to_crs=str(graph.graph['crs']) if graph is not None else None)
    points_gdf_utm['x'] = points_gdf_utm['geometry'].map(lambda point: point.x)
    points_gdf_utm['y'] = points_gdf_utm['geometry'].map(lambda point: point.y)
    return np.squeeze(np.array(points_gdf_utm[['x', 'y']]))









def interpolate_path(graph: MultiDiGraph,
                     path: np.ndarray,
                     d_refine: float = 1,
                     t_column: bool = False) -> np.ndarray:
    """
    Turns path into a discrete collection of positions to be plotted
    :param graph: simplified cam_graph
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
                                    + d_refine, point[start_col + 3] * edge_length, d_refine)
        else:
            edge_metres = np.arange(0, point[start_col + 3] * edge_length, d_refine)
        edge_alphas = edge_metres / edge_length
        append_arr = np.zeros((len(edge_alphas), out_arr.shape[1]))
        append_arr[:, start_col:(start_col + 3)] = point[start_col:(start_col + 3)]
        append_arr[:, start_col + 3] = edge_alphas
        out_arr = np.append(out_arr, append_arr, axis=0)
        prev_point = point
    return out_arr


def cartesianise_path(graph, path, t_column=True, observation_time_only=False):
    """
    Converts particle or array of edges and alphas into cartesian points.
    :param path: numpy.ndarray, shape=(_, 5+)
        columns - (t), u, v, k, alpha, ...
    :param t_column: boolean
        boolean describing if input has a first column for the time variable
    :return: numpy.ndarray, shape = (_, 2)
        cartesian points
    """
    start_col = 1 * t_column

    if observation_time_only:
        path = observation_time_rows(path)

    cart_points = np.zeros(shape=(path.shape[0], 2))

    for i, point in enumerate(path):
        edge_geom = get_geometry(graph, point[start_col:(3 + start_col)])
        cart_points[i, :] = edge_interpolate(edge_geom, point[3 + start_col])

    return np.atleast_2d(cart_points)
