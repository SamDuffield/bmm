################################################################################
# Module: inference/optimal_pf.py
# Description: Infer route taken by vehicle given sparse observations
#              through the use of SMC/particle filtering with optimal proposal.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import numpy as np
import data
from tools.graph import load_graph, plot_graph
import tools.edges
import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import Point
from shapely.geometry import LineString

ess_resample_threshold = 1


def get_geometry(edge_array):
    """
    Extract geometry of an edge from global graph object. If geometry doesn't exist set to straight line.
    :param edge_array: np.array or list, length 3
        u, v, k
    :return: geometry object
    """
    global graph

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


def sample_x0(y_0, n_sample):
    """
    Samples from truncated Gaussian centred around y0, constrained to the road network.
    :param y_0: np.array, length 2, observation point (cartesian)
    :param n_sample: int. number of samples
    :return: list of np.arrays
        length of list = n_sample
        array rows = one for each time - either intersection or observation time, simply 1 in this case.
        array columns = [t, u, v, k, alpha, n_inter, d]
            t = time (assuming constant speed between observations)
            u, v, k = edge
            alpha = position on edge
            n_inter = number of roads at intersection (zero if alpha != 1)
            d = distance travelled since previous observation time
        t, n_inter, d all zero in this case
    """
    global graph_edges

    # Discretize nearby edges
    dis_points = tools.edges.get_truncated_discrete_edges(graph_edges, y_0)

    # Calculate likelihood weights
    weights = np.exp(-0.5 / tools.edges.sigma2_GPS * dis_points['distance_to_obs'].to_numpy() ** 2)
    weights /= np.sum(weights)

    n_cols = 7

    # Convert to np.array with columns t, u, v, k, alpha
    dis_points_array = np.zeros((dis_points.shape[0], n_cols))
    dis_points_array[:, 1:5] = dis_points[['u', 'v', 'key', 'alpha']].to_numpy()

    # Sample indices according to weights
    sampled_indices = np.random.choice(len(weights), n_sample, True, weights)

    # Sampled points
    sampled_points = [dis_points_array[i, :].reshape(1, n_cols) for i in sampled_indices]

    return sampled_points


def get_all_possible_routes(in_route, max_distance_to_travel):
    """
    Given a route so far and maximum distance to travel, calculate and return all possible routes on graph.
    :param in_route: np.array, shape = (6)
        starting edge and position on edge
        u, v, k, alpha, n_inter, d
        Note no time parameter
    :param max_distance_to_travel: float
        maximum possible distance to travel
    :return:
        list of np.arrays, shape=(_ , 6)
        each array describes a possible route
    """

    global graph

    # Extract final position from inputted route
    start_edge_and_position = in_route[-1]

    # Extract edge geometry
    start_edge_geom = get_geometry(start_edge_and_position[:3])

    # Distance left on edge before intersection
    # Use NetworkX length rather than OSM length
    distance_left_on_edge = (1 - start_edge_and_position[3]) * start_edge_geom.length

    if distance_left_on_edge > max_distance_to_travel:
        # Remain on edge
        # Propagate and return
        start_edge_and_position[3] += max_distance_to_travel / start_edge_geom.length
        start_edge_and_position[5] += max_distance_to_travel
        return [in_route]
    else:
        # Reach intersection at end of edge
        # Propagate to intersection and recurse
        max_distance_to_travel -= distance_left_on_edge
        start_edge_and_position[3] = 1.
        start_edge_and_position[5] += distance_left_on_edge

        intersection_edges = np.atleast_2d([[u, v, k] for u, v, k in graph.out_edges(start_edge_and_position[1],
                                                                                     keys=True)])

        if intersection_edges.shape[1] == 0:
            # Dead-end and one-way
            return [in_route]

        n_inter = max(1, sum(intersection_edges[:, 1] != start_edge_and_position[0]))
        start_edge_and_position[4] = n_inter

        if len(intersection_edges) == 1 and intersection_edges[0][1] == start_edge_and_position[0]:
            # Dead-end and two-way -> Only option is u-turn
            return [in_route]
        else:
            new_routes = []
            for new_edge in intersection_edges:
                if new_edge[1] != start_edge_and_position[0]:
                    new_route = np.append(in_route,
                                          np.atleast_2d(np.append(
                                              new_edge, [0, 0, start_edge_and_position[5]]
                                          )),
                                          axis=0)

                    new_routes += get_all_possible_routes(new_route, max_distance_to_travel)

        return [in_route] + new_routes


def optimal_particle_filter(polyline, delta_y, d_max, n_samps):
    """
    Runs optimal particle filter for sampling vehicle paths given noisy observations.
    This particle filter is expensive but serves as the gold standard in terms of accuracy for a given sample size.
    :param polyline: np.array, shape=(M, 2)
        M observations, cartesian coordinates
    :param delta_y: float
        inter-observation time, assumed constant
    :param n_samps: int
        number of samples to output
    :param d_max: float
        maximum distance possible to travel in delta_y
    :return: (particles, weights)
        particles: list of np.arrays, list length = n_samps, array shape=(num_t_steps, 7)
            array columns = t, u, v, k, alpha, n_inter, d
            each array represents a sampled vehicle path
        weights: np.array, shape=(M, n_samps)
            probability assigned to each particle at each time points
    """
    global graph, graph_edges

    # Number of observations
    M = len(polyline)

    # Sample from p(x_0|y_0)
    xd_particles = sample_x0(polyline[0], n_samps)

    # Set initial weights
    log_weights = np.zeros((M, n_samps))
    log_weights[0, :] = np.log(1/n_samps)

    # Initiate ESS
    ess = np.zeros(M)
    ess[0] = n_samps

    for m in range(1, M):
        # Store particles (to be resampled)
        old_particles = xd_particles.copy()
        xd_particles = []
        for j in range(n_samps):
            # Resample if ESS below threshold
            if ess[m-1] < n_samps * ess_resample_threshold:
                sample_index = np.random.choice(n_samps, 1, True, np.exp(log_weights[m - 1, :]))[0]
                temp_weight = 1 / n_samps
            else:
                sample_index = j
                temp_weight = log_weights[m - 1, j]
            old_particle = old_particles[sample_index].copy()

            # Get all possible routes (no t column)
            possible_routes = get_all_possible_routes(old_particle[-1, 1:].copy(), d_max)

            dis_route_list = []
            for i, route in enumerate(possible_routes):
                # Get geometry of final edge
                last_edge_geom = get_geometry(route[-1])

                # Discretise edge
                last_edge_discretised_alphas = tools.edges.discretise_edge(last_edge_geom)

                # Remove backward alphas if route remains on previous edge
                if route.shape[0] == 1:
                    last_edge_discretised_alphas = \
                        last_edge_discretised_alphas[last_edge_discretised_alphas > old_particle[-1, 4]]

                # Number of discretised points on final edge
                n_discretised = len(last_edge_discretised_alphas)

                # Matrix with columns:
                # route_index, alpha, distance travelled (in observation time), squared distance to new observations
                dis_route_matrix = np.zeros((n_discretised, 4))
                dis_route_matrix[:, 0] = i
                dis_route_matrix[:, 1] = last_edge_discretised_alphas

                # Distances travelled
                base_dist = - old_particle[-1, 4] * last_edge_geom.length if route.shape[0] == 1 else route[-2, -1]
                dis_route_matrix[:, 2] = base_dist + last_edge_discretised_alphas * last_edge_geom.length

                # Cartesianise points
                dis_route_cart_points = np.array([[tools.edges.edge_interpolate(last_edge_geom, alpha)]
                                                  for alpha in last_edge_discretised_alphas])

                # Squared distance to observation
                dis_route_matrix[:, 3] = np.sum((dis_route_cart_points - polyline[m]) ** 2, axis=1)

                dis_route_list += [dis_route_matrix]

            # Concatenate sample space into single array
            dis_route_array = np.concatenate(dis_route_list)

            # Calculate log-probabilities






def cartesianise_numpy(point_np):
    """
    Converts numpy array of u, v, k, alpha into cartesian coordinate.
    :param point_np: np.array. u, v, k, alpha
    :return: np.array, length 2 (cartesian)
    """
    edge_geom = get_geometry(point_np[:3])

    return tools.edges.edge_interpolate(edge_geom, point_np[-1])


def cartesianise_path(path, intersection_indicator=False):
    """
    Converts particle stored as edge, alpha into cartesian points.
    :param path: np.array, shape=(T,7)
        columns - t, u, v, k, alpha, n_inter, d
    :param intersection_indicator: boolean
        whether to also return np.array of length T with boolean indicating if point is an intersection
    :return:
        if not intersection_indicator:
            np.array, shape=(T,2) cartesian points
        elif intersection_indicator:
            list of length 2
                np.array, shape=(T,2) cartesian points
                np.array, shape=(T) boolean, True if intersection (alpha == 1)
    """

    cart_points = np.zeros(shape=(path.shape[0], 2))

    for i, point in enumerate(path):
        cart_points[i, :] = cartesianise_numpy(point[-6:-2])

    if not intersection_indicator:
        return cart_points
    else:
        intersection_bool = np.array([point[-3] == 1 for point in path])
        return [cart_points, intersection_bool]


def plot_particles(particles, polyline=None, weights=None):
    """
    Plot paths (output from particle filter).
    :param particles: List of np.arrays representing paths
    :param polyline: np.array, shape=(M,2) for M trajectory observations
    :param weights: np.array, same length as the list particles
    :return: fig, ax, graph with polyline and sampled trajectories
    """
    # global graph

    if type(particles) is np.ndarray:
        particles = [particles]

    n_samps = len(particles)

    fig, ax = plot_graph(graph, polyline=polyline)

    min_alpha = 0.3

    xlim = [None, None]
    ylim = [None, None]

    for i, path in enumerate(particles):
        cart_path, inter_bool = cartesianise_path(path, True)

        xlim[0] = np.min(cart_path[:, 0]) if i == 0 else min(xlim[0], np.min(cart_path[:, 0]))
        xlim[1] = np.max(cart_path[:, 0]) if i == 0 else max(xlim[1], np.max(cart_path[:, 0]))
        ylim[0] = np.min(cart_path[:, 1]) if i == 0 else min(ylim[0], np.min(cart_path[:, 1]))
        ylim[1] = np.max(cart_path[:, 1]) if i == 0 else max(ylim[1], np.max(cart_path[:, 1]))

        path_weight = 1 / n_samps if weights is None else weights[i]
        alpha = min_alpha + (1 - min_alpha) * path_weight

        ax.scatter(cart_path[:, 0], cart_path[:, 1],
                   linewidths=[0.2 if inter else 3 for inter in inter_bool],
                   alpha=alpha, color='orange')

    expand_coef = 0.25

    x_range = xlim[1] - xlim[0]
    xlim[0] -= x_range * expand_coef
    xlim[1] += x_range * expand_coef

    y_range = ylim[1] - ylim[0]
    ylim[0] -= y_range * expand_coef
    ylim[1] += y_range * expand_coef

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    return fig, ax


if __name__ == '__main__':
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Load networkx graph and edges gdf
    graph = load_graph()
    graph_edges = tools.edges.graph_edges_gdf(graph)

    # Load taxi data
    data_path = data.utils.choose_data()
    raw_data = data.utils.read_data(data_path, 100).get_chunk()

    # Select single polyline
    single_index = np.random.choice(100, 1)[0]
    poly_single_list = raw_data['POLYLINE_UTM'][single_index]
    poly_single = np.asarray(poly_single_list)

    # Sample single x0 | y0
    single_x0 = np.atleast_2d(sample_x0(poly_single[0], 1)[0][0][1:])

    # Get all possible routes from single x0
    poss_routes = get_all_possible_routes(single_x0.copy(), 50)

    # Plot
    plot_particles([single_x0] + poss_routes, poly_single)

    plt.show(block=True)

