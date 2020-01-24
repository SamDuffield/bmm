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
from scipy.special import gamma as gamma_func
from shapely.geometry import Point
from shapely.geometry import LineString

# Prior mean for distance
dist_prior_mean = 108

# Prior variance for distance
dist_prior_variance = 10700

# Resampling ESS threshold
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


def pdf_gamma_mv(vals, mean, var):
    """
    Evaluates Gamma pdf with parameters adjusted to give an inputted mean and variance.
    :param vals: np.array, values to be evaluated
    :param mean: float, inputted distribution mean
    :param var: float, inputted distribution variance
    :return: np.array, same length as vals, Gamma pdf evaulations
    """
    gamma_beta = mean / var
    gamma_alpha = mean * gamma_beta

    if any(np.atleast_1d(vals) <= 0):
        raise ValueError("Gamma pdf takes only positive values")

    return gamma_beta ** gamma_beta / gamma_func(gamma_alpha) * vals ** (gamma_alpha - 1) * np.exp(-gamma_beta * vals)


def distance_prior(distance):
    """
    Evaluates prior probability of distance, assumes time interval of 15 seconds.
    :param distance: float or np.array, values to be evaluated
    :return: float or np.array same length as distance, prior pdf evaluations
    """
    return pdf_gamma_mv(distance, dist_prior_mean, dist_prior_variance)


def sample_x0(y_0, n_sample, edge_refinement):
    """
    Samples from truncated Gaussian centred around y0, constrained to the road network.
    :param y_0: np.array, length 2, observation point (cartesian)
    :param n_sample: int. number of samples
    :param edge_refinement: float, discretisation increment of edges (metres)
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
    dis_points = tools.edges.get_truncated_discrete_edges(graph_edges, y_0, edge_refinement)

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


def optimal_particle_filter(polyline, n_samps, delta_y, d_refine, d_max):
    """
    Runs optimal particle filter for sampling vehicle paths given noisy observations.
    This particle filter is expensive but serves as the gold standard in terms of accuracy for a given sample size.
    :param polyline: np.array, shape=(M, 2)
        M observations, cartesian coordinates
    :param n_samps: int
        number of samples to output
    :param delta_y: float
        inter-observation time, assumed constant
    :param d_refine: float
        discretisation increment of edges (metres)
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
    xd_particles = sample_x0(polyline[0], n_samps, d_refine)

    # Set initial weights
    weights = np.zeros((M, n_samps))
    weights[0, :] = 1/n_samps

    # Initiate ESS
    ess = np.zeros(M)
    ess[0] = n_samps

    # Discretise d
    d_discrete = np.arange(0.001, d_max, d_refine)

    print("Assimilated observation {}/{} with ESS of {:0.2f} ({} samples)".format(1, M, ess[0], n_samps))

    for m in range(1, M):
        # Store particles (to be resampled)
        old_particles = xd_particles.copy()
        xd_particles = []
        for j in range(n_samps):
            # Resample if ESS below threshold
            if ess[m-1] < n_samps * ess_resample_threshold:
                sample_index = np.random.choice(n_samps, 1, True, weights[m - 1, :])[0]
                temp_weight = 1 / n_samps
            else:
                sample_index = j
                temp_weight = weights[m - 1, j]
            old_particle = old_particles[sample_index].copy()

            # Get all possible routes (no t column)
            possible_routes = get_all_possible_routes(old_particle[-1:, 1:].copy(), d_max)

            dis_routes_list = []
            for i, route in enumerate(possible_routes):
                # Minimum distance travelled to be in route
                route_d_min = 0 if route.shape[0] == 1 else route[-2, -1]

                # Maximum distance travelled to be in route
                route_d_max = route[-1, -1]

                # Possible discrete distance for route
                route_ds = d_discrete[(route_d_min <= d_discrete) & (d_discrete < route_d_max)]

                if len(route_ds) > 0:
                    # Matrix with columns:
                    # route_index
                    # alpha
                    # distance travelled (in observation time)
                    # product of 1 / n_i where n_i is the number of possible edges at each intersection i traversed
                    # squared distance to new observation

                    # Initiatilisation, route index and distance
                    dis_route_matrix = np.zeros((len(route_ds), 5))
                    dis_route_matrix[:, 0] = i
                    dis_route_matrix[:, 2] = route_ds

                    # Product of 1 / number of intersection choices
                    intersection_col = route[:, -2]
                    dis_route_matrix[:, 3] = np.prod(1 / intersection_col[intersection_col > 1])

                    # Get last edge geometry
                    last_edge_geom = get_geometry(route[-1])

                    # Convert distances to alphas
                    dis_route_matrix[:, 1] = alphas = \
                        route_ds / last_edge_geom.length + old_particle[-1, 4]\
                        if route.shape[0] == 1 else \
                        (route_ds - route_d_min) / last_edge_geom.length

                    # Cartesianise points
                    dis_route_cart_points = np.array([tools.edges.edge_interpolate(last_edge_geom, alpha)
                                                      for alpha in alphas])

                    # Squared distance to observation
                    dis_route_matrix[:, 4] = np.sum((dis_route_cart_points - polyline[m]) ** 2, axis=1)

                    dis_routes_list += [dis_route_matrix]

            # Concatenate list into large array
            dis_routes = np.concatenate(dis_routes_list)

            # Calculate probabilities
            sample_probs = distance_prior(dis_routes[:, 2])\
                * dis_routes[:, 3]\
                * np.exp(- 0.5 / tools.edges.sigma2_GPS * dis_routes[:, 4])

            # Normalising constant = p(y_m | x_m-1^j)
            sample_probs_norm_const = sum(sample_probs)

            # Sample an edge and distance
            sampled_dis_route_index = np.random.choice(len(dis_routes), 1, p=sample_probs/sample_probs_norm_const)[0]
            sampled_dis_route = dis_routes[sampled_dis_route_index]

            # Append sampled route to old particle
            sampled_route = possible_routes[int(sampled_dis_route[0])]
            new_route_append = np.zeros((len(sampled_route), 7))
            new_route_append[-1, 0] = old_particle[-1, 0] + delta_y
            new_route_append[:, 1:] = sampled_route
            new_route_append[-1, 4] = sampled_dis_route[1]
            new_route_append[-1, 6] = sampled_dis_route[2]

            xd_particles += [np.append(old_particle, new_route_append, axis=0)]

            # Calculate weight (unnormalised)
            weights[m, j] = sample_probs_norm_const * temp_weight

        # Normalise weights
        weights[m, :] /= sum(weights[m, :])

        # Update ESS
        ess[m] = 1 / sum(weights[m, :] ** 2)

        print("Assimilated observation {}/{} with ESS of {:0.2f} ({} samples)".format(m+1, M, ess[m], n_samps))

    return xd_particles, weights


def first_occurence(twod_array, oned_array):
    for i, obj in enumerate(twod_array):
        if np.all(obj == oned_array):
            return i
    raise ValueError("Couldn't find occurence")


def fixed_lag_resample_all(particles, current_weights, lag):

    observation_times_full = particles[0][:, 0]
    observation_times = observation_times_full[(observation_times_full != 0)
                                               | (np.arange(len(observation_times_full)) == 0)]
    m = len(observation_times)
    n_samps = len(particles)

    # Standard resampling if not reached lag yet
    if m <= lag:
        resampled_indices = np.random.choice(n_samps, n_samps, replace=True, p=current_weights)
        return [particles[i] for i in resampled_indices]

    max_fixed_time = observation_times[max(m - lag - 1, 0)]
    max_fixed_time_next = observation_times[max(m - lag, 0)]

    fixed_particles = []
    newer_particles = []
    out_particles = []
    max_fix_next_indices = []
    for particle in particles:
        max_fix_index = np.where(particle[:, 0] == max_fixed_time)[0][0]
        max_fix_next_indices += [(np.where(particle[:, 0] == max_fixed_time_next)[0][0]) - max_fix_index]
        fixed_particles += [particle[:(max_fix_index + 1), :].copy()]
        newer_particles += [particle[max_fix_index:, :].copy()]

    for i in range(n_samps):
        resample_prob = np.zeros(n_samps)
        fixed_last_edge = fixed_particles[i][-1, 1:4]
        fixed_last_edge_geom = get_geometry(fixed_last_edge)
        fixed_edge_length = fixed_last_edge_geom.length

        newer_particles_adjusted = []

        for j in range(n_samps):
            if j == i or np.all(fixed_particles[i][-1, :-1] == newer_particles[j][0, :-1]):
                newer_particles_adjusted += [newer_particles[j][1:]]
                resample_prob[j] = current_weights[j]\
                    * distance_prior(newer_particles[j][max_fix_next_indices[j], -1])
            else:
                other_particle_edges_until_observation = newer_particles[j][:(max_fix_next_indices[j] + 1), 1:4]
                if fixed_last_edge.tolist() in other_particle_edges_until_observation.tolist():

                    # Check if other route doesn't overtake fixed position
                    if np.array_equal(fixed_last_edge, other_particle_edges_until_observation[-1]) and \
                            (newer_particles[j][max_fix_next_indices[j], 4] < fixed_particles[i][-1, 4]):
                        newer_particles_adjusted += [None]
                        continue

                    # First occurence of fixed edge on other particle
                    first_occur_edge_other_particle_index = first_occurence(other_particle_edges_until_observation,
                                                                            fixed_last_edge)

                    # Intersection (at end of common edge)
                    if newer_particles[j][first_occur_edge_other_particle_index, 4] == 1:
                        newer_particles_adjusted += [newer_particles[j][first_occur_edge_other_particle_index:].copy()]
                        next_obs_index = max_fix_next_indices[j] - first_occur_edge_other_particle_index
                        newer_particles_adjusted[j][:(next_obs_index + 1), -1] += \
                            (1 - fixed_particles[i][-1, 4]) * fixed_edge_length\
                            - newer_particles[j][first_occur_edge_other_particle_index, -1]

                        resample_prob[j] = current_weights[j] * distance_prior(newer_particles_adjusted[j][next_obs_index, -1])

                    # Other particle finishes (at next observation time) on fixed edge
                    elif np.array_equal(fixed_last_edge, other_particle_edges_until_observation[-1]):
                        newer_particles_adjusted += [np.atleast_2d(newer_particles[j][max_fix_next_indices[j]].copy())]
                        newer_particles_adjusted[j][0, -1] = (newer_particles_adjusted[j][0, 4]
                                                              - fixed_particles[i][-1, 4]) * fixed_edge_length

                        resample_prob[j] = current_weights[j] * distance_prior(newer_particles_adjusted[j][0, -1])

                    # Other particle starts on fixed edge and ends on a different edge
                    else:
                        newer_particles_adjusted += [newer_particles[j][(first_occur_edge_other_particle_index + 1):].copy()]
                        next_obs_index = max_fix_next_indices[j] - first_occur_edge_other_particle_index - 1
                        newer_particles_adjusted[j][:(next_obs_index + 1), -1] += (newer_particles[j][first_occur_edge_other_particle_index, 4]
                                                                                            - fixed_particles[i][-1, 4]) * fixed_edge_length

                        resample_prob[j] = current_weights[j] * distance_prior(
                            newer_particles_adjusted[j][next_obs_index, -1])

                else:
                    newer_particles_adjusted += [None]

        # If fixed edge isn't shared by any other particles, do full resampling
        if all([a is None for a in newer_particles_adjusted]) or sum(resample_prob) == 0:
            resample_index = np.random.choice(n_samps, 1, p=current_weights)[0]
            out_particle = particles[resample_index]
        # Otherwise sampled a new adjusted particle and append to fixed
        else:
            # Normalise
            resample_prob /= sum(resample_prob)

            # Resample
            resample_index = np.random.choice(n_samps, 1, p=resample_prob)[0]

            # Append new particle to fixed
            out_particle = np.append(fixed_particles[i], newer_particles_adjusted[resample_index], axis=0)

        out_particles += [out_particle]

    return out_particles


def optimal_particle_filter_fixed_lag(polyline, n_samps, delta_y, d_refine, d_max, lag):
    """
    Runs optimal particle filter for sampling vehicle paths given noisy observations.
    This particle filter is expensive but serves as the gold standard in terms of accuracy for a given sample size.
    :param polyline: np.array, shape=(M, 2)
        M observations, cartesian coordinates
    :param n_samps: int
        number of samples to output
    :param delta_y: float
        inter-observation time, assumed constant
    :param d_refine: float
        discretisation increment of edges (metres)
    :param d_max: float
        maximum distance possible to travel in delta_y
    :param lag: int
        lag before which to stop resampling
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
    xd_particles = sample_x0(polyline[0], n_samps, d_refine)

    # Set initial weights
    weights = np.zeros((M, n_samps))
    weights[0, :] = 1/n_samps

    # Initiate ESS
    ess = np.zeros(M)
    ess[0] = n_samps

    # Discretise d
    d_discrete = np.arange(0.01, d_max, d_refine)

    print("Assimilated observation {}/{} with ESS of {:0.2f} ({} samples)".format(1, M, ess[0], n_samps))

    for m in range(1, M):
        old_particles = xd_particles.copy()
        xd_particles = []
        for j in range(n_samps):
            old_particle = old_particles[j].copy()

            # Get all possible routes (no t column)
            start_point = old_particle[-1:, 1:].copy()
            start_point[0, -1] = 0
            possible_routes = get_all_possible_routes(start_point, d_max)

            dis_routes_list = []
            for i, route in enumerate(possible_routes):

                # Minimum distance travelled to be in route
                route_d_min = 0 if route.shape[0] == 1 else route[-2, -1]

                # Maximum distance travelled to be in route
                route_d_max = route[-1, -1]

                # Possible discrete distance for route
                route_ds = d_discrete[(route_d_min <= d_discrete) & (d_discrete < route_d_max)]

                if len(route_ds) > 0:
                    # Matrix with columns:
                    # route_index
                    # alpha
                    # distance travelled (in observation time)
                    # product of 1 / n_i where n_i is the number of possible edges at each intersection i traversed
                    # squared distance to new observation

                    # Initiatilisation, route index and distance
                    dis_route_matrix = np.zeros((len(route_ds), 5))
                    dis_route_matrix[:, 0] = i
                    dis_route_matrix[:, 2] = route_ds

                    # Product of 1 / number of intersection choices
                    intersection_col = route[:, -2]
                    dis_route_matrix[:, 3] = np.prod(1 / intersection_col[intersection_col > 1])

                    # Get last edge geometry
                    last_edge_geom = get_geometry(route[-1])

                    # Convert distances to alphas
                    dis_route_matrix[:, 1] = \
                        route_ds / last_edge_geom.length + old_particle[-1, 4]\
                        if route.shape[0] == 1 else \
                        (route_ds - route_d_min) / last_edge_geom.length

                    # Cartesianise points
                    dis_route_cart_points = np.array([tools.edges.edge_interpolate(last_edge_geom, alpha)
                                                      for alpha in dis_route_matrix[:, 1]])

                    # Squared distance to observation
                    dis_route_matrix[:, 4] = np.sum((dis_route_cart_points - polyline[m]) ** 2, axis=1)

                    dis_routes_list += [dis_route_matrix.copy()]

            # Concatenate list into large array
            dis_routes = np.concatenate(dis_routes_list)

            # Calculate probabilities
            sample_probs = distance_prior(dis_routes[:, 2])\
                * dis_routes[:, 3]\
                * np.exp(- 0.5 / tools.edges.sigma2_GPS * dis_routes[:, 4])

            # Normalising constant = p(y_m | x_m-1^j)
            sample_probs_norm_const = sum(sample_probs)

            # Check if path diverged so much all probabilities are 0
            if sample_probs_norm_const == 0:
                # In this case just set arbitrarily and it should get resampled away
                sampled_dis_route_index = 0
            # Otherwise sample accordingly
            else:
                sampled_dis_route_index = np.random.choice(len(dis_routes), 1, p=sample_probs/sample_probs_norm_const)[0]

            sampled_dis_route = dis_routes[sampled_dis_route_index]

            # Append sampled route to old particle
            sampled_route = possible_routes[int(sampled_dis_route[0])]
            new_route_append = np.zeros((len(sampled_route), 7))
            new_route_append[-1, 0] = old_particle[-1, 0] + delta_y
            new_route_append[:, 1:] = sampled_route
            new_route_append[-1, 4] = sampled_dis_route[1]
            new_route_append[-1, 6] = sampled_dis_route[2]

            xd_particles += [np.append(old_particle, new_route_append, axis=0)]

            # Calculate weight (unnormalised)
            weights[m, j] = sample_probs_norm_const


        # Normalise weights
        weights[m] /= sum(weights[m])

        # Update ESS
        ess[m] = 1 / sum(weights[m, :] ** 2)

        # Fixed lag resample
        xd_particles = fixed_lag_resample_all(xd_particles, weights[m], lag)

        print("Assimilated observation {}/{} with ESS of {:0.2f} ({} samples)".format(m+1, M, ess[m], n_samps))

    return xd_particles, weights


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
    # single_index = np.random.choice(100, 1)[0]
    single_index = 0
    poly_single_list = raw_data['POLYLINE_UTM'][single_index]
    poly_single = np.asarray(poly_single_list)

    # Edge refinement
    edge_refinement_dist = 1

    # Sample size
    N_samps = 10

    # Observation time increment (s)
    delta_obs = 15

    # Max distance (in observation time)
    distance_max = 500

    # # Run optimal particle filter
    # particles, weights = optimal_particle_filter(poly_single[:5, :], N_samps,
    #                                              delta_obs, edge_refinement_dist, distance_max)
    # ess = np.array([1 / sum(w ** 2) for w in weights])
    #
    # # Plot
    # plot_particles(particles, poly_single, weights=weights[-1, :])\

    # Fixed lag
    fixed_lag = 1

    # Run optimal particle filter with fixed lag
    particles, weights = optimal_particle_filter_fixed_lag(poly_single[:20], N_samps,
                                                 delta_obs, edge_refinement_dist, distance_max, fixed_lag)
    ess = np.array([1 / sum(w ** 2) for w in weights])

    alpha_max = [max(p[:, 4]) for p in particles]

    # Plot
    plot_particles(particles, poly_single, weights=weights[-1, :])

    plt.show(block=True)
