################################################################################
# Module: map_matching_mcmc.py
# Description: Infer route taken by vehicle given sparse observations.
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
from scipy.special import gammainc as gammainc_func
from shapely.geometry import Point
from shapely.geometry import LineString


# Prior mean for distance
dist_prior_mean = 108

# Prior variance for distance
dist_prior_variance = 10700

# Variance of distance given x_n-1 and y_n
# dist_cond_variance = 1000
dist_cond_variance = 100

# Resample threshold
ess_resample_threshold = 0.5


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


def cartesianise_numpy(point_np):
    """
    Converts numpy array of u, v, k, alpha into cartesian coordinate.
    :param point_np: np.array. u, v, k, alpha
    :return: np.array, length 2 (cartesian)
    """
    global graph

    edge_geom = get_geometry(point_np[:3])

    return np.asarray(tools.edges.edge_interpolate(edge_geom, point_np[-1]))


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
        cart_points[i, :] = cartesianise_numpy(point[1:5])

    if not intersection_indicator:
        return cart_points
    else:
        intersection_bool = np.array([point[4] == 1 for point in path])
        return [cart_points, intersection_bool]


def euclidean_distance(car_point_1, car_point_2, squared=False):
    """
    Gets euclidean distance between two cartesian points.
    :param car_point_1: list of floats or np.array, [x, y] of first point
    :param car_point_2: list of floats or np.array, [x, y] of second point
    :param squared: boolean, whether to return distance squared or not
    :return: float
    """
    square_dist = (car_point_1[0] - car_point_2[0]) ** 2 + (car_point_1[1] - car_point_2[1]) ** 2

    return square_dist if squared else square_dist ** 0.5


def sample_dist(mean, var, size=None):
    """
    Samples from Gamma distribution with parameters adjusted to give an inputted mean and variance.
    :param mean: float, inputted mean
    :param var: float, inputted variance
    :param size: int, number of samples
    :return: float, random sample from Gamma if size=None otherwise np.array of length size
    """
    gamma_beta = mean / var
    gamma_alpha = mean * gamma_beta

    return np.random.gamma(gamma_alpha, 1/gamma_beta, size=size)


def prob_dist(vals, mean, var):
    """
    Evaluates Gamma pdf with parameters adjusted to give an inputted mean and variance.
    :param vals: np.array, values to be evaluated
    :param mean: float, inputted distribution mean
    :param var: float, inputted distribution variance
    :return: np.array, same length as vals, Gamma pdf evaulations
    """
    gamma_beta = mean / var
    gamma_alpha = mean * gamma_beta

    return gamma_beta ** gamma_beta / gamma_func(gamma_alpha) * vals ** (gamma_alpha - 1) * np.exp(-gamma_beta * vals)


def cdf_prob_dist(d_upper, mean, var):
    """
    Evaluates Gamma cdf with parameters adjusted to give an inputted mean and variance. I.e. P(dist < d_upper)
    :param d_upper: float, upper bound
    :param mean: float, inputted distribution mean
    :param var: float, inputted distribution variance
    :return: np.array, same length as vals, Gamma pdf evaulations
    """
    gamma_beta = mean / var
    gamma_alpha = mean * gamma_beta
    return gammainc_func(gamma_alpha, gamma_beta * d_upper)


def prob_dist_prior(distance):
    """
    Evaluates prior probability of distance, assumes time interval of 15 seconds.
    :param distance: float or np.array, values to be evaluated
    :return: float or np.array same length as distance, prior pdf evaluations
    """
    return prob_dist(distance, dist_prior_mean, dist_prior_variance)


def sample_dist_given_xnmin1_yn(old_point, new_observation):
    """
    Calculates distance between old point and new observations for input to sample_dist.
    Returns sample form sample_dist
    :param old_point: np.array. u, v, k, alpha
    :param new_observation: np.array, length 2 (cartesian)
    :return: float
    """
    cartesian_old_point = cartesianise_numpy(old_point)
    old_p_new_o_dist = euclidean_distance(cartesian_old_point, new_observation)

    return sample_dist(old_p_new_o_dist, dist_cond_variance)


def prob_dist_given_xnmin1_yn(distance, old_point, new_observation):
    """
    Calculates distance between old point and new observations for input to sample_dist.
    Returns pdf evaluations.
    :param distance: float or np.array, values to be evaluated
    :param old_point: np.array. u, v, k, alpha
    :param new_observation: np.array, length 2 (cartesian)
    :return: float or np.array same length as distance, conditional pdf evaluations
    """
    cartesian_old_point = cartesianise_numpy(old_point)
    old_p_new_o_dist = euclidean_distance(cartesian_old_point, new_observation)

    return prob_dist(distance, old_p_new_o_dist, dist_cond_variance)


def cdf_prob_dist_given_xnmin1_yn(d_upper, old_point, new_observation):
    """
    Evaluate probability of
    :param d_upper: float or np.array, upper bound
    :param old_point: np.array. u, v, k, alpha
    :param new_observation: np.array, length 2 (cartesian)
    :return:
    """
    cartesian_old_point = cartesianise_numpy(old_point)
    old_p_new_o_dist = ox.euclidean_dist_vec(cartesian_old_point[0], cartesian_old_point[1],
                                             new_observation[0], new_observation[1])

    return cdf_prob_dist(d_upper, old_p_new_o_dist, dist_cond_variance)


def propagate_x(old_particle_in, distance_to_travel, time_to_travel=0, return_intermediate_routes=False):
    """
    Propagates vehicle a given travel distance from a given start position.
    Returns list of possible routes of given distance including all choices at intersections
    :param old_particle_in: np.array, shape=(1, 7)
        t, u, v, k, alpha, n_inter, d
        Initial call with d=0
    :param distance_to_travel: float
        distance to travel until next observation in metres
    :param time_to_travel: float
        inter-observation time in seconds, only required to update t column
    :param return_intermediate_routes: boolean
        whether to return a route each time an intersection is encountered
    :return: list of np.arrays, shape=(num_t_steps, 7)
        describes possible routes taken between observations
        num_t_steps = 2 + number intersections encountered (different for each array)
    """
    global graph

    # Copy input particle to be safe
    old_particle = old_particle_in.copy()

    # Easier to work with 1D array than 2D for now
    old_particle_1d = old_particle[-1, :]

    # Extract edge geometry
    old_edge_geom = get_geometry(old_particle_1d[1:4])

    # Distance left on edge before intersection
    # Use NetworkX length rather than OSM length
    distance_left_on_edge = (1 - old_particle_1d[4]) * old_edge_geom.length

    if distance_left_on_edge > distance_to_travel:
        # Case where propagation remains on the same edge (i.e. doesn't reach an intersection)
        old_particle[-1, 0] += time_to_travel
        old_particle[-1, 4] += distance_to_travel / old_edge_geom.length
        old_particle[-1, 6] += distance_to_travel
        return [old_particle]
    else:
        # Case where intersection is encountered
        time_to_intersection = distance_left_on_edge / distance_to_travel * time_to_travel
        distance_to_travel -= distance_left_on_edge

        time_to_travel -= time_to_intersection

        old_particle[-1, 0] += time_to_intersection
        old_particle[-1, 4] = 1.0
        old_particle[-1, 6] += distance_left_on_edge

        intersection_edges = np.atleast_2d([[u, v, k] for u, v, k in graph.out_edges(old_particle_1d[2], keys=True)])
        n_inter = max(1, sum(intersection_edges[:, 1] != old_particle_1d[1]))

        old_particle[-1, 5] = n_inter

        out_particles = []
        for new_edge in intersection_edges:
            # Don't allow U-turn
            if new_edge[1] != old_particle_1d[1] or len(intersection_edges) == 1:
                new_particle_1d = old_particle[-1:, :].copy()
                new_particle_1d[0, 1:4] = new_edge
                new_particle_1d[0, 4:6] = 0

                new_particle = np.append(old_particle, new_particle_1d, axis=0)

                if return_intermediate_routes:
                    out_particles += [old_particle]
                out_particles += propagate_x(new_particle, distance_to_travel, time_to_travel)

        return out_particles


def naive_particle_filter(polyline, delta_y, n_samps):
    """
    Runs particle filter sampling vehicle paths given noisy observations.
    :param polyline: np.array, shape=(M, 2)
        M observations, cartesian coordinates
    :param delta_y: float
        inter-observation time, assumed constant
    :param n_samps: int
        number of samples to output
    :return: [particles, weights]
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
    weights = np.zeros((M, n_samps))
    weights[0, :] = 1/n_samps

    for m in range(1, M):
        old_particles = xd_particles.copy()
        xd_particles = []
        for j in range(n_samps):
            # Resample
            resample_index = np.random.choice(n_samps, 1, True, weights[m-1, :])[0]
            old_particle = old_particles[resample_index].copy()

            # Sample distance
            new_dist = sample_dist_given_xnmin1_yn(old_particle[-1, 1:5].copy(), polyline[m, :])

            # Possible routes
            pos_routes = propagate_x(old_particle[-1:, :], new_dist, delta_y)

            # Only one possible route
            if len(pos_routes) == 1:
                # Append route to particle
                old_particle = np.append(old_particle, pos_routes[0], axis=0)

                # Calculate weight (unnormalised)
                x_t_cart = cartesianise_numpy(pos_routes[0][-1, 1:5])
                distance_to_obs = euclidean_distance(x_t_cart, polyline[m, :], squared=True)
                weights[m, j] = weights[m - 1, j] * prob_dist_prior(new_dist) \
                    / prob_dist_given_xnmin1_yn(new_dist, old_particle[-1, 1:5], polyline[m, :]) \
                    * np.exp(-0.5 / tools.edges.sigma2_GPS * distance_to_obs) / (2 * np.pi * tools.edges.sigma2_GPS)

            else:
                route_probs = np.zeros(len(pos_routes))
                for i, route in enumerate(pos_routes):
                    # Distance to observation
                    last_pos = route[-1, :]
                    x_t_cart = cartesianise_numpy(last_pos[1:5])
                    distance_to_obs = euclidean_distance(x_t_cart, polyline[m, :], squared=True)

                    # Number of intersections and possibilities at each one
                    intersection_col = route[:, 5]
                    intersection_options = intersection_col[intersection_col > 0]

                    # Probability of travelling route given observation (unnormalised)
                    route_probs[i] = np.exp(-0.5 / tools.edges.sigma2_GPS * distance_to_obs) \
                        / (2 * np.pi * tools.edges.sigma2_GPS) \
                        * np.prod(1 / intersection_options)

                # Probability of generating observation (for all routes)
                prob_yn_given_xnmin1_d_n = sum(route_probs)

                # Normalise route probabilities
                route_probs_normalised = route_probs / prob_yn_given_xnmin1_d_n

                # Sample a route
                sampled_route_index = np.random.choice(len(pos_routes), 1, p=route_probs_normalised)[0]

                # Append sampled route to particle
                old_particle = np.append(old_particle, pos_routes[sampled_route_index], axis=0)

                # Calculate weight (unnormalised)
                weights[m, j] = prob_dist_prior(new_dist) * prob_yn_given_xnmin1_d_n\
                    / prob_dist_given_xnmin1_yn(new_dist, old_particle[-1, 1:5], polyline[m, :])

            xd_particles += [old_particle]
        weights[m, :] /= sum(weights[m, :])

    return xd_particles, weights


def route_probs(routes, new_observation):

    route_probs_out = np.zeros(len(routes))
    for i, route in enumerate(routes):
        # Distance of last edge to observation
        last_pos = route[-1, :]
        last_edge_geom = get_geometry(last_pos[1:4])
        distance_to_obs = Point(new_observation).distance(last_edge_geom)

        # Number of intersections and possibilities at each one
        intersection_col = route[:, 5]
        intersection_options = intersection_col[intersection_col > 0]

        # Probability of travelling route given observation (unnormalised)
        route_probs_out[i] = np.exp(-0.5 / tools.edges.sigma2_GPS * distance_to_obs) \
            / (2 * np.pi * tools.edges.sigma2_GPS) \
            * np.prod(1 / intersection_options)

    return route_probs_out


def get_last_edge_d_bounds(route):

    d_min = 0 if route.shape[0] == 1 else route[-2, -1]

    last_edge_geom = get_geometry(route[-1, 1:4])
    d_max = d_min + last_edge_geom.length

    return d_min, d_max


def extend_routes(routes, extension_distance):

    extended_routes = []
    for route in routes:
        extended_routes += propagate_x(route, extension_distance - 0.0001, return_intermediate_routes=True)

    return extended_routes


def auxiliary_variable_particle_filter(polyline, delta_y, n_samps):
    """
    Runs auxiliary variable particle filter sampling vehicle paths given noisy observations.
    :param polyline: np.array, shape=(M, 2)
        M observations, cartesian coordinates
    :param delta_y: float
        inter-observation time, assumed constant
    :param n_samps: int
        number of samples to output
    :return: [particles, weights]
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
    weights = np.zeros((M, n_samps))
    weights[0, :] = 1/n_samps

    # Initiate ESS
    ess = np.zeros(M)
    ess[0] = n_samps

    for m in range(1, M):
        old_particles = xd_particles.copy()
        xd_particles = []
        for j in range(n_samps):
            # Resample if ESS below threshold
            if ess[m-1] < n_samps * ess_resample_threshold:
                sample_index = np.random.choice(n_samps, 1, True, weights[m - 1, :])[0]
            else:
                sample_index = j
            old_particle = old_particles[sample_index].copy()

            # Sample auxiliary distance variable
            intermediate_dist = sample_dist_given_xnmin1_yn(old_particle[-1, 1:5].copy(), polyline[m, :])

            # Possible routes
            pos_routes = propagate_x(old_particle[-1:, :], intermediate_dist, delta_y)

            if len(pos_routes) == 1:
                sampled_route = pos_routes[0]
            else:
                # Calculate probabilities of chosing routes
                route_sample_probs = route_probs(pos_routes, polyline[m, :])

                # Probability of generating observation (for all routes) given auxiliary distance variable
                prob_yn_given_xnmin1_int_d_n = sum(route_sample_probs)

                # Normalise route probabilities
                route_sample_probs_normalised = route_sample_probs / prob_yn_given_xnmin1_int_d_n

                # Sample a route
                sampled_route_index = np.random.choice(len(pos_routes), 1, p=route_sample_probs_normalised)[0]
                sampled_route = pos_routes[sampled_route_index]

            # Get distances to enter and exit last edge of chosen route
            d_min, d_max = get_last_edge_d_bounds(sampled_route)

            # Get routes up to d_min
            pos_routes_d_min = propagate_x(old_particle[-1:, :], d_min, delta_y)

            # Extend routes
            extended_routes = extend_routes(pos_routes_d_min, d_max - d_min)

            # Probability of sampling extended routes
            extended_route_probs = route_probs(extended_routes, polyline[m, :])

            # Extended route end distances
            extended_routes_end_ds = np.unique([route[-1, -1] for route in extended_routes])





def plot_particles(particles, polyline=None, weights=None):
    """
    Plot paths (output from particle filter).
    :param particles: List of np.arrays representing paths
    :param polyline: np.array, shape=(M,2) for M trajectory observations
    :param weights: np.array, same length as the list particles
    :return: fig, ax, graph with polyline and sampled trajectories
    """
    global graph

    if type(particles) is np.ndarray:
        particles = [particles]

    n_samps = len(particles)

    fig, ax = plot_graph(graph, polyline=polyline)

    min_alpha = 0.3

    for i, path in enumerate(particles):
        cart_path, inter_bool = cartesianise_path(path, True)
        
        path_weight = 1/n_samps if weights is None else weights[i]
        alpha = min_alpha + (1 - min_alpha) * path_weight
        
        ax.scatter(cart_path[:, 0], cart_path[:, 1],
                   linewidths=[0.2 if inter else 3 for inter in inter_bool],
                   alpha=alpha, color='orange')
        
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
    single_index = 0
    poly_single = raw_data['POLYLINE_UTM'][single_index]
    poly_single_array = np.asarray(poly_single)

    # Sample size
    N_samps = 100

    # Observation time increment (s)
    delta_obs = 15

    # Run particle filter
    particles, weights = naive_particle_filter(poly_single_array[:4, :], delta_obs, N_samps)
    ess = np.array([1 / sum(w ** 2) for w in weights])
    print(ess)

    # Plot
    plot_particles(particles, poly_single_array, weights=weights[-1, :])

    # Run auxiliary variable particle filter
    av_particles, av_weights = auxiliary_variable_particle_filter(poly_single_array[:5, :], delta_obs, N_samps)
    av_ess = np.array([1 / sum(w ** 2) for w in av_weights])
    print(av_ess)

    # Plot
    plot_particles(av_particles, poly_single_array, weights=av_weights[-1, :])

    plt.show(block=True)
