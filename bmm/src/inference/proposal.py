########################################################################################################################
# Module: inference/proposal.py
# Description: Proposal mechanisms to extend particles (series of positions/edges/distances) and re-weight
#              in light of a newly received observation.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

from functools import lru_cache

import numpy as np
from numba import njit

from bmm.src.tools.edges import get_geometry, edge_interpolate, discretise_edge
from bmm.src.inference.model import pdf_gamma_mv, cdf_gamma_mv, MapMatchingModel


@lru_cache(maxsize=2 ** 8)
def get_out_edges(graph, node: int):
    return np.atleast_2d([[u, v, k] for u, v, k in graph.out_edges(node, keys=True)])


def get_possible_routes(graph, in_route, dist, all_routes=False):
    """
    Given a route so far and maximum distance to travel, calculate and return all possible routes on graph.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param in_route: np.ndarray, shape = (_, 7)
        starting edge and position on edge
        columns: t, u, v, k, alpha, n_inter, d
        t: float, time
        u: int, edge start node
        v: int, edge end node
        k: int, edge key
        alpha: in [0,1], position along edge
        n_inter: int, number of options if intersection
        d: metres, distance travelled
    :param dist: float
        metres
        maximum possible distance to travel
    :param all_routes: bool
        if true return all routes possible <= d
        else return only routes of length d
    :return: list of numpy.ndarrays
        each numpy.ndarray with shape = (_, 7) as in_route
        each array describes a possible route
    """
    # Extract final position from inputted route
    start_edge_and_position = in_route[-1]

    # Extract edge geometry
    start_edge_geom = get_geometry(graph, start_edge_and_position[1:4])
    start_edge_geom_length = start_edge_geom.length

    # Distance left on edge before intersection
    # Use NetworkX length rather than OSM length
    distance_left_on_edge = (1 - start_edge_and_position[4]) * start_edge_geom_length

    if distance_left_on_edge > dist:
        # Remain on edge
        # Propagate and return
        start_edge_and_position[4] += dist / start_edge_geom_length
        start_edge_and_position[6] += dist
        return [in_route]

    # Reach intersection at end of edge
    # Propagate to intersection and recurse
    dist -= distance_left_on_edge
    start_edge_and_position[4] = 1.
    start_edge_and_position[6] += distance_left_on_edge

    intersection_edges = get_out_edges(graph, start_edge_and_position[2]).copy()

    if intersection_edges.shape[1] == 0:
        # Dead-end and one-way
        if all_routes:
            return [in_route]
        else:
            return [None]

    n_inter = max(1, np.sum(intersection_edges[:, 1] != start_edge_and_position[0]))

    start_edge_and_position[5] = n_inter

    if len(intersection_edges) == 1 and intersection_edges[0][1] == start_edge_and_position[0]:
        # Dead-end and two-way -> Only option is u-turn
        if all_routes:
            return [in_route]
    else:
        new_routes = []
        for new_edge in intersection_edges:
            if new_edge[1] != start_edge_and_position[1]:
                new_route = np.append(in_route,
                                      np.atleast_2d(np.concatenate(
                                          [[0], new_edge, [0, 0, start_edge_and_position[6]]]
                                      )),
                                      axis=0)

                new_routes += get_possible_routes(graph, new_route, dist, all_routes)
        if all_routes:
            return [in_route] + new_routes
        else:
            return new_routes


def extend_routes(graph, routes, add_distance, all_routes=True):
    """
    Extend routes to a further distance.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param routes: list of np.ndarrays
        np.ndarray, shape = (_, 7)
        starting edge and position on edge
        columns: t, u, v, k, alpha, n_inter, d
        t: float, time
        u: int, edge start node
        v: int, edge end node
        k: int, edge key
        alpha: in [0,1], position along edge
        n_inter: int, number of options if intersection
        d: metres, distance travelled
    :param add_distance: float
        metres
        additional distance to travel
    :param all_routes: bool
        if true return all routes possible <= d
        else return only routes of length d
    :return: list of numpy.ndarrays
        each numpy.ndarray with shape = (_, 7)
        each array describes a possible route
    """
    out_routes = []
    for route in routes:
        out_routes += get_possible_routes(graph, route, add_distance, all_routes=all_routes)

    return out_routes


@njit
def pre_discretise_route(route, intersection_penalisation):
    route_d_min = 0 if route.shape[0] == 1 else route[-2, -1]

    # Maximum distance travelled to be in route
    route_d_max = route[-1, -1]

    # Product of 1 / number of intersection choices
    intersection_col = route[:-1, 5]
    intersection_choices_prob = np.prod(1 / intersection_col[intersection_col > 1]) \
                                * intersection_penalisation ** len(intersection_col)

    return route_d_min, route_d_max, intersection_choices_prob


@njit
def post_discretise_route(route, dis_last_edge_mat, route_d_min, route_d_max, intersection_choices_prob, trim_routes):
    if dis_last_edge_mat.size == 0:
        return None

    # Convert distances to full route distances
    dis_last_edge_mat[:, 1] += route_d_min

    # Remove cases that exceed max distance
    if route.shape[0] != 1 and trim_routes:
        dis_last_edge_mat = dis_last_edge_mat[dis_last_edge_mat[:, 1] <= route_d_max]

    # Combine likelihood and prior edge probabilities
    dis_last_edge_mat[:, 2] *= intersection_choices_prob

    return dis_last_edge_mat


def discretise_route(graph, route, d_refine, observation,
                     mm_model, trim_routes=True):
    """
    Discretise route into copies with all possible end positions given a distance discretisation sequence.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param route: np.ndarray, shape = (_, 7)
        columns: t, u, v, k, alpha, n_inter, d
        t: float, time
        u: int, edge start node
        v: int, edge end node
        k: int, edge key
        alpha: in [0,1], position along edge
        n_inter: int, number of options if intersection
        d: metres, distance travelled
    :param d_refine: float
        metres
        resolution of distance discretisation
        increase of speed, decrease for accuracy
    :param observation: numpy.ndarray, shape = (2,)
        UTM projection
        coordinate of first observation
    :param mm_model: MapMatchingModel
        from inference/model
    :param trim_routes: bool
        if true only discretise up to the final distance in route
        if false discretise entire edge
    :return: numpy.ndarray, shape = (_,3)
        columns alpha, d, p_inter_likelihood
            alpha: in [0,1], position along edge
            d: metres, distance travelled since previous observation time
            p_inter_likelihood: likelihood times prior edge probability
    """
    intersection_penalisation = mm_model.intersection_penalisation
    gps_sd = mm_model.gps_sd

    route_d_min, route_d_max, intersection_choices_prob = pre_discretise_route(route, intersection_penalisation)

    # Extract last edge
    last_edge = route[-1, 1:4]

    # Discretise edge
    # Columns: alphas, distances from start of edge, likelihood
    dis_last_edge_mat = discretise_edge(graph, last_edge, d_refine, observation, gps_sd)

    return post_discretise_route(route, dis_last_edge_mat, route_d_min, route_d_max,
                                 intersection_choices_prob, trim_routes)


def process_output(particle, sampled_route, sampled_dis_route, time_interval, full_smoothing):
    # Append sampled route to old particle
    new_route_append = sampled_route
    new_route_append[0, 0] = 0
    new_route_append[-1, 0] = particle[-1, 0] + time_interval
    new_route_append[-1, 4] = sampled_dis_route[1]
    new_route_append[-1, 5] = 0
    new_route_append[-1, 6] = sampled_dis_route[2]

    if full_smoothing:
        return np.append(particle, new_route_append, axis=0)
    else:
        return np.append(particle[-1:], new_route_append, axis=0)


def optimal_proposal(graph, particle, new_observation, time_interval,
                     mm_model,
                     full_smoothing=True,
                     d_refine=1):
    """
    Samples a single particle from the (distance discretised) optimal proposal.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param particle: numpy.ndarray, shape = (_, 7)
        single element of MMParticles.particles
    :param new_observation: numpy.ndarray, shape = (2,)
        UTM projection
        coordinate of first observation
    :param time_interval: float
        seconds
        time between last observation and newly received observation
    :param mm_model: MapMatchingModel
        from inference/model
    :param full_smoothing: bool
        if True returns full trajectory
        else returns only x_t-1 to x_t
    :param d_refine: float
        metres
        resolution of distance discretisation
        increase of speed, decrease for accuracy
    :return: tuple, particle with appended proposal and weight
        particle: numpy.ndarray, shape = (_, 7)
        weight: float, not normalised
    """
    if particle is None:
        return None, 0.

    d_max = mm_model.d_max(time_interval)

    # Extract all possible routes from previous position
    start_position = particle[-1:].copy()
    start_position[0, -1] = 0
    possible_routes = get_possible_routes(graph, start_position, d_max, all_routes=True)

    # Get all possible positions on each route
    discretised_routes_list = []
    for i, route in enumerate(possible_routes):
        # All possible end positions of route
        discretised_route_matrix = discretise_route(graph, route, d_refine, new_observation, mm_model)

        if route.shape[0] == 1:
            discretised_route_matrix = discretised_route_matrix[discretised_route_matrix[:, 0] >= particle[-1, 4]]
            discretised_route_matrix[:, 1] -= discretised_route_matrix[-1, 1]

        # Track route index and append to list
        if discretised_route_matrix is not None:
            discretised_routes_list += [np.append(np.ones((discretised_route_matrix.shape[0], 1)) * i,
                                                  discretised_route_matrix, axis=1)]

    # Concatenate into numpy.ndarray
    discretised_routes = np.concatenate(discretised_routes_list)

    # Calculate sample probabilities
    sample_probs = mm_model.distance_prior_evaluate(discretised_routes[:, 2], time_interval) \
                   * discretised_routes[:, 3]

    # Normalising constant = p(y_m | x_m-1^j)
    sample_probs_norm_const = np.sum(sample_probs)

    if sample_probs_norm_const < 1e-200:
        return None, 0.

    # Sample an edge and distance
    sampled_dis_route_index = np.random.choice(len(discretised_routes), 1, p=sample_probs / sample_probs_norm_const)[0]
    sampled_dis_route = discretised_routes[sampled_dis_route_index]

    # Append sampled route to old particle
    sampled_route = possible_routes[int(sampled_dis_route[0])]

    return process_output(particle, sampled_route, sampled_dis_route, time_interval, full_smoothing), \
           sample_probs_norm_const


class DistanceProposal:
    """
    Abstract class that can sample a distance or evaluate proposal pdf.
    Can take as input particle position, new observation, time interval, GPS standard deviation
    plus any hyperparameters.
    """

    def sample(self, *args, **kwargs):
        """
        Samples from proposal distribution
        :param args:
            additional arguments for proposal
        :param kwargs:
            additional keyword arguments for proposal
        :return: float
            single sample from distance proposal
        """
        raise NotImplementedError

    def pdf(self, x, *args, **kwargs):
        """
        Evaluate proposal pdf.
        :param x: float (>0)
            point(s) to be evaluated
        :param args:
            additional arguments for proposal
        :param kwargs:
            additional keyword arguments for proposal
        :return: float or np.ndarray like x
            pdf evaluation
        """
        raise NotImplementedError

    def cdf(self, x, *args, **kwargs):
        """
        Evaluate proposal cdf.
        :param x: float (>0)
            point(s) to be evaluated
        :param args:
            additional arguments for proposal
        :param kwargs:
            additional keyword arguments for proposal
        :return: float or np.ndarray like x
            pdf evaluation
        """
        raise NotImplementedError


class EuclideanLengthDistanceProposal(DistanceProposal):
    """
    Distance proposal distribution informed only by Euclidean distance between previous position and new observation.
    In particular a Gamma distribution with mean centred around the Euclidean distance and inputted variance.
    """

    @staticmethod
    def sample(euclidean_distance, var=20):
        """
        Single sample from proposal.
        :param euclidean_distance: float (>0)
            mean
        :param var: float (>0)
            variance
        :return: float (>0)
        """
        gamma_beta = euclidean_distance / var
        gamma_alpha = euclidean_distance * gamma_beta
        return np.random.gamma(gamma_alpha, 1 / gamma_beta)

    @staticmethod
    def pdf(x, euclidean_distance, var=20):
        """
        Evaluate proposal pdf
        :param x: float or np.array (>0)
            value(s) to be evaluated
        :param euclidean_distance: float (>0)
            mean
        :param var: float (>0)
            variance
        :return: float or np.array (>0)
        """
        return pdf_gamma_mv(x, euclidean_distance, var)

    @staticmethod
    def cdf(x, euclidean_distance, var=20):
        """
        :param x: float (>0)
        :param euclidean_distance: float (>0)
            mean
        :param var: float (>0)
            variance
        :return: float or np.array in [0,1]
        """
        return cdf_gamma_mv(x, euclidean_distance, var)


def get_route_ranges(routes):
    d_ranges_all = np.zeros((2, len(routes)))
    for i in range(len(routes)):
        d_ranges_all[0, i] = 0 if routes[i].shape[0] == 1 else routes[i][-2, -1]
        d_ranges_all[1, i] = routes[i][-1, -1]
    return d_ranges_all


def auxiliary_distance_proposal(graph, particle, new_observation, time_interval, mm_model, full_smoothing=True,
                                d_refine=1, dist_expand=50,
                                dist_prop=EuclideanLengthDistanceProposal(), **kwargs):


    if particle is None:
        return None, 0.

    gps_sd = mm_model.gps_sd

    # Extract all possible routes from previous position
    start_position = particle[-1:].copy()
    start_position[0, -1] = 0

    # Get geometry
    start_geom = get_geometry(graph, start_position[0, 1:4])

    # Cartesianise
    cart_start = edge_interpolate(start_geom, start_position[0, 4])

    # Get Euclidean distance between particle and new observation
    euclid_dist = np.linalg.norm(cart_start - new_observation)

    # Sample distance
    dist_samp = dist_prop.sample(euclid_dist, **kwargs)

    # Expand sampled distance
    min_dist_range = max(0, dist_samp - dist_expand)
    dist_range = (min_dist_range, min_dist_range + 2 * dist_expand)

    # Get possible routes of length dist_samp
    all_routes = get_possible_routes(graph, start_position, dist_range[1], all_routes=True)

    # Remove routes that don't finish in dist_range
    d_ranges_all = get_route_ranges(all_routes)
    dist_samp_keep = d_ranges_all[1] >= dist_range[0]
    routes = [all_routes[i] for i in np.where(dist_samp_keep)[0]]

    # No routes implies reach and overflow dead end
    if len(routes) == 0:
        return None, 0

    # Get all possible positions on each route
    discretised_routes_list = []
    for i, route in enumerate(routes):
        # All possible end positions of route
        discretised_route_matrix = discretise_route(graph, route, d_refine, new_observation, mm_model, trim_routes=False)

        if route.shape[0] == 1:
            discretised_route_matrix = discretised_route_matrix[discretised_route_matrix[:, 0] >= particle[-1, 4]]
            discretised_route_matrix[:, 1] -= discretised_route_matrix[-1, 1]

        # Track route index and append to list
        if discretised_route_matrix is not None:
            discretised_routes_list += [np.append(np.ones((discretised_route_matrix.shape[0], 1)) * i,
                                                  discretised_route_matrix, axis=1)]

    # Concatenate into numpy.ndarray
    discretised_routes = np.concatenate(discretised_routes_list)

    # Remove points outside range
    discretised_routes = discretised_routes[np.logical_and(discretised_routes[:, 2] >= dist_range[0],
                                                           discretised_routes[:, 2] < dist_range[1])]

    # Calculate sample probabilities
    sample_probs = mm_model.distance_prior_evaluate(discretised_routes[:, 2], time_interval) * discretised_routes[:, 3]

    # Normalising constant
    sample_probs_norm_const = np.sum(sample_probs)

    if sample_probs_norm_const < 1e-200:
        return None, 0.

    # Sample an edge and distance
    sampled_dis_route_index = np.random.choice(len(discretised_routes), 1, p=sample_probs / sample_probs_norm_const)[0]
    sampled_dis_route = discretised_routes[sampled_dis_route_index]

    # Append sampled route to old particle
    sampled_route = routes[int(sampled_dis_route[0])]
    out_particle = process_output(particle, sampled_route, sampled_dis_route, time_interval, full_smoothing)

    # Sampled position distance
    selected_dist = sampled_dis_route[2]

    # Range of distances that need checking
    dist_check_range = (max(0, selected_dist - 2 * dist_expand), selected_dist + 2 * dist_expand)

    # All possible routes up to max_aux_dist
    min_dist_keep = d_ranges_all[1, :] >= dist_check_range[0]
    check_routes = [all_routes[i] for i in np.where(min_dist_keep)[0]]

    # Extend routes
    check_routes = extend_routes(graph, check_routes, dist_check_range[1] - dist_range[1])

    d_ranges_check = get_route_ranges(check_routes)

    check_routes_keep = np.logical_and(d_ranges_check[1, :] >= dist_check_range[0],
                                       d_ranges_check[1, :] <= dist_check_range[1])
    check_routes = [check_routes[i] for i in np.where(check_routes_keep)[0]]

    # Discretise routes
    discretised_check_routes_list = []
    for i, route in enumerate(check_routes):
        # All possible end positions of route
        discretised_route_matrix = discretise_route(graph, route, d_refine, new_observation, mm_model, trim_routes=False)

        if route.shape[0] == 1:
            discretised_route_matrix = discretised_route_matrix[discretised_route_matrix[:, 0] >= particle[-1, 4]]
            discretised_route_matrix[:, 1] -= discretised_route_matrix[-1, 1]

        # Track route index and append to list
        if discretised_route_matrix is not None:
            discretised_check_routes_list += [np.append(np.ones((discretised_route_matrix.shape[0], 1)) * i,
                                                        discretised_route_matrix, axis=1)]

    # Concatenate into numpy.ndarray
    discretised_check_routes = np.concatenate(discretised_check_routes_list)

    # Remove points outside range
    discretised_check_routes = discretised_check_routes[
        np.logical_and(discretised_check_routes[:, 2] >= dist_check_range[0],
                       discretised_check_routes[:, 2] <= dist_check_range[1])]

    # Calculate sample probabilities
    dis_check_probs = mm_model.distance_prior_evaluate(discretised_check_routes[:, 2], time_interval)\
                      * discretised_check_routes[:, 3]

    # All possible (discrete) distances
    all_check_distances = np.unique(discretised_check_routes[:, 2])

    # possible extrema of stratum
    poss_min_strata = all_check_distances[all_check_distances <= max(dist_samp, 2 * dist_expand)]
    if poss_min_strata[0] < dist_expand:
        poss_min_strata[0] = 0
        poss_min_strata = poss_min_strata[np.logical_or(poss_min_strata == 0,
                                                        poss_min_strata >= dist_expand)]
    poss_max_stata = all_check_distances[all_check_distances >= max(dist_samp, 2 * dist_expand)]
    poss_min_strata = np.unique(np.append(poss_min_strata, poss_max_stata - 2 * dist_expand))
    poss_max_strata = poss_min_strata + 2 * dist_expand
    poss_mid_strata = poss_min_strata + dist_expand

    mid_strata_cdf_evals = dist_prop.cdf(poss_mid_strata, euclid_dist, **kwargs)
    mid_strata_cdf_evals[1:] -= mid_strata_cdf_evals[:-1]

    unnorm_weight_denom = aux_dist_expand_weight(discretised_check_routes, dis_check_probs,
                                                 poss_min_strata, poss_max_strata, mid_strata_cdf_evals)

    if unnorm_weight_denom == 0:
        return out_particle, 0
    else:
        return out_particle, 1 / unnorm_weight_denom


@njit
def aux_dist_expand_weight(discretised_check_routes, dis_check_probs,
                           poss_min_strata, poss_max_strata, mid_strata_cdf_evals):
    unnorm_weight_denom = 0
    for i in range(len(poss_min_strata)):
        if mid_strata_cdf_evals[i] == 0:
            continue

        min_stratum = poss_min_strata[i]
        max_stratum = poss_max_strata[i]

        partial_prob_denom = np.sum(dis_check_probs[np.logical_and(discretised_check_routes[:, 2] >= min_stratum,
                                                                   discretised_check_routes[:, 2] <= max_stratum)])

        if partial_prob_denom == 0:
            continue

        unnorm_weight_denom += mid_strata_cdf_evals[i] / partial_prob_denom

    return unnorm_weight_denom

#
#
# def dist_then_edge_proposal(graph, particle, new_observation, time_interval, gps_sd,
#                             dist_prop=EuclideanLengthDistanceProposal(), **kwargs):
#     """
#     Samples distance first then chooses edges travelled. Outputs updated particle and (unnormalised) weight.
#     :param graph: NetworkX MultiDiGraph
#         UTM projection
#         encodes road network
#         generating using OSMnx, see tools.graph.py
#     :param particle: numpy.ndarray, shape = (_, 7)
#         single element of MMParticles.particles
#     :param new_observation: numpy.ndarray, shape = (2,)
#         UTM projection
#         coordinate of first observation
#     :param time_interval: float
#         seconds
#         time between last observation and newly received observation
#     :param gps_sd: float
#         metres
#         standard deviation of GPS noise
#     :param dist_prop: DistanceProposal
#         class that can propose a distance and evaluate pdf of said proposal
#     :param kwargs: optional parameters to pass to distance proposal
#         i.e. variance of proposal (var=10)
#     :return: tuple, particle with appended proposal and weight
#         particle: numpy.ndarray, shape = (_, 7)
#         weight: float, not normalised
#     """
#     if particle is None:
#         return None, 0.
#
#     # Extract position at last observation time
#     start_position = particle[-1:].copy()
#     start_position[0, -1] = 0
#
#     # Get geometry
#     start_geom = get_geometry(graph, start_position[0, 1:4])
#
#     # Cartesianise
#     cart_start = edge_interpolate(start_geom, start_position[0, 4])
#
#     # Get Euclidean distance between particle and new observation
#     euclid_dist = np.linalg.norm(cart_start - new_observation)
#
#     # Sample distance
#     dist_samp = dist_prop.sample(euclid_dist, **kwargs)
#
#     # Get possible routes of length dist_samp
#     routes = get_possible_routes(graph, start_position, dist_samp, all_routes=False)
#
#     # No routes implies reached dead end
#     if len(routes) == 0:
#         start_position[-1, 0] = particle[-1, 0] + time_interval
#         start_position[-1, 5] = 0
#         start_position[-1, -1] = 0
#         out_particle = np.append(particle, start_position, axis=0)
#         return out_particle, 0
#
#     # Initiate cartesian position of end of routes
#     routes_end_cart_pos = np.zeros((len(routes), 2))
#
#     # Initiate prod 1/number of choices at intersection
#     intersection_probs = np.zeros(len(routes))
#
#     # Iterate through routes
#     for i, route in enumerate(routes):
#
#         if route is None:
#             continue
#
#         end_position = route[-1]
#
#         end_geom = get_geometry(graph, end_position[1:4])
#
#         routes_end_cart_pos[i] = edge_interpolate(end_geom, end_position[4])
#
#         intersection_col = route[:-1, 5]
#         intersection_probs[i] = np.prod(1 / intersection_col[intersection_col > 1]) \
#                                 * intersection_penalisation ** len(intersection_col)
#
#     # Distances of end points to new observation
#     obs_distances_sqr = np.sum((routes_end_cart_pos - new_observation) ** 2, axis=1)
#
#     # Unnormalised sample probablilites
#     route_sample_weights = np.exp(- 0.5 / gps_sd ** 2 * obs_distances_sqr) * intersection_probs
#
#     # Normalising constant
#     prob_y_given_x_prev_d = np.sum(route_sample_weights)
#
#     # Normalise
#     route_sample_weights /= prob_y_given_x_prev_d
#
#     # Sample route
#     sampled_route_ind = np.random.choice(len(routes), 1, p=route_sample_weights)[0]
#     sampled_route = routes[sampled_route_ind]
#
#     # Append to old particle
#     sampled_route[0, 0] = 0
#     sampled_route[-1, 0] = particle[-1, 0] + time_interval
#     sampled_route[-1, 5] = 0
#     out_particle = np.append(particle, sampled_route, axis=0)
#
#     # Weight
#     weight = prob_y_given_x_prev_d * distance_prior(dist_samp, time_interval) \
#              / dist_prop.pdf(dist_samp, euclid_dist, **kwargs)
#
#     return out_particle, weight
#
#
# def auxiliary_distance_proposal_edge(graph, particle, new_observation, time_interval, gps_sd,
#                                      d_refine=1, dist_prop=EuclideanLengthDistanceProposal(), **kwargs):
#
#     if particle is None:
#         return None, 0.
#
#     # Extract all possible routes from previous position
#     start_position = particle[-1:].copy()
#     start_position[0, -1] = 0
#
#     # Get geometry
#     start_geom = get_geometry(graph, start_position[0, 1:4])
#
#     # Cartesianise
#     cart_start = edge_interpolate(start_geom, start_position[0, 4])
#
#     # Get Euclidean distance between particle and new observation
#     euclid_dist = np.linalg.norm(cart_start - new_observation)
#
#     # Sample distance
#     dist_samp = dist_prop.sample(euclid_dist, **kwargs)
#
#     # Get possible routes of length dist_samp
#     all_routes = get_possible_routes(graph, start_position, dist_samp, all_routes=True)
#
#     # Remove routes that don't reach dist_samp
#     d_ranges_all = np.zeros((2, len(all_routes)))
#     for i in range(len(all_routes)):
#         d_ranges_all[0, i] = 0 if all_routes[i].shape[0] == 1 else all_routes[i][-2, -1]
#         d_ranges_all[1, i] = all_routes[i][-1, -1]
#     dist_samp_keep = d_ranges_all[1, :] == dist_samp
#     routes = [all_routes[i] for i in np.where(dist_samp_keep)[0]]
#
#     # No routes implies reached dead end
#     if len(routes) == 0:
#         start_position[-1, 0] = particle[-1, 0] + time_interval
#         start_position[-1, 5] = 0
#         start_position[-1, -1] = 0
#         out_particle = np.append(particle, start_position, axis=0)
#         return out_particle, 0
#
#     # Get all possible positions on each route
#     discretised_routes_list = []
#     for i, route in enumerate(routes):
#         # All possible end positions of route
#         discretised_route_matrix = discretise_route(graph, route, d_refine, new_observation, gps_sd, trim_routes=False)
#
#         if route.shape[0] == 1:
#             discretised_route_matrix = discretised_route_matrix[discretised_route_matrix[:, 0] >= particle[-1, 4]]
#             discretised_route_matrix[:, 1] -= discretised_route_matrix[-1, 1]
#
#         # Track route index and append to list
#         if discretised_route_matrix is not None:
#             discretised_routes_list += [np.append(np.ones((discretised_route_matrix.shape[0], 1)) * i,
#                                                   discretised_route_matrix, axis=1)]
#
#     # Concatenate into numpy.ndarray
#     discretised_routes = np.concatenate(discretised_routes_list)
#
#     # Calculate sample probabilities
#     sample_probs = distance_prior(discretised_routes[:, 2], time_interval) \
#                    * discretised_routes[:, 3]
#
#     # Normalising constant
#     sample_probs_norm_const = np.sum(sample_probs)
#
#     if sample_probs_norm_const < 1e-200:
#         return None, 0.
#
#     # Sample an edge and distance
#     sampled_dis_route_index = np.random.choice(len(discretised_routes), 1, p=sample_probs / sample_probs_norm_const)[0]
#     sampled_dis_route = discretised_routes[sampled_dis_route_index]
#
#     # Append sampled route to old particle
#     sampled_route = routes[int(sampled_dis_route[0])]
#     new_route_append = sampled_route.copy()
#     new_route_append[0, 0] = 0
#     new_route_append[-1, 0] = particle[-1, 0].copy() + time_interval
#     new_route_append[-1, 4] = sampled_dis_route[1]
#     new_route_append[-1, 5] = 0
#     new_route_append[-1, 6] = sampled_dis_route[2]
#
#     # Minimum possible sampled auxiliary distance
#     min_aux_dist = 0 if sampled_route.shape[0] == 1 else sampled_route[-2, -1]
#
#     # Distance to end of selected edge
#     sampled_edge_geom = get_geometry(graph, new_route_append[-1, 1:4])
#     sampled_edge_length = sampled_edge_geom.length
#     dist_to_end_of_sampled_edge = (1 - new_route_append[-1, 4]) * sampled_edge_length
#
#     # Maximum possible sampled auxiliary distance
#     max_aux_dist = new_route_append[-1, -1] + dist_to_end_of_sampled_edge
#
#     # All possible routes up to max_aux_dist
#     min_dist_keep = d_ranges_all[0, :] >= min_aux_dist
#     possible_sampled_routes = [all_routes[i] for i in np.where(min_dist_keep)[0]]
#     possible_sampled_routes = extend_routes(graph, possible_sampled_routes, max_aux_dist)
#
#     # Store min and max distances for each route
#     d_ranges = np.zeros((2, len(possible_sampled_routes)))
#     for i in range(len(possible_sampled_routes)):
#         d_ranges[0, i] = 0 if possible_sampled_routes[i].shape[0] == 1 else possible_sampled_routes[i][-2, -1]
#         d_ranges[1, i] = possible_sampled_routes[i][-1, -1]
#
#     discretised_poss_sampled_routes_list = []
#     for i, route in enumerate(possible_sampled_routes):
#         # All possible end positions of route
#         discretised_route_matrix = discretise_route(graph, route, d_refine, new_observation, gps_sd, trim_routes=False)
#
#         if route.shape[0] == 1:
#             discretised_route_matrix = discretised_route_matrix[discretised_route_matrix[:, 0] >= particle[-1, 4]]
#             discretised_route_matrix[:, 1] -= discretised_route_matrix[-1, 1]
#
#         # Track route index and append to list
#         if discretised_route_matrix is not None:
#             discretised_poss_sampled_routes_list += [np.append(np.ones((discretised_route_matrix.shape[0], 1)) * i,
#                                                                discretised_route_matrix, axis=1)]
#
#     # Concatenate into numpy.ndarray
#     discretised_poss_sampled_routes = np.concatenate(discretised_poss_sampled_routes_list)
#
#     # Calculate sample probabilities
#     poss_sample_probs = distance_prior(discretised_poss_sampled_routes[:, 2], time_interval) \
#                         * discretised_poss_sampled_routes[:, 3]
#
#     # Sort all min and max distances
#     d_ranges_all_sort = np.unique(np.concatenate(d_ranges))
#
#     # Pre-evaluate required cdfs
#     aux_cdf_evals = np.zeros_like(d_ranges_all_sort)
#     aux_cdf_evals[d_ranges_all_sort != 0] = \
#         dist_prop.cdf(d_ranges_all_sort[d_ranges_all_sort != 0], euclid_dist, **kwargs)
#     aux_cdf_evals[1:] -= aux_cdf_evals[:-1]
#
#     # Iteratively calculate unnormalised weights
#     unnorm_weight_denom = 0
#     for i in range(1, len(d_ranges_all_sort)):
#         if aux_cdf_evals[i] == 0:
#             continue
#
#         min_samp_dist = d_ranges_all_sort[i - 1]
#         max_samp_dist = d_ranges_all_sort[i]
#
#         possible_routes_indices = np.where((d_ranges[0] <= min_samp_dist) & (d_ranges[1] >= max_samp_dist))[0]
#
#         partial_prob_denom = np.sum(poss_sample_probs[np.isin(discretised_poss_sampled_routes[:, 0],
#                                                               possible_routes_indices)])
#
#         if partial_prob_denom == 0:
#             continue
#
#         unnorm_weight_denom += aux_cdf_evals[i] / partial_prob_denom
#
#     if unnorm_weight_denom == 0:
#         return np.append(particle, new_route_append, axis=0), 0
#     else:
#         return np.append(particle, new_route_append, axis=0), 1 / unnorm_weight_denom
