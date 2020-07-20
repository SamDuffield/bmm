########################################################################################################################
# Module: inference/proposal.py
# Description: Proposal mechanisms to extend particles (series of positions/edges/distances) and re-weight
#              in light of a newly received observation.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

from functools import lru_cache
from typing import Tuple, Union

import numpy as np
from numba import njit
from networkx.classes import MultiDiGraph

from bmm.src.tools.edges import get_geometry, edge_interpolate, discretise_edge
from bmm.src.inference.model import pdf_gamma_mv, cdf_gamma_mv, MapMatchingModel


@lru_cache(maxsize=2 ** 8)
def get_out_edges(graph: MultiDiGraph,
                  node: int) -> np.ndarray:
    """
    Extracts out edges from a given node
    :param graph: encodes road network, simplified and projected to UTM
    :param node: graph index to a single node
    :return: array with columns u, v, k with u = node
    """
    return np.atleast_2d([[u, v, k] for u, v, k in graph.out_edges(node, keys=True)])


def get_possible_routes(graph: MultiDiGraph,
                        in_route: np.ndarray,
                        dist: float,
                        all_routes: bool = False,
                        num_inter_cut_off: int = np.inf) -> list:
    """
    Given a route so far and maximum distance to travel, calculate and return all possible routes on graph.
    :param graph: encodes road network, simplified and projected to UTM
    :param in_route: shape = (_, 9)
        columns: t, u, v, k, alpha, x, y, n_inter, d
        t: float, time
        u: int, edge start node
        v: int, edge end node
        k: int, edge key
        alpha: in [0,1], position along edge
        x: float, metres, cartesian x coordinate
        y: float, metres, cartesian y coordinate
        n_inter: int, number of options if intersection
        d: metres, distance travelled
    :param dist: metres, maximum possible distance to travel
    :param all_routes: if true return all routes possible <= d
        otherwise return only routes of length d
    :param num_inter_cut_off: maximum number of intersections to cross in the time interval
    :return: list of arrays
        each array with shape = (_, 9) as in_route
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
        start_edge_and_position[-1] += dist
        return [in_route]

    # Reach intersection at end of edge
    # Propagate to intersection and recurse
    dist -= distance_left_on_edge
    start_edge_and_position[4] = 1.
    start_edge_and_position[-1] += distance_left_on_edge

    intersection_edges = get_out_edges(graph, start_edge_and_position[2]).copy()

    if intersection_edges.shape[1] == 0:
        # Dead-end and one-way
        if all_routes:
            return [in_route]
        else:
            return [None]

    n_inter = max(1, np.sum(~((intersection_edges[:, 1] == start_edge_and_position[1]) *
                              (intersection_edges[:, 2] == start_edge_and_position[3]))))

    start_edge_and_position[-2] = n_inter

    if len(intersection_edges) == 1 and intersection_edges[0][1] == start_edge_and_position[1]\
            and intersection_edges[0][2] == start_edge_and_position[3]:
        # Dead-end and two-way -> Only option is u-turn
        if all_routes:
            return [in_route]
    else:
        new_routes = []
        for new_edge in intersection_edges:
            # If not u-turn or loop continue route search on new edge
            if not (new_edge[1] == start_edge_and_position[1] and new_edge[2] == start_edge_and_position[3])\
                    and not (new_edge == in_route[:, 1:4]).all(1).any() \
                    and len(in_route) < num_inter_cut_off:
                add_edge = np.array([[0, *new_edge, 0, 0, 0, 0, start_edge_and_position[-1]]])
                new_route = np.append(in_route,
                                      add_edge,
                                      axis=0)

                new_routes += get_possible_routes(graph, new_route, dist, all_routes, num_inter_cut_off)
        if all_routes:
            return [in_route] + new_routes
        else:
            return new_routes


def extend_routes(graph, routes, add_distance, all_routes=True):
    """
    Extend routes to a further distance.
    :param graph: encodes road network, simplified and projected to UTM
    :param routes: list of arrays
        columns: t, u, v, k, alpha, x, y, n_inter, d
        t: float, time
        u: int, edge start node
        v: int, edge end node
        k: int, edge key
        alpha: in [0,1], position along edge
        x: float, metres, cartesian x coordinate
        y: float, metres, cartesian y coordinate
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


def process_proposal_output(particle: np.ndarray,
                            sampled_route: np.ndarray,
                            sampled_dis_route: np.ndarray,
                            time_interval: float,
                            full_smoothing: bool) -> np.ndarray:
    """
    Append sampled route to previous particle
    :param particle: route up to previous observation
    :param sampled_route: route since previous observation
    :param sampled_dis_route: alpha, x, y, distance
    :param time_interval: time between last observation and newly received observation
    :param full_smoothing: whether to append to full particle or only last row
    :return: appended particle
    """
    # Append sampled route to old particle
    new_route_append = sampled_route
    new_route_append[0, 0] = 0
    new_route_append[0, 5:7] = 0
    new_route_append[-1, 0] = particle[-1, 0] + time_interval
    new_route_append[-1, 4:7] = sampled_dis_route[0:3]
    new_route_append[-1, -2] = 0
    new_route_append[-1, -1] = sampled_dis_route[-1]

    if full_smoothing:
        return np.append(particle, new_route_append, axis=0)
    else:
        return np.append(particle[-1:], new_route_append, axis=0)


def intersection_prior_evaluate(routes: list,
                                mm_model: MapMatchingModel) -> np.ndarray:
    """
    Evaluate intersection prior/transition density
    :param routes: list of arrays which describe edges traversed since last observation
    :param mm_model: MapMatchingModel
    :return: intersection prior evaluations
    """
    evals = np.zeros(len(routes))

    for i, route in enumerate(routes):
        evals[i] = mm_model.intersection_prior_evaluate(route)
    return evals


def optimal_proposal(graph: MultiDiGraph,
                     particle: np.ndarray,
                     new_observation: np.ndarray,
                     time_interval: float,
                     mm_model: MapMatchingModel,
                     full_smoothing: bool = True,
                     d_refine: float = 1.,
                     num_inter_cut_off: int = None,
                     store_dev_norm_quants: bool = False) -> Union[Tuple[Union[None, np.ndarray], float],
                                                                   Tuple[Union[None, np.ndarray],
                                                                         float,
                                                                         np.ndarray]]:
    """
    Samples a single particle from the (distance discretised) optimal proposal.
    :param graph: encodes road network, simplified and projected to UTM
    :param particle: single element of MMParticles.particles
    :param new_observation: cartesian coordinate in UTM
    :param time_interval: time between last observation and newly received observation
    :param mm_model: MapMatchingModel
    :param full_smoothing: if True returns full trajectory
        otherwise returns only x_t-1 to x_t
    :param d_refine: metres, resolution of distance discretisation
    :param num_inter_cut_off: maximum number of intersections to cross in the time interval
    :param store_dev_norm_quants: whether to additionally return quantities needed for gradient EM step
        assuming deviation prior is used
    :return: (particle, unnormalised weight) or (particle, unnormalised weight, dev_norm_quants)
    """
    if particle is None:
        return (None, 0., 0.) if store_dev_norm_quants else (None, 0.)

    if num_inter_cut_off is None:
        num_inter_cut_off = max(int(time_interval / 1.5), 10)

    d_max = mm_model.d_max(time_interval)

    # Extract all possible routes from previous position
    start_position = particle[-1:].copy()
    start_position[0, -1] = 0
    possible_routes = get_possible_routes(graph, start_position, d_max, all_routes=True,
                                          num_inter_cut_off=num_inter_cut_off)

    # Get all possible positions on each route
    discretised_routes_indices_list = []
    discretised_routes_list = []
    for i, route in enumerate(possible_routes):
        # All possible end positions of route
        discretised_edge_matrix = discretise_edge(graph, route[-1, 1:4], d_refine)

        if route.shape[0] == 1:
            discretised_edge_matrix = discretised_edge_matrix[discretised_edge_matrix[:, 0] >= particle[-1, 4]]
            discretised_edge_matrix[:, -1] -= discretised_edge_matrix[-1, -1]
        else:
            discretised_edge_matrix[:, -1] += route[-2, -1]

        discretised_edge_matrix = discretised_edge_matrix[discretised_edge_matrix[:, -1] < d_max]

        # Track route index and append to list
        if discretised_edge_matrix is not None and len(discretised_edge_matrix) > 0:
            discretised_routes_indices_list += [np.ones(discretised_edge_matrix.shape[0], dtype=int) * i]
            discretised_routes_list += [discretised_edge_matrix]

    # Concatenate into numpy.ndarray
    discretised_routes_indices = np.concatenate(discretised_routes_indices_list)
    discretised_routes = np.concatenate(discretised_routes_list)

    # Likelihood evaluations
    likelihood_evals = mm_model.likelihood_evaluate(discretised_routes[:, 1:3], new_observation)

    # Trim
    if not store_dev_norm_quants:
        positive_lik_inds = likelihood_evals > 0
        discretised_routes_indices = discretised_routes_indices[positive_lik_inds]
        discretised_routes = discretised_routes[positive_lik_inds]
        likelihood_evals = likelihood_evals[positive_lik_inds]

    if len(discretised_routes) == 0:
        return (None, 0., 0.) if store_dev_norm_quants else (None, 0.)

    # Distance prior evals
    distances = discretised_routes[:, -1]
    distance_prior_evals = mm_model.distance_prior_evaluate(distances, time_interval)

    # Intersection prior evals
    route_intersection_prior_evals = intersection_prior_evaluate(possible_routes, mm_model)[discretised_routes_indices]

    # Deviation prior evals
    deviation_prior_evals = mm_model.deviation_prior_evaluate(particle[-1, 5:7],
                                                              discretised_routes[:, 1:3],
                                                              discretised_routes[:, -1])

    # Normalise prior/transition probabilities
    prior_probs = distance_prior_evals \
                  * route_intersection_prior_evals \
                  * deviation_prior_evals

    if len(discretised_routes) == 1 and discretised_routes[0][-1] == 0:
        return (None, 0., np.sum(prior_probs)) if store_dev_norm_quants else (None, 0.)

    if store_dev_norm_quants:
        prior_probs_norm_const = prior_probs[distances > 0].sum()
        prior_probs[distances > 0] *= (1 - prior_probs[distances == 0][0]) / prior_probs_norm_const

    else:
        prior_probs_norm_const = prior_probs.sum()
        prior_probs /= prior_probs_norm_const

    # Calculate sample probabilities
    sample_probs = prior_probs * likelihood_evals

    # p(y_m | x_m-1^j)
    prop_weight = sample_probs.sum()

    if prop_weight < 1e-200:
        proposal_out = None
        prop_weight = 0.
    else:
        # Sample an edge and distance
        sampled_dis_route_index = np.random.choice(len(discretised_routes), 1, p=sample_probs / prop_weight)[0]
        sampled_dis_route = discretised_routes[sampled_dis_route_index]

        # Append sampled route to old particle
        sampled_route = possible_routes[discretised_routes_indices[sampled_dis_route_index]]

        proposal_out = process_proposal_output(particle, sampled_route, sampled_dis_route, time_interval,
                                               full_smoothing)

    if store_dev_norm_quants:
        deviations = np.sqrt(np.sum((particle[-1, 5:7] - discretised_routes[:, 1:3]) ** 2, axis=1))
        deviations = np.abs(deviations - discretised_routes[:, -1])

        # Z, dz/d alpha, dZ/d beta (alpha is all distance parameters, beta is deviation parameter)
        dev_norm_quants = np.array([prior_probs_norm_const,
                                    *np.sum(mm_model.distance_prior_gradient(distances, time_interval)
                                            * route_intersection_prior_evals[discretised_routes_indices]
                                            * deviation_prior_evals, axis=-1),
                                    -np.sum(deviations
                                            * distance_prior_evals
                                            * route_intersection_prior_evals[discretised_routes_indices]
                                            * deviation_prior_evals)
                                    ])

        if prior_probs_norm_const == 0:
            raise

        return proposal_out, prop_weight, dev_norm_quants
    else:
        return proposal_out, prop_weight


class DistanceProposal:
    """
    Abstract class that can sample a distance or evaluate proposal pdf.
    Can take as input particle position, new observation, time interval, GPS standard deviation
    plus any hyperparameters.
    """

    def sample(self, *args, **kwargs) -> Union[float, np.ndarray]:
        """
        Samples from proposal distribution
        :param args: additional arguments for proposal
        :param kwargs: additional keyword arguments for proposal
        :return: samples(s) from proposal
        """
        raise NotImplementedError

    def pdf(self, *args, **kwargs) -> Union[float, np.ndarray]:
        """
        Evaluate proposal pdf.
        :param x: point(s) to be evaluated
        :param args: additional arguments for proposal
        :param kwargs: additional keyword arguments for proposal
        :return: pdf evaluation(s)
        """
        raise NotImplementedError

    def cdf(self, *args, **kwargs) -> Union[float, np.ndarray]:
        """
        Evaluate proposal cdf.
        :param x: point(s) to be evaluated
        :param args: additional arguments for proposal
        :param kwargs: additional keyword arguments for proposal
        :return: cdf evaluation(s)
        """
        raise NotImplementedError


class EuclideanLengthDistanceProposal(DistanceProposal):
    """
    Distance proposal distribution informed only by Euclidean distance between previous position and new observation.
    In particular a Gamma distribution with mean centred around the Euclidean distance and inputted variance.
    """

    @staticmethod
    def sample(euclidean_distance: Union[float, np.ndarray],
               var: Union[float, np.ndarray] = 20,
               n: int = None) -> Union[float, np.ndarray]:
        """
        Single sample from proposal.
        :param euclidean_distance: mean
        :param var: variance
        :param n: number of samples (default is np.broadcast(euclidean_distance, var).size)
        :return: samples from proposal
        """
        gamma_beta = euclidean_distance / var
        gamma_alpha = euclidean_distance * gamma_beta
        return np.random.gamma(gamma_alpha, 1 / gamma_beta, size=n)

    @staticmethod
    def pdf(x: Union[float, np.ndarray],
            euclidean_distance: Union[float, np.ndarray],
            var: Union[float, np.ndarray] = 20) -> Union[float, np.ndarray]:
        """
        Evaluate proposal pdf
        :param x: point(s) to be evaluated
        :param euclidean_distance: mean
        :param var: variance
        :return: pdf evaluation(s)
        """
        return pdf_gamma_mv(x, euclidean_distance, var)

    @staticmethod
    def cdf(x: Union[float, np.ndarray],
            euclidean_distance: Union[float, np.ndarray],
            var: Union[float, np.ndarray] = 20) -> Union[float, np.ndarray]:
        """
        Evaluate proposal cdf
        :param x: point(s) to be evaluated
        :param euclidean_distance: mean
        :param var: variance
        :return: cdf evaluation(s)
        """
        return cdf_gamma_mv(x, euclidean_distance, var)


def get_route_ranges(routes: list) -> np.ndarray:
    """
    Extract min and max possible distances travelled for each inputted route
    :param routes: list of arrays
    :return: shape = (2, _)
        first row: min distances
        second row: max distances
    """
    d_ranges_all = np.zeros((2, len(routes)))
    for i in range(len(routes)):
        d_ranges_all[0, i] = 0 if routes[i].shape[0] == 1 else routes[i][-2, -1]
        d_ranges_all[1, i] = routes[i][-1, -1]
    return d_ranges_all


def auxiliary_distance_proposal(graph: MultiDiGraph,
                                particle: np.ndarray,
                                new_observation: np.ndarray,
                                time_interval: float,
                                mm_model: MapMatchingModel,
                                full_smoothing: bool = True,
                                d_refine: float = 1.,
                                dist_expand: float = 50,
                                dist_prop: DistanceProposal = EuclideanLengthDistanceProposal(),
                                **kwargs) -> Tuple[Union[None, np.ndarray], float]:
    """
    Samples a single particle from the (distance discretised) optimal proposal.
    :param graph: encodes road network, simplified and projected to UTM
    :param particle: single element of MMParticles.particles
    :param new_observation: cartesian coordinate in UTM
    :param time_interval: time between last observation and newly received observation
    :param mm_model: MapMatchingModel
    :param full_smoothing: if True returns full trajectory
        otherwise returns only x_t-1 to x_t
    :param d_refine: metres, resolution of distance discretisation
    :param dist_expand: metres, how far to expand space away from sampled distance
    :param dist_prop: class that contains methods for sampling and evalutaing pdf/cdf of distance proposal
    :return: tuple, (particle, unnormalised weight)
    """
    if particle is None:
        return None, 0.

    # Extract all possible routes from previous position
    start_position = particle[-1:].copy()
    start_position[0, -1] = 0

    # Cartesianise
    cart_start = start_position[0, 5:7]

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
    discretised_routes_indices_list = []
    discretised_routes_list = []
    for i, route in enumerate(routes):
        # All possible end positions of route
        discretised_edge_matrix = discretise_edge(graph, route[-1, 1:4], d_refine)

        if route.shape[0] == 1:
            discretised_edge_matrix = discretised_edge_matrix[discretised_edge_matrix[:, 0] >= particle[-1, 4]]
            discretised_edge_matrix[:, -1] -= discretised_edge_matrix[-1, -1]
        else:
            discretised_edge_matrix[:, -1] += route[-2, -1]

        # Track route index and append to list
        if discretised_edge_matrix is not None and len(discretised_edge_matrix) > 0:
            discretised_routes_indices_list += [np.ones(discretised_edge_matrix.shape[0], dtype=int) * i]
            discretised_routes_list += [discretised_edge_matrix]

    # Concatenate into numpy.ndarray
    discretised_routes_indices = np.concatenate(discretised_routes_indices_list)
    discretised_routes = np.concatenate(discretised_routes_list)

    # Remove points outside range
    inside_range_indices = np.logical_and(discretised_routes[:, -1] >= dist_range[0],
                                          discretised_routes[:, -1] < dist_range[1])
    discretised_routes_indices = discretised_routes_indices[inside_range_indices]
    discretised_routes = discretised_routes[inside_range_indices]

    # Likelihood evaluations
    likelihood_evals = mm_model.likelihood_evaluate(discretised_routes[:, 1:3], new_observation)

    # Trim
    positive_lik_inds = likelihood_evals > 0
    discretised_routes_indices = discretised_routes_indices[positive_lik_inds]
    discretised_routes = discretised_routes[positive_lik_inds]
    likelihood_evals = likelihood_evals[positive_lik_inds]

    # Distance prior evals
    distance_prior_evals = mm_model.distance_prior_evaluate(discretised_routes[:, -1], time_interval)

    # Intersection prior evals
    route_intersection_prior_evals = intersection_prior_evaluate(routes, mm_model)

    # Deviation prior evals
    deviation_prior_evals = mm_model.deviation_prior_evaluate(particle[-1, 5:7],
                                                              discretised_routes[:, 1:3],
                                                              discretised_routes[:, -1])

    # Calculate sample probabilities
    sample_probs = distance_prior_evals \
                   * route_intersection_prior_evals[discretised_routes_indices] \
                   * deviation_prior_evals \
                   * likelihood_evals

    # Normalising constant
    sample_probs_norm_const = sample_probs.sum()

    if sample_probs_norm_const < 1e-200:
        return None, 0.

    # Sample an edge and distance
    sampled_dis_route_index = np.random.choice(len(discretised_routes), 1, p=sample_probs / sample_probs_norm_const)[0]
    sampled_dis_route = discretised_routes[sampled_dis_route_index]

    # Append sampled route to old particle
    sampled_route = routes[discretised_routes_indices[sampled_dis_route_index]]
    out_particle = process_proposal_output(particle, sampled_route, sampled_dis_route, time_interval, full_smoothing)

    # Sampled position distance
    selected_dist = sampled_dis_route[-1]

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
    discretised_check_routes_indices_list = []
    discretised_check_routes_list = []
    for i, route in enumerate(check_routes):
        # All possible end positions of route
        discretised_edge_matrix = discretise_edge(graph, route[-1, 1:4], d_refine)

        if route.shape[0] == 1:
            discretised_edge_matrix = discretised_edge_matrix[discretised_edge_matrix[:, 0] >= particle[-1, 4]]
            discretised_edge_matrix[:, -1] -= discretised_edge_matrix[-1, -1]
        else:
            discretised_edge_matrix[:, -1] += route[-2, -1]

        # Track route index and append to list
        if discretised_edge_matrix is not None and len(discretised_edge_matrix) > 0:
            discretised_check_routes_indices_list += [np.ones(discretised_edge_matrix.shape[0], dtype=int) * i]
            discretised_check_routes_list += [discretised_edge_matrix]

    # Concatenate into numpy.ndarray
    discretised_check_routes_indices = np.concatenate(discretised_routes_indices_list)
    discretised_check_routes = np.concatenate(discretised_routes_list)

    # Remove points outside range
    inside_check_range_indices = np.logical_and(discretised_check_routes[:, -1] >= dist_check_range[0],
                                                discretised_check_routes[:, -1] <= dist_check_range[1])
    discretised_check_routes_indices = discretised_check_routes_indices[inside_check_range_indices]
    discretised_check_routes = discretised_check_routes[inside_check_range_indices]

    # Likelihood evaluations
    likelihood_check_evals = mm_model.likelihood_evaluate(discretised_check_routes[:, 1:3], new_observation)

    # Trim
    positive_lik_check_inds = likelihood_check_evals > 0
    discretised_check_routes_indices = discretised_check_routes_indices[positive_lik_check_inds]
    discretised_check_routes = discretised_check_routes[positive_lik_check_inds]
    likelihood_check_evals = likelihood_check_evals[positive_lik_check_inds]

    # Distance prior evals
    distance_prior_check_evals = mm_model.distance_prior_evaluate(discretised_check_routes[:, -1], time_interval)

    # Intersection prior evals
    route_intersection_prior_check_evals = intersection_prior_evaluate(check_routes, mm_model)

    # Deviation prior evals
    deviation_prior_check_evals = mm_model.deviation_prior_evaluate(particle[-1, 5:7],
                                                                    discretised_check_routes[:, 1:3],
                                                                    discretised_check_routes[:, -1])

    # Calculate sample probabilities
    dis_check_probs = distance_prior_check_evals \
                      * route_intersection_prior_check_evals[discretised_check_routes_indices] \
                      * deviation_prior_check_evals \
                      * likelihood_check_evals

    # All possible (discrete) distances
    all_check_distances = np.unique(discretised_check_routes[:, -1])

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
def aux_dist_expand_weight(discretised_check_routes: np.ndarray,
                           dis_check_probs: np.ndarray,
                           poss_min_strata: np.ndarray,
                           poss_max_strata: np.ndarray,
                           mid_strata_cdf_evals: np.ndarray) -> float:
    """
    For a series of strata, sum over sampling probabilities for all positions within strata
    :param discretised_check_routes: alpha, x, y, dist array
    :param dis_check_probs: position sampling probabilities
    :param poss_min_strata: shape = (_,) lower distance bound for each stratum
    :param poss_max_strata: shape = (_,) upper distance bound for each stratum
    :param mid_strata_cdf_evals: strata sampling probabilities
    :return: resulting weight
    """
    unnorm_weight_denom = 0
    for i in range(len(poss_min_strata)):
        if mid_strata_cdf_evals[i] == 0:
            continue

        min_stratum = poss_min_strata[i]
        max_stratum = poss_max_strata[i]

        partial_prob_denom = np.sum(dis_check_probs[np.logical_and(discretised_check_routes[:, -1] >= min_stratum,
                                                                   discretised_check_routes[:, -1] <= max_stratum)])

        if partial_prob_denom == 0:
            continue

        unnorm_weight_denom += mid_strata_cdf_evals[i] / partial_prob_denom

    return unnorm_weight_denom


def dist_then_edge_proposal(graph: MultiDiGraph,
                            particle: np.ndarray,
                            new_observation: np.ndarray,
                            time_interval: float,
                            mm_model: MapMatchingModel,
                            full_smoothing: bool = True,
                            dist_prop: DistanceProposal = EuclideanLengthDistanceProposal(),
                            **kwargs) -> Tuple[Union[None, np.ndarray], float]:
    """
    Samples a single particle from the (distance discretised) optimal proposal.
    :param graph: encodes road network, simplified and projected to UTM
    :param particle: single element of MMParticles.particles
    :param new_observation: cartesian coordinate in UTM
    :param time_interval: time between last observation and newly received observation
    :param mm_model: MapMatchingModel
    :param full_smoothing: if True returns full trajectory
        otherwise returns only x_t-1 to x_t
    :param dist_prop: class that contains methods for sampling and evalutaing pdf/cdf of distance proposal
    :return: tuple, (particle, unnormalised weight)
    """
    if particle is None:
        return None, 0.

    # Extract position at last observation time
    start_position = particle[-1:].copy()
    start_position[0, -1] = 0

    # Cartesianise
    cart_start = start_position[0, 5:7]

    # Get Euclidean distance between particle and new observation
    euclid_dist = np.linalg.norm(cart_start - new_observation)

    # Sample distance
    dist_samp = dist_prop.sample(euclid_dist, **kwargs)

    # Get possible routes of length dist_samp
    routes = get_possible_routes(graph, start_position, dist_samp, all_routes=False)

    # No routes implies reached dead end
    if len(routes) == 0:
        return process_proposal_output(particle,
                                       particle[-1:],
                                       particle[-1, [4, 5, 6, 8]],
                                       time_interval,
                                       full_smoothing), 0

    # Initiate cartesian position of end of routes
    routes_end_cart_pos = np.zeros((len(routes), 2))

    # Iterate through routes
    for i, route in enumerate(routes):

        if route is None:
            continue

        end_position = route[-1]

        end_geom = get_geometry(graph, end_position[1:4])

        routes_end_cart_pos[i] = route[-1, 5:7] = edge_interpolate(end_geom, end_position[4])

    # Intersection prior
    intersection_probs = intersection_prior_evaluate(routes, mm_model)

    # Unnormalised sample probablilites
    route_sample_weights = intersection_probs * mm_model.deviation_prior_evaluate(particle[-1, 5:7],
                                                                                  routes_end_cart_pos,
                                                                                  dist_samp) \
                           * mm_model.likelihood_evaluate(routes_end_cart_pos, new_observation)

    # Normalising constant
    prob_y_given_x_prev_d = np.sum(route_sample_weights)

    if prob_y_given_x_prev_d == 0:
        return None, 0

    # Normalise
    route_sample_weights /= prob_y_given_x_prev_d

    # Sample route
    sampled_route_ind = np.random.choice(len(routes), 1, p=route_sample_weights)[0]
    sampled_route = routes[sampled_route_ind]

    out_particle = process_proposal_output(particle,
                                           sampled_route,
                                           sampled_route[-1, [4, 5, 6, 8]],
                                           time_interval,
                                           full_smoothing)

    dist_samp_prop_eval = 1e-7 if dist_samp == 0 else dist_samp

    # Weight
    weight = prob_y_given_x_prev_d * mm_model.distance_prior_evaluate(dist_samp, time_interval) \
             / dist_prop.pdf(dist_samp_prop_eval, euclid_dist, **kwargs)

    if np.isnan(weight):
        weight = 0.

    return out_particle, weight
