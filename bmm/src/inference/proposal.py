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
from bmm.src.inference.model import MapMatchingModel


@lru_cache(maxsize=2 ** 8)
def get_out_edges(graph: MultiDiGraph,
                  node: int) -> np.ndarray:
    """
    Extracts out edges from a given node
    :param graph: encodes road network, simplified and projected to UTM
    :param node: cam_graph index to a single node
    :return: array with columns u, v, k with u = node
    """
    return np.atleast_2d([[u, v, k] for u, v, k in graph.out_edges(node, keys=True)])


@lru_cache(maxsize=2 ** 7)
def get_possible_routes_all_cached(graph: MultiDiGraph,
                                   u: int,
                                   v: int,
                                   k: int,
                                   d_max: float,
                                   num_inter_cut_off: int) -> list:
    in_route = np.array([[0., u, v, k, 1., 0., 0., 0.]])
    return get_possible_routes(graph, in_route, d_max, all_routes=True, num_inter_cut_off=num_inter_cut_off)


def get_all_possible_routes_overshoot(graph: MultiDiGraph,
                                      in_edge: np.ndarray,
                                      d_max: float,
                                      num_inter_cut_off: int = np.inf) -> list:
    in_edge_geom = get_geometry(graph, in_edge[-1, 1:4])
    in_edge_length = in_edge_geom.length
    extra_dist = (1 - in_edge[-1, 4]) * in_edge_length

    if extra_dist > d_max:
        return get_possible_routes(graph, in_edge, d_max, all_routes=True, num_inter_cut_off=num_inter_cut_off)

    all_possible_routes_overshoot = get_possible_routes_all_cached(graph, *in_edge[-1, 1:4],
                                                                   d_max, num_inter_cut_off)

    out_routes = []
    for i in range(len(all_possible_routes_overshoot)):
        temp_route = all_possible_routes_overshoot[i].copy()
        temp_route[:, -1] += extra_dist
        out_routes.append(temp_route)
    return out_routes


def get_possible_routes(graph: MultiDiGraph,
                        in_route: np.ndarray,
                        dist: float,
                        all_routes: bool = False,
                        num_inter_cut_off: int = np.inf) -> list:
    """
    Given a route so far and maximum distance to travel, calculate and return all possible routes on cam_graph.
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

    if intersection_edges.shape[1] == 0 or len(in_route) >= num_inter_cut_off:
        # Dead-end and one-way or exceeded max intersections
        if all_routes:
            return [in_route]
        else:
            return [None]

    if len(intersection_edges) == 1 and intersection_edges[0][1] == start_edge_and_position[1] \
            and intersection_edges[0][2] == start_edge_and_position[3]:
        # Dead-end and two-way -> Only option is u-turn
        if all_routes:
            return [in_route]
    else:
        new_routes = []
        for new_edge in intersection_edges:
            # If not u-turn or loop continue route search on new edge
            if (not (new_edge[1] == start_edge_and_position[1] and new_edge[2] == start_edge_and_position[3])) \
                    and not (new_edge == in_route[:, 1:4]).all(1).any():
                add_edge = np.array([[0, *new_edge, 0, 0, 0, start_edge_and_position[-1]]])
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
    new_route_append[-1, -1] = sampled_dis_route[-1]

    if full_smoothing:
        return np.append(particle, new_route_append, axis=0)
    else:
        return np.append(particle[-1:], new_route_append, axis=0)


def optimal_proposal(graph: MultiDiGraph,
                     particle: np.ndarray,
                     new_observation: Union[None, np.ndarray],
                     time_interval: float,
                     mm_model: MapMatchingModel,
                     full_smoothing: bool = True,
                     d_refine: float = 1.,
                     d_max: float = None,
                     d_max_fail_multiplier: float = 1.5,
                     d_max_threshold: tuple = (0.9, 0.1),
                     num_inter_cut_off: int = None,
                     only_norm_const: bool = False,
                     store_norm_quants: bool = False,
                     resample_fails: bool = True) -> Union[Tuple[Union[None, np.ndarray],
                                                                 float,
                                                                 Union[float, np.ndarray]], float]:
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
    :param d_max: optional override of d_max = mm_model.d_max(time_interval)
    :param d_max_fail_multiplier: extension of d_max in case all probs are 0
    :param d_max_threshold: tuple defining when to extend d_max
        extend if total sample prob of distances > d_max * d_max_threshold[0] larger than d_max_threshold[1]
    :param num_inter_cut_off: maximum number of intersections to cross in the time interval
    :param only_norm_const: if true only return prior normalising constant (don't sample)
    :param store_norm_quants: whether to additionally return quantities needed for gradient EM step
        assuming deviation prior is used
    :param resample_fails: whether to return None (and induce later resampling of whole trajectory)
        if proposal fails to find route with positive probability
        if False assume distance=0
    :return: (particle, unnormalised weight, prior_norm) or (particle, unnormalised weight, dev_norm_quants)
    """
    if particle is None:
        return 0. if only_norm_const else (None, 0., 0.)

    if num_inter_cut_off is None:
        num_inter_cut_off = max(int(time_interval / 1.5), 10)

    if d_max is None:
        d_max = mm_model.d_max(time_interval)

    # Extract all possible routes from previous position
    start_position = particle[-1:].copy()
    start_position[0, -1] = 0
    # possible_routes = get_possible_routes(cam_graph, start_position, d_max, all_routes=True,
    #                                       num_inter_cut_off=num_inter_cut_off)
    possible_routes = get_all_possible_routes_overshoot(graph, start_position, d_max,
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

        discretised_edge_matrix = discretised_edge_matrix[discretised_edge_matrix[:, -1] < d_max + 1e-5]

        # Track route index and append to list
        if discretised_edge_matrix is not None and len(discretised_edge_matrix) > 0:
            discretised_routes_indices_list += [np.ones(discretised_edge_matrix.shape[0], dtype=int) * i]
            discretised_routes_list += [discretised_edge_matrix]

    # Concatenate into numpy.ndarray
    discretised_routes_indices = np.concatenate(discretised_routes_indices_list)
    discretised_routes = np.concatenate(discretised_routes_list)

    if len(discretised_routes) == 0 or (len(discretised_routes) == 1 and discretised_routes[0][-1] == 0):
        if only_norm_const:
            return 0
        if resample_fails:
            return None, 0., 0.
        else:
            sampled_dis_route = discretised_routes[0]

            # Append sampled route to old particle
            sampled_route = possible_routes[0]

            proposal_out = process_proposal_output(particle, sampled_route, sampled_dis_route, time_interval,
                                                   full_smoothing)
            return proposal_out, 0., 0.

    # Distance prior evals
    distances = discretised_routes[:, -1]
    distance_prior_evals = mm_model.distance_prior_evaluate(distances, time_interval)

    # Deviation prior evals
    deviation_prior_evals = mm_model.deviation_prior_evaluate(particle[-1, 5:7],
                                                              discretised_routes[:, 1:3],
                                                              discretised_routes[:, -1])

    # Normalise prior/transition probabilities
    prior_probs = distance_prior_evals * deviation_prior_evals

    prior_probs_norm_const = prior_probs.sum()
    if only_norm_const:
        if store_norm_quants:
            deviations = np.sqrt(np.sum((particle[-1, 5:7] - discretised_routes[:, 1:3]) ** 2, axis=1))
            deviations = np.abs(deviations - discretised_routes[:, -1])

            # Z, dZ/d(dist_params), dZ/d(deviation_beta)
            dev_norm_quants = np.array([prior_probs_norm_const,
                                        *np.sum(mm_model.distance_prior_gradient(distances, time_interval)
                                                .reshape(len(mm_model.distance_params), len(distances))
                                                * deviation_prior_evals, axis=-1),
                                        -np.sum(deviations
                                                * distance_prior_evals
                                                * deviation_prior_evals)
                                        ])
            return dev_norm_quants
        else:
            return prior_probs_norm_const
    prior_probs /= prior_probs_norm_const

    # Likelihood evaluations
    likelihood_evals = mm_model.likelihood_evaluate(discretised_routes[:, 1:3], new_observation)

    # Calculate sample probabilities
    sample_probs = prior_probs[likelihood_evals > 0] * likelihood_evals[likelihood_evals > 0]
    # sample_probs = prior_probs * likelihood_evals

    # p(y_m | x_m-1^j)
    prop_weight = sample_probs.sum()

    model_d_max = mm_model.d_max(time_interval)

    if prop_weight < 1e-100 \
            or (np.sum(sample_probs[np.where(distances[likelihood_evals > 0]
                                             > (d_max * d_max_threshold[0]))[0]])/prop_weight > d_max_threshold[1]\
                and (not d_max > model_d_max)):
        if (d_max - np.max(distances)) < d_refine + 1e-5 \
                and d_max_fail_multiplier > 1 and (not d_max > model_d_max):
            return optimal_proposal(graph,
                                    particle,
                                    new_observation,
                                    time_interval,
                                    mm_model,
                                    full_smoothing,
                                    d_refine,
                                    d_max=d_max * d_max_fail_multiplier,
                                    num_inter_cut_off=num_inter_cut_off,
                                    only_norm_const=only_norm_const,
                                    store_norm_quants=store_norm_quants,
                                    resample_fails=resample_fails)
        if resample_fails:
            proposal_out = None
        else:
            sampled_dis_route_index = np.where(discretised_routes[:, -1] == 0)[0][0]
            sampled_dis_route = discretised_routes[sampled_dis_route_index]

            # Append sampled route to old particle
            sampled_route = possible_routes[discretised_routes_indices[sampled_dis_route_index]]

            proposal_out = process_proposal_output(particle, sampled_route, sampled_dis_route, time_interval,
                                                   full_smoothing)
        prop_weight = 0.
    else:
        # Sample an edge and distance
        sampled_dis_route_index = np.random.choice(len(sample_probs), 1, p=sample_probs / prop_weight)[0]
        sampled_dis_route = discretised_routes[likelihood_evals > 0][sampled_dis_route_index]

        # Append sampled route to old particle
        sampled_route = possible_routes[discretised_routes_indices[likelihood_evals > 0][sampled_dis_route_index]]

        proposal_out = process_proposal_output(particle, sampled_route, sampled_dis_route, time_interval,
                                               full_smoothing)

    if store_norm_quants:
        deviations = np.sqrt(np.sum((particle[-1, 5:7] - discretised_routes[:, 1:3]) ** 2, axis=1))
        deviations = np.abs(deviations - discretised_routes[:, -1])

        # Z, dZ/d(dist_params), dZ/d(deviation_beta)
        dev_norm_quants = np.array([prior_probs_norm_const,
                                    *np.sum(mm_model.distance_prior_gradient(distances, time_interval)
                                            .reshape(len(mm_model.distance_params), len(distances))
                                            * deviation_prior_evals, axis=-1),
                                    -np.sum(deviations
                                            * distance_prior_evals
                                            * deviation_prior_evals)
                                    ])

        return proposal_out, prop_weight, dev_norm_quants
    else:
        return proposal_out, prop_weight, prior_probs_norm_const
