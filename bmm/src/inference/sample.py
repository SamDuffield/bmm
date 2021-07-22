########################################################################################################################
# Module: inference/sample.py
# Description: Generate route and polyline from map-matching model.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################
import warnings
from typing import Union, Tuple

import numpy as np
from networkx.classes import MultiDiGraph

from bmm.src.inference import proposal
from bmm.src.tools import edges
from bmm.src.inference.model import MapMatchingModel, ExponentialMapMatchingModel
from bmm.src.inference.smc import get_time_interval_array


def random_positions(graph: MultiDiGraph,
                     n: int = 1) -> np.ndarray:
    """
    Sample random positions on a graph.
    :param graph: encodes road network, simplified and projected to UTM
    :param n: int number of positions to sample, default 1
    :return: array of positions (u, v, key, alpha) - shape (n, 4)
    """
    edges_arr = np.array(graph.edges)
    n_edges = len(edges_arr)
    edge_selection_indices = np.random.choice(n_edges, n)
    edge_selection = edges_arr[edge_selection_indices]
    random_alphas = np.random.uniform(size=(n, 1))
    positions = np.concatenate((edge_selection, random_alphas), axis=1)
    return positions


def sample_route(graph: MultiDiGraph,
                 timestamps: Union[float, np.ndarray],
                 num_obs: int = None,
                 mm_model: MapMatchingModel = ExponentialMapMatchingModel(),
                 d_refine: float = 1.,
                 start_position: np.ndarray = None,
                 num_inter_cut_off: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs offline map-matching. I.e. receives a full polyline and returns an equal probability collection
    of trajectories.
    Forward-filtering backward-simulation implementation - no fixed-lag approximation needed for offline inference.
    :param graph: encodes road network, simplified and projected to UTM
    :param timestamps: seconds
        either float if all times between observations are the same, or a series of timestamps in seconds/UNIX timestamp
    :param num_obs: int length of observed polyline to generate
    :param mm_model: MapMatchingModel
    :param d_refine: metres, resolution of distance discretisation
    :param start_position: optional start position; array (u, v, k, alpha)
    :param num_inter_cut_off: maximum number of intersections to cross in the time interval
    :return: tuple with sampled route (array with same shape as a single MMParticles)
        and polyline (array with shape (num_obs, 2))
    """

    if isinstance(timestamps, np.ndarray):
        num_obs = len(timestamps) + 1

    time_interval_arr = get_time_interval_array(timestamps, num_obs)

    if start_position is None:
        start_position = random_positions(graph, 1)[0]

    start_geom = edges.get_geometry(graph, start_position)
    start_coords = edges.edge_interpolate(start_geom, start_position[-1])

    full_sampled_route = np.concatenate([[0.], start_position, start_coords, [0.]])[np.newaxis]

    for k in range(num_obs - 1):
        time_interval = time_interval_arr[k]
        d_max = mm_model.d_max(time_interval)

        num_inter_cut_off_i = max(int(time_interval / 1.5), 10) if num_inter_cut_off is None else num_inter_cut_off

        prev_pos = full_sampled_route[-1:].copy()
        prev_pos[0, 0] = 0.
        prev_pos[0, -1] = 0.

        possible_routes = proposal.get_all_possible_routes_overshoot(graph, prev_pos, d_max,
                                                                     num_inter_cut_off=num_inter_cut_off_i)

        # Get all possible positions on each route
        discretised_routes_indices_list = []
        discretised_routes_list = []
        for i, route in enumerate(possible_routes):
            # All possible end positions of route
            discretised_edge_matrix = edges.discretise_edge(graph, route[-1, 1:4], d_refine)

            if route.shape[0] == 1:
                discretised_edge_matrix = discretised_edge_matrix[discretised_edge_matrix[:, 0]
                                                                  >= full_sampled_route[-1, 4]]
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
            warnings.warn('sample_route exited prematurely')
            break

        # Distance prior evals
        distances = discretised_routes[:, -1]
        distance_prior_evals = mm_model.distance_prior_evaluate(distances, time_interval)

        # Deviation prior evals
        deviation_prior_evals = mm_model.deviation_prior_evaluate(full_sampled_route[-1, 5:7],
                                                                  discretised_routes[:, 1:3],
                                                                  discretised_routes[:, -1])

        # Normalise prior/transition probabilities
        prior_probs = distance_prior_evals * deviation_prior_evals
        prior_probs_norm_const = prior_probs.sum()

        sampled_dis_route_index = np.random.choice(len(prior_probs), 1, p=prior_probs / prior_probs_norm_const)[0]
        sampled_dis_route = discretised_routes[sampled_dis_route_index]

        # Append sampled route to old particle
        sampled_route = possible_routes[discretised_routes_indices[sampled_dis_route_index]]

        full_sampled_route = proposal.process_proposal_output(full_sampled_route, sampled_route, sampled_dis_route,
                                                              time_interval, True)

    polyline = full_sampled_route[edges.observation_time_indices(full_sampled_route[:, 0]), 5:7] \
               + mm_model.gps_sd * np.random.normal(size=(num_obs, 2))

    return full_sampled_route, polyline

