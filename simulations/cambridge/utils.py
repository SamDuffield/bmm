import functools
import gc

import numpy as np
import osmnx as ox
from networkx import write_gpickle, read_gpickle

import bmm


def download_cambridge_graph(save_path):

    cambridge_ll_bbox = [52.245, 52.150, 0.220, 0.025]
    raw_graph = ox.graph_from_bbox(*cambridge_ll_bbox,
                                   truncate_by_edge=True,
                                   simplify=False,
                                   network_type='drive')

    projected_graph = ox.project_graph(raw_graph)

    simplified_graph = ox.simplify_graph(projected_graph)

    write_gpickle(simplified_graph, save_path)


# Load cam_graph of Cambridge
def load_graph(path):
    graph = read_gpickle(path)
    return graph


# Clear lru_cache
def clear_cache():
    gc.collect()
    wrappers = [
        a for a in gc.get_objects()
        if isinstance(a, functools._lru_cache_wrapper)]

    for wrapper in wrappers:
        wrapper.cache_clear()


# Function to sample a random point on the cam_graph
def random_positions(graph, n=1):
    edges_arr = np.array(graph.edges)
    n_edges = len(edges_arr)

    edge_selection_indices = np.random.choice(n_edges, n)
    edge_selection = edges_arr[edge_selection_indices]

    random_alphas = np.random.uniform(size=(n, 1))

    positions = np.concatenate((edge_selection, random_alphas), axis=1)
    return positions


# Function to sample a route (given a start position, route length and time_interval (assumed constant))
def sample_route(graph, model, time_interval, length, start_position=None, cart_route=False, observations=False,
                 d_refine=1, num_inter_cut_off=None, num_pos_route_cap=np.inf):
    route = np.zeros((1, 8))

    if start_position is None:
        start_position = random_positions(graph, 1)

    route[0, 1:5] = start_position
    start_geom = bmm.get_geometry(graph, start_position[0, :3])
    route[0, 5:7] = bmm.src.tools.edges.edge_interpolate(start_geom, start_position[0, 3])

    d_max = model.d_max(time_interval)
    if num_inter_cut_off is None:
        num_inter_cut_off = max(int(time_interval / 1.5), 10)

    for t in range(1, length):
        prev_pos = route[-1:].copy()
        prev_pos[0, -1] = 0

        possible_routes = bmm.get_possible_routes(graph, prev_pos, d_max, all_routes=True,
                                                  num_inter_cut_off=num_inter_cut_off)

        if len(possible_routes) > num_pos_route_cap:
            break

        # Get all possible positions on each route
        discretised_routes_indices_list = []
        discretised_routes_list = []
        for i, sub_route in enumerate(possible_routes):
            # All possible end positions of route
            discretised_edge_matrix = bmm.discretise_edge(graph, sub_route[-1, 1:4], d_refine)

            if sub_route.shape[0] == 1:
                discretised_edge_matrix = discretised_edge_matrix[discretised_edge_matrix[:, 0] >= route[-1, 4]]
                discretised_edge_matrix[:, -1] -= discretised_edge_matrix[-1, -1]
            else:
                discretised_edge_matrix[:, -1] += sub_route[-2, -1]

            discretised_edge_matrix = discretised_edge_matrix[discretised_edge_matrix[:, -1] < d_max + 1e-5]

            # Track route index and append to list
            if discretised_edge_matrix is not None and len(discretised_edge_matrix) > 0:
                discretised_routes_indices_list += [np.ones(discretised_edge_matrix.shape[0], dtype=int) * i]
                discretised_routes_list += [discretised_edge_matrix]

        if len(discretised_routes_indices_list) == 0 \
                or (len(discretised_routes_indices_list) == 1 and len(discretised_routes_indices_list[0]) == 1
                    and discretised_routes_list[0][0, -1] == 0.):
            break

        # Concatenate into numpy.ndarray
        discretised_routes_indices = np.concatenate(discretised_routes_indices_list)
        discretised_routes = np.concatenate(discretised_routes_list)

        # Distance prior evals
        distances = discretised_routes[:, -1]

        if np.max(distances) < d_max * 0.8:
            break

        distance_prior_evals = model.distance_prior_evaluate(distances, time_interval)

        # Deviation prior evals
        deviation_prior_evals = model.deviation_prior_evaluate(route[-1, 5:7],
                                                               discretised_routes[:, 1:3],
                                                               discretised_routes[:, -1])

        # Normalise prior/transition probabilities
        prior_probs = distance_prior_evals * deviation_prior_evals
        prior_probs /= prior_probs.sum()

        # Choose one
        sampled_route_index = np.random.choice(len(discretised_routes), 1, p=prior_probs)[0]
        sampled_route = possible_routes[discretised_routes_indices[sampled_route_index]]

        sampled_route[0, 0] = 0
        sampled_route[0, 5:7] = 0
        sampled_route[-1, 0] = route[-1, 0] + time_interval
        sampled_route[-1, 4:7] = discretised_routes[sampled_route_index][0:3]
        sampled_route[-1, -1] = discretised_routes[sampled_route_index][-1]

        route = np.append(route, sampled_route, axis=0)

        if np.all(route[-3:, -1] == 0):
            break

    if cart_route or observations:
        cartesianised_route_out = bmm.cartesianise_path(graph, route, t_column=True, observation_time_only=True)

        if observations:
            observations_out = cartesianised_route_out \
                               + model.gps_sd * np.random.normal(size=cartesianised_route_out.shape)
            if cart_route:
                return route, cartesianised_route_out, observations_out
            else:
                return route, observations_out
        else:
            return route, cartesianised_route_out
    else:
        return route

