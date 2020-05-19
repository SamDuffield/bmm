import numpy as np

import bmm


# Function to sample a random point on the graph
def random_positions(graph, n=1):
    edges_arr = np.array(graph.edges)
    n_edges = len(edges_arr)

    edge_selection_indices = np.random.choice(n_edges, n)
    edge_selection = edges_arr[edge_selection_indices]

    random_alphas = np.random.uniform(size=(n, 1))

    positions = np.concatenate((edge_selection, random_alphas), axis=1)
    return positions


# Function to sample a route (given a start position, route length and time_interval (assumed constant))
def sample_route(graph, model, time_interval, length, start_position=None):
    route = np.zeros((1, 7))

    if start_position is None:
        start_position = random_positions(graph, 1)

    route[0, 1:5] = start_position

    for t in range(1, length):
        prev_pos = route[-1:].copy()
        prev_pos[0, 0] = 0

        # Sample a distance
        sampled_dist = model.distance_prior_sample(time_interval)

        # Evaluate all possible routes
        possible_routes = bmm.get_possible_routes(graph, prev_pos, sampled_dist)

        if possible_routes is None or all(p is None for p in possible_routes):
            return route

        # Prior route probabilities given distance
        num_poss_routes = len(possible_routes)
        if num_poss_routes == 0:
            return route
        possible_routes_probs = np.zeros(num_poss_routes)
        for i in range(num_poss_routes):
            if possible_routes[i] is None:
                continue

            intersection_col = possible_routes[i][:-1, 5]
            possible_routes_probs[i] = np.prod(1 / intersection_col[intersection_col > 1]) \
                                       * model.intersection_penalisation ** len(intersection_col)

        # Normalise
        possible_routes_probs /= np.sum(possible_routes_probs)

        # Choose one
        sampled_route_index = np.random.choice(num_poss_routes, 1, p=possible_routes_probs)[0]
        sampled_route = possible_routes[sampled_route_index]

        sampled_route[-1, 0] = route[-1, 0] + time_interval

        route = np.append(route, sampled_route, axis=0)

    return route


# RMSE given particle cloud
def rmse(graph, particles, truth, each_time=False):
    # N x T x 2
    obs_time_particles = np.zeros((len(particles), len(truth), 2))
    for i, particle in enumerate(particles):
        obs_time_particles[i] = bmm.cartesianise_path(graph, particle, t_column=True, observation_time_only=True)

    squared_error = np.square(obs_time_particles - truth)

    if each_time:
        return np.sqrt(np.mean(squared_error), axis=(0,2))
    else:
        return np.sqrt(np.mean(squared_error))

