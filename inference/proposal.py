########################################################################################################################
# Module: inference/proposal.py
# Description: Proposal mechanisms to extend particles (series of positions/edges/distances) and re-weight
#              in light of a newly received observation.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import numpy as np

from tools.edges import get_geometry, edge_interpolate
from inference.model import distance_prior, default_d_max, pdf_gamma_mv


def get_possible_routes(graph, in_route, dist, all=False):
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
    :param all: bool
        if true return all routes possible <= d
        else return only routes of length d
    :return: list of numpy.ndarrays, length = n_samps
        each numpy.ndarray with shape = (_, 7) as in_route
        each array describes a possible route
    """
    # Extract final position from inputted route
    start_edge_and_position = in_route[-1]

    # Extract edge geometry
    start_edge_geom = get_geometry(graph, start_edge_and_position[1:4])

    # Distance left on edge before intersection
    # Use NetworkX length rather than OSM length
    distance_left_on_edge = (1 - start_edge_and_position[4]) * start_edge_geom.length

    if distance_left_on_edge > dist:
        # Remain on edge
        # Propagate and return
        start_edge_and_position[4] += dist / start_edge_geom.length
        start_edge_and_position[6] += dist
        return [in_route]
    else:
        # Reach intersection at end of edge
        # Propagate to intersection and recurse
        dist -= distance_left_on_edge
        start_edge_and_position[4] = 1.
        start_edge_and_position[6] += distance_left_on_edge

        intersection_edges = np.atleast_2d([[u, v, k] for u, v, k in graph.out_edges(start_edge_and_position[2],
                                                                                     keys=True)])

        if intersection_edges.shape[1] == 0:
            # Dead-end and one-way
            if all:
                return [in_route]

        n_inter = max(1, np.sum(intersection_edges[:, 1] != start_edge_and_position[0]))

        start_edge_and_position[5] = n_inter

        if len(intersection_edges) == 1 and intersection_edges[0][1] == start_edge_and_position[0]:
            # Dead-end and two-way -> Only option is u-turn
            if all:
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

                    new_routes += get_possible_routes(graph, new_route, dist, all)
            if all:
                return [in_route] + new_routes
            else:
                return new_routes


def discretise_route(graph, route, discrete_distances):
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
    :param discrete_distances: numpy.ndarray, shape = (_,)
        metres
        distance discretisation sequence
    :return: numpy.ndarray, shape = (_,5)
        columns alpha, d, n_inter, x, y
            alpha: in [0,1], position along edge
            d: metres, distance travelled since previous observation time
            1/prod(n_inter): in [0,1], product of number of options at intersections treaversed
            x: metres, cartesian x position
            y: metres, cartesian y position
    """
    # Minimum distance travelled to be in route
    route_d_min = 0 if route.shape[0] == 1 else route[-2, -1]

    # Maximum distance travelled to be in route
    route_d_max = route[-1, -1]

    # Possible discrete distance for route
    route_ds = discrete_distances[(route_d_min <= discrete_distances) & (discrete_distances < route_d_max)]

    if len(route_ds) > 0:
        # Initiatilisation, route index and distance
        dis_route_matrix = np.zeros((len(route_ds), 5))
        dis_route_matrix[:, 1] = route_ds

        # Product of 1 / number of intersection choices
        intersection_col = route[:-1, 5]
        dis_route_matrix[:, 2] = np.prod(1 / intersection_col[intersection_col > 1])

        # Get last edge geometry
        last_edge_geom = get_geometry(graph, route[-1, 1:4])

        # Convert distances to alphas
        if route.shape[0] == 1:
            # Stayed on same edge
            dis_route_matrix[:, 0] = route[0, 4] - (route[0, -1] - route_ds) / last_edge_geom.length
        else:
            # New edge
            dis_route_matrix[:, 0] = (route_ds - route_d_min) / last_edge_geom.length

        # Cartesianise positions
        dis_route_matrix[:, 3:] = np.array([edge_interpolate(last_edge_geom, alpha)
                                            for alpha in dis_route_matrix[:, 0]])

    else:
        dis_route_matrix = None
    return dis_route_matrix


def optimal_proposal(graph, particle, new_observation, time_interval, gps_sd, d_refine=1, d_max=None):
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
    :param gps_sd: float
        metres
        standard deviation of GPS noise
    :param d_refine: float
        metres
        resolution of distance discretisation
        increase of speed, decrease for accuracy
    :param d_max: float
        metres
        maximum distance for vehicle to travel in time_interval
        defaults to time_interval * 35 (35m/s ≈ 78mph)
    :return: tuple, particle with appended proposal and weight
        particle: numpy.ndarray, shape = (_, 7)
        weight: float, not normalised
    """
    # Default d_max
    d_max = default_d_max(d_max, time_interval)

    # Discretise distance
    d_discrete = np.arange(0.01, d_max, d_refine)

    # Extract all possible routes from previous position
    start_position = particle[-1:].copy()
    start_position[0, -1] = 0
    possible_routes = get_possible_routes(graph, start_position, d_max, all=True)

    # Get all possible positions on each route
    discretised_routes_list = []
    for i, route in enumerate(possible_routes):
        # All possible end positions of route
        discretised_route_matrix = discretise_route(graph, route, d_discrete)

        # Track route index and append to list
        if discretised_route_matrix is not None:
            discretised_routes_list += [np.append(np.ones((discretised_route_matrix.shape[0], 1)) * i,
                                                  discretised_route_matrix, axis=1)]

    # Concatenate into numpy.ndarray
    discretised_routes = np.concatenate(discretised_routes_list)

    # Distance from end position to observation
    obs_distance_sqr = np.sum((discretised_routes[:, 4:] - new_observation) ** 2, axis=1)

    # Calculate sample probabilities
    sample_probs = distance_prior(discretised_routes[:, 2], time_interval) \
        * discretised_routes[:, 3] \
        * np.exp(- 0.5 / gps_sd ** 2 * obs_distance_sqr)

    # Normalising constant = p(y_m | x_m-1^j)
    sample_probs_norm_const = sum(sample_probs)

    # Sample an edge and distance
    sampled_dis_route_index = np.random.choice(len(discretised_routes), 1, p=sample_probs / sample_probs_norm_const)[0]
    sampled_dis_route = discretised_routes[sampled_dis_route_index]

    # Append sampled route to old particle
    sampled_route = possible_routes[int(sampled_dis_route[0])]
    new_route_append = sampled_route
    new_route_append[0, 0] = 0
    new_route_append[-1, 0] = particle[-1, 0] + time_interval
    new_route_append[-1, 4] = sampled_dis_route[1]
    new_route_append[-1, 5] = 0
    new_route_append[-1, 6] = sampled_dis_route[2]

    return np.append(particle, new_route_append, axis=0), sample_probs_norm_const


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
        :param kwargs:
        :return: float
            single sample from distance proposal
        """
        raise NotImplementedError

    def pdf(self, x, *args, **kwargs):
        """
        Evaluate proposal pdf.
        :param x: value(s) to be evaluated
        :param args:
        :param kwargs:
        :return: float or np.ndarray like x
            pdf evaluation
        """
        raise NotImplementedError


class EuclideanLengthDistanceProposal:
    def sample(self, euclidean_distance, var=10):
        gamma_beta = euclidean_distance / var
        gamma_alpha = euclidean_distance * gamma_beta
        return np.random.gamma(gamma_alpha, 1/gamma_beta)

    def pdf(self, x, euclidean_distance, var=10):
        return pdf_gamma_mv(x, euclidean_distance, var)


def dist_then_edge_proposal(graph, particle, new_observation, time_interval, gps_sd,
                            dist_prop=EuclideanLengthDistanceProposal(), **kwargs):
    """
    Samples distance first then chooses edges travelled. Outputs updated particle and (unnormalised) weight.
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
    :param gps_sd: float
        metres
        standard deviation of GPS noise
    :param dist_prop: DistanceProposal
        class that can propose a distance and evaluate pdf of said proposal
    :param **kwargs: optional parameters to pass to distance proposal
        i.e. variance of proposal (var=10)
    :return: tuple, particle with appended proposal and weight
        particle: numpy.ndarray, shape = (_, 7)
        weight: float, not normalised
    """

    # Extract position at last observation time
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

    # Get possible routes of length dist_samp
    routes = get_possible_routes(graph, start_position, dist_samp, all=False)

    # Initiate cartesian position of end of routes
    routes_end_cart_pos = np.zeros((len(routes), 2))

    # Initiate prod 1/number of choices at intersection
    intersection_probs = np.zeros(len(routes))

    # Iterate through routes
    for i, route in enumerate(routes):
        end_position = route[-1]

        end_geom = get_geometry(graph, end_position[1:4])

        routes_end_cart_pos[i] = edge_interpolate(end_geom, end_position[4])

        intersection_col = route[:-1, 5]
        intersection_probs[i] = np.prod(1 / intersection_col[intersection_col > 1])

    # Distances of end points to new observation
    obs_distances_sqr = np.sum((routes_end_cart_pos - new_observation) ** 2, axis=1)

    # Unnormalised sample probablilites
    route_sample_weights = np.exp(- 0.5 / gps_sd ** 2 * obs_distances_sqr) * intersection_probs

    # Normalising constant
    prob_y_given_x_prev_d = np.sum(route_sample_weights)

    # Normalise
    route_sample_weights /= prob_y_given_x_prev_d

    # Sample route
    sampled_route_ind = np.random.choice(len(routes), 1, p=route_sample_weights)[0]
    sampled_route = routes[sampled_route_ind]

    # Append to old particle
    sampled_route[0, 0] = 0
    sampled_route[-1, 0] = particle[-1, 0] + time_interval
    sampled_route[-1, 5] = 0
    out_particle = np.append(particle, sampled_route, axis=0)

    # Weight
    weight = prob_y_given_x_prev_d * distance_prior(dist_samp, time_interval)\
        / dist_prop.pdf(dist_samp, euclid_dist, **kwargs)

    return out_particle, weight

