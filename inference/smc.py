########################################################################################################################
# Module: inference/smc.py
# Description: Implementation of sequential Monte Carlo map-matching. Both offline and online.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################


import numpy as np

import tools.edges


def initiate_particles(graph, first_observation, n_samps, d_refine=1, gps_sd=7, truncation_distance=None):
    '''
    Initiate start of a trajectory by sampling points around the first observation.
    Note that coordinate system of inputs must be the same, typically a UTM projection (not longtitude-latitude!).
    See tools/graph.py and data/preprocess.py to ensure both your grapha and polyline data are projected to UTM.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        object encoding road network
        generating using OSMnx, see tools.graph.py
    :param first_observation: numpy.ndarray, shape = (2,)
        UTM projection
        coordinate of first observation
    :param n_samps: int
        number of samples to generate
    :param d_refine: float
        metres
        resolution of distance discretisation
        increase of speed, decrease for accuracy
    :param gps_sd: float
        metres
        standard deviation of GPS noise
    :param truncation_distance: float
        metres
        distance beyond which to assume zero likelihood probability
        required only for first observation
    :return: list of numpy.ndarrays, length = n_samps
        each numpy.ndarray with shape = (1, 7)
        columns: t, u, v, k, alpha, n_inter, d
            t: seconds, observation time (= 0 here)
            u: int, edge start node
            v: int, edge end node
            k: int, edge key
            alpha: in [0,1], position along edge
            n_inter: int, number of options if intersection (= 0 here)
            d: metres, distance travelled since previous observation (= 0 here)
    '''
    if truncation_distance is None:
        truncation_distance = gps_sd * 3

    # Discretize edges within truncation
    dis_points = tools.edges.get_truncated_discrete_edges(graph, first_observation, d_refine, truncation_distance)

    # Likelihood weights
    weights = np.exp(-0.5 / gps_sd ** 2 * dis_points[:, 4] ** 2)
    weights /= np.sum(weights)

    # Sample indices according to weights
    sampled_indices = np.random.choice(len(weights), n_samps, replace=True, p=weights)

    # Initiate array
    sampled_points = np.zeros((n_samps, 7))
    sampled_points[:, 1:5] = dis_points[sampled_indices, :4]

    # Change to list of numpy.ndarrays
    return [np.atleast_2d(sampled_points[i, :]) for i in range(n_samps)]


def update_particles(graph, particles, time_interval, new_observation, d_refine=1, d_max=None,
                     proposal, resampling_scheme):

    if d_max is None:
        d_max = time_interval * 35






















