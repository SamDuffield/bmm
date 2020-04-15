########################################################################################################################
# Module: inference/resampling.py
# Description: Resampling schemes for converting weighted particles (series of positions/edges/distances) to
#              unweighted. Notably multinomial resampling and fixed-lag resampling (with stitching).
#
# Web: https://github.com/SamDuffield/bayesian-traffic
########################################################################################################################

import numpy as np

from inference.model import distance_prior, get_distance_prior_bound
from inference.particles import MMParticles
from tools.edges import get_geometry


def multinomial(particles, weights):
    """
    Full multinomial resampling scheme. Lengths of particles and weights must conform.
    :param particles: list-like or MMParticles object (from inference.smc)
        collection of objects to to resample
    :param weights: list-like
        resampling probabilities
    :return: list-like or MMParticles object
        unweighted collection of objects in the same form as input
    """

    # Number of samples
    n = len(weights)

    # Check weights are normalised
    weights_sum = np.sum(weights)
    if weights_sum != 1:
        weights /= weights_sum

    # Sample indices according to weights (with replacement)
    sampled_indices = np.random.choice(n, n, replace=True, p=weights)

    # Update and output particles
    if isinstance(particles, MMParticles):
        if particles.n != n:
            raise ValueError("Length of MMParticles to be resampled and weights do not conform")
        out_particles = particles.copy()
        out_particles.particles = [out_particles.particles[i].copy() for i in sampled_indices]
    elif isinstance(particles, np.ndarray):
        if len(particles) != n:
            raise ValueError("Length of particles (numpy.ndarray) to be resampled and weights do not conform")
        out_particles = particles[sampled_indices]
    else:
        if len(particles) != n:
            raise ValueError("Length of particles to be resampled and weights do not conform")
        out_particles = [particles[i] for i in sampled_indices]

    return out_particles


def full_fixed_lag_stitch(j, fixed_particle, last_edge_fixed, last_edge_fixed_length,
                          new_particles, pf_weights,
                          min_resample_time, stitch_time_interval,
                          min_resample_time_indices, originial_stitching_distances, distance_prior_evals,
                          return_ess_stitch=False):

    n = len(new_particles)

    # Possible particles to be resampled placeholder
    newer_particles_adjusted = [None] * n

    # Stitching distances
    new_stitching_distances = np.empty(n)
    new_stitching_distances[:] = np.nan

    for k in range(n):
        if k == j:
            newer_particles_adjusted[k] = new_particles[k][1:]
            new_stitching_distances[k] = originial_stitching_distances[k]
            continue

        if pf_weights[k] == 0:
            continue

        new_particle = new_particles[k].copy()

        # Check both particles start from same edge
        if np.array_equal(last_edge_fixed[1:4], new_particle[0, 1:4]):
            # Check that new edge overtakes fixed edge. i.e. distance isn't negative
            if np.array_equal(last_edge_fixed[1:4], new_particle[1, 1:4]) and \
                    new_particle[1, 4] < last_edge_fixed[4]:
                continue

            # Calculate distance modification
            first_distance_j_to_k = (new_particle[1, 4] - last_edge_fixed[4]) * last_edge_fixed_length
            first_distance_k = new_particle[1, 6]

            change_dist = first_distance_j_to_k - first_distance_k

            new_particle[:min_resample_time_indices[j], 6] += change_dist

            new_stitching_distances[k] = new_particle[new_particle[:, 0] <= min_resample_time][-1, 6]

            # Store adjusted particle
            newer_particles_adjusted[k] = new_particle[1:]

    # Calculate adjusted weight
    res_weights = np.zeros(n)
    possible_inds = ~np.isnan(new_stitching_distances)
    res_weights[possible_inds] = pf_weights[possible_inds] \
                                 * distance_prior(new_stitching_distances[possible_inds], stitch_time_interval) \
                                 / distance_prior_evals[possible_inds]

    # Normalise adjusted resample weights
    res_weights /= res_weights.sum()

    # If only particle on fixed edge resample full trajectory
    if max(res_weights) == 1 or max(res_weights) == 0:
        out_particle = None
        ess_stitch = 1 / np.sum(pf_weights ** 2)

    # Otherwise fixed-lag resample and stitch
    else:
        # Resample index
        res_index = np.random.choice(n, 1, p=res_weights)[0]

        # Update output
        out_particle = np.append(fixed_particle, newer_particles_adjusted[res_index], axis=0)

        # Track ESS
        ess_stitch = 1 / np.sum(res_weights ** 2)

    if return_ess_stitch:
        return out_particle, ess_stitch
    else:
        return out_particle


def rejection_fixed_lag_stitch(j, fixed_particle, last_edge_fixed, last_edge_fixed_length,
                               new_particles, adjusted_weights,
                               min_resample_time, stitch_time_interval,
                               min_resample_time_indices,
                               distance_prior_bound, max_rejections):

    n = len(new_particles)

    for k in range(max_rejections):
        new_index = np.random.choice(n, 1, p=adjusted_weights)[0]
        new_particle = new_particles[new_index].copy()

        # Reject if new_particle starts from differen edge
        if not np.array_equal(last_edge_fixed[1:4], new_particle[0, 1:4]):
            continue
        # Reject if new_particle doesn't overtake fixed_particles
        elif np.array_equal(last_edge_fixed[1:4], new_particle[1, 1:4]) and \
                new_particle[1, 4] < last_edge_fixed[4]:
            continue

        # Calculate stitching distance
        first_distance_j_to_k = (new_particle[1, 4] - last_edge_fixed[4]) * last_edge_fixed_length
        first_distance_k = new_particle[1, 6]

        change_dist = first_distance_j_to_k - first_distance_k

        new_particle[:min_resample_time_indices[j], 6] += change_dist

        new_stitching_distance = new_particle[new_particle[:, 0] <= min_resample_time][-1, 6]

        # Evaluate distance prior
        new_stitching_distance_prior = distance_prior(new_stitching_distance, stitch_time_interval)

        if np.random.uniform() < new_stitching_distance_prior / distance_prior_bound:
            out_particle = np.append(fixed_particle, new_particle[1:], axis=0)
            return out_particle

    return None


def fixed_lag_stitching(graph, particles, weights, lag, max_rejections=100):
    """
    Resamples only elements of particles after a certain time - defined by the lag parameter.
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param particles: MMParticles object (from inference.smc)
        trajectories generated
    :param weights: list-like, length = n
        weights at latest observation time
    :param lag: int or None
        lag parameter
        trajectories before this will be fixed
        None indicates full multinomial resampling
    :param max_rejections: int
        maximum number of rejections
        0 indicates full sampling rather than rejection sampling
    :return: MMParticles object
        unweighted collection of trajectories post resampling + stitching
    """
    # Bool whether to store ESS stitch quantities
    full_fixed_lag_resample = max_rejections == 0

    # Check weights are normalised
    weights_sum = np.sum(weights)
    if weights_sum != 1:
        weights /= weights_sum

    # Extract basic quantities
    observation_times = particles.observation_times
    m = len(observation_times)
    n = particles.n
    ess_pf = 1 / np.sum(weights ** 2)

    # Initiate output
    out_particles = particles.copy()

    # If not reached lag yet do standard resampling
    if lag is None or m <= lag:
        if full_fixed_lag_resample:
            out_particles.ess_stitch = np.append(particles.ess_stitch, np.ones((1, n))*ess_pf,
                                                 axis=0)
        return multinomial(out_particles, weights)

    # Largest time not to be resampled
    max_fixed_time = observation_times[m - lag - 1]

    # Smallest time to be resampled
    min_resample_time = observation_times[m - lag]

    # Time between stitching observations
    stitch_time_interval = min_resample_time - max_fixed_time

    # Pre-process a bit
    fixed_particles = [None] * n
    new_particles = [None] * n
    max_fixed_time_indices = [0] * n
    min_resample_time_indices = [0] * n
    originial_stitching_distances = np.zeros(n)

    for j in range(n):
        if weights[j] == 0:
            continue

        max_fixed_time_indices[j] = np.where(out_particles[j][:, 0] == max_fixed_time)[0][0]
        fixed_particles[j] = out_particles[j][:(max_fixed_time_indices[j] + 1)]
        new_particles[j] = out_particles[j][max_fixed_time_indices[j]:]
        min_resample_time_indices[j] = np.where(new_particles[j][:, 0] == min_resample_time)[0][0]
        originial_stitching_distances[j] = new_particles[j][min_resample_time_indices[j], -1]
    distance_prior_evals = distance_prior(originial_stitching_distances, stitch_time_interval)

    # Initiate some required quantities depending on whether to do rejection sampling or not
    if full_fixed_lag_resample:
        ess_stitch_track = np.zeros(n)

        distance_prior_bound = None
        adjusted_weights = None
    else:
        ess_stitch_track = None

        distance_prior_bound = get_distance_prior_bound()
        adjusted_weights = weights / distance_prior_evals
        adjusted_weights /= np.sum(adjusted_weights)

    # Iterate through particles
    for j in range(n):
        # Check if particle has probability 0 then do full resampling
        # i.e. fixed lag approx has failed
        if weights[j] == 0:
            out_particles[j] = out_particles[np.random.choice(n, 1, p=weights)[0]]
            if full_fixed_lag_resample:
                ess_stitch_track[j] = ess_pf
            continue

        fixed_particle = fixed_particles[j]
        last_edge_fixed = fixed_particle[-1]
        last_edge_fixed_geom = get_geometry(graph, last_edge_fixed[1:4])
        last_edge_fixed_length = last_edge_fixed_geom.length

        if full_fixed_lag_resample:
            #### Full resampling
            out_particles[j], ess_stitch_track[j] = full_fixed_lag_stitch(j, fixed_particle,
                                                                          last_edge_fixed, last_edge_fixed_length,
                                                                          new_particles,
                                                                          weights,
                                                                          min_resample_time, stitch_time_interval,
                                                                          min_resample_time_indices,
                                                                          originial_stitching_distances,
                                                                          distance_prior_evals,
                                                                          True)

        else:
            out_particles[j] = rejection_fixed_lag_stitch(j, fixed_particle, last_edge_fixed, last_edge_fixed_length,
                                                          new_particles, adjusted_weights,
                                                          min_resample_time, stitch_time_interval,
                                                          min_resample_time_indices,
                                                          distance_prior_bound, max_rejections)
            if out_particles[j] is None:
                out_particles[j] = full_fixed_lag_stitch(j, fixed_particle, last_edge_fixed, last_edge_fixed_length,
                                                         new_particles,
                                                         weights,
                                                         min_resample_time, stitch_time_interval,
                                                         min_resample_time_indices,
                                                         originial_stitching_distances,
                                                         distance_prior_evals,
                                                         False)

        if out_particles[j] is None:
            out_particles[j] = out_particles[np.random.choice(n, 1, p=weights)[0]]
            if full_fixed_lag_resample:
                ess_stitch_track[j] = ess_pf

    if full_fixed_lag_resample:
        out_particles.ess_stitch = np.append(particles.ess_stitch, np.atleast_2d(ess_stitch_track), axis=0)

    return out_particles


