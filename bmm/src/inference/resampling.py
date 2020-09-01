########################################################################################################################
# Module: inference/resampling.py
# Description: Resampling schemes for converting weighted particles (series of positions/edges/distances) to
#              unweighted. Notably multinomial resampling and fixed-lag resampling (with stitching).
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

from typing import Union, Tuple

import numpy as np
from networkx.classes import MultiDiGraph

from bmm.src.inference.particles import MMParticles
from bmm.src.inference.model import MapMatchingModel
from bmm.src.tools.edges import get_geometry


def multinomial(particles: Union[list, np.ndarray, MMParticles],
                weights: np.ndarray) -> Union[list, np.ndarray, MMParticles]:
    """
    Full multinomial resampling scheme. Lengths of particles and weights must conform.
    :param particles: list-like or MMParticles object to be resampled
    :param weights: resampling probabilities
    :return: unweighted collection of objects in the same form as input
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
        out_particles.particles = [out_particles.particles[i] for i in sampled_indices]
        if hasattr(out_particles, 'prior_norm'):
            if out_particles.prior_norm.shape[1] == n:
                out_particles.prior_norm = out_particles.prior_norm[:, sampled_indices]
            else:
                out_particles.prior_norm = out_particles.prior_norm[sampled_indices]
    elif isinstance(particles, np.ndarray):
        if len(particles) != n:
            raise ValueError("Length of particles (numpy.ndarray) to be resampled and weights do not conform")
        out_particles = particles[sampled_indices]
    else:
        if len(particles) != n:
            raise ValueError("Length of particles to be resampled and weights do not conform")
        out_particles = [particles[i] for i in sampled_indices]

    return out_particles


def full_fixed_lag_stitch(fixed_particle: np.ndarray,
                          last_edge_fixed: np.ndarray,
                          last_edge_fixed_length: float,
                          new_particles: MMParticles,
                          adjusted_weights: np.ndarray,
                          stitch_time_interval: float,
                          min_resample_time_indices: Union[list, np.ndarray],
                          mm_model: MapMatchingModel,
                          return_ess_stitch: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Evaluate full interacting weights, normalise and sample (stitch) for a single fixed particle
    :param fixed_particle: trajectory prior to stitching time
    :param last_edge_fixed: row of last fixed particle
    :param last_edge_fixed_length: length of last fixed edge (so don't have to call get_geometry)
    :param new_particles: particles proposed to stitching
    :param adjusted_weights: non-interacting weights for new_particles
    :param stitch_time_interval: time between stitching observations
    :param min_resample_time_indices: indices for row of min_resample_time in new_particles
    :param mm_model: MapMatchingModel
    :param return_ess_stitch: whether to calculate and return the ESS of the full stitching weights
    :return: stitched particle (and ess_stitch if return_ess_stitch)
    """
    n = len(new_particles)

    # Possible particles to be resampled placeholder
    newer_particles_adjusted = [None] * n

    # Stitching distances
    new_stitching_distances = np.empty(n)
    new_stitching_distances[:] = np.nan

    new_cart_coords = np.empty((n, 2))

    for k in range(n):
        # if adjusted_weights[k] == 0:
        #     continue

        if new_particles[k] is None:
            continue

        new_particle = new_particles[k].copy()

        # Check both particles start from same edge
        if np.array_equal(last_edge_fixed[1:4], new_particle[0, 1:4]):
            # Check that new edge overtakes fixed edge. i.e. distance isn't negative
            if np.array_equal(last_edge_fixed[1:4], new_particle[1, 1:4]) and \
                    new_particle[1, 4] < (last_edge_fixed[4] - 1e-6):
                continue

            new_cart_coords[k] = new_particle[min_resample_time_indices[k], 5:7]

            # Calculate distance modification
            first_distance_j_to_k = (new_particle[1, 4] - last_edge_fixed[4]) * last_edge_fixed_length
            first_distance_k = new_particle[1, -1]

            change_dist = np.round(first_distance_j_to_k - first_distance_k, 5)

            new_particle[1:(min_resample_time_indices[k] + 1), -1] += change_dist

            new_stitching_distances[k] = new_particle[min_resample_time_indices[k], -1]

            # Store adjusted particle
            newer_particles_adjusted[k] = new_particle[1:]

    # Calculate adjusted weight
    res_weights = np.zeros(n)
    possible_inds = ~np.isnan(new_stitching_distances)

    new_stitching_distances_trimmed = new_stitching_distances[possible_inds]
    new_cart_coords_trimmed = new_cart_coords[possible_inds]

    adjusted_weights_trimmed = adjusted_weights[possible_inds]
    if adjusted_weights_trimmed.sum() == 0:
        adjusted_weights_trimmed[:] = 1
    stitched_distance_prior_evals_trimmed = mm_model.distance_prior_evaluate(new_stitching_distances_trimmed,
                                                                             stitch_time_interval)

    stitched_deviation_prior_trimmed = mm_model.deviation_prior_evaluate(fixed_particle[-1, 5:7],
                                                                         new_cart_coords_trimmed,
                                                                         new_stitching_distances_trimmed)

    res_weights[possible_inds] = adjusted_weights_trimmed \
                                 * stitched_distance_prior_evals_trimmed \
                                 * stitched_deviation_prior_trimmed

    # Normalise adjusted resample weights
    with np.errstate(invalid='ignore'):
        res_weights /= res_weights.sum()

    # If only particle on fixed edge resample full trajectory
    if max(res_weights) == 0 or np.all(np.isnan(res_weights)):
        out_particle = None
        ess_stitch = 1 / np.sum(adjusted_weights ** 2)

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


def rejection_fixed_lag_stitch(fixed_particle: np.ndarray,
                               last_edge_fixed: np.ndarray,
                               last_edge_fixed_length: float,
                               new_particles: MMParticles,
                               adjusted_weights: np.ndarray,
                               stitch_time_interval: float,
                               min_resample_time_indices: Union[list, np.ndarray],
                               dist_prior_bound: float,
                               mm_model: MapMatchingModel,
                               max_rejections: int,
                               break_on_zero: bool = False) -> Union[np.ndarray, None, int]:
    """
    Attempt up to max_rejections of rejection sampling to stitch a single fixed particle
    :param fixed_particle: trajectory prior to stitching time
    :param last_edge_fixed: row of last fixed particle
    :param last_edge_fixed_length: length of last fixed edge (so don't have to call get_geometry)
    :param new_particles: particles proposed to stitching
    :param adjusted_weights: non-interacting stitching weights
    :param stitch_time_interval: time between stitching observations
    :param min_resample_time_indices: indices for row of min_resample_time in new_particles
    :param dist_prior_bound: bound on distance transition density (given positive if break_on_zero)
    :param mm_model: MapMatchingModel
    :param max_rejections: number of rejections to attempt, if none succeed return None
    :param break_on_zero: whether to return 0 if new_stitching_distance=0
    :return: stitched particle
    """
    n = len(new_particles)

    for reject_ind in range(max_rejections):
        new_index = np.random.choice(n, 1, p=adjusted_weights)[0]
        new_particle = new_particles[new_index].copy()

        # Reject if new_particle starts from different edge
        if not np.array_equal(last_edge_fixed[1:4], new_particle[0, 1:4]):
            continue
        # Reject if new_particle doesn't overtake fixed_particles
        elif np.array_equal(last_edge_fixed[1:4], new_particle[1, 1:4]) and \
                new_particle[1, 4] < last_edge_fixed[4]:
            continue

        # Calculate stitching distance
        first_distance_j_to_k = (new_particle[1, 4] - last_edge_fixed[4]) * last_edge_fixed_length
        first_distance_k = new_particle[1, -1]

        change_dist = np.round(first_distance_j_to_k - first_distance_k, 5)

        new_particle[1:(min_resample_time_indices[new_index] + 1), -1] += change_dist

        new_stitching_distance = new_particle[min_resample_time_indices[new_index], -1]

        if break_on_zero and new_stitching_distance < 1e-5:
            return 0

        # Evaluate distance prior
        new_stitching_distance_prior = mm_model.distance_prior_evaluate(new_stitching_distance, stitch_time_interval)
        new_stitching_deviation_prior = mm_model.deviation_prior_evaluate(fixed_particle[-1, 5:7],
                                                                          new_particle[None,
                                                                          min_resample_time_indices[new_index], 5:7],
                                                                          new_stitching_distance)

        accept_prob = new_stitching_distance_prior * new_stitching_deviation_prior / dist_prior_bound
        if accept_prob > (1 - 1e-5) or np.random.uniform() < accept_prob:
            out_particle = np.append(fixed_particle, new_particle[1:], axis=0)
            return out_particle
    return None


def fixed_lag_stitch_post_split(graph: MultiDiGraph,
                                fixed_particles: MMParticles,
                                new_particles: MMParticles,
                                new_weights: np.ndarray,
                                mm_model: MapMatchingModel,
                                max_rejections: int) -> MMParticles:
    """
    Stitch together fixed_particles with samples from new_particles according to joint fixed-lag posterior
    :param graph: encodes road network, simplified and projected to UTM
    :param fixed_particles: trajectories before stitching time (won't be changed)
    :param new_particles: trajectories after stitching time (to be resampled)
        one observation time overlap with fixed_particles
    :param new_weights: weights applied to new_particles
    :param mm_model: MapMatchingModel
    :param max_rejections: number of rejections to attempt before doing full fixed-lag stitching
        0 will do full fixed-lag stitching and track ess_stitch
    :return: MMParticles object
    """

    n = len(fixed_particles)
    full_fixed_lag_resample = max_rejections == 0

    min_resample_time = new_particles.observation_times[1]
    min_resample_time_indices = [np.where(particle[:, 0] == min_resample_time)[0][0] if particle is not None else 0
                                 for particle in new_particles]
    originial_stitching_distances = np.array([new_particles[j][min_resample_time_indices[j], -1]
                                              if new_particles[j] is not None else 0 for j in range(n)])

    max_fixed_time = fixed_particles._first_non_none_particle[-1, 0]

    stitch_time_interval = min_resample_time - max_fixed_time

    distance_prior_evals = mm_model.distance_prior_evaluate(originial_stitching_distances, stitch_time_interval)

    fixed_last_coords = np.array([part[0, 5:7] if part is not None else [0, 0] for part in new_particles])
    new_coords = np.array([new_particles[j][min_resample_time_indices[j], 5:7]
                           if new_particles[j] is not None else [0, 0] for j in range(n)])
    deviation_prior_evals = mm_model.deviation_prior_evaluate(fixed_last_coords,
                                                              new_coords,
                                                              originial_stitching_distances)

    original_prior_evals = np.zeros(n)
    pos_inds = new_particles.prior_norm > 1e-5
    original_prior_evals[pos_inds] = distance_prior_evals[pos_inds] \
                                     * deviation_prior_evals[pos_inds] \
                                     * new_particles.prior_norm[pos_inds]

    out_particles = fixed_particles

    # Initiate some required quantities depending on whether to do rejection sampling or not
    if full_fixed_lag_resample:
        ess_stitch_track = np.zeros(n)

        # distance_prior_bound = None
        # adjusted_weights = None
    else:
        ess_stitch_track = None

        pos_prior_bound = mm_model.pos_distance_prior_bound(stitch_time_interval)
        prior_bound = mm_model.distance_prior_bound(stitch_time_interval)
        store_out_parts = fixed_particles.copy()

    adjusted_weights = new_weights.copy()
    adjusted_weights[original_prior_evals > 1e-5] /= original_prior_evals[original_prior_evals > 1e-5]
    adjusted_weights[original_prior_evals < 1e-5] = 0
    adjusted_weights /= np.sum(adjusted_weights)

    resort_to_full = False

    # Iterate through particles
    for j in range(n):
        fixed_particle = fixed_particles[j]

        # Check if particle is None
        # i.e. fixed lag approx has failed
        if fixed_particle is None:
            out_particles[j] = None
            if full_fixed_lag_resample:
                ess_stitch_track[j] = 0
            continue

        last_edge_fixed = fixed_particle[-1]
        last_edge_fixed_geom = get_geometry(graph, last_edge_fixed[1:4])
        last_edge_fixed_length = last_edge_fixed_geom.length

        if full_fixed_lag_resample:
            # Full resampling
            out_particles[j], ess_stitch_track[j] = full_fixed_lag_stitch(fixed_particle,
                                                                          last_edge_fixed, last_edge_fixed_length,
                                                                          new_particles,
                                                                          adjusted_weights,
                                                                          stitch_time_interval,
                                                                          min_resample_time_indices,
                                                                          mm_model,
                                                                          True)

        else:
            # Rejection sampling
            out_particles[j] = rejection_fixed_lag_stitch(fixed_particle, last_edge_fixed, last_edge_fixed_length,
                                                          new_particles, adjusted_weights,
                                                          stitch_time_interval,
                                                          min_resample_time_indices,
                                                          pos_prior_bound,
                                                          mm_model,
                                                          max_rejections,
                                                          break_on_zero=True)
            if out_particles[j] is None:
                # Rejection sampling reached max_rejections -> try full resampling
                out_particles[j] = full_fixed_lag_stitch(fixed_particle, last_edge_fixed, last_edge_fixed_length,
                                                         new_particles,
                                                         adjusted_weights,
                                                         stitch_time_interval,
                                                         min_resample_time_indices,
                                                         mm_model,
                                                         False)

            if isinstance(out_particles[j], int) and out_particles[j] == 0:
                resort_to_full = True
                break

    if resort_to_full:
        for j in range(n):
            fixed_particle = store_out_parts[j]

            # Check if particle is None
            # i.e. fixed lag approx has failed
            if fixed_particle is None:
                out_particles[j] = None
                if full_fixed_lag_resample:
                    ess_stitch_track[j] = 0
                continue

            last_edge_fixed = fixed_particle[-1]
            last_edge_fixed_geom = get_geometry(graph, last_edge_fixed[1:4])
            last_edge_fixed_length = last_edge_fixed_geom.length

            # Rejection sampling with full bound
            out_particles[j] = rejection_fixed_lag_stitch(fixed_particle, last_edge_fixed, last_edge_fixed_length,
                                                          new_particles, adjusted_weights,
                                                          stitch_time_interval,
                                                          min_resample_time_indices,
                                                          prior_bound,
                                                          mm_model,
                                                          max_rejections)
            if out_particles[j] is None:
                # Rejection sampling reached max_rejections -> try full resampling
                out_particles[j] = full_fixed_lag_stitch(fixed_particle, last_edge_fixed, last_edge_fixed_length,
                                                         new_particles,
                                                         adjusted_weights,
                                                         stitch_time_interval,
                                                         min_resample_time_indices,
                                                         mm_model,
                                                         False)

    if full_fixed_lag_resample:
        out_particles.ess_stitch = np.append(out_particles.ess_stitch, np.atleast_2d(ess_stitch_track), axis=0)

    # Do full resampling where fixed lag approx broke
    none_inds = np.array([p is None for p in out_particles])
    good_inds = ~none_inds
    n_good = good_inds.sum()

    if n_good == 0:
        raise ValueError("Map-matching failed: all stitching probabilities zero,"
                         "try increasing the lag or number of particles")

    if n_good < n:
        none_inds_res_indices = np.random.choice(n, n - n_good, p=good_inds / n_good)
        for i, j in enumerate(np.where(none_inds)[0]):
            out_particles[j] = out_particles[none_inds_res_indices[i]]
        if full_fixed_lag_resample:
            out_particles.ess_stitch[-1, none_inds] = 1 / (new_weights ** 2).sum()

    return out_particles


def fixed_lag_stitching(graph: MultiDiGraph,
                        mm_model: MapMatchingModel,
                        particles: MMParticles,
                        weights: np.ndarray,
                        lag: int,
                        max_rejections: int) -> MMParticles:
    """
    Split particles and resample (with stitching) coordinates after a certain time - defined by the lag parameter.
    :param graph: encodes road network, simplified and projected to UTM
    :param mm_model: MapMatchingModel
    :param particles: MMParticles object
    :param weights: shape = (n,) weights at latest observation time
    :param lag: fixed lag for resampling/stitching
        None indicates full multinomial resampling
    :param max_rejections: number of rejections to attempt before doing full fixed-lag stitching
        0 will do full fixed-lag stitching and track ess_stitch
    :return: MMParticles object
    """
    # Bool whether to store ESS stitch quantities
    full_fixed_lag_resample = max_rejections == 0

    # Check weights are normalised
    weights_sum = np.sum(weights)
    if weights_sum != 1:
        weights /= weights_sum

    # Extract basic quantities
    observation_times = particles.observation_times
    m = len(observation_times) - 1
    n = particles.n
    ess_pf = 1 / np.sum(weights ** 2)

    # Initiate output
    out_particles = particles.copy()

    # If not reached lag yet do standard resampling
    if lag is None or m <= lag:
        if full_fixed_lag_resample:
            out_particles.ess_stitch = np.append(particles.ess_stitch, np.ones((1, n)) * ess_pf,
                                                 axis=0)
        return multinomial(out_particles, weights)

    # Largest time not to be resampled
    max_fixed_time = observation_times[m - lag - 1]

    # Pre-process a bit
    fixed_particles = out_particles.copy()
    new_particles = out_particles.copy()
    max_fixed_time_indices = [0] * n

    for j in range(n):
        if out_particles[j] is None:
            continue

        max_fixed_time_indices[j] = np.where(out_particles[j][:, 0] == max_fixed_time)[0][0]
        fixed_particles[j] = out_particles[j][:(max_fixed_time_indices[j] + 1)]
        new_particles[j] = out_particles[j][max_fixed_time_indices[j]:]

    new_particles.prior_norm = out_particles.prior_norm[m - lag - 1]

    # Stitch
    out_particles = fixed_lag_stitch_post_split(graph,
                                                fixed_particles,
                                                new_particles,
                                                weights,
                                                mm_model,
                                                max_rejections)

    return out_particles
