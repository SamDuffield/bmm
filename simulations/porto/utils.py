import functools
import gc
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import bmm


def read_data(path, chunksize=None):
    """
    Read vehicle trajectory data from csv, automatically converting polylines stored as JSON
    :param path: location of csv
    :param chunksize: size of chunks to iterate through, None loads full file at once
    :return: generator iterating over (chunks of) data from csv
    """
    data_reader = pd.read_csv(path, chunksize=10)
    data_columns = data_reader.get_chunk().columns
    polyline_converters = {col_name: json.loads for col_name in data_columns
                           if 'POLYLINE' in col_name}

    return pd.read_csv(path, converters=polyline_converters, chunksize=chunksize)



def clear_cache():
    gc.collect()
    for a in gc.get_objects():
        if isinstance(a, functools._lru_cache_wrapper):
            a.cache_clear()


#
# def total_variation_dists(dists_one,
#                           dists_two,
#                           bins=None):
#     n1 = len(dists_one)
#     n2 = len(dists_two)
#
#     all_dists = np.concatenate([dists_one, dists_two])
#     all_dists = np.unique(all_dists)
#
#     if bins is None:
#         tv = 0.
#         for dist in all_dists:
#             p_1 = np.sum(dists_one == dist) / n1
#             p_2 = np.sum(dists_two == dist) / n2
#             tv = tv + np.abs(p_1 - p_2)
#     else:
#         min_dist = np.min(dists_one)
#         max_dist = np.max(dists_one)
#         bin_int = (max_dist - min_dist) / bins
#         tv = 0
#         bin_linsp = np.arange(min_dist, max_dist, bin_int)
#
#         # Below min
#         tv += np.sum(dists_two < min_dist) / n2
#
#         # Above max
#         tv += np.sum(dists_two > max_dist) / n2
#
#         for i in range(len(bin_linsp)):
#             int_min = bin_linsp[i]
#             int_max = int_min + bin_int + 1e-5 if i == len(bin_linsp) - 1 else int_min + bin_int
#             p_1 = np.sum((dists_one >= int_min) * (dists_one < int_max)) / n1
#             p_2 = np.sum((dists_two >= int_min) * (dists_two < int_max)) / n2
#             tv += np.abs(p_1 - p_2)
#     return tv / 2
#

def total_variation_dists(dists_one,
                          dists_two,
                          bin_width=3):
    n1 = len(dists_one)
    n2 = len(dists_two)

    all_dists = np.concatenate([dists_one, dists_two])
    all_dists = np.unique(all_dists)

    if bin_width is None:
        tv = 0.
        for dist in all_dists:
            p_1 = np.sum(dists_one == dist) / n1
            p_2 = np.sum(dists_two == dist) / n2
            tv = tv + np.abs(p_1 - p_2)
    else:
        min_dist = np.min(dists_one)
        max_dist = np.max(dists_one)
        tv = 0
        bin_linsp = np.arange(min_dist, max_dist, bin_width)

        # Below min
        tv += np.sum(dists_two < min_dist) / n2

        # Above max
        tv += np.sum(dists_two >= max_dist) / n2

        for i in range(len(bin_linsp)):
            int_min = bin_linsp[i]
            int_max = int_min + bin_width
            p_1 = np.sum((dists_one >= int_min) * (dists_one < int_max)) / n1
            p_2 = np.sum((dists_two >= int_min) * (dists_two < int_max)) / n2
            tv += np.abs(p_1 - p_2)
    return tv / 2


def obs_rows_trim(particles, trail_zero_lim=3):
    particles_obs_rows = []
    for p in particles:
        obs_rows = bmm.observation_time_rows(p)
        zero_dist_bools = obs_rows[:, -1] == 0
        if np.all(zero_dist_bools[-trail_zero_lim:]):
            count = 3
            is_zero = True
            while is_zero and count < len(obs_rows):
                count += 1
                is_zero = zero_dist_bools[-count]
            particles_obs_rows.append(obs_rows[:-(count - 1)])
        else:
            particles_obs_rows.append(obs_rows)

        particles_obs_rows.append(bmm.observation_time_rows(p))
    return particles_obs_rows


def interval_tv_dists(particles_one,
                      particles_two,
                      interval=60,
                      speeds=False,
                      bins=None,
                      trim_zeros=3):
    observation_times = particles_one.observation_times
    obs_int = observation_times[1]
    if interval % obs_int != 0:
        raise ValueError('interval must be a multiple of inter-observation times')

    obs_per_int = int(interval / obs_int)

    num_ints = int(observation_times[-1] / interval)

    tv_each_time = np.zeros(num_ints)

    particles_one_obs_rows = obs_rows_trim(particles_one, trim_zeros)
    particles_two_obs_rows = obs_rows_trim(particles_two, trim_zeros)

    for i in range(1, num_ints + 1):
        start_time = observation_times[(i - 1) * obs_per_int]
        end_time = observation_times[i * obs_per_int]

        p1_dists = -np.ones(particles_one.n) * 2
        for j in range(particles_one.n):
            obs_rows = particles_one_obs_rows[j]
            if end_time in obs_rows[:, 0]:
                p1_dists[j] = np.sum(
                    obs_rows[np.logical_and(obs_rows[:, 0] >= start_time, obs_rows[:, 0] <= end_time), -1])

        p2_dists = -np.ones(particles_two.n) * 3
        for k in range(particles_two.n):
            obs_rows = particles_two_obs_rows[k]
            if end_time in obs_rows:
                p2_dists[k] = np.sum(
                    obs_rows[np.logical_and(obs_rows[:, 0] >= start_time, obs_rows[:, 0] <= end_time), -1])

        if speeds:
            p1_dists /= interval
            p2_dists /= interval

        tv_each_time[i - 1] = total_variation_dists(p1_dists, p2_dists, bins)

    return tv_each_time


def total_variation_edges(edges_one,
                          edges_two,
                          round_alpha=None):
    n1 = len(edges_one)
    n2 = len(edges_two)

    if edges_one.shape[-1] == 4 and round_alpha is not None:
        edges_one[:, -1] = np.round(edges_one[:, -1], round_alpha)
        edges_two[:, -1] = np.round(edges_two[:, -1], round_alpha)

    all_edges = np.concatenate([edges_one, edges_two])
    all_edges = np.unique(all_edges, axis=0)

    axes = tuple(i for i in range(1, all_edges.ndim))

    tv = 0.
    for edge in all_edges:
        p_1 = np.sum(np.all(np.abs(edges_one - edge) < 1e-3, axis=axes)) / n1
        p_2 = np.sum(np.all(np.abs(edges_two - edge) < 1e-3, axis=axes)) / n2
        tv = tv + np.abs(p_1 - p_2)

    return tv / 2


def each_edge_route_total_variation(particles_one,
                                    particles_two,
                                    observation_times,
                                    include_alpha=False,
                                    round_alpha=None):
    m = observation_times.size
    tv_each_time = np.zeros(m)

    alpha_extend = include_alpha * 1

    for i in range(m):
        if i == 0:
            p1_first_edges = np.array([p[:1, 1:(4 + alpha_extend)] for p in particles_one])
            p2_first_edges = np.array([p[:1, 1:(4 + alpha_extend)] for p in particles_two])

            tv_each_time[i] = total_variation_edges(p1_first_edges, p2_first_edges, round_alpha)
        else:
            prev_time = observation_times[i - 1]
            current_time = observation_times[i]

            p1_edges = np.zeros((len(particles_one), 1, (3 + alpha_extend)))
            for j, p1 in enumerate(particles_one):
                prev_ind = np.where(p1[:, 0] == prev_time)[0][0]
                curr_ind = np.where(p1[:, 0] == current_time)[0][0]

                p1_ed = p1[(prev_ind + 1):(curr_ind + 1), 1:(4 + alpha_extend)].copy()
                p1_ed_len = len(p1_ed)
                if p1_ed_len > p1_edges.shape[1]:
                    p1_edges = np.append(p1_edges, np.zeros((p1_edges.shape[0],
                                                             p1_ed_len - p1_edges.shape[1],
                                                             (3 + alpha_extend))),
                                         axis=1)

                p1_edges[j, :p1_ed_len] = p1_ed

            p2_edges = np.zeros((len(particles_two), p1_edges.shape[1], (3 + alpha_extend)))
            for j, p2 in enumerate(particles_two):
                prev_ind = np.where(p2[:, 0] == prev_time)[0][0]
                curr_ind = np.where(p2[:, 0] == current_time)[0][0]

                p2_ed = p2[(prev_ind + 1):(curr_ind + 1), 1:(4 + alpha_extend)].copy()
                p2_ed_len = len(p2_ed)
                if p2_ed_len > p2_edges.shape[1]:
                    p2_edges = np.append(p2_edges, np.zeros((p2_edges.shape[0],
                                                             p2_ed_len - p2_edges.shape[1], (3 + alpha_extend))),
                                         axis=1)

                p2_edges[j, :p2_ed_len] = p2_ed

            if p1_edges.shape[1] < p2_edges.shape[1]:
                p1_edges = np.append(p1_edges, np.zeros((p1_edges.shape[0],
                                                         p2_edges.shape[1] - p1_edges.shape[1],
                                                         (3 + alpha_extend))),
                                     axis=1)

            tv_each_time[i] = total_variation_edges(p1_edges, p2_edges, round_alpha)

    return tv_each_time


def each_obs_edge_route_total_variation(particles_one,
                                        particles_two,
                                        observation_times,
                                        include_alpha=False,
                                        round_alpha=None):
    m = observation_times.size
    tv_each_time = np.zeros(m)

    alpha_extend = include_alpha * 1

    for i in range(m):
        if i == 0:
            p1_first_edges = np.array([p[:1, 1:(4 + alpha_extend)] for p in particles_one])
            p2_first_edges = np.array([p[:1, 1:(4 + alpha_extend)] for p in particles_two])

            tv_each_time[i] = total_variation_edges(p1_first_edges, p2_first_edges, round_alpha)
        else:
            current_time = observation_times[i]

            p1_edges = np.zeros((len(particles_one), (3 + alpha_extend)))
            for j, p1 in enumerate(particles_one):
                curr_ind = np.where(p1[:, 0] == current_time)[0][0]
                p1_edges[j] = p1[curr_ind, 1:(4 + alpha_extend)].copy()

            p2_edges = np.zeros((len(particles_two), (3 + alpha_extend)))
            for j, p2 in enumerate(particles_two):
                curr_ind = np.where(p2[:, 0] == current_time)[0][0]
                p2_edges[j] = p2[curr_ind, 1:(4 + alpha_extend)].copy()

            tv_each_time[i] = total_variation_edges(p1_edges, p2_edges, round_alpha)

    return tv_each_time


def append_zeros(list_arr, max_len):
    for i in range(len(list_arr)):
        path = list_arr[i]
        if len(path) < max_len:
            list_arr[i] = np.append(path, np.zeros(max_len - len(path)))
    return list_arr


def all_edges_total_variation(particles_one,
                              particles_two):
    n1 = particles_one.n
    n2 = particles_two.n

    route_nodes_one = particles_one.route_nodes()
    route_nodes_two = particles_two.route_nodes()

    len_route_nodes_one = np.unique([len(p) for p in route_nodes_one])
    len_route_nodes_two = np.unique([len(p) for p in route_nodes_two])

    max_len = np.max(np.concatenate([len_route_nodes_one, len_route_nodes_two]))

    # Extend route nodes to find unique
    route_nodes_one = append_zeros(route_nodes_one, max_len)
    route_nodes_two = append_zeros(route_nodes_two, max_len)

    unique_route_nodes = np.unique(np.concatenate([route_nodes_one, route_nodes_two]), axis=0)

    tv = 0.
    for edge in unique_route_nodes:
        p_1 = np.sum(np.all(route_nodes_one == edge, axis=-1)) / n1
        p_2 = np.sum(np.all(route_nodes_two == edge, axis=-1)) / n2
        tv = tv + np.abs(p_1 - p_2)
    return tv / 2


# def total_variation_dists(dists_one,
#                           dists_two,
#                           round_dists=None):
#     n1 = len(dists_one)
#     n2 = len(dists_two)
#
#     if round_dists is not None:
#         dists_one = np.round(dists_one, round_dists)
#         dists_two = np.round(dists_two, round_dists)
#
#     all_dists = np.concatenate([dists_one, dists_two])
#     all_dists = np.unique(all_dists)
#
#     tv = 0.
#     for dist in all_dists:
#         p_1 = np.sum(dists_one == dist) / n1
#         p_2 = np.sum(dists_two == dist) / n2
#         tv = tv + np.abs(p_1 - p_2)
#     return tv / 2

#
# def each_distance_route_total_variation(particles_one,
#                                         particles_two,
#                                         observation_times,
#                                         round_dists=None):
#     m = observation_times.size
#     tv_each_time = np.zeros(m)
#
#     for i in range(1, m):
#         current_time = observation_times[i]
#
#         p1_dists = np.zeros(len(particles_one))
#         for j, p1 in enumerate(particles_one):
#             curr_ind = np.where(p1[:, 0] == current_time)[0][0]
#             p1_dists[j] = p1[curr_ind, -1]
#
#         p2_dists = np.zeros(len(particles_two))
#         for j, p2 in enumerate(particles_two):
#             curr_ind = np.where(p2[:, 0] == current_time)[0][0]
#             p2_dists[j] = p2[curr_ind, -1]
#
#         tv_each_time[i] = total_variation_dists(p1_dists, p2_dists, round_dists)
#
#     return tv_each_time


def plot_metric_over_time(setup_dict, fl_pf_metric, fl_bsi_metric, fl_pf_time=None, fl_bsi_time=None, save_dir=None,
                          ffbsi_metric=None, ffbsi_time=None, t_linspace=None, x_lab='t', x_ticks=None):
    lags = setup_dict['lags']

    m = fl_pf_metric.shape[-1]

    if t_linspace is None:
        t_linspace = np.arange(m)

    if ffbsi_metric is not None:
        if ffbsi_metric.ndim == 1:
            ffbsi_metric = np.repeat(ffbsi_metric[np.newaxis], len(setup_dict['fl_n_samps']), axis=0)
        if ffbsi_time.ndim == 1:
            ffbsi_metric = np.repeat(ffbsi_time[np.newaxis], len(setup_dict['fl_n_samps']), axis=0)

    fontsize = 9
    title_runtime = 9
    shift = 0.09

    left_start = 0.005
    up_start = 0.19

    lines = [None] * (len(lags) + 1)

    fig, axes = plt.subplots(len(setup_dict['fl_n_samps']), 2, sharex='all', sharey='all', figsize=(8, 6))
    for j, n in enumerate(setup_dict['fl_n_samps']):
        for k, lag in enumerate(lags):
            axes[j, 0].plot(t_linspace, fl_pf_metric[j, k], label=f'Lag: {lag}')
            lines[k], = axes[j, 1].plot(t_linspace, fl_bsi_metric[j, k], label=f'Lag: {lag}')

        if ffbsi_metric is not None:
            lines[len(lags)], = axes[j, 0].plot(t_linspace, ffbsi_metric[j], label='FFBSi')
            axes[j, 1].plot(t_linspace, ffbsi_metric[j], label='FFBSi')

        axes[j, 0].set_ylabel(f'N={n}')
        # axes[j, 0].set_yticks([0, 0.5, 1])
        # axes[j, 1].set_yticks([0, 0.5, 1])

    if fl_pf_time is not None:
        for j, n in enumerate(setup_dict['fl_n_samps']):
            for k, lag in enumerate(lags):
                axes[j, 0].text(left_start, up_start - k * shift, "{:.1f}".format(fl_pf_time[j, k]),
                                color=lines[k].get_color(),
                                fontsize=fontsize, transform=axes[j, 0].transAxes)
                axes[j, 1].text(left_start, up_start - k * shift, "{:.1f}".format(fl_bsi_time[j, k]),
                                color=lines[k].get_color(),
                                fontsize=fontsize, transform=axes[j, 1].transAxes)

            if ffbsi_metric is not None:
                axes[j, 0].text(left_start, up_start - len(lags) * shift, "{:.1f}".format(ffbsi_time[j]),
                                color=lines[len(lags)].get_color(),
                                fontsize=fontsize, transform=axes[j, 0].transAxes)
                axes[j, 1].text(left_start, up_start - len(lags) * shift, "{:.1f}".format(ffbsi_time[j]),
                                color=lines[len(lags)].get_color(),
                                fontsize=fontsize, transform=axes[j, 1].transAxes)

            axes[j, 0].text(left_start, up_start + shift, "Runtime (s)",
                            fontsize=title_runtime, transform=axes[j, 0].transAxes)

            axes[j, 1].text(left_start, up_start + shift, "Runtime (s)",
                            fontsize=title_runtime, transform=axes[j, 1].transAxes)

    axes[-1, 0].set_xlabel(x_lab)
    axes[-1, 1].set_xlabel(x_lab)

    if x_ticks is not None:
        axes[-1, 0].set_xticks(x_ticks)
        axes[-1, 1].set_xticks(x_ticks)

    axes[0, 0].set_title('FL Particle Filter')
    axes[0, 1].set_title('FL Backward Simulation')

    plt.legend(loc='upper right')

    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir, dpi=400)

    return fig, axes


def plot_conv_metric(tv_mat, n_samps, lags, mins=None, maxs=None, leg=False, save_dir=None):
    fig, ax = plt.subplots()
    for i in range(tv_mat.shape[1]):
        line, = ax.plot(n_samps, tv_mat[:, i], label=f'Lag: {lags[i]}', zorder=2)
        if maxs is not None and mins is not None:
            ax.fill_between(n_samps, mins[:, i], maxs[:, i], color=line.get_color(), alpha=0.2, zorder=1)

    plt.tight_layout()
    if leg:
        plt.legend()

    if save_dir:
        plt.savefig(save_dir, dpi=400)

    return fig, ax
