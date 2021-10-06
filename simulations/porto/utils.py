import functools
import gc
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import bmm


def read_data(path, chunksize=None):
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


def plot_metric_over_time(setup_dict, fl_pf_metric, fl_bsi_metric, save_dir=None,
                          t_linspace=None, x_lab='t', x_ticks=None):
    lags = setup_dict['lags']

    m = fl_pf_metric.shape[-1]

    if t_linspace is None:
        t_linspace = np.arange(m)

    lines = [None] * (len(lags) + 1)

    fig_pf, axes_pf = plt.subplots(len(setup_dict['fl_n_samps']), sharex='all', sharey='all', figsize=(8, 6))
    fig_bs, axes_bs = plt.subplots(len(setup_dict['fl_n_samps']), sharex='all', sharey='all', figsize=(8, 6))
    for j, n in enumerate(setup_dict['fl_n_samps']):
        for k, lag in enumerate(lags):
            axes_pf[j].plot(t_linspace, fl_pf_metric[j, k], label=f'Lag: {lag}')
            lines[k], = axes_bs[j].plot(t_linspace, fl_bsi_metric[j, k], label=f'Lag: {lag}')

        axes_pf[j].set_ylabel(f'N={n}', fontsize=18)
        axes_bs[j].set_ylabel(f'N={n}', fontsize=18)

        # axes_pf[j].set_ylim(0, 0.7)
        # axes_bs[j].set_ylim(0, 0.7)
        # axes[j, 0].set_yticks([0, 0.5, 1])
        # axes[j, 1].set_yticks([0, 0.5, 1])

    axes_pf[-1].set_xlabel(x_lab, fontsize=16)
    axes_bs[-1].set_xlabel(x_lab, fontsize=16)

    if x_ticks is not None:
        axes_pf[-1].set_xticks(x_ticks)
        axes_bs[-1].set_xticks(x_ticks)

    plt.legend(loc='upper right', bbox_to_anchor=(0.8, 0.99))

    fig_pf.set_figwidth(5)
    fig_bs.set_figwidth(5)
    # fig_pf.set_figheight(7)
    # fig_bs.set_figheight(7)
    fig_pf.set_figheight(11)
    fig_bs.set_figheight(11)

    fig_pf.tight_layout()
    fig_bs.tight_layout()

    if save_dir is not None:
        fig_pf.savefig(save_dir + '_pf', dpi=400)
        fig_bs.savefig(save_dir + '_bs', dpi=400)

    return (fig_pf, axes_pf), (fig_bs, axes_bs)

