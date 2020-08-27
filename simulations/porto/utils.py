import functools
import gc

import numpy as np
import matplotlib.pyplot as plt


def clear_cache():
    gc.collect()
    for a in gc.get_objects():
        if isinstance(a, functools._lru_cache_wrapper):
            a.cache_clear()


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
        p_1 = np.sum(np.all(edges_one == edge, axis=axes)) / n1
        p_2 = np.sum(np.all(edges_two == edge, axis=axes)) / n2
        tv = tv + np.abs(p_1 - p_2)
    return tv / 2


def append_zeros(list_arr, max_len):
    for i in range(len(list_arr)):
        path = list_arr[i]
        if len(path) < max_len:
            list_arr[i] = np.append(path, np.zeros(max_len - len(path)))
    return list_arr


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
            prev_time = observation_times[i - 1]
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


def plot_metric_over_time(setup_dict, fl_pf_metric, fl_pf_time, fl_bsi_metric, fl_bsi_time, save_dir=None,
                          ffbsi_metric=None, ffbsi_time=None):
    lags = setup_dict['lags']

    m = fl_pf_metric.shape[-1]

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

        axes[j, 0].set_ylabel(f'N={n}')
        axes[j, 0].set_yticks([0, 0.5, 1])
        axes[j, 1].set_yticks([0, 0.5, 1])

    axes[-1, 0].set_xlabel('t')
    axes[-1, 1].set_xlabel('t')

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
