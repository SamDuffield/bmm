import functools
import gc

import numpy as np
import matplotlib.pyplot as plt


import bmm


def clear_cache():
    gc.collect()
    for a in gc.get_objects():
        if isinstance(a, functools._lru_cache_wrapper):
            a.cache_clear()


def total_variation_edges(edges_one,
                          edges_two):
    n1 = len(edges_one)
    n2 = len(edges_two)

    all_edges = np.concatenate([edges_one, edges_two])
    all_edges = np.unique(all_edges, axis=0)

    tv = 0.
    for edge in all_edges:
        p_1 = np.sum(np.all(edges_one == edge, axis=(1, 2))) / n1
        p_2 = np.sum(np.all(edges_two == edge, axis=(1, 2))) / n2
        tv = tv + np.abs(p_1 - p_2)

    return tv / 2


def each_edge_route_total_variation(particles_one,
                                    particles_two,
                                    observation_times):
    m = observation_times.size
    tv_each_time = np.zeros(m)

    for i in range(m):
        if i == 0:
            p1_first_edges = np.array([p[:1, 1:5] for p in particles_one])
            p2_first_edges = np.array([p[:1, 1:5] for p in particles_two])

            tv_each_time[i] = total_variation_edges(p1_first_edges, p2_first_edges)
        else:
            prev_time = observation_times[i - 1]
            current_time = observation_times[i]

            p1_edges = np.zeros((len(particles_one), 1, 4))
            for j, p1 in enumerate(particles_one):
                prev_ind = np.where(p1[:, 0] == prev_time)[0][0]
                curr_ind = np.where(p1[:, 0] == current_time)[0][0]

                p1_ed = p1[prev_ind:(curr_ind + 1), 1:5].copy()
                p1_ed[0, 3] = 0.
                p1_ed_len = len(p1_ed)
                if p1_ed_len > p1_edges.shape[1]:
                    p1_edges = np.append(p1_edges, np.zeros((p1_edges.shape[0],
                                                             p1_ed_len - p1_edges.shape[1],
                                                             4)),
                                         axis=1)

                p1_edges[j, :p1_ed_len] = p1_ed

            p2_edges = np.zeros((len(particles_two), p1_edges.shape[1], 4))
            for j, p2 in enumerate(particles_two):
                prev_ind = np.where(p2[:, 0] == prev_time)[0][0]
                curr_ind = np.where(p2[:, 0] == current_time)[0][0]

                p2_ed = p2[prev_ind:(curr_ind + 1), 1:5].copy()
                p2_ed[0, 3] = 0
                p2_ed_len = len(p2_ed)
                if p2_ed_len > p2_edges.shape[1]:
                    p2_edges = np.append(p2_edges, np.zeros((p2_edges.shape[0],
                                                             p2_ed_len - p2_edges.shape[1], 4)),
                                         axis=1)

                p2_edges[j, :p2_ed_len] = p2_ed

            if p1_edges.shape[1] < p2_edges.shape[1]:
                p1_edges = np.append(p1_edges, np.zeros((p1_edges.shape[0],
                                                         p2_edges.shape[1] - p1_edges.shape[1], 4)),
                                     axis=1)

            tv_each_time[i] = total_variation_edges(p1_edges, p2_edges)

    return tv_each_time


# Plot proportion routes correct
def plot_metric_over_time(setup_dict, save_dir, fl_pf_metric, fl_pf_time, fl_bsi_metric, fl_bsi_time,
                          ffbsi_metric=None, ffbsi_time=None):
    lags = setup_dict['lags']

    m = fl_pf_metric.shape[-1]

    # t_linspace = np.linspace(0, (m - 1) * setup_dict['time_interval'], m)
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

    fig, axes = plt.subplots(3, 2, sharex='all', sharey='all', figsize=(8, 6))
    for j, n in enumerate(setup_dict['fl_n_samps']):
        for k, lag in enumerate(lags):
            axes[j, 0].plot(t_linspace, fl_pf_metric[k, j], label=f'Lag: {lag}')
            lines[k], = axes[j, 1].plot(t_linspace, fl_bsi_metric[k, j], label=f'Lag: {lag}')

        if ffbsi_metric is not None:
            lines[len(lags)], = axes[j, 0].plot(t_linspace, ffbsi_metric[j], label='FFBSi')
            axes[j, 1].plot(t_linspace, ffbsi_metric[j], label='FFBSi')

    for j, n in enumerate(setup_dict['fl_n_samps']):
        for k, lag in enumerate(lags):
            axes[j, 0].text(left_start, up_start - k * shift, "{:.1f}".format(fl_pf_time[k, j]),
                            color=lines[k].get_color(),
                            fontsize=fontsize, transform=axes[j, 0].transAxes)
            axes[j, 1].text(left_start, up_start - k * shift, "{:.1f}".format(fl_bsi_time[k, j]),
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

    plt.savefig(save_dir + 'route_tv_compare.png', dpi=400)

    return fig, axes
