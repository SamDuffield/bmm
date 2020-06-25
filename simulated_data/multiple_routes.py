import os

import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt

import bmm

from simulated_data.clean_intersections import clean_intersections_graph
from simulated_data.utils import sample_route, rmse

########################################################################################################################
# Setup

np.random.seed(100)

num_routes = 100

# Model parameters
time_interval = 15
route_length = 50
gps_sd = 7.5

# Inference parameters
n_samps = np.array([20, 40, 60, 80, 100])

lags = np.array([1, 3, 5])

max_rejections = 3

initial_truncation = 30


save_dir = '/Users/samddd/Main/Data/bayesian-map-matching/simulations/' + str(num_routes) + '_routes/'
########################################################################################################################


# Where to store graph
graph_dir = '/Users/samddd/Main/Data/bayesian-map-matching/graphs/Cambridge/'
graph_name = 'cambridge_latest_utm_simplified_clean_int_rde'
graph_path = graph_dir + graph_name + '.graphml'

# Download/load graph
if not os.path.exists(graph_path):
    cambridge_ll_bbox = [52.245, 52.150, 0.220, 0.025]
    raw_graph = ox.graph_from_bbox(*cambridge_ll_bbox,
                                   truncate_by_edge=True,
                                   simplify=False,
                                   network_type='drive')

    projected_graph = ox.project_graph(raw_graph)
    simplified_graph = ox.simplify_graph(projected_graph)

    clean_int_simplified_graph = clean_intersections_graph(simplified_graph, rebuild_graph=True)


    def remove_dead_ends(graph):

        pruned_graph = graph.copy()

        for u in graph.nodes:

            u_in_edges = graph.in_edges(u)
            u_out_edges = graph.out_edges(u)

            if len(u_out_edges) == 0 or (len(u_out_edges) == 1 and len(u_in_edges) == 1
                                         and list(u_in_edges)[0][::-1] in list(u_out_edges)):
                pruned_graph.remove_node(u)

        return pruned_graph


    simplified_clean_int_rde_graph = clean_int_simplified_graph

    # Removing dead-ends creates more dead-ends - so repeat a few times
    prune_iters = 10
    for i in range(prune_iters):
        simplified_clean_int_rde_graph = remove_dead_ends(simplified_clean_int_rde_graph)

    ox.save_graphml(simplified_clean_int_rde_graph, graph_path)

    graph = simplified_clean_int_rde_graph

    del raw_graph, projected_graph, simplified_graph, clean_int_simplified_graph, simplified_clean_int_rde_graph

else:
    graph = ox.load_graphml(graph_path)

# Initiate map-matching probabilistic model
mm_model = bmm.GammaMapMatchingModel()
mm_model.gps_sd = gps_sd

end_time = time_interval * (route_length - 1)

sampled_routes = [sample_route(graph, mm_model, time_interval, route_length) for _ in range(num_routes)]
fin_times = np.array([route[-1, 0] for route in sampled_routes])

# Ensure all routes are of full length
while not all(fin_times == end_time):
    for i in range(num_routes):
        if fin_times[i] != end_time:
            sampled_routes[i] = sample_route(graph, mm_model, time_interval, route_length)
            fin_times[i] = sampled_routes[i][-1, 0]

# Extract true positions at observation times
cartesianised_routes = [bmm.cartesianise_path(graph, path, t_column=True, observation_time_only=True)
                        for path in sampled_routes]

# Add noise to generate observations
observations = [path + mm_model.gps_sd * np.random.normal(size=path.shape) for path in cartesianised_routes]


fl_pf_rmse = np.zeros((num_routes, len(lags), len(n_samps)))
fl_pf_time = np.zeros((num_routes, len(lags), len(n_samps)))
fl_bsi_rmse = np.zeros((num_routes, len(lags), len(n_samps)))
fl_bsi_time = np.zeros((num_routes, len(lags), len(n_samps)))
ffbsi_rmse = np.zeros((num_routes, len(n_samps)))
ffbsi_time = np.zeros((num_routes, len(n_samps)))


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(num_routes):
    try:
        for j, n in enumerate(n_samps):
            for k, lag in enumerate(lags):
                fl_pf_inferred_route = bmm._offline_map_match_fl(graph,
                                                                 observations[i],
                                                                 n,
                                                                 time_interval=time_interval,
                                                                 lag=lag,
                                                                 update='PF',
                                                                 max_rejections=max_rejections,
                                                                 initial_d_truncate=initial_truncation)
                fl_pf_rmse[i, k, j] = rmse(graph, fl_pf_inferred_route, cartesianised_routes[i])
                fl_pf_time[i, k, j] = fl_pf_inferred_route.time
                print(i, j, k)

                fl_bsi_inferred_route = bmm._offline_map_match_fl(graph,
                                                                  observations[i],
                                                                  n,
                                                                  time_interval=time_interval,
                                                                  lag=lag,
                                                                  update='BSi',
                                                                  max_rejections=max_rejections,
                                                                  initial_d_truncate=initial_truncation)
                fl_bsi_rmse[i, k, j] = rmse(graph, fl_bsi_inferred_route, cartesianised_routes[i])
                fl_bsi_time[i, k, j] = fl_bsi_inferred_route.time

                print(i, j, k)

            ffbsi_inferred_route = bmm.offline_map_match(graph,
                                                         observations[i],
                                                         n,
                                                         time_interval=time_interval,
                                                         max_rejections=max_rejections,
                                                         initial_d_truncate=initial_truncation)
            ffbsi_rmse[i, j] = rmse(graph, ffbsi_inferred_route, cartesianised_routes[i])
            ffbsi_time[i, j] = ffbsi_inferred_route.time
    except:
        fl_pf_rmse[i] *= 0
        continue

    np.save(save_dir + 'fl_pf_rmse', fl_pf_rmse[:i])
    np.save(save_dir + 'fl_bsi_rmse', fl_bsi_rmse[:i])
    np.save(save_dir + 'ffbsi_rmse', ffbsi_rmse[:i])

    np.save(save_dir + 'fl_pf_time', fl_pf_time[:i])
    np.save(save_dir + 'fl_bsi_time', fl_bsi_time[:i])
    np.save(save_dir + 'ffbsi_time', ffbsi_time[:i])

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
for k, lag in enumerate(lags):
    ax1.plot(n_samps, np.mean(fl_pf_rmse[:, k, :], axis=0), label="FL-PF Lag = " + str(lag))
    ax2.plot(n_samps, np.mean(fl_bsi_rmse[:, k, :], axis=0), label="FL-BSi Lag = " + str(lag))

ax1.plot(n_samps, np.mean(ffbsi_rmse, axis=0), label="FFBSi")
ax2.plot(n_samps, np.mean(ffbsi_rmse, axis=0), label="FFBSi")


def make_pretty(fig, ax):
    ax.legend()
    ax.set_ylabel('RMSE')
    ax.set_xlabel('n')
    fig.tight_layout()


make_pretty(fig1, ax1)
make_pretty(fig2, ax2)

plt.show()


#
# fl_pf_rmse = np.load(save_dir + 'fl_pf_rmse.npy')
# fl_bsi_rmse = np.load(save_dir + 'fl_bsi_rmse.npy')
# ffbsi_rmse = np.load(save_dir + 'ffbsi_rmse.npy')
#
# fl_pf_time = np.load(save_dir + 'fl_pf_time.npy')
# fl_bsi_time = np.load(save_dir + 'fl_bsi_time.npy')
# ffbsi_time = np.load(save_dir + 'ffbsi_time.npy')
#
# is_zero = np.array([0. in mat for mat in ffbsi_time])
# keep_arr = ~np.array(is_zero)
# np.save(save_dir + 'fl_pf_rmse', fl_pf_rmse[keep_arr])
# np.save(save_dir + 'fl_bsi_rmse', fl_bsi_rmse[keep_arr])
# np.save(save_dir + 'ffbsi_rmse', ffbsi_rmse[keep_arr])
#
# np.save(save_dir + 'fl_pf_time', fl_pf_time[keep_arr])
# np.save(save_dir + 'fl_bsi_time', fl_bsi_time[keep_arr])
# np.save(save_dir + 'ffbsi_time', ffbsi_time[keep_arr])


