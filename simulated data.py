import os

import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from clean_intersections import clean_intersections_graph

import bmm

########################################################################################################################
# Setup

np.random.seed(0)

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
    prune_iters = 100
    for i in range(prune_iters):
        simplified_clean_int_rde_graph = remove_dead_ends(simplified_clean_int_rde_graph)

    ox.save_graphml(simplified_clean_int_rde_graph, graph_path)

    graph = simplified_clean_int_rde_graph

    del raw_graph, projected_graph, simplified_graph, clean_int_simplified_graph, simplified_clean_int_rde_graph

else:
    graph = ox.load_graphml(graph_path)

# Initiate map-matching probabilistic model
mm_model = bmm.SimpleMapMatchingModel()
mm_model.gps_sd = gps_sd


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


# RMSE given particle cloud
def rmse(particles, truth):
    obs_time_particles = np.zeros((len(particles), len(truth), 2))
    for i, particle in enumerate(particles):
        obs_time_particles[i] = bmm.cartesianise_path(graph, particle, t_column=True, observation_time_only=True)

    squared_error = np.square(obs_time_particles - truth)
    return np.sqrt(np.mean(squared_error))


fl_pf_rmse = np.zeros((num_routes, len(lags), len(n_samps)))
fl_pf_time = np.zeros((num_routes, len(lags), len(n_samps)))
fl_bsi_rmse = np.zeros((num_routes, len(lags), len(n_samps)))
fl_bsi_time = np.zeros((num_routes, len(lags), len(n_samps)))
ffbsi_rmse = np.zeros((num_routes, len(n_samps)))
ffbsi_time = np.zeros((num_routes, len(n_samps)))

save_dir = '/Users/samddd/Main/Data/bayesian-map-matching/simulations/' + str(num_routes) + '_routes/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(num_routes):
    for j, n in enumerate(n_samps):
        for k, lag in enumerate(lags):
            fl_pf_inferred_route = bmm._offline_map_match_fl(graph,
                                                             observations[i],
                                                             n,
                                                             time_interval=time_interval,
                                                             lag=lag,
                                                             update='PF',
                                                             max_rejections=max_rejections,
                                                             initial_truncation=initial_truncation)
            fl_pf_rmse[i, k, j] = rmse(fl_pf_inferred_route, cartesianised_routes[i])
            fl_pf_time[i, k, j] = fl_pf_inferred_route.time
            print(i, j, k)

            fl_bsi_inferred_route = bmm._offline_map_match_fl(graph,
                                                              observations[i],
                                                              n,
                                                              time_interval=time_interval,
                                                              lag=lag,
                                                              update='BSi',
                                                              max_rejections=max_rejections,
                                                              initial_truncation=initial_truncation)
            fl_bsi_rmse[i, k, j] = rmse(fl_bsi_inferred_route, cartesianised_routes[i])
            fl_bsi_time[i, k, j] = fl_bsi_inferred_route.time

            print(i, j, k)

        ffbsi_inferred_route = bmm.offline_map_match(graph,
                                                     observations[i],
                                                     n,
                                                     time_interval=time_interval,
                                                     max_rejections=max_rejections,
                                                     initial_truncation=initial_truncation)
        ffbsi_rmse[i, j] = rmse(ffbsi_inferred_route, cartesianised_routes[i])
        ffbsi_time[i, j] = ffbsi_inferred_route.time

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
ax1.legend()

ax2.plot(n_samps, np.mean(ffbsi_rmse, axis=0), label="FFBSi")
ax2.legend()

plt.tight_layout()

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
