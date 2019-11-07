################################################################################
# Module: map_matching_mcmc.py
# Description: Infer route taken by vehicle given sparse observations.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import data
from tools.graph import load_graph, remove_unconnected_islands
import tools.edges
import archive.sampling
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx

# Observation time interval
time_bet_obs = 15

# Maximum possible speed in m/s
v_max = 20

# Maximum distance travelled between observations
d_obs_max = v_max * time_bet_obs


def trim_graph_polyline(graph, polyline, distance):
    """
    Trims graph to only nodes within a distance of polyline.
    Choose large distance to avoid discarding long edges.
    :param graph: networkx object
    :param polyline: list of coordinates
    :param distance: distance beyond which to discard nodes
    :return: subgraph
    """
    graph_coords = [[node, data['x'], data['y']] for node, data in graph.nodes(data=True)]
    df = pd.DataFrame(graph_coords, columns=['node', 'x', 'y']).set_index('node')

    node_subset = set({})

    for coord in polyline:
        df['reference_x'] = coord[0]
        df['reference_y'] = coord[1]

        distances = ox.euclidean_dist_vec(y1=df['reference_y'],
                                          x1=df['reference_x'],
                                          y2=df['y'],
                                          x2=df['x'])

        node_subset.update(df[distances < distance].index.to_list())

    # copy nodes into new graph
    G2 = graph.__class__()
    G2.add_nodes_from((n, graph.nodes[n]) for n in node_subset)

    # copy edges to new graph, including parallel edges
    G2.add_edges_from((n, nbr, key, d)
                      for n, nbrs in graph.adj.items() if n in node_subset
                      for nbr, keydict in nbrs.items() if nbr in node_subset
                      for key, d in keydict.items())

    # update graph attribute dict
    G2.graph.update(graph.graph)

    # Remove isolated nodes
    G2 = ox.remove_isolated_nodes(G2)

    # Remove islands
    G2 = remove_unconnected_islands(G2)

    return G2


def induce_subgraph_within_distance(graph, polyline, distance=v_max*time_bet_obs/2):
    """
    Returns smaller graph with only edges that are deemed possible to traverse given a polyline.
    Modified ox.induce_subgraph
    :param graph: full networkx road network
    :param polyline: list of coordinates
    :param distance: distance beyond which to trim edges
    :return: networkx subgraph
    """

    trimmed_graph = trim_graph_polyline(graph, polyline, distance*2)

    graph_edges = tools.edges.graph_edges_gdf(trimmed_graph)

    node_subset = set({})
    # graph_coords = [[node, data['x'], data['y']] for node, data in graph.nodes(data=True)]
    # df = pd.DataFrame(graph_coords, columns=['node', 'x', 'y']).set_index('node')

    for coord in polyline:
        # df['reference_x'] = coord[0]
        # df['reference_y'] = coord[1]
        #
        # distances = ox.euclidean_dist_vec(y1=df['reference_y'],
        #                                   x1=df['reference_x'],
        #                                   y2=df['y'],
        #                                   x2=df['x'])

        close_edges = tools.edges.get_edges_within_dist(graph_edges, coord, distance)

        close_nodes = set(close_edges['u'].values.tolist() + close_edges['v'].values.tolist())

        node_subset.update(close_nodes)

    # copy nodes into new graph
    G2 = graph.__class__()
    G2.add_nodes_from((n, graph.nodes[n]) for n in node_subset)

    # copy edges to new graph, including parallel edges
    G2.add_edges_from((n, nbr, key, d)
                      for n, nbrs in graph.adj.items() if n in node_subset
                      for nbr, keydict in nbrs.items() if nbr in node_subset
                      for key, d in keydict.items())

    # update graph attribute dict
    G2.graph.update(graph.graph)

    # Remove isolated nodes
    G2 = ox.remove_isolated_nodes(G2)

    # Remove islands
    G2 = remove_unconnected_islands(G2)

    return G2


def sample_close_edge(graph_edges, coord, prob=False):

    close_edges = tools.edges.get_edges_within_dist(graph_edges, coord)

    weights = np.exp(-0.5 / tools.edges.sigma2_GPS * close_edges['distance_to_obs'].values ** 2)
    weights /= np.sum(weights)

    sampled_index = np.random.choice(len(weights), 1, True, weights)[0]
    sampled_edge = close_edges.iloc[sampled_index, :3].values.tolist()

    if prob:
        return sampled_edge, weights[sampled_index]

    else:
        return sampled_edge


def path_sample(sub_graph, polyline, sub_graph_edges=None):

    if sub_graph_edges is None:
        sub_graph_edges = tools.edges.graph_edges_gdf(sub_graph)

    # Sample x0
    x_gdf = archive.sampling.sample_x0(sub_graph_edges, polyline[0], 1)

    for m in range(1, len(polyline)):
        prev_u, prev_v = x_gdf.iloc[-1, :2].values

        edge_m, prob_edge_m = sample_close_edge(sub_graph_edges, polyline[m], prob=True)


def possible_paths(graph, polyline):

    x0 = polyline[0]
    x1 = polyline[1]

    x0x1_subgraph = induce_subgraph_within_distance(graph, polyline[:2])
    x0x1_subgraph_edges = tools.edges.graph_edges_gdf(x0x1_subgraph)

    x0_near_edges = tools.edges.get_edges_within_dist(x0x1_subgraph_edges, x0)
    x1_near_edges = tools.edges.get_edges_within_dist(x0x1_subgraph_edges, x1)

    routes = []

    for v in x0_near_edges['v']:
        for u in x1_near_edges['u']:
            routes += [simp_path for simp_path in nx.all_simple_paths(x0x1_subgraph, v, u, d_obs_max)]

    return routes


def sample_trajectory(sub_graph, polyline, sub_graph_edges=None, max_iters=100):

    if sub_graph_edges is None:
        sub_graph_edges = tools.edges.graph_edges_gdf(sub_graph)

    for i in range(max_iters):
        x = archive.sampling.sample_x0(sub_graph_edges, polyline[0], 1)
        x_prev = x.iloc[0]

        for j in range(1, len(polyline)):
            y_j = polyline[j]
            route_found = False
            while not route_found:
                # Sample x_j
                x_j = archive.sampling.sample_x0(sub_graph_edges, polyline[j], 1).iloc[0]

                if x_prev['u'] == x_j['u'] and x_prev['v'] == x_j['v'] and x_prev['k'] == x_j['k']:
                    x = x.append(x_j)
                    x_prev = x_j
                    route_found = True
                else:
                    yjmin1_yj_subgraph = induce_subgraph_within_distance(sub_graph, polyline[j-1:j+1])

                    # Get all paths from x_prev[v] to x_j[u]
                    all_paths = [path for path
                                 in nx.all_simple_paths(yjmin1_yj_subgraph, x_prev['v'], x_j['u'], d_obs_max)]



                    if len(all_paths) == 0:
                        route_found = False
                    else:


    return routes





if __name__ == '__main__':
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Load networkx graph and edges gdf
    graph = load_graph()
    edges_gdf = tools.edges.graph_edges_gdf(graph)

    # Load taxi data
    data_path = data.utils.choose_data()
    raw_data = data.utils.read_data(data_path)

    # Select single polyline
    single_index = 6
    poly_single = raw_data['POLYLINE_UTM'][single_index]

    # Number of observations
    M_obs = len(poly_single)

    # Plot full graph with polyline
    # tools.edges.plot_graph(graph, raw_data['POLYLINE_UTM'][single_index])

    # Induce subgraph around polyline
    subgraph_poly = induce_subgraph_within_distance(graph, poly_single)

    # Subgraph edges
    subgraph_poly_edges = tools.edges.graph_edges_gdf(subgraph_poly)

    # Plot subgraph
    tools.edges.plot_graph(subgraph_poly, poly_single)


