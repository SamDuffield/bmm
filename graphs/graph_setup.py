################################################################################
# Module: graph_setup.py
# Description: Trim raw data by timestamp (if required - working on a small data
#              set initially is advised).
#              Trim and split to ensure all coordinates within a bounding box.
#              Convert longitude-latitude coordinates to UTM.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import data.utils
from data.preprocess import bbox_ll
import osmnx as ox
import geopandas as gpd
import os
import matplotlib.pyplot as plt


def download_full_graph():
    """
    Returns a graph of road network in the form of a networkx multidigraph
    :return: networkx object
    """

    graph_full = ox.graph_from_bbox(bbox_ll[0], bbox_ll[1], bbox_ll[2], bbox_ll[3], network_type='drive_service',
                                    truncate_by_edge=True, simplify=False)

    return graph_full


def save_graph(graph, path):
    """
    Saves graph as an OSM (XML) file
    :param graph: networkx object
    :param path: location to save
    """
    ox.save_graphml(graph, folder=os.path.dirname(path), filename=os.path.basename(path))


def load_graph(path):
    """
    Loads saved graph
    :param path: location of graph
    :return: loaded graph
    """
    return ox.load_graphml(folder=os.path.dirname(path), filename=os.path.basename(path))


def get_bbox_from_graph(graph):
    """
    Extract bounding box (extreme coordinates of nodes) from graph
    :param graph:
    :return: [north, south, east, west]
    """
    nodes, data = zip(*graph.nodes(data=True))
    gdf_nodes = gpd.GeoDataFrame(list(data), index=nodes)

    bbox = [max(gdf_nodes['y']), min(gdf_nodes['y']),
            max(gdf_nodes['x']), min(gdf_nodes['x'])]

    return bbox


if __name__ == "__main__":
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Directory to save graphs
    graph_dir = process_data_path + '/graphs/'

    # Download full graph (many nodes, each edge a straight line)
    full_graph = download_full_graph()
    save_graph(full_graph, graph_dir + data.utils.project_title + '_graph_full_LL.graphml')

    # Project graph to UTM
    project_graph = ox.project_graph(full_graph)
    save_graph(project_graph, graph_dir + data.utils.project_title + '_graph_full.graphml')

    # Simplify graph (fewer nodes, edges incorporate non-straight geometry)
    simplified_graph = ox.simplify_graph(full_graph)
    save_graph(simplified_graph, graph_dir + data.utils.project_title + '_graph_simple.graphml')

    # Plot simplified projected graph
    ox.plot_graph(simplified_graph, fig_height=50, show=False, close=False, equal_aspect=True)
    plt.tight_layout()
    plt.show(block=False)
