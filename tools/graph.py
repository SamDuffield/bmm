################################################################################
# Module: graph.py
# Description: Download, project, simplify and save road network using OSMNx.
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


def load_graph(path=None):
    """
    Loads saved graph
    :param path: location of graph
    :return: loaded graph
    """
    if path is None:
        _, process_data_path = data.utils.source_data()
        path = process_data_path + '/graphs/' + data.utils.project_title + '_graph_simple.graphml'

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


def polyline_axis(polyline, axis):
    """
    Takes polyline and returns only latitudes or longitudes. (Skips any None coordinates)
    :param polyline: Standard format
    :param axis: 0 for longitude, 1 for latitude
    :return: elements of polyline along single axis
    """
    return [coordinate[axis] for coordinate in polyline if coordinate is not None]


def plot_graph(graph, polyline=None, edges_to_highlight=None):
    """
    Plot OSMNx graph, with optional polyline and highlighted edges.
    :param graph:
    :param polyline:
    :param edges_to_highlight:
    :return: fig, ax of plotted road network (plus polyline)
    """
    if edges_to_highlight is not None:
        edge_colours = ['blue' if [u, v] in edges_to_highlight or [v, u] in edges_to_highlight
                        else 'grey' for u, v, d in graph.edges]
    else:
        edge_colours = 'grey'

    fig, ax = ox.plot_graph(graph, show=False, close=False, equal_aspect=True, edge_color=edge_colours)

    if polyline is not None:
        if len(polyline) > 1:
            ax.scatter(polyline_axis(polyline, 0), polyline_axis(polyline, 1), c='red')
            ax.scatter(polyline[-1][0], polyline[-1][1], c='red', edgecolor='blue', linewidth=2)
        ax.scatter(polyline[0][0], polyline[0][1], c='red', edgecolor='green', linewidth=2)

    plt.tight_layout()

    return fig, ax


if __name__ == "__main__":
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Directory to save graphs
    graph_dir = process_data_path + '/graphs/'

    # Create graph folder in process_data_path if it doesn't already exist
    if not os.path.exists(process_data_path + '/graphs/'):
        os.mkdir(process_data_path + '/graphs/')

    # Download full graph (many nodes, each edge a straight line)
    full_graph = download_full_graph()
    save_graph(full_graph, graph_dir + data.utils.project_title + '_graph_full_LL.graphml')

    # Project graph to UTM
    project_graph = ox.project_graph(full_graph)
    save_graph(project_graph, graph_dir + data.utils.project_title + '_graph_full.graphml')

    # Simplify graph (fewer nodes, edges incorporate non-straight geometry)
    simplified_graph = ox.simplify_graph(project_graph)
    save_graph(simplified_graph, graph_dir + data.utils.project_title + '_graph_simple.graphml')

    # Plot simplified projected graph
    fig, ax = plot_graph(simplified_graph)
    plt.show(block=True)

    # Load simplified projected graph with
    # from tools.graph import load_graph
    # graph = load_graph()
