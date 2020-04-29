################################################################################
# Module: graph.py
# Description: Download/prune, project, simplify
#              and save road network using OSMNx.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import bmm.src.data.utils
from bmm.src.data.preprocess import bbox_ll
import osmnx as ox
import geopandas as gpd
import networkx as nx
import os
import json
import tkinter.messagebox
import tkinter.filedialog


def download_full_graph():
    """
    Returns a graph of road network in the form of a networkx multidigraph, based off latest OSM data.
    :return: networkx object
    """

    graph_full = ox.graph_from_bbox(bbox_ll[0], bbox_ll[1], bbox_ll[2], bbox_ll[3], network_type='drive_service',
                                    truncate_by_edge=True, simplify=False, name='latest')

    return graph_full


def load_road_types():
    """
    Load pre-defined list of road types accessible by car.
    :return: list of road types (strings)
    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    source_file = open(curdir + "/road_types.txt", "r")
    json_road_types = source_file.read()
    source_file.close()

    road_types = json.loads(json_road_types)

    return road_types


def prune_non_highways(graph):
    """
    Removes all edges from graph that aren't for highways (cars) and subsequent isolated nodes.
    :param graph: Full graph with variety of edges
    :return: Graph with only highway type edges
    """
    pruned_graph = graph.copy()

    road_types = load_road_types()

    for u, v, k in graph.edges:
        edge_data = graph.get_edge_data(u, v, k)

        if 'highway' not in edge_data.keys():
            pruned_graph.remove_edge(u, v, k)
        else:
            highway = edge_data['highway']
            if type(highway) == str:
                highway = [highway]

            remove = True
            for val in highway:
                if val in road_types:
                    remove = False

            if remove:
                pruned_graph.remove_edge(u, v, k)

    pruned_graph = ox.remove_isolated_nodes(pruned_graph)

    return pruned_graph


def extract_full_graph_from_osm(osm_path, osm_data_name="external"):
    """
    Extracts graph from a pre-downloaded OSM file (either xml or pbf format).
    Typically useful if using older data and thus latest OSM map might be too up to date.
    You can download old OSM files from http://download.geofabrik.de/
    e.g. http://download.geofabrik.de/europe/portugal.html -> raw directory index - > *.osm.pbf
    If OSM file, this function requires osmconvert, which can be installed with:
        Linux: apt install osmctools
        Mac OS: brew install interline-io/planetutils/osmctools
    :param osm_path: path of OSM pbf or xml file
    :param osm_data_name: name of data (i.e. location and data)
    :return: networkx graph, pruned but not simplified or projected
    """

    # Source data paths
    _, process_data_path = bmm.src.data.utils.source_data()

    # Check extension of data
    if osm_path[-3:] == 'pbf':
        # Convert pbf to xml
        _, process_data_path = bmm.src.data.utils.source_data()

        convert_path = process_data_path + '/graphs/' + os.path.basename(osm_path)[:-3] + 'xml'

        # Use osmconvert to convert
        osmconvert_cmd = "osmconvert " + osm_path \
                         + " -b={},{},{},{}".format(bbox_ll[3], bbox_ll[1], bbox_ll[2], bbox_ll[0]) \
                         + "--drop-broken-refs --complete-ways" \
                         + " -o=" + convert_path

        os.system(osmconvert_cmd)

    elif osm_path[-3:] == 'xml':
        convert_path = osm_path
    else:
        raise ValueError('OSM file must be either .xml or .pbf')

    graph_full = ox.graph_from_file(convert_path, name=osm_data_name, simplify=False)

    graph_full = ox.truncate_graph_bbox(graph_full, bbox_ll[0], bbox_ll[1], bbox_ll[2], bbox_ll[3])

    graph_full = prune_non_highways(graph_full)

    return graph_full


def remove_unconnected_islands(graph):
    """
    Removes road islands not connected to the main graph.
    :param graph: networkx object
    :return: trimmmed graph, networkx object
    """

    # Find largest island
    largest_component = max(nx.weakly_connected_components(graph), key=len)

    # Create a subgraph of G consisting only of this component:
    G2 = graph.subgraph(largest_component)

    return G2


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
        # path = os.environ.get('GRAPH_PATH')
        curdir = os.path.dirname(os.path.abspath(__file__))
        source_file = open(curdir + "/graph_source", "r")
        path = source_file.read()
        source_file.close()

    return ox.load_graphml(folder=os.path.dirname(path), filename=os.path.basename(path))


def get_bbox_from_graph(graph):
    """
    Extract bounding box (extreme coordinates of nodes) from graph
    :param graph: networkx object
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
    :param polyline: standard format
    :param axis: 0 for longitude, 1 for latitude
    :return: elements of polyline along single axis
    """
    return [coordinate[axis] for coordinate in polyline if coordinate is not None]


if __name__ == "__main__":
    # Source data paths
    _, process_data_path = bmm.src.data.utils.source_data()

    # Directory to save graphs
    graph_dir = process_data_path + '/graphs/'

    # Create graph folder in process_data_path if it doesn't already exist
    if not os.path.exists(process_data_path + '/graphs/'):
        os.mkdir(process_data_path + '/graphs/')

    # Ask user if they have pre-downloaded OSM data
    #     You can download old OSM files from http://download.geofabrik.de/
    #     e.g. http://download.geofabrik.de/europe/portugal.html -> raw directory index - > *.osm.pbf
    root = tkinter.Tk()
    osm_data = tkinter.messagebox.askquestion("OSM data",
                                              "Do you have OSM data already downloaded (xml or pbf) to construct a "
                                              "graph from? (otherwise we'll just download the latest graph)")
    if osm_data == 'yes':
        # Get data location
        root.update()
        osm_data_path = tkinter.filedialog.askopenfilename(parent=root, title='Locate OSM file (xml or pbf)')
        root.update()
        root.destroy()

        # Get name of data
        data_name = os.path.basename(osm_data_path)[:-3]

        # Extract full graph (many nodes, each edge a straight line)
        full_graph = extract_full_graph_from_osm(osm_data_path, data_name)

        # Remove islands
        full_graph = remove_unconnected_islands(full_graph)

    else:
        root.destroy()
        # Download full graph (many nodes, each edge a straight line)
        full_graph = download_full_graph()


    # Save full graph
    graph_base_name = bmm.src.data.utils.project_title + '_graph_' + full_graph.name
    save_graph(full_graph, graph_dir + graph_base_name + 'LL_full.graphml')

    # Project graph to UTM and save
    project_graph = ox.project_graph(full_graph)
    save_graph(project_graph, graph_dir + graph_base_name + '_full.graphml')

    # Simplify graph (fewer nodes, edges incorporate non-straight geometry) and save
    simplified_graph = ox.simplify_graph(project_graph)
    save_graph(simplified_graph, graph_dir + graph_base_name + '_simple.graphml')

    # Save in text file path for load_graph to call later
    curdir = os.path.dirname(os.path.abspath(__file__))
    source_file = open(curdir + "/graph_source", "w")
    source_file.write(graph_dir + graph_base_name + '_simple.graphml')
    source_file.close()

    # # Plot simplified projected graph
    # fig, ax = plot_graph(simplified_graph)
    # plt.show(block=True)

    # Load simplified projected graph with
    # from tools.graph import load_graph
    # graph = load_graph()
