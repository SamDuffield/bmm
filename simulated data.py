import os

import bmm
import osmnx as ox

# Load (or download) graph for Cambridge
graph_dir = '/Users/samddd/Main/Data/bayesian-map-matching/graphs/Cambridge/'
graph_name = 'cambridge_latest_utm_simplified'
graph_path = graph_dir + graph_name + '.graphml'

if not os.path.exists(graph_path):
    # location_str = 'Cambridge, UK'
    # raw_graph = ox.graph_from_place(location_str,
    #                                 truncate_by_edge=True,
    #                                 simplify=False,
    #                                 network_type='drive_service')

    cambridge_ll_bbox = [52.245, 52.150, 0.220, 0.025]
    raw_graph = ox.graph_from_bbox(*cambridge_ll_bbox,
                                   truncate_by_edge=True,
                                   simplify=False,
                                   network_type='drive_service')

    projected_graph = ox.project_graph(raw_graph)
    simplified_graph = ox.simplify_graph(projected_graph)

    ox.save_graphml(simplified_graph, graph_path)

    graph = simplified_graph

    del projected_graph, simplified_graph, raw_graph

else:
    graph = ox.load_graphml(graph_path)











