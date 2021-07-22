import functools
import gc

import numpy as np
import osmnx as ox
from networkx import write_gpickle, read_gpickle

import bmm


def download_cambridge_graph(save_path):

    cambridge_ll_bbox = [52.245, 52.150, 0.220, 0.025]
    raw_graph = ox.graph_from_bbox(*cambridge_ll_bbox,
                                   truncate_by_edge=True,
                                   simplify=False,
                                   network_type='drive')

    projected_graph = ox.project_graph(raw_graph)

    simplified_graph = ox.simplify_graph(projected_graph)

    write_gpickle(simplified_graph, save_path)


# Load cam_graph of Cambridge
def load_graph(path):
    graph = read_gpickle(path)
    return graph


# Clear lru_cache
def clear_cache():
    gc.collect()
    wrappers = [
        a for a in gc.get_objects()
        if isinstance(a, functools._lru_cache_wrapper)]

    for wrapper in wrappers:
        wrapper.cache_clear()

