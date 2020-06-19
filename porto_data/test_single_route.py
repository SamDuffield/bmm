
import json
import os

import numpy as np
import pandas as pd

import bmm

from bmm.src.tools.graph import load_graph
from bmm.src.data.utils import source_data, read_data

_, process_data_path = source_data()

graph = load_graph()

run_indicator = 2

# Load taxi data
# data_path = data.utils.choose_data()
data_path = process_data_path + "/data/portotaxi_06052014_06052014_utm_1730_1745.csv"
raw_data = read_data(data_path, 100).get_chunk()

# Select single route
route_index = 0
route_polyline = np.asarray(raw_data['POLYLINE_UTM'][route_index])

# # Load particles
# save_dir = f'/Users/samddd/Main/Data/bayesian-map-matching/simulations/porto/{route_index}/{run_indicator}/'
# fl_pf_routes = np.load(save_dir + 'fl_pf.npy', allow_pickle=True)
# fl_bsi_routes = np.load(save_dir + 'fl_bsi.npy', allow_pickle=True)


def route_error_inds(route_particles):
    error_inds = []

    for n, particle in enumerate(route_particles):
        for i in range(1, len(particle)):
            if not ((particle[i, 1] == particle[i-1, 2])
                    or (particle[i, 1] == particle[i-1, 1] and particle[i, 2] == particle[i-1, 2])):
                error_inds.append([n, i])
    return error_inds


np.random.seed(0)
particles = bmm._offline_map_match_fl(graph, route_polyline, 200, 15, lag=10, update='PF')

error_inds = route_error_inds(particles)
print(error_inds)

