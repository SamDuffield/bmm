


import json
import os

import numpy as np
import pandas as pd

import bmm

from bmm.src.tools.graph import load_graph
from bmm.src.data.utils import source_data, read_data
from bmm.src.data.preprocess import longlat_polys_to_utm

_, process_data_path = source_data()

graph = load_graph()

osrm_data_path = '/Users/samddd/Main/Data/bayesian-map-matching/osrm/'
osrm_response_path = osrm_data_path + 'single_route_match_response'

with open(osrm_response_path) as f:
    response = json.load(f)


long_lat_mm_polyline = [a['location'] for a in response['tracepoints'] if a is not None]
utm_mm_polyline = longlat_polys_to_utm([long_lat_mm_polyline], to_crs=graph.graph['crs']).iloc[0]
utm_mm_polyline_arr = np.array(utm_mm_polyline)

bmm.plot(graph, polyline=utm_mm_polyline)


mm_model = bmm.GammaMapMatchingModel()
mm_model.gps_sd = 0.5
particles = bmm.offline_map_match(graph, utm_mm_polyline, 50, time_interval=15, mm_model=mm_model)

bmm.plot(graph, particles[0], utm_mm_polyline)

np.save(osrm_data_path + 'single_route_matched_particle', particles[0])

arr = np.load(osrm_data_path + 'single_route_matched_particle.npy')


data_path = process_data_path + "/data/portotaxi_06052014_06052014_utm_1730_1745.csv"
raw_data = read_data(data_path, 100).get_chunk()

# Select single route
route_index = 0
route_polyline = np.asarray(raw_data['POLYLINE_UTM'][route_index])


sub_route_plot_xlim = (532399.6154033017, 532860.7133234204)
sub_route_plot_ylim = (4556923.901931656, 4557388.957773835)


fig, ax = bmm.plot(graph, arr, route_polyline)
ax.set_xlim(sub_route_plot_xlim)
ax.set_ylim(sub_route_plot_ylim)







