import os
import json

import numpy as np
import osmnx as ox
import pandas as pd
import bmm

porto_sim_dir = os.getcwd()
graph_path = porto_sim_dir + '/portotaxi_graph_portugal-140101.osm._simple.graphml'
graph = ox.load_graphml(graph_path)

test_route_data_path = ''   # Download from https://archive.ics.uci.edu/ml/datasets/Taxi+Service+Trajectory+-+Prediction+Challenge,+ECML+PKDD+2015
save_dir = porto_sim_dir + '/bulk_output/'


# Load long-lat polylines
polylines_ll = pd.read_csv(test_route_data_path, chunksize=5000).get_chunk()['POLYLINE'].apply(json.loads)
polylines_ll = [np.array(a) for a in polylines_ll]

num_routes = 500
min_length = 20
max_length = 60
polylines_ll = [c for c in polylines_ll if min_length <= len(c) <= max_length]

mm_routes = np.empty(num_routes, dtype=object)
failed_routes = []
i = 0
j = 0
while j < num_routes:
    poly = bmm.long_lat_to_utm(polylines_ll[i], graph)
    print('Route attempt:', i, len(poly))
    print('Successful routes:', j)

    try:
        mm_route = bmm.offline_map_match(graph, poly, 100, 15.)
        mm_routes[j] = mm_route
        j += 1

    except:
        # Typically missing data in the polyline or the polyline leaves the graph
        print(i, 'mm failed')
        failed_routes.append(i)
    i += 1

failed_routes = np.array(failed_routes)
np.save(save_dir + 'mm_routes', mm_routes)
np.save(save_dir + 'failed_routes', failed_routes)

mm_routes = np.load(save_dir + 'mm_routes.npy', allow_pickle=True)
failed_routes = np.load(save_dir + 'failed_routes.npy')


def is_multi_modal(particles):
    route_nodes = particles.route_nodes()
    return any([not np.array_equal(route_nodes[0], a) for a in route_nodes[1:]])


mm_routes_multi_modal = np.array([is_multi_modal(a) for a in mm_routes])
print(mm_routes_multi_modal.sum(), '/', mm_routes_multi_modal.size, ' routes multi-modal')

