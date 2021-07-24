
import numpy as np
import pandas as pd
import osmnx as ox
import json

import bmm

# Download and project graph
graph = ox.graph_from_place('London, UK')
graph = ox.project_graph(graph)

# Generate synthetic route and polyline
generated_route, generated_polyline = bmm.sample_route(graph, timestamps=15, num_obs=20)

# Map-match
matched_particles = bmm.offline_map_match(graph, generated_polyline, n_samps=100, timestamps=15)

# Plot true route
bmm.plot(graph, generated_route, generated_polyline, particles_color='green')

# Plot map-matched particles
bmm.plot(graph, matched_particles, generated_polyline)
