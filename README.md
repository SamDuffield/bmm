# bmm: bayesian-map-matching
Map-matching using particle smoothing methods.

## Install
```
pip install bmm
```

## Load graph and convert to UTM
```python
import numpy as np
import pandas as pd
import osmnx as ox
import json

import bmm

graph = ox.graph_from_place('Porto, Portugal')
graph = ox.project_graph(graph)
```

## Load polyline and convert to UTM
```python
data_path = 'simulations/porto/test_route.csv'
polyline_longlat = json.loads(pd.read_csv(data_path)['POLYLINE'][0])
polyline_utm = bmm.long_lat_to_utm(polyline_longlat, graph)
```

## Offline map-matching
```python
matched_particles = bmm.offline_map_match(graph, polyline=polyline_utm, n_samps=100, timestamps=15)
```

## Online map-matching
```python
# Initiate with first observation
matched_particles = bmm.initiate_particles(graph, first_observation=polyline_utm[0], n_samps=100)

# Update when new observation comes in
matched_particles = bmm.update_particles(graph, matched_particles, new_observation=polyline_utm[1], time_interval=15)
```

## Plot
```python
bmm.plot(graph, particles=matched_particles, polyline=polyline_utm)
```
![porto_mm](simulations/porto/test_route.png?raw=true "Map-matched route - Porto")




