################################################################################
# Module: prior_distance_exploration.py
# Description: Explore the distribution of the distances/speeds/travel times
#              of the vehicles.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import numpy as np
from bmm.src.tools.graph import load_graph
from bmm.src.data.utils import source_data, read_data
import bmm.src.tools.edges
import matplotlib.pyplot as plt
from scipy import stats
from tweedie import tweedie


def polyline_to_euclidean_distance(polyline):
    """
    Convert polyline into M-1 euclidean distances between points
    :param polyline: UTM polyline
    :return: list of distances in metres
    """
    poly_arr = np.asarray(polyline)
    return np.sqrt(np.sum(np.square(poly_arr[1:] - poly_arr[:-1]), axis=1))

# Source data paths
_, process_data_path = source_data()

# Load networkx graph and edges gdf
graph = load_graph()
edges_gdf = bmm.src.tools.edges.graph_edges_gdf(graph)

# Load taxi data
# data_path = bmm.src.data.utils.choose_data()
# raw_data = bmm.src.data.utils.read_data(data_path)
data_path = process_data_path + "/data/portotaxi_06052014_06052014_utm_1730_1745.csv"
raw_data = read_data(data_path)
# raw_data = read_data(data_path, 100).get_chunk()

# Extract distances
euclidean_dists = np.array([])
for poly in raw_data['POLYLINE_UTM']:
    euclidean_dists = np.append(euclidean_dists, polyline_to_euclidean_distance(poly))

# Inter-observation time
delta_obs = 15

euclidean_speeds = euclidean_dists / delta_obs

# Set maximum possible distance to travel between observations
v_max = 32

# Remove massive values (outliers)
euclidean_speeds = euclidean_speeds[euclidean_speeds < v_max]

# Probability of zero speed
zero_cut_off = 0
zero_prob = sum(euclidean_speeds < zero_cut_off) / len(euclidean_speeds)

# Remove zero values
euclidean_speeds = euclidean_speeds[euclidean_speeds >= zero_cut_off]

# Plot histogram
plt.figure()
plt.hist(euclidean_speeds, bins=100, density=True, alpha=0.5)
plt.xlabel('||y_0-y_1||/t')

# Define linspace
linsp = np.linspace(0, v_max, 1000)

# Fit Gamma manually
mean_v = np.mean(euclidean_speeds)
print(mean_v)
var_v = np.var(euclidean_speeds)
print(var_v)


bg = mean_v / var_v
ag = mean_v * bg
pdf_gamma = stats.gamma.pdf(linsp, ag, loc=0, scale=1/bg)
plt.plot(linsp, pdf_gamma, label="Fitted Gamma")

# Log data
log_euclidean_speeds = np.log(euclidean_speeds[euclidean_speeds > 0])

# # Fit Gaussian to log data (logGaussian to raw data)
mean_log_v = np.mean(log_euclidean_speeds)
var_log_v = np.var(log_euclidean_speeds)
pdf_loggaussian = stats.lognorm.pdf(linsp, scale=np.exp(mean_log_v), s=np.sqrt(var_log_v))
plt.plot(linsp, pdf_loggaussian, label="Fitted logGaussian")

# Add tweedie
# pdf_tweedie = tweedie.pdf(linsp, p=1.91, mu=8.09, phi=3.18)
pdf_tweedie = tweedie.pdf(linsp, p=1.9, mu=8.65, phi=2.53)
plt.plot(linsp, pdf_tweedie, label='Fitted Tweedie')

for p in np.linspace(1.1, 1.9, 9):
    pdf_tweedie = tweedie.pdf(linsp, p=p, mu=mean_v, phi=var_v / mean_v ** p)
    plt.plot(linsp, pdf_tweedie, label=f'p={p}')

# Add legend
plt.legend()

plt.show()

