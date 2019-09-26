################################################################################
# Module: prior_speed_exploration.py
# Description: Explore the distribution of the distances/speeds/travel times
#              of the vehicles.
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import numpy as np
import data
from tools.graph import load_graph
import tools.edges
import tools.sampling
import matplotlib.pyplot as plt
import osmnx as ox
from scipy import stats

import seaborn
seaborn.set_style(
    "whitegrid", {'axes.grid': False, 'axes.spines.right': False, 'axes.spines.top': False})


def polyline_to_euclidean_distance(polyline):
    """
    Convert polyline into M-1 euclidean distances between points
    :param polyline: UTM polyline
    :return: list of distances in metres
    """
    return [ox.euclidean_dist_vec(polyline[i][1], polyline[i][0], polyline[i+1][1], polyline[i+1][0])
            for i in range(len(polyline) - 1)]


if __name__ == '__main__':
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Load networkx graph and edges gdf
    graph = load_graph()
    edges_gdf = tools.edges.graph_edges_gdf(graph)

    # Load taxi data
    data_path = data.utils.choose_data()
    raw_data = data.utils.read_data(data_path)

    # Extract distances
    euclidean_dists = []
    for poly in raw_data['POLYLINE_UTM']:
        euclidean_dists += polyline_to_euclidean_distance(poly)
    euclidean_dists = np.asarray(euclidean_dists)

    # Arbitrarily account for non-linear travel
    arbitrary_deflation_dists = euclidean_dists * 1.2

    # Convert to average speeds
    approx_speeds = arbitrary_deflation_dists / 15

    # Remove massive values (outliers)
    approx_speeds_trim = approx_speeds[approx_speeds < 40]

    # Plot histogram
    plt.hist(approx_speeds_trim, bins=100, normed=True)
    linsp = np.linspace(0, 40, 300)

    # Fit Gaussian distribution
    m, s = stats.norm.fit(approx_speeds_trim)  # get mean and standard deviation
    pdf_g = stats.norm.pdf(linsp, m, s)  # now get theoretical values in our interval
    plt.plot(linsp, pdf_g, label="Norm")  # plot it

    plt.show()












