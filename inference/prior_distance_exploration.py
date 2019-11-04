################################################################################
# Module: prior_distance_exploration.py
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

    # Inter-observation time
    delta_obs = 15

    # Set maximum possible distance to travel between observations
    v_max = 32
    d_max = delta_obs * v_max

    # Remove massive values (outliers)
    euclidean_dists = euclidean_dists[euclidean_dists < d_max]

    # Remove zero values
    euclidean_dists = euclidean_dists[euclidean_dists > 0.01]

    # Plot histogram
    plt.figure()
    plt.hist(euclidean_dists, bins=100, density=True, alpha=0.5)
    plt.xlabel('||y_0-y_1||')

    # Define linspace
    linsp = np.linspace(0, d_max, 1000)

    # Fit Gamma manually
    mean_dist = np.mean(euclidean_dists)
    var_dist = np.var(euclidean_dists)
    bg = mean_dist / var_dist
    ag = mean_dist * bg
    pdf_gamma = stats.gamma.pdf(linsp, ag, loc=0, scale=1/bg)
    plt.plot(linsp, pdf_gamma, label="Fitted Gamma")

    # Log data
    log_euclidean_dists = np.log(euclidean_dists)

    # Fit Gaussian to log data (logGaussian to raw data)
    mean_log_dist = np.mean(log_euclidean_dists)
    var_log_dist = np.var(log_euclidean_dists)
    pdf_loggaussian = stats.lognorm.pdf(linsp, s=np.sqrt(var_log_dist), scale=np.exp(mean_log_dist))
    plt.plot(linsp, pdf_loggaussian, label="Fitted logGaussian")

    # Add legend
    plt.legend()

    # Observation variance
    obs_var = 20

    # Single observed distance
    d_obs = 100

    # # Plot Gamma prior
    # plt.figure()
    # plt.plot(linsp, pdf_gamma, label="Gamma Prior")
    #
    # # Gamma Likelihood
    # bg_lik = d_obs / obs_var
    # ag_lik = d_obs * bg_lik
    # pdf_gamma_lik = stats.gamma.pdf(linsp, ag_lik, loc=0, scale=1/bg_lik)
    # pdf_gamma_lik *= np.max(pdf_gamma)/np.max(pdf_gamma_lik)
    # plt.plot(linsp, pdf_gamma_lik, label="Gamma Likelihood")
    #
    # # Gamma_posterior
    # ag_post = ag
    # bg_post = bg - obs_var*np.log(d_obs)
    # pdf_post_gamma = stats.gamma.pdf(linsp, ag_post, loc=0, scale=1/bg_post)
    # plt.figure()
    # plt.plot(linsp, pdf_post_gamma, label="Gamma Posterior")

    # Plot logGaussian prior
    plt.figure()
    plt.plot(linsp, pdf_loggaussian, label="logGaussian Prior")

    # Plot logGaussian likelihood
    pdf_loggaussian_lik = stats.lognorm.pdf(linsp, s=np.sqrt(obs_var), scale=np.exp(d_obs))
    pdf_loggaussian_lik *= np.max(pdf_loggaussian) / np.max(pdf_loggaussian_lik)
    plt.plot(linsp, pdf_loggaussian_lik, label="logGaussian Likelihood")







    plt.show(block=True)

