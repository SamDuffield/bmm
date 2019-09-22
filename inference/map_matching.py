################################################################################
# Module: map_matching.py
# Description: Infer route taken by vehicle given sparse observations.
#
# Web: https://github.com/SamDuffield/bayesian-traffic
################################################################################

import numpy as np
import data
from tools.graph import load_graph
import tools.edges
import tools.sampling
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


def sample_TGaus_speed(mu, sigma, vmax, size=1):
    """
    Sample a speed from univariate Gaussian truncated to [0,vmax].
    All quantities in m/s.
    :param mu: mean of (pre-truncated) Gaussian
    :param sigma: standard deviation of (pre-truncated Gaussian)
    :param vmax: maximum speed (upper truncation)
    :return: float (or array if size>1), speed in m/s.
    """
    sample = truncnorm.rvs(a=-mu/sigma, b=(vmax-mu)/sigma, loc=mu, scale=sigma, size=size)

    if len(sample) == 1:
        return sample.item()
    else:
        return sample


def plot_TGaus(mu, sigma, vmax):
    """
    Plots Gaussian truncated to [0, vmax]
    :param mu: mean of (pre-truncated) Gaussian
    :param sigma: standard deviation of (pre-truncated Gaussian)
    :param vmax: upper truncation
    :return: fig, ax
    """
    fig, ax = plt.subplots(1, 1)

    rv = truncnorm(-mu/sigma, (vmax-mu)/sigma, scale=sigma)

    x = np.linspace(-mu/sigma, (vmax-mu)/sigma, 100)
    ax.plot(x*sigma + mu, rv.pdf(x))

    return fig, ax




if __name__ == '__main__':
    # Source data paths
    _, process_data_path = data.utils.source_data()

    # Load networkx graph
    graph = load_graph()

    # Load small taxi data set (i.e. only 15 minutes)
    data_path = data.utils.choose_data()
    raw_data = data.utils.read_data(data_path)

    # Select single polyline
    single_index = 0
    poly_single = raw_data['POLYLINE_UTM'][single_index]

    # Number of observations
    M_obs = len(poly_single)

    # Sample size
    N_samps = 10

    # Observation time increment (s)
    delta_obs = 15

    # Inference time discretisation
    delta_x = 3

    # Initiate sample storage, preallocate?
    xv_samples = []

    # Sample x_t0|y_t0
    xv_samples += [tools.sampling.sample_x0(graph, poly_single[0], N_samps)]

    # Speed prior hyperparameters (all links a priori iid)
    v_mu = 9
    v_sigma = 5
    v_max = v_mu*2.5

    # Plot speed prior (all links a priori iid)
    plot_TGaus(v_mu, v_sigma, v_max)
    # plt.hist(sample_TGaus_speed(v_mu, v_sigma, v_max, 1000))

    # Sample initial speeds and set initial weights
    xv_samples[0] = [[edge, alpha, sample_TGaus_speed(v_mu, v_sigma, v_max), 1/N_samps] for edge, alpha in xv_samples[0]]









