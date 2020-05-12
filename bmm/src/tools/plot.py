import numpy as np
import osmnx as ox
from matplotlib import pyplot as plt

from bmm.src.tools.edges import interpolate_path, cartesianise_path, observation_time_rows


def plot(graph, particles=None, polyline=None, particles_alpha=None):
    """
    Plots particle approximation of trajectory
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.graph.py
    :param particles: MMParticles object (from inference.particles)
        particle approximation
    :param polyline: list-like, each element length 2
        UTM - metres
        series of GPS coordinate observations
    :param particles_alpha: float in [0, 1]
        plotting parameter
        opacity of routes
    :return: fig, ax
    """
    fig, ax = ox.plot_graph(graph, show=False, close=False, equal_aspect=True, edge_color='lightgrey',
                            node_size=0, edge_linewidth=3)

    if particles is not None:
        if isinstance(particles, np.ndarray):
            particles = [particles]

        alpha_min = 0.3

        if particles_alpha is None:
            particles_alpha = 1 / len(particles) * (1 - alpha_min) + alpha_min

        xlim = [None, None]
        ylim = [None, None]

        for i, particle in enumerate(particles):

            if len(particle) > 1:
                int_path = interpolate_path(graph, particle, t_column=True)

                cart_int_path = cartesianise_path(graph, int_path, t_column=True)
                ax.plot(cart_int_path[:, 0], cart_int_path[:, 1], color='orange', linewidth=1.5,
                        alpha=particles_alpha)

                cart_path = cartesianise_path(graph, observation_time_rows(particle), t_column=True)
            else:
                cart_path = cartesianise_path(graph, particle, t_column=True)[None, :]

            ax.scatter(cart_path[:, 0], cart_path[:, 1], color='orange', alpha=particles_alpha, zorder=2)

            xlim[0] = np.min(cart_path[:, 0]) if xlim[0] is None else min(np.min(cart_path[:, 0]), xlim[0])
            xlim[1] = np.max(cart_path[:, 0]) if xlim[1] is None else max(np.max(cart_path[:, 0]), xlim[1])
            ylim[0] = np.min(cart_path[:, 1]) if ylim[0] is None else min(np.min(cart_path[:, 1]), ylim[0])
            ylim[1] = np.max(cart_path[:, 1]) if ylim[1] is None else max(np.max(cart_path[:, 1]), ylim[1])

        expand_coef = 0.1

        x_range = max(xlim[1] - xlim[0], 200)
        xlim[0] -= x_range * expand_coef
        xlim[1] += x_range * expand_coef

        y_range = max(ylim[1] - ylim[0], 200)
        ylim[0] -= y_range * expand_coef
        ylim[1] += y_range * expand_coef

        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])

    if polyline is not None:
        poly_arr = np.array(polyline)
        ax.scatter(poly_arr[:, 0],
                   poly_arr[:, 1],
                   marker='x', c='red', s=100, linewidth=3, zorder=10)

    plt.tight_layout()

    return fig, ax



