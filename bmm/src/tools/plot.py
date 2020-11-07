########################################################################################################################
# Module: plot.py
# Description: Plot cam_graph, inferred route and/or polyline.
#
# Web: https://github.com/SamDuffield/bmm
########################################################################################################################

import numpy as np
import osmnx as ox
from matplotlib import pyplot as plt

from bmm.src.tools.edges import interpolate_path, cartesianise_path, observation_time_rows


def plot(graph, particles=None, polyline=None, particles_alpha=None, label_start_end=True,
         bgcolor='white', node_color='grey', node_size=0, edge_color='lightgrey', edge_linewidth=3, **kwargs):
    """
    Plots particle approximation of trajectory
    :param graph: NetworkX MultiDiGraph
        UTM projection
        encodes road network
        generating using OSMnx, see tools.cam_graph.py
    :param particles: MMParticles object (from inference.particles)
        particle approximation
    :param polyline: list-like, each element length 2
        UTM - metres
        series of GPS coordinate observations
    :param particles_alpha: float in [0, 1]
        plotting parameter
        opacity of routes
    :param label_start_end: bool
        whether to label the start and end points of the route
    :param bgcolor: str
        background colour
    :param node_color: str
        node (intersections) colour
    :param node_size: float
        size of nodes (intersections)
    :param edge_color: str
        colour of edges (roads)
    :param edge_linewidth: float
        width of edges (roads
    :param kwargs:
        additional parameters to ox.plot_graph
    :return: fig, ax
    """
    fig, ax = ox.plot_graph(graph, show=False, close=False,
                            bgcolor=bgcolor, node_color=node_color, node_size=node_size,
                            edge_color=edge_color, edge_linewidth=edge_linewidth,
                            **kwargs)
    ax.set_aspect("equal")

    start_end_points = None

    if particles is not None:
        if isinstance(particles, np.ndarray):
            particles = [particles]

        start_end_points = np.zeros((2, 2))

        alpha_min = 0.1

        if particles_alpha is None:
            particles_alpha = 1 / len(particles) * (1 - alpha_min) + alpha_min

        xlim = [None, None]
        ylim = [None, None]

        for i, particle in enumerate(particles):
            if particle is None:
                continue

            if len(particle) > 1:
                int_path = interpolate_path(graph, particle, t_column=True)

                cart_int_path = cartesianise_path(graph, int_path, t_column=True)
                ax.plot(cart_int_path[:, 0], cart_int_path[:, 1], color='orange', linewidth=1.5,
                        alpha=particles_alpha)

                cart_path = cartesianise_path(graph, observation_time_rows(particle), t_column=True)
            else:
                cart_path = cartesianise_path(graph, particle, t_column=True)

            ax.scatter(cart_path[:, 0], cart_path[:, 1], color='orange', alpha=particles_alpha, zorder=2)

            start_end_points[0] += cart_path[0] / len(particles)
            start_end_points[1] += cart_path[-1] / len(particles)

            xlim[0] = np.min(cart_path[:, 0]) if xlim[0] is None else min(np.min(cart_path[:, 0]), xlim[0])
            xlim[1] = np.max(cart_path[:, 0]) if xlim[1] is None else max(np.max(cart_path[:, 0]), xlim[1])
            ylim[0] = np.min(cart_path[:, 1]) if ylim[0] is None else min(np.min(cart_path[:, 1]), ylim[0])
            ylim[1] = np.max(cart_path[:, 1]) if ylim[1] is None else max(np.max(cart_path[:, 1]), ylim[1])

        xlim, ylim = expand_lims(xlim, ylim, 0.1)

        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])

    if polyline is not None:
        poly_arr = np.array(polyline)
        ax.scatter(poly_arr[:, 0],
                   poly_arr[:, 1],
                   marker='x', c='red', s=100, linewidth=3, zorder=10)

        if particles is None:
            start_end_points = poly_arr[np.array([0, -1])]

            xlim = [np.min(poly_arr[:, 0]), np.max(poly_arr[:, 0])]
            ylim = [np.min(poly_arr[:, 1]), np.max(poly_arr[:, 1])]

            xlim, ylim = expand_lims(xlim, ylim, 0.1)

            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])

    if start_end_points is not None and label_start_end:
        plt.annotate('Start', start_end_points[0] + 25, zorder=12)
        plt.annotate('End', start_end_points[1] + 25, zorder=12)

    plt.tight_layout()

    return fig, ax


def expand_lims(xlim, ylim, inflation):
    x_range = max(xlim[1] - xlim[0], 200)
    xlim[0] -= x_range * inflation
    xlim[1] += x_range * inflation

    y_range = max(ylim[1] - ylim[0], 200)
    ylim[0] -= y_range * inflation
    ylim[1] += y_range * inflation

    return xlim, ylim

