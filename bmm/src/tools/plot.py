import numpy as np
import osmnx as ox
from matplotlib import pyplot as plt

from bmm.src.tools.edges import interpolate_path, cartesianise_path, observation_time_rows
from bmm.src.tools.graph import polyline_axis


def plot_graph(graph, polyline=None, edges_to_highlight=None):
    """
    Plot OSMnx graph, with optional polyline and highlighted edges.
    :param graph: networkx object
    :param polyline: standard format
    :param edges_to_highlight: edges to highlight in blue
    :return: fig, ax of plotted road network (plus polyline)
    """
    if edges_to_highlight is not None:
        edge_colours = ['blue' if [u, v] in edges_to_highlight or [v, u] in edges_to_highlight
                        else 'lightgrey' for u, v, d in graph.edges]
    else:
        edge_colours = 'lightgrey'

    fig, ax = ox.plot_graph(graph, show=False, close=False, equal_aspect=True, edge_color=edge_colours,
                            node_size=0, edge_linewidth=3)

    if polyline is not None:
        if len(polyline) > 1:
            ax.scatter(polyline_axis(polyline, 0), polyline_axis(polyline, 1),
                       marker='x', c='red', s=100, linewidth=3, zorder=10)
            # ax.scatter(polyline[-1][0], polyline[-1][1], marker='x', c='blue', s=100, linewidth=3)
        # ax.scatter(polyline[0][0], polyline[0][1], marker='x', c='blue', s=100, linewidth=3)

    plt.tight_layout()

    return fig, ax


def plot_particles(graph, particles, polyline=None, alpha=None):
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
    :param alpha: float in [0, 1]
        plotting parameter
        opacity of routes
    :return: fig, ax
    """
    fig, ax = plot_graph(graph, polyline=polyline)

    if isinstance(particles, np.ndarray):
        particles = [particles]

    alpha_min = 0.3

    if alpha is None:
        alpha = 1 / len(particles) * (1 - alpha_min) + alpha_min

    xlim = [None, None]
    ylim = [None, None]

    for i, particle in enumerate(particles):

        if len(particle) > 1:
            int_path = interpolate_path(graph, particle, t_column=True)

            cart_int_path = cartesianise_path(graph, int_path, t_column=True)
            ax.plot(cart_int_path[:, 0], cart_int_path[:, 1], color='orange', linewidth=1.5,
                    alpha=alpha)

            cart_path = cartesianise_path(graph, observation_time_rows(particle), t_column=True)
            ax.scatter(cart_path[:, 0], cart_path[:, 1], color='orange', alpha=alpha, zorder=2)
        else:
            cart_path = cartesianise_path(graph, particle, t_column=True)
            ax.scatter(cart_path[:, 0], cart_path[:, 1], color='orange', alpha=alpha)

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

    return fig, ax

