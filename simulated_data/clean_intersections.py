import time
import collections
import logging as lg


import numpy as np
from geopandas import gpd
from shapely.geometry import Point, Polygon, LineString
from osmnx import graph_to_gdfs, log, settings, get_nearest_nodes
import networkx as nx


def clean_intersections_graph(G, rebuild_graph=False, tolerance=15, method='kdtree'):
    """
    From https://github.com/gboeing/osmnx/blob/clean_intersections/osmnx/simplify.py
    Clean-up intersections comprising clusters of nodes by merging them and
    returning either their centroids or a rebuilt graph.
    Divided roads are represented by separate centerline edges. The intersection
    of two divided roads thus creates 4 nodes, representing where each edge
    intersects a perpendicular edge. These 4 nodes represent a single
    intersection in the real world. This function cleans them up by buffering
    their points to an arbitrary distance, merging overlapping buffers, and
    taking their centroid. Then, it constructs a new graph with the centroids
    as nodes. Edges between centroids are calculated such that the new graph
    remains topologically (approximately) equal to G.
    Cleaning occurs in the graph's current units, but the use of unprojected
    units is not recommended. For best results, the tolerance argument should be
    adjusted to approximately match street design standards in the specific
    street network.
    Parameters
    ----------
    G : networkx multidigraph
    rebuild_graph : bool
    	if False, just return the cleaned intersection centroids. if True,
    	rebuild the graph around the cleaned intersections and return it.
    tolerance : float
        nodes within this distance (in graph's geometry's units) will be
        dissolved into a single intersection
    dead_ends : bool
        if False, discard dead-end nodes to return only street-intersection
        points
    method : str {None, 'kdtree', 'balltree'}
        Which method to use for finding nearest node to each point.
        If None, we manually find each node one at a time using
        osmnx.utils.get_nearest_node and haversine. If 'kdtree' we use
        scipy.spatial.cKDTree for very fast euclidean search. If
        'balltree', we use sklearn.neighbors.BallTree for fast
        haversine search.
    Returns
    ----------
    G2 : networkx multidigraph
        the new cleaned graph
    """

    start_time = time.time()

    if G.graph['crs'] == settings.default_crs:
        log(("The graph seems to be using unprojected units which is not recommended."),
            level = lg.WARNING)

    # Let's make a copy of G
    G = G.copy()

    # create a GeoDataFrame of nodes, buffer to passed-in distance, merge
    # overlaps
    gdf_nodes = graph_to_gdfs(G, edges=False)
    buffered_nodes = gdf_nodes.buffer(tolerance).unary_union
    if isinstance(buffered_nodes, Polygon):
        # if only a single node results, make it iterable so we can turn it into
        # a GeoSeries
        buffered_nodes = [buffered_nodes]

    # get the centroids of the merged intersection polygons
    unified_intersections = gpd.GeoSeries(list(buffered_nodes))
    intersection_centroids = unified_intersections.centroid

    # if we are not rebuilding the graph around the cleaned intersections,
    # just return the intersection_centroids now to exit the function
    if not rebuild_graph:
        log('Calculated/cleaned intersection points in {:,.2f} seconds'.format(time.time()-start_time))
        return intersection_centroids

    # THE REMAINDER OF THIS FUNCTION ONLY EXECUTES WHEN rebuild_graph=True
    log('Finding centroids (checkpoint 0-1) took {:,.2f} seconds'\
         .format(time.time()-start_time))

    start_time2 = time.time()

    # To make things simpler, every edge should have a geometry
    # (this avoids KeyError later when choosing between several
    #  geometries or when the centroid inherits a geometry directly)
    for u, v, data in G.edges(keys=False, data=True):
        if 'geometry' not in data:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            data['geometry'] = LineString([(x1, y1), (x2, y2)])

    # Let's first create the Graph first
    G__ = nx.MultiDiGraph(crs  = G.graph['crs'])

    # And add the nodes (without the osmid attributes just yet)
    # The centroids are given new ids = 0 .. total_centroids-1
    for i in range(len(intersection_centroids)):
        G__.add_node(i,
                     x = intersection_centroids[i].x,
                     y = intersection_centroids[i].y)

    # We can now run ox.get_nearest_nodes to calculate the closest centroid
    # to each node
    osmids = [ osmid for osmid in G.nodes()           ]
    X      = [ x     for _, x  in G.nodes(data = 'x') ]
    Y      = [ y     for _, y  in G.nodes(data = 'y') ]

    # Mapping: i -> centroid
    nearest_centroid = get_nearest_nodes(G__, X, Y, method = method)

    # For ease of use, we build the mappings:
    # osmid -> centroid
    nearest_centroid = { osmids[i] : centroid for i, centroid in enumerate(nearest_centroid) }

    # centroid -> osmids
    centroid_elements = {i : [] for i in range(len(intersection_centroids))}
    for osmid, centroid in nearest_centroid.items():
        centroid_elements[centroid].append(osmid)

    # Add osmids as attribute
    nx.set_node_attributes(G__, centroid_elements, 'osmids')
    # For compability, set attribute the 'osmid' with a single value
    nx.set_node_attributes(G__, { centroid : elements[0] for centroid, elements in centroid_elements.items() } , 'osmid')

    log('Getting nearest nodes and computing the mapping and reverse mapping of '
        'centroid->elements (checkpoint 1-2) took {:,.2f} seconds'\
        .format(time.time()-start_time2))

    start_time2 = time.time()

    # --------
    # Independently of the cases below, in the main loop, we need to
    # fix edge geometries and make sure they intersect their centroids,
    # otherwise we may have gaps on the resulting graph when plotted.
    # --------
    # We define this behavior as an inner function here because it will be
    # extensively used in the main loop, thus avoiding clutter in the main loop.
    # --------
    def fix_geometry(centroid_p, neighbor_p, geom):
        # make sure the resulting geometry intersects the centroid
        coords = geom.coords[:]
        if not geom.intersects(centroid_point):
            coords = centroid_point.coords[:] + coords

        # make sure the resulting geometry intersects the neighbor's centroid
        if not geom.intersects(neighbor_point):
            coords = coords + neighbor_point.coords[:]

        # return updated geometry
        return LineString(coords)

    # Now we're ready to start adding the edges to the cleaned graph
    for centroid, elements in centroid_elements.items():
            # get the outgoing edges for all elements in the centroid
            out_edges = []
            for node in elements:
                out_edges += list(G.out_edges(node))

            # select the neighbor nodes of each element in the centroid
            out_nodes = map(lambda edge: edge[1], out_edges)

            # and get the corresponding centroid of each element
            neighbors = list(map(lambda osmid: nearest_centroid[osmid],
                                 out_nodes))

            # remove duplicate neighbor centroids
            neighbors = set(neighbors)
            # and self-loops
            if centroid in neighbors:
                neighbors.remove(centroid)

            # helper for fixing geometries
            # initialised outside the for loop even thoug it's used later on
            # to avoid multiple redundant calls
            centroid_point = Point(
                intersection_centroids[centroid].x,
                intersection_centroids[centroid].y)

            # add an outgoing edge to each resulting neighbor
            for neighbor in neighbors:
                # retrieve the elements of this neighbor
                neighbor_elements = centroid_elements[neighbor]

                # select the edges (c1,c2) where c1 is an element of this centroid,
                # and c2 is an element of this neighbor's centroid
                out_edges_neighbor = set(filter(lambda x: x[1] in neighbor_elements, out_edges))

                # We're gonna need this to fix any problems with resulting geometries
                neighbor_point = Point(
                    intersection_centroids[neighbor].x,
                    intersection_centroids[neighbor].y)

                # --------
                # Cases 1 and 2:
                # --------
                # This centroid and my neighbor centroid have only one element each,
                # OR
                # There are multiple elements in one or both of this centroid and
                # its neighbor, but only a single edge exists between them.
                #
                # This means that we can keep any parallel edges between them and
                # reuse any existing attributes. Case 2 may still need corrections
                # for the resulting geometry (see below, after case 3)
                if (len(elements) == 1 and len(neighbor_elements) == 1) or \
                   (len(out_edges_neighbor) == 1):
                   # There is only one edge in either case
                   u, v = list(out_edges_neighbor)[0]
                   # Iterate through existing keys
                   for key in list(G[u][v].keys()):
                       attr = G.edges[u, v, key]
                       # Case 1 does not need to fix_geometry but case 2 might.
                       # fix_geometry does not affect an already correct geometry.
                       attr['geometry'] = fix_geometry(centroid_point,
                                                       neighbor_point,
                                                       attr['geometry'])
                       # update length based on geometry
                       attr['length'] = attr['geometry'].length
                       # Time to add the edge
                       G__.add_edge(centroid, neighbor, key = key, **attr)

                # --------
                # Case 3:
                # --------
                # There are multiple elements in one or both of this centroid and
                # its neighbor, and multiple edges between them.
                # The result is a single edge in the new graph, whose attributes
                # are combined among the existing edges, except for geometry
                # and length, whose value must be a single value (and not a list)
                else:
                    edges_attrs = [ G.edges[edge[0], edge[1], 0] for edge in out_edges_neighbor ]

                    # Not every edge has the same set of keys, so we compute the union of the individual
                    # sets, and also count their occurrence for initialisation purposes
                    all_keys = []
                    for edge_attr in edges_attrs:
                        all_keys += edge_attr.keys()

                    key_counter = collections.Counter(all_keys)

                    # If if the key counter is greater than 1, we initialise the new
                    # dictionary with empty lists, so that we can later call append()
                    attr = {key : [] if count > 1 else '' for key, count in key_counter.items()}

                    for edge_attr in edges_attrs:
                        for key in edge_attr.keys():
                            val = edge_attr[key]

                            # This key only occurs once, so we can set its value
                            if key_counter[key] == 1:
                                attr[key] = val
                            # Otherwise, append
                            elif isinstance(val, list):
                                attr[key].extend(val)
                            else:
                                attr[key].append(val)

                    # We pick the geometry with largest length
                    # but, there may be a better way to make this decision
                    whichmax = np.argmax(attr['length'])
                    geom = attr['geometry'][whichmax]
                    attr['geometry'] = fix_geometry(centroid_point, neighbor_point, geom)

                    # update length based on geometry
                    attr['length'] = attr['geometry'].length

                    # Time to add the edge
                    G__.add_edge(centroid, neighbor, **attr)

    log('Adding edges to the new graph (checkpoint 2-3) took {:,.2f} seconds'\
        .format(time.time()-start_time2))

    msg = 'Cleaned intersections of graph (from {:,} to {:,} nodes and from {:,} to {:,} edges) in {:,.2f} seconds'
    log(msg.format(len(list(G.nodes())), len(list(G__.nodes())),
                   len(list(G.edges())), len(list(G__.edges())), time.time()-start_time))

    return G__

#
# def coarse_grain_osm_network(G_proj, tolerance=10):
#     """
#     From https://github.com/gboeing/osmnx/blob/coarse/osmnx/simplify.py
#     Accepts an (unprojected) osmnx graph and coarse grains it, i.e.
#     creates a new graph H with less nodes and edges. H is constructed
#     as follows:
#     1. Draws a circle of radius tolerance (meters) around each node of G.
#     2. If the circles of two (or more) nodes (say u1, u2) have an overlap, they
#     are collapsed into a single node v1 in H.
#     3. For each edge (u1, u2) in G, if u1 and u2 has been collapsed into a
#     single node in H, H gets no edge corresponding to (u1, u2). Otherwise, an
#     edge is added in H between the corresponding nodes of u1 and u2.
#     Parameters
#     ----------
#     G : networkx multidigraph
#     tolerance : float
#         nodes within this distance (in graph's geometry's units) will be
#         dissolved into a single node
#     Returns
#     ----------
#     networkx.Graph
#     """
#     # G_proj = project_graph(G)
#     gdf_nodes = graph_to_gdfs(G_proj, edges=False)
#     buffered_nodes = gdf_nodes.buffer(tolerance).unary_union
#     old2new = dict()
#
#     if not hasattr(buffered_nodes, '__iter__'):
#         # if tolerance is very high, buffered_nodes won't be an iterable of Point's
#         # but a single point. To take care of this case, we make it into a list.
#         buffered_nodes = [buffered_nodes]
#
#     for node, data in G_proj.nodes(data=True):
#         x, y = data['x'], data['y']
#         lon, lat = data['lon'], data['lat']
#         osm_id = data['osmid']
#         for poly_idx, polygon in enumerate(buffered_nodes):
#             if polygon.contains(Point(x, y)):
#                 poly_centroid = polygon.centroid
#                 old2new[node] = dict(label=poly_idx, x=poly_centroid.x, y=poly_centroid.y)
#                 break
#
#     H = Graph()
#     for node in G_proj.nodes():
#         new_node_data = old2new[node]
#         new_label = new_node_data['label']
#         H.add_node(new_label, **new_node_data)
#     for u, v in G_proj.edges():
#         u2, v2 = old2new[u]['label'], old2new[v]['label']
#         if u2 != v2:
#             H.add_edge(u2, v2)
#     H.graph['crs'] = G_proj.graph['crs']
#     return H
#
#




#
#
#
# def clean_intersections_graph(G, tolerance=15, dead_ends=False):
#     """
#     From https://gist.github.com/timlrx/3522d6a79ddf438857228e9dea92025d
#     Clean-up intersections comprising clusters of nodes by merging them and
#     returning a modified graph.
#     Divided roads are represented by separate centerline edges. The intersection
#     of two divided roads thus creates 4 nodes, representing where each edge
#     intersects a perpendicular edge. These 4 nodes represent a single
#     intersection in the real world. This function cleans them up by buffering
#     their points to an arbitrary distance, merging overlapping buffers, and
#     taking their centroid. For best results, the tolerance argument should be
#     adjusted to approximately match street design standards in the specific
#     street network.
#     Parameters
#     ----------
#     G : networkx multidigraph
#     tolerance : float
#         nodes within this distance (in graph's geometry's units) will be
#         dissolved into a single intersection
#     dead_ends : bool
#         if False, discard dead-end nodes to return only street-intersection
#         points
#     Returns
#     ----------
#     Networkx graph with the new aggregated vertices and induced edges
#     """
#
#     # if dead_ends is False, discard dead-end nodes to only work with edge
#     # intersections
#     if not dead_ends:
#         if 'streets_per_node' in G.graph:
#             streets_per_node = G.graph['streets_per_node']
#         else:
#             streets_per_node = count_streets_per_node(G)
#
#         dead_end_nodes = [node for node, count in streets_per_node.items() if count <= 1]
#         G = G.copy()
#         G.remove_nodes_from(dead_end_nodes)
#
#     # create a GeoDataFrame of nodes, buffer to passed-in distance, merge
#     # overlaps
#     gdf_nodes, gdf_edges = graph_to_gdfs(G)
#     buffered_nodes = gdf_nodes.buffer(tolerance).unary_union
#     if isinstance(buffered_nodes, Polygon):
#         # if only a single node results, make it iterable so we can turn it into
#         # a GeoSeries
#         buffered_nodes = [buffered_nodes]
#
#     # Buffer points by tolerance and union the overlapping ones
#     gdf_nodes, gdf_edges = graph_to_gdfs(G)
#     buffered_nodes = gdf_nodes.buffer(15).unary_union
#     unified_intersections = gpd.GeoSeries(list(buffered_nodes))
#     unified_gdf = gpd.GeoDataFrame(unified_intersections).rename(columns={0:'geometry'}).set_geometry('geometry')
#     unified_gdf.crs = gdf_nodes.crs
#
#     ### Merge original nodes with the aggregated shapes
#     intersections = gpd.sjoin(gdf_nodes, unified_gdf, how="right", op='intersects')
#     intersections['geometry_str'] = intersections['geometry'].map(lambda x: str(x))
#     intersections['new_osmid'] = intersections.groupby('geometry_str')['index_left'].transform('min').astype(str)
#     intersections['num_osmid_agg'] = intersections.groupby('geometry_str')['index_left'].transform('count')
#
#     ### Create temporary lookup with the agg osmid and the new one
#     lookup = intersections[intersections['num_osmid_agg']>1][['osmid', 'new_osmid', 'num_osmid_agg']]
#     lookup = lookup.rename(columns={'osmid': 'old_osmid'})
#     intersections = intersections[intersections['osmid'].astype(str)==intersections['new_osmid']]
#     intersections = intersections.set_index('index_left')
#
#     ### Make everything else similar to original node df
#     intersections = intersections[gdf_nodes.columns]
#     intersections['geometry'] = intersections.geometry.centroid
#     intersections['x'] = intersections.geometry.x
#     intersections['y'] = intersections.geometry.y
#     # del intersections.index.name
#     intersections.gdf_name = gdf_nodes.gdf_name
#
#     # Replace aggregated osimid with the new ones
#     # 3 cases - 1) none in lookup, 2) either u or v in lookup, 3) u and v in lookup
#     # Ignore case 1. Append case 3 to case 2. ignore distance but append linestring.
#
#     agg_gdf_edges = pd.merge(gdf_edges.assign(u=gdf_edges.u.astype(str)),
#                         lookup.rename(columns={'new_osmid': 'new_osmid_u', 'old_osmid': 'old_osmid_u'}),
#                         left_on='u', right_on='old_osmid_u', how='left')
#     agg_gdf_edges = pd.merge(agg_gdf_edges.assign(v=agg_gdf_edges.v.astype(str)),
#                         lookup.rename(columns={'new_osmid': 'new_osmid_v', 'old_osmid': 'old_osmid_v'}),
#                         left_on='v', right_on='old_osmid_v', how='left')
#
#     # Remove all u-v edges that are between the nodes that are aggregated together (case 3)
#     agg_gdf_edges_c3 = agg_gdf_edges[((agg_gdf_edges['new_osmid_v'].notnull()) &
#         (agg_gdf_edges['new_osmid_u'].notnull()) &
#         (agg_gdf_edges['new_osmid_u'] == agg_gdf_edges['new_osmid_v']))]
#
#     agg_gdf_edges = agg_gdf_edges[~agg_gdf_edges.index.isin(agg_gdf_edges_c3.index)]
#
#     # Create a self loop containing all the joint geometries of the aggregated nodes where both u and v are agg
#     # Set onway to false to prevent duplication if someone were to create bidrectional edges
#     agg_gdf_edges_int = agg_gdf_edges_c3[~((agg_gdf_edges_c3['new_osmid_u'] == agg_gdf_edges_c3['u']) |
#                                         (agg_gdf_edges_c3['new_osmid_v'] == agg_gdf_edges_c3['v']))]
#     agg_gdf_edges_int = agg_gdf_edges_int.dissolve(by=['new_osmid_u', 'new_osmid_v']).reset_index()
#     agg_gdf_edges_int['u'] = agg_gdf_edges_int['new_osmid_u']
#     agg_gdf_edges_int['v'] = agg_gdf_edges_int['new_osmid_v']
#     agg_gdf_edges_int = agg_gdf_edges_int[gdf_edges.columns]
#     agg_gdf_edges_int['oneway'] = False
#
#     # Simplify by removing edges that do not involve the chosen agg point
#     # at least one of them must contain the new u or new v
#     agg_gdf_edges_c3 = agg_gdf_edges_c3[(agg_gdf_edges_c3['new_osmid_u'] == agg_gdf_edges_c3['u']) |
#                                         (agg_gdf_edges_c3['new_osmid_v'] == agg_gdf_edges_c3['v'])]
#
#     agg_gdf_edges_c3 = agg_gdf_edges_c3[['geometry', 'u', 'v', 'new_osmid_u', 'new_osmid_v']]
#     agg_gdf_edges_c3.columns = ['old_geometry', 'old_u', 'old_v', 'new_osmid_u', 'new_osmid_v']
#
#     # Merge back the linestring for case 2
#     # Ignore u and v if they are on the merging / agg node
#     # Copy over the linestring only on the old node
#     subset_gdf = agg_gdf_edges_c3[agg_gdf_edges_c3['new_osmid_v']!=agg_gdf_edges_c3['old_v']]
#     agg_gdf_edges = pd.merge(agg_gdf_edges, subset_gdf[['old_geometry', 'old_v']],
#                              how='left', left_on='u', right_on='old_v')
#
#     geom = agg_gdf_edges[['geometry', 'old_geometry']].values.tolist()
#     agg_gdf_edges['geometry'] = [linemerge([r[0], r[1]]) if isinstance(r[1], (LineString, MultiLineString)) else r[0] for r in geom]
#     agg_gdf_edges.drop(['old_geometry', 'old_v'], axis=1, inplace=True)
#
#     # If new osmid matches on u, merge in the existing u-v string
#     # where u is the aggregated vertex and v is the old one to be removed
#
#     subset_gdf = agg_gdf_edges_c3[agg_gdf_edges_c3['new_osmid_u']!=agg_gdf_edges_c3['old_u']]
#     agg_gdf_edges = pd.merge(agg_gdf_edges, subset_gdf[['old_geometry', 'old_u']],
#                              how='left', left_on='v', right_on='old_u')
#
#     geom = agg_gdf_edges[['geometry', 'old_geometry']].values.tolist()
#     agg_gdf_edges['geometry'] = [linemerge([r[0], r[1]]) if isinstance(r[1], (LineString, MultiLineString)) else r[0] for r in geom]
#     agg_gdf_edges.drop(['old_geometry', 'old_u'], axis=1, inplace=True)
#
#     agg_gdf_edges['u'] = np.where(agg_gdf_edges['new_osmid_u'].notnull(), agg_gdf_edges['new_osmid_u'], agg_gdf_edges['u'])
#     agg_gdf_edges['v'] = np.where(agg_gdf_edges['new_osmid_v'].notnull(), agg_gdf_edges['new_osmid_v'], agg_gdf_edges['v'])
#     agg_gdf_edges = agg_gdf_edges[gdf_edges.columns]
#     agg_gdf_edges = gpd.GeoDataFrame(pd.concat([agg_gdf_edges, agg_gdf_edges_int], ignore_index=True),
#                                      crs=agg_gdf_edges.crs)
#
#     agg_gdf_edges['u'] = agg_gdf_edges['u'].astype(np.int64)
#     agg_gdf_edges['v'] = agg_gdf_edges['v'].astype(np.int64)
#
#     return gdfs_to_graph(intersections, agg_gdf_edges)
