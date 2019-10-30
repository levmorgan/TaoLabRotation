#!/usr/bin/env python
# coding: utf-8

# !/usr/bin/env python
# coding: utf-8

import os
import pickle
import datetime

import numpy as np
from mayavi import mlab
from plyfile import PlyData
import networkx as nx
from scipy import sparse, spatial

ELLIPTICAL_DISTANCE_PLOT = False


def cartesian_product(x, y):
    return np.transpose([np.repeat(y, len(x)), np.tile(x, len(y))])


def log_print(string):
    print("{}: {}".format(datetime.datetime.now(), string))


def load_skels_from_dir(directory):
    pickle_path = os.path.join(directory, "parsed_graph.pickle")

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as fi:
            log_print("Loading skeleton data from pickle")
            vertices, edges, vertex_properties, G = pickle.load(fi)
            return [vertices, edges, vertex_properties, G]
    else:
        log_print("Loading skeleton data from directory")
        files = [os.path.join(directory, fi)
                 for fi in os.listdir(directory) if fi.endswith(".ply")]
        skel1 = PlyData.read(files[0])
        vertices = np.array(skel1["vertex"].data.tolist())
        vertex_properties = vertices[:, [0, 1]]
        vertices = vertices[:, 2:]
        edges = np.array(skel1["edge"].data.tolist())
        begin_verts = vertices[edges[:, 0], :]
        end_verts = vertices[edges[:, 1], :]
        vector_components = end_verts - begin_verts

        # Load a list of ply files
        for fi in files[1:]:
            skel = PlyData.read(fi)
            _vertices = np.array(skel["vertex"].data.tolist())
            _vertex_properties = _vertices[:, [0, 1]]
            _vertices = _vertices[:, 2:]
            _edges = np.array(skel["edge"].data.tolist())

            # Increment all the indices in edges so that they match up with the full vertex array
            edges = np.concatenate([edges, (_edges + vertices.shape[0])], axis=0)

            vertices = np.concatenate([vertices, _vertices], axis=0)
            vertex_properties = np.concatenate([vertex_properties, _vertex_properties], axis=0)

        G = nx.Graph()
        G.add_edges_from(edges)

        with open(pickle_path, "wb") as fi:
            log_print("Writing skeleton data as pickle")
            pickle.dump([vertices, edges, vertex_properties, G], fi)

    return [vertices, edges, vertex_properties, G]


def make_filtered_graph(_vertices, _vertex_properties, _edges, threshold):
    vertex_filter = _vertex_properties[:, 1] > threshold
    good_indices = np.arange(_vertices.shape[0])[vertex_filter]
    edge_filter = np.logical_and(np.isin(_edges[:, 0], good_indices),
                                 np.isin(_edges[:, 1], good_indices))
    filtered_verts = _vertices[vertex_filter, :]
    filtered_edges = _edges[edge_filter, :]
    filtered_begin_verts = _vertices[filtered_edges[:, 0], :]
    filtered_end_verts = _vertices[filtered_edges[:, 1], :]
    filtered_vector_components = filtered_end_verts - filtered_begin_verts
    filtered_distances = np.sqrt(np.sum(np.square(filtered_end_verts - filtered_begin_verts), axis=1))
    mlab.points3d(filtered_verts[:, 0], filtered_verts[:, 1], filtered_verts[:, 2],
                  np.sqrt(_vertex_properties[vertex_filter, 1]),
                  opacity=0.1)

    mlab.quiver3d(filtered_begin_verts[:, 0], filtered_begin_verts[:, 1], filtered_begin_verts[:, 2],
                  filtered_vector_components[:, 0], filtered_vector_components[:, 1],
                  filtered_vector_components[:, 2], scalars=filtered_distances, scale_mode="scalar", scale_factor=1,
                  mode="2ddash", opacity=1.)


def plot_subgraph_old(vertices, edges, vertex_properties, subgraph_no):
    _, subgraph_edges = get_subgraph(edges, subgraph_no)
    begin_verts = vertices[subgraph_edges[:, 0], :]
    end_verts = vertices[subgraph_edges[:, 1], :]
    vector_components = end_verts - begin_verts
    distances = np.sqrt(np.sum(np.square(end_verts - begin_verts), axis=1))
    vertex_mask = np.isin(np.arange(vertices.shape[0]), subgraph_edges)

    mlab.points3d(vertices[vertex_mask, 0], vertices[vertex_mask, 1], vertices[vertex_mask, 2],
                  vertex_properties[vertex_mask, 1], scale_mode='scalar', scale_factor=1, opacity=0.1)
    mlab.quiver3d(begin_verts[:, 0], begin_verts[:, 1], begin_verts[:, 2], vector_components[:, 0],
                  vector_components[:, 1],
                  vector_components[:, 2], scalars=distances, mode="2ddash", opacity=1., scale_mode="scalar",
                  scale_factor=1, color=(1, 1, 1))


def plot_subgraph(subgraph, vertices, points_kwargs={}, quiver_kwargs={}):
    subgraph_edges = np.array(subgraph.edges)
    begin_verts = vertices[subgraph_edges[:, 0], :]
    end_verts = vertices[subgraph_edges[:, 1], :]
    vector_components = end_verts - begin_verts
    distances = np.sqrt(np.sum(np.square(end_verts - begin_verts), axis=1))
    vertex_mask = np.unique(subgraph_edges.reshape((-1, 1)))

    mlab.points3d(vertices[vertex_mask, 0], vertices[vertex_mask, 1], vertices[vertex_mask, 2],
                  scale_mode='scalar', scale_factor=1, opacity=0.1, **points_kwargs)
    mlab.quiver3d(begin_verts[:, 0], begin_verts[:, 1], begin_verts[:, 2], vector_components[:, 0],
                  vector_components[:, 1],
                  vector_components[:, 2], scalars=distances, mode="2ddash", opacity=1., scale_mode="scalar",
                  scale_factor=1, **quiver_kwargs)


def get_subgraph(edges, subgraph_no):
    G = nx.Graph()
    G.add_edges_from(edges)
    subgraphs = list(sorted(nx.connected_components(G), key=len, reverse=True))
    root_subgraph = G.subgraph(subgraphs[subgraph_no])
    subgraph_edges = np.array(root_subgraph.edges)
    return root_subgraph, subgraph_edges


def plot_skeleton(skel):
    vertices, edges, vertex_properties = get_skel_data(skel)

    G = nx.Graph()
    G.add_edges_from(edges)

    plot_edges(edges, vertices)

    return plot_nodes(vertices, vertex_properties)


def plot_nodes(vertices, vertex_properties, **kwargs):
    plot_args = {"opacity": 0.05, "scale_mode": 'scalar', "scale_factor": 1, }
    plot_args.update(kwargs)
    return mlab.points3d(vertices[:, 0], vertices[:, 1], vertices[:, 2], vertex_properties[:, 1], **kwargs)


def get_skel_data(skel):
    vertices = np.array(skel["vertex"].data.tolist())
    vertex_properties = vertices[:, [0, 1]]
    vertices = vertices[:, 2:]
    edges = np.array(skel["edge"].data.tolist())
    return vertices, edges, vertex_properties


def plot_edges(edges, vertices, **kwargs):
    plot_args = {"mode": "2ddash", "opacity": 1., "scale_mode": "scalar",
                 "scale_factor": 1.}
    plot_args.update(kwargs)
    begin_verts = vertices[edges[:, 0], :]
    end_verts = vertices[edges[:, 1], :]
    vector_components = end_verts - begin_verts
    distances = np.sqrt(np.sum(np.square(end_verts - begin_verts), axis=1))
    return mlab.quiver3d(begin_verts[:, 0], begin_verts[:, 1], begin_verts[:, 2], vector_components[:, 0],
                         vector_components[:, 1],
                         vector_components[:, 2], scalars=distances, **plot_args)

def get_adjacent_points(center, vertices, side_length):
    half = side_length / 2.
    cx, cy, cz = center
    vx = vertices[:, 0]
    vy = vertices[:, 1]
    vz = vertices[:, 2]
    selected = vertices[np.logical_and.reduce(
        (vx >= (cx - half), vx <= (cx + half),
         vy >= (cy - half), vy <= (cy + half),
         vz >= (cz - half), vz <= (cz + half))), :]
    return selected


def get_subgraph_vector(subgraph, vertices, mean_root_vert, get_distal_leaf=False):
    degrees = np.array(subgraph.degree())
    leaf_indices = degrees[degrees[:, 1] == np.min(degrees[:, 1]), 0]
    leaf_verts = vertices[leaf_indices, :]
    #     try:
    distal_leaf = leaf_verts[np.argmax(spatial.distance.cdist(mean_root_vert, leaf_verts)), :]
    #     except Exception as e:
    #         log_print("Degrees: {}".format(degrees))
    #         log_print("Leaf verts:\n{}".format(leaf_verts))
    #         log_print("Leaf indices:\n{}".format(leaf_indices))
    #         raise e
    root_vector = np.mean(vertices[subgraph.nodes, :] - distal_leaf, axis=0)
    norm = np.linalg.norm(root_vector)
    if norm != 0:
        root_vector = root_vector / norm
    if not get_distal_leaf:
        return root_vector.reshape([1, -1])
    else:
        return root_vector.reshape([1, -1]), distal_leaf.reshape([1, -1])


def get_elliptical_distance(s_verts, d_verts, alpha, vector):
    # s_verts: m x 3
    # d_verts: o x 3
    # f(a,b): o x 1 x m x 3
    d_verts_br = d_verts[:, None, None, :]

    dxdxdz = d_verts_br - s_verts
    if ELLIPTICAL_DISTANCE_PLOT:
        vectors = dxdxdz[0, 0, :, :]
        norms = np.linalg.norm(vectors, axis=1)
        log_print(s_verts)
        log_print(vectors)
        log_print(norms)

        dest_vertex = d_verts[0, :].reshape([1, -1])
        mlab.points3d(dest_vertex[:, 0], dest_vertex[:, 1], dest_vertex[:, 2], color=(0, 1, 0))

        mlab.quiver3d(s_verts[:, 0], s_verts[:, 1], s_verts[:, 2], vectors[:, 0],
                      vectors[:, 1],
                      vectors[:, 2], scalars=norms, scale_mode="scalar", scale_factor=1, line_width=5)

    dx = dxdxdz[:, :, :, 0]
    dy = dxdxdz[:, :, :, 1]
    dz = dxdxdz[:, :, :, 2]
    sq_distance = dx * dx + dy * dy + dz * dz

    final_distance = np.sqrt(sq_distance +
                             alpha * np.square((dx * vector[:, 0] + dy * vector[:, 1] + dz * vector[:, 2])))

    lt0 = final_distance < 0
    if np.any(lt0):
        log_print("PROBLEM! Got final_distance < 0:")
        nonzeros = np.nonzero(lt0)
        log_print(np.vstack([nonzeros[0], nonzeros[1], final_distance[lt0]]).transpose())
        raise ValueError()

    final_distance = np.transpose(final_distance.reshape(d_verts.shape[0], s_verts.shape[0]))
    return final_distance


def get_elliptical_distance_bi(s_verts, d_verts, alpha, s_vectors, beta, d_vectors):
    # s_verts: m x 3
    # d_verts: o x 3
    # f(a,b): o x 1 x m x 3
    d_verts_br = d_verts[:, None, None, :]

    dxdxdz = d_verts_br - s_verts
    if ELLIPTICAL_DISTANCE_PLOT:
        vectors = dxdxdz[0, 0, :, :]
        norms = np.linalg.norm(vectors, axis=1)
        log_print(s_verts)
        log_print(vectors)
        log_print(norms)

        dest_vertex = d_verts[0, :].reshape([1, -1])
        mlab.points3d(dest_vertex[:, 0], dest_vertex[:, 1], dest_vertex[:, 2], color=(0, 1, 0))

        mlab.quiver3d(s_verts[:, 0], s_verts[:, 1], s_verts[:, 2], vectors[:, 0],
                      vectors[:, 1],
                      vectors[:, 2], scalars=norms, scale_mode="scalar", scale_factor=1, line_width=5)

    dx = dxdxdz[:, :, :, 0]
    dy = dxdxdz[:, :, :, 1]
    dz = dxdxdz[:, :, :, 2]
    sq_distance = dx * dx + dy * dy + dz * dz

    final_distance = np.sqrt(sq_distance -
                             alpha * np.square((dx * s_vectors[:, 0] + dy * s_vectors[:, 1] + dz * s_vectors[:, 2])) -
                             beta * np.square((dx * d_vectors[:, 0] + dy * d_vectors[:, 1] + dz * d_vectors[:, 2])))
    final_distance = np.transpose(final_distance.reshape(d_verts.shape[0], s_verts.shape[0]))
    return final_distance


def get_best_edge_between_subgraphs(subgraph_s, subgraph_d, s_vector, verts_and_idx):
    # Get nodes with smallest distance between them and the distance
    s_verts = verts_and_idx[subgraph_s.nodes, :]
    d_verts = verts_and_idx[subgraph_d.nodes, :]

    distances = get_elliptical_distance(s_verts, d_verts, 0.8, s_vector)
    min_distance_idx = np.unravel_index(distances.argmin(), distances.shape)
    min_distance_edge = (s_verts[min_distance_idx[0], 3], d_verts[min_distance_idx[1], 3], distances[min_distance_idx])

    return min_distance_edge


def link_by_steiner(directory):
    G, edges, root_indices, root_subgraph, subgraph_vectors, subgraphs, nx_subgraphs, vertices, verts_and_idx = \
        get_graph_data(directory)

    log_print("Setting up matrices")
    matrix_name = "edge_matrix_steiner_{}_alpha_0.8".format(DIRECTORY)
    # For coordinate [x, y, :], subgraph_node_edges should give the triple:
    # [closest node in x to y, closest node in y to x, distance between the two nodes]
    subgraph_node_edges = np.zeros((len(subgraphs), len(subgraphs), 3))
    node_deg_sg = np.array(
        [[row[0], row[1], idx] for idx, table in enumerate(subgraph.degree for subgraph in nx_subgraphs)
         for row in table])
    leaf_nodes = node_deg_sg[node_deg_sg[:, 1] == 1, :]
    leaf_nodes = leaf_nodes[:, (0, 2)]
    subgraph_lens = np.array([len(subgraph) for subgraph in subgraphs])
    if os.path.exists(matrix_name+".npy"):
        log_print("Loading best nodes and distance matrix")
        subgraph_node_edges = np.load(matrix_name+".npy")
    else:
        # Find shortest path between all subgraphs
        # Allow distances from larger subgraphs to overwrite smaller.
        # Small subgraphs have bad vectors, so skip them.
        log_print("Finding optimal paths among all connected components")
        good_subgraphs = reversed(list(np.nonzero(subgraph_lens > 5)[0]))

        for i in good_subgraphs:
            # Skip the root graph
            if i == 0:
                continue
            s_inds = leaf_nodes[leaf_nodes[:, 1] == i, 0]

            # If the subgraph has no leaf nodes, it's a loop. Skip it.
            if s_inds.size == 0:
                continue

            s_vector = subgraph_vectors[i]
            s_verts = vertices[s_inds, :]
            d_leaves = leaf_nodes[leaf_nodes[:, 1] != i, :]
            d_inds = d_leaves[:, 0]
            d_verts = vertices[d_inds, :]

            distances = get_elliptical_distance(s_verts, d_verts, alpha=0.8,
                                                vector=s_vector)

            best_dists = np.zeros(len(subgraphs))
            best_edges = np.zeros((len(subgraphs), 2))
            dist_table = distances.view().transpose()

            # Iterate over the table, finding the shortest path for each subgraph.
            # I tried iterating over each subgraph and doing a 2D argmin, but it was slower! Weird, huh?
            for row_idx in range(dist_table.shape[0]):
                d_idx, d_sg = d_leaves[row_idx, :].astype(int)
                best_col = np.argmin(dist_table[row_idx, :])
                best_s_idx = s_inds[best_col]
                best_dist = dist_table[row_idx, best_col]
                if best_dist < best_dists[d_sg] or best_dists[d_sg] == 0:
                    best_dists[d_sg] = best_dist
                    best_edges[d_sg] = [best_s_idx, d_idx]
            subgraph_node_edges[i, :, :] = np.hstack([best_edges, best_dists.reshape(-1, 1)])
            subgraph_node_edges[:, i, :] = np.hstack([best_edges, best_dists.reshape(-1, 1)])
            sg_no = len(subgraphs) - i
            if sg_no % 50 == 0:
                log_print("Processed {} subgraphs".format(sg_no))
        np.save(matrix_name, subgraph_node_edges)

    log_print("Rearrange data to make the Steiner tree")
    # Rearrange our data for making the Steiner tree
    tree_inds = cartesian_product(np.arange(subgraph_node_edges.shape[0]),
                                  np.arange(subgraph_node_edges.shape[0]))

    weights = subgraph_node_edges[:, :, 2]
    edge_mask = (weights.copy() > 0).flatten()
    edges = tree_inds[edge_mask, :]
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0).astype(np.int64)

    real_nodes = np.unique(edges)

    # We can't have any gaps in our node numbering
    nodes = np.arange(np.max(real_nodes + 1))
    prizes = np.zeros_like(nodes)
    prizes[real_nodes] = subgraph_lens[real_nodes]

    prizes = prizes.astype(np.float64)
    costs_indices = np.ravel_multi_index((edges[:, 0], edges[:, 1]), weights.shape)
    costs = weights.flatten()[costs_indices].astype(np.float64)

    output_nodes, output_edges = pcst_fast(edges, prizes * 10, costs, 0, 1, "gw", 1)
    steiner_edges = edges[output_edges, :]
    rendering_edges = subgraph_node_edges[tuple(steiner_edges[:, 0]), tuple(steiner_edges[:, 1]), :]
    rendering_edges = rendering_edges[:, :2].astype(int)

    rendering_nodes = np.unique(rendering_edges.flatten())
    sg_nodes = [index for i in output_nodes for index in subgraphs[i]]
    rendering_nodes = np.concatenate([rendering_nodes, sg_nodes])

    edge_mask = np.isin(edges, rendering_nodes)
    edge_mask = np.logical_or(edge_mask[:, 0], edge_mask[:, 1])

    rendering_edges = np.vstack([rendering_edges, edges[edge_mask, :]])

    plot_edges(rendering_edges, vertices)
    return rendering_edges


def link_by_distance(directory, max_distance, threshold, distance_metric="elliptical", min_neighbors=0,
                     bad=False, load_from_pickle=True, **plot_params):
    log_print("Loading skeletons, creating graphs")
    G, edges, root_indices, root_subgraph, subgraph_vectors, subgraphs, nx_subgraphs, vertices, verts_and_idx = \
        get_graph_data(directory)
    pickle_name = "distance_matrix_{}_{}_{}.pickle".format(directory, distance_metric, max_distance)
    if os.path.exists(pickle_name) and load_from_pickle:
        log_print("Reading distance matrix from file: {}".format(pickle_name))
        with open(pickle_name, "rb") as fi:
            distance_matrix = pickle.load(fi)
    else:
        log_print("Calculating distance matrix:")
        nrG = G.subgraph(item for subgraph in subgraphs[1:] for item in subgraph)

        log_print("Enumerating end nodes")
        nr_degrees = np.array(nrG.degree())
        nr_leaf_indices = nr_degrees[nr_degrees[:, 1] == 1, 0]

        indices_by_subgraph = np.array(
            [[node_idx, subgraph_idx] for subgraph_idx, subgraph in enumerate(subgraphs) for node_idx in subgraph])
        end_indices = np.concatenate([nr_leaf_indices, root_indices])
        end_indices_sub = indices_by_subgraph[np.isin(indices_by_subgraph, end_indices)[:, 0], :]

        log_print("Enumerating start nodes")
        subgraph_sizes = np.array([len(subgraph) for subgraph in subgraphs])
        # Don't start from root nodes
        start_nodes = end_indices_sub[end_indices_sub[:, 1] != 0]
        # Get distances from small subgraphs first, so larger ones can overwrite them
        start_nodes = start_nodes[np.argsort(subgraph_sizes[start_nodes[:, 1]]), :]
        start_nodes = start_nodes[subgraph_sizes[start_nodes[:, 1]] > 5, :]

        distance_matrix = sparse.lil_matrix((vertices.shape[0], vertices.shape[0]))

        distance_matrix[edges[:, 0], edges[:, 1]] = 10 ** -6

        ct = 0
        pruned = 0

        log_print("Finding neighbors")
        # TODO: Find optimal path (according to the cost function) between all pairwise subgraphs, then optimize that
        if bad:
            start_nodes = end_indices_sub[end_indices_sub[:, 1] != 0, :]

        for leaf_idx, subgraph in start_nodes:
            leaf = vertices[leaf_idx, :]
            _end_indices = end_indices_sub[end_indices_sub[:, 1] != subgraph, 0]
            end_verts = verts_and_idx[_end_indices, :]
            _max_distance = max_distance

            neighbors = get_adjacent_points(leaf, end_verts, _max_distance)
            while len(neighbors) < min_neighbors:
                neighbors = get_adjacent_points(leaf, end_verts, _max_distance)
                _max_distance = _max_distance + 10
            leaf = leaf.reshape(1, -1)

            if distance_metric == "elliptical":
                sg_vector = subgraph_vectors[subgraph]
                distances = get_elliptical_distance(leaf, neighbors[:, :3], alpha=0.8, vector=sg_vector)

            elif distance_metric == "bi_elliptical":
                sg_vector = subgraph_vectors[subgraph]
                neighbor_subgraphs = indices_by_subgraph[indices_by_subgraph[:, 0] == neighbors[:, 3], 1]

                d_vectors = subgraph_vectors[neighbor_subgraphs.astype(int), :]
                distances = get_elliptical_distance_bi(leaf, neighbors[:, :3], alpha=0.8, s_vectors=sg_vector,
                                                       beta=0.8, d_vectors=d_vectors)
            elif distance_metric == "euclidean":
                distances = spatial.distance_matrix(leaf, neighbors[:, :3])
            else:
                raise ValueError("distance_metric must be one of: elliptical, bi_elliptical, euclidean")

            distance_matrix[leaf_idx, neighbors[:, 3].astype(int)] = distances
            distance_matrix[neighbors[:, 3].astype(int), leaf_idx] = distances.transpose()
            ct = ct + 1

        if load_from_pickle:
            log_print("Writing distance matrix: {}".format(pickle_name))
            with open(pickle_name, "wb") as fi:
                pickle.dump(distance_matrix, fi)

    threshold_mask = distance_matrix > threshold
    pruned = np.sum(threshold_mask)
    distance_matrix[threshold_mask] = 0

    log_print("Pruned {} edges based on threshold {}".format(pruned, threshold))

    log_print("Calculating MST")
    weighted_G = nx.from_scipy_sparse_matrix(distance_matrix)
    mst = nx.minimum_spanning_tree(weighted_G)

    mst_subgraphs = list(sorted(nx.connected_components(mst), key=len, reverse=True))
    root_mst = mst.subgraph(mst_subgraphs[0])
    mst_edges = np.array(root_mst.edges)
    log_print("Plotting")
    plot_edges(mst_edges, vertices, **plot_params)
    root_edges = np.array(root_subgraph.edges)
    plot_edges(root_edges, vertices, color=(0, 0, 0), line_width=3.0)
    return mst_edges


def get_graph_data(directory):
    vertices, edges, vertex_properties, G = load_skels_from_dir(directory)
    verts_and_idx = np.hstack([vertices, np.arange(len(vertices)).reshape(-1, 1)])
    subgraphs = list(sorted(nx.connected_components(G), key=len, reverse=True))
    nx_subgraphs = [G.subgraph(subgraph) for subgraph in subgraphs]
    root_subgraph = nx_subgraphs[0]
    root_indices = root_subgraph.nodes
    root_verts = vertices[root_indices, :]
    root_mean = np.mean(root_verts, axis=0).reshape((1, 3))
    subgraph_vectors = np.array([get_subgraph_vector(subgraph, vertices, root_mean) for subgraph in nx_subgraphs])
    return G, edges, root_indices, root_subgraph, subgraph_vectors, subgraphs, nx_subgraphs, vertices, verts_and_idx


def intersect2d(A, B):
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, ncols)
    return C


if __name__ == "__main__":
    DIRECTORY = "brady129"
    MAX_DISTANCE = 100
    THRESHOLD = 30
    MIN_NEIGHBORS = 0
    BIDIRECTIONAL_DISTANCE = False
    PICKLE_NAME = "distance_matrix_{}.pickle".format(DIRECTORY)
    if os.path.exists(PICKLE_NAME):
        os.unlink(PICKLE_NAME)

    G, edges, root_indices, root_subgraph, subgraph_vectors, subgraphs, nx_subgraphs, vertices, verts_and_idx = \
        get_graph_data(DIRECTORY)
    MODE = "euclidean"
    if MODE == "steiner":
        from pcst_fast import pcst_fast  # From here: https://github.com/fraenkel-lab/pcst_fast
        link_by_steiner(DIRECTORY)
    else:
        new_edges = link_by_distance(DIRECTORY, MAX_DISTANCE, THRESHOLD, distance_metric="elliptical",
                                     min_neighbors=MIN_NEIGHBORS, load_from_pickle=True,
                                     color=(1, 0, 0))
        if MODE == "euclidean":
            old_edges = link_by_distance(DIRECTORY, MAX_DISTANCE, THRESHOLD, distance_metric="euclidean",
                                         min_neighbors=MIN_NEIGHBORS,
                                         color=(0, 0, 1))
        elif MODE == "elliptical_bad":
            old_edges = link_by_distance(DIRECTORY, MAX_DISTANCE, THRESHOLD, distance_metric="elliptical",
                                         min_neighbors=MIN_NEIGHBORS, bad=True, load_from_pickle=False,
                                         color=(0, 0, 1))

        elif MODE == "bi_elliptical":
            old_edges = link_by_distance(DIRECTORY, MAX_DISTANCE, THRESHOLD * 2.0, distance_metric="bi_elliptical",
                                         min_neighbors=MIN_NEIGHBORS,
                                         color=(0, 1, 0))
        elif MODE == "elliptical":
            old_edges = link_by_distance(DIRECTORY, MAX_DISTANCE, THRESHOLD, distance_metric="elliptical",
                                         min_neighbors=MIN_NEIGHBORS, load_from_pickle=True,
                                         color=(1, 0, 0))
        else:
            old_edges = np.array([])

        shared_edges = intersect2d(old_edges, new_edges)
        plot_edges(shared_edges, vertices, color=(0, 0, 0), line_width=10)

    mlab.show()
