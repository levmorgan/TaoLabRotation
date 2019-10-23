#!/usr/bin/env python
# coding: utf-8

# !/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
from mayavi import mlab
from plyfile import PlyData
import networkx as nx
from scipy import sparse, spatial


def load_skels_from_dir(directory):
    pickle_path = os.path.join(directory, "parsed_graph.pickle")

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as fi:
            print("Loading skeleton data from pickle")
            vertices, edges, vertex_properties, G = pickle.load(fi)
            return [vertices, edges, vertex_properties, G]
    else:
        print("Loading skeleton data from directory")
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
            print("Writing skeleton data as pickle")
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


# plot_subgraph(vertices, edges, vertex_properties, 0)

# skel1 = PlyData.read("v2highres_skel.ply")
# vertices, edges, vertex_properties = get_skel_data(skel1)
# # plot_skeleton(skel1)
# plot_edges(edges, vertices)
# plot_edges(cycle1, vertices, color=(1, 0, 0), line_width=10.0)

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
    #         print("Degrees: {}".format(degrees))
    #         print("Leaf verts:\n{}".format(leaf_verts))
    #         print("Leaf indices:\n{}".format(leaf_indices))
    #         raise e
    root_vector = np.mean(vertices[subgraph.nodes, :] - distal_leaf, axis=0)
    norm = np.linalg.norm(root_vector)
    if norm != 0:
        root_vector = root_vector / norm
    if not get_distal_leaf:
        return root_vector.reshape([1, -1])
    else:
        return root_vector.reshape([1, -1]), distal_leaf.reshape([1, -1])

ELLIPTICAL_DISTANCE_PLOT = False

def get_elliptical_distance(s_verts, d_verts, alpha, vector):
    # s_verts: m x 3
    # d_verts: o x 3
    # f(a,b): o x 1 x m x 3
    d_verts_br = d_verts[:, None, None, :]

    dxdxdz = d_verts_br - s_verts
    if ELLIPTICAL_DISTANCE_PLOT:
        vectors = dxdxdz[0,0,:,:]
        norms = np.linalg.norm(vectors, axis=1)
        print(s_verts)
        print(vectors)
        print(norms)

        dest_vertex = d_verts[0, :].reshape([1, -1])
        mlab.points3d(dest_vertex[:, 0], dest_vertex[:, 1], dest_vertex[:,2], color=(0,1,0))

        mlab.quiver3d(s_verts[:, 0], s_verts[:, 1], s_verts[:, 2], vectors[:, 0],
                         vectors[:, 1],
                         vectors[:, 2], scalars=norms, scale_mode="scalar", scale_factor=1, line_width=5)

    dx = dxdxdz[:, :, :, 0]
    dy = dxdxdz[:, :, :, 1]
    dz = dxdxdz[:, :, :, 2]
    sq_distance = dx * dx + dy * dy + dz * dz

    final_distance = np.sqrt(sq_distance -
                             alpha * np.square((dx * vector[:, 0] + dy * vector[:, 1] + dz * vector[:, 2])))
                             
    final_distance = np.transpose(final_distance.reshape(d_verts.shape[0], s_verts.shape[0]))
    return final_distance


def get_elliptical_distance_bi(s_verts, d_verts, alpha, s_vectors, beta, d_vectors):
    # s_verts: m x 3
    # d_verts: o x 3
    # f(a,b): o x 1 x m x 3
    d_verts_br = d_verts[:, None, None, :]

    dxdxdz = d_verts_br - s_verts
    if ELLIPTICAL_DISTANCE_PLOT:
        vectors = dxdxdz[0,0,:,:]
        norms = np.linalg.norm(vectors, axis=1)
        print(s_verts)
        print(vectors)
        print(norms)

        dest_vertex = d_verts[0, :].reshape([1, -1])
        mlab.points3d(dest_vertex[:, 0], dest_vertex[:, 1], dest_vertex[:,2], color=(0,1,0))

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

    pickle_name = "distance_matrix_{}_alpha_0.8.pickle".format(DIRECTORY)
    if not os.path.exists(pickle_name):
        subgraph_node_edges = np.zeros((len(subgraphs), len(subgraphs), 3))

        # Find shortest path between subgraphs
        for i, subgraph in enumerate(subgraphs):
            # Skip the root graph
            if i == 0:
                continue
            s_vector = subgraph_vectors[i]
            _subgraph = nx_subgraphs[i]
            best_edges = np.array(
                [get_best_edge_between_subgraphs(_subgraph, subgraph, s_vector, verts_and_idx) for subgraph in nx_subgraphs])
            subgraph_node_edges[i, :, :] = best_edges

        with open(pickle_name, "wb") as pickle_fi:
            pickle.dump(subgraph_node_edges, pickle_fi)
    else:
        with open(pickle_name, "rb") as pickle_fi:
            subgraph_node_edges = pickle.load(pickle_fi)

    distance_matrix = subgraph_node_edges[:, :, 3]




def link_by_distance(directory, max_distance, min_neighbors=0):
    print("Loading skeletons, creating graphs")
    G, edges, root_indices, root_subgraph, subgraph_vectors, subgraphs, nx_subgraphs, vertices, verts_and_idx = \
        get_graph_data(directory)
    pickle_name = "distance_matrix_{}.pickle".format(directory)
    if os.path.exists(pickle_name):
        print("Reading distance matrix from file: {}".format(pickle_name))
        with open(pickle_name, "rb") as fi:
            distance_matrix, mst = pickle.load(fi)
    else:
        print("Calculating distance matrix:")
        nrG = G.subgraph(item for subgraph in subgraphs[1:] for item in subgraph)

        print("Finding connected component leaves")
        nr_degrees = np.array(nrG.degree())
        nr_leaf_indices = nr_degrees[nr_degrees[:, 1] == 1, 0]

        indices_by_subgraph = np.array([[item, idx] for idx, subgraph in enumerate(subgraphs) for item in subgraph])
        end_indices = np.concatenate([nr_leaf_indices, root_indices])
        end_indices_sub = indices_by_subgraph[np.isin(indices_by_subgraph, end_indices)[:, 0], :]

        distance_matrix = sparse.lil_matrix((vertices.shape[0], vertices.shape[0]))

        distance_matrix[edges[:, 0], edges[:, 1]] = 10 ^ -6

        ct = 0

        print("Finding neighbors")
        # TODO: Find optimal path (according to the cost function) between all pairwise subgraphs, then optimize that
        for leaf_idx, subgraph in end_indices_sub[end_indices_sub[:, 1] != 0, :]:
            leaf = vertices[leaf_idx, :]
            _end_indices = end_indices_sub[end_indices_sub[:, 1] != subgraph, 0]
            end_verts = verts_and_idx[_end_indices, :]
            _max_distance = max_distance

            neighbors = get_adjacent_points(leaf, end_verts, _max_distance)
            while len(neighbors) < min_neighbors:
                neighbors = get_adjacent_points(leaf, end_verts, _max_distance)
                _max_distance = _max_distance + 10
            leaf = leaf.reshape(1, -1)

            if not BIDIRECTIONAL_DISTANCE:
                sg_vector = subgraph_vectors[subgraph]
                distances = get_elliptical_distance(leaf, neighbors[:, :3], alpha=0.8, vector=sg_vector)
            else:
                sg_vector = subgraph_vectors[subgraph]
                d_vectors = subgraph_vectors[:, 3]
                distances = get_elliptical_distance_bi(leaf, neighbors[:, :3], alpha=0.8, s_vectors=sg_vector,
                                                       beta=0.8, d_vectors=d_vectors)

            distance_matrix[leaf_idx, neighbors[:, 3].astype(int)] = distances
            ct = ct + 1

        print("Calculating MST")
        weighted_G = nx.from_scipy_sparse_matrix(distance_matrix)
        mst = nx.minimum_spanning_tree(weighted_G)

        print("Writing distance matrix and MST: {}".format(pickle_name))
        with open(pickle_name, "wb") as fi:
            pickle.dump([distance_matrix, mst], fi)
    mst_subgraphs = list(sorted(nx.connected_components(mst), key=len, reverse=True))
    root_mst = mst.subgraph(mst_subgraphs[0])
    mst_edges = np.array(root_mst.edges)
    print("Plotting")
    mlab.clf()
    plot_edges(mst_edges, vertices)
    root_edges = np.array(root_subgraph.edges)
    plot_edges(root_edges, vertices, color=(0, 0, 0), line_width=3.0)
    mlab.show()


def get_graph_data(directory):
    vertices, edges, vertex_properties, G = load_skels_from_dir(directory)
    verts_and_idx = np.hstack([vertices, np.arange(len(vertices)).reshape(-1, 1)])
    subgraphs = list(sorted(nx.connected_components(G), key=len, reverse=True))
    nx_subgraphs = [G.subgraph(subgraph) for subgraph in subgraphs]
    root_subgraph = nx_subgraphs[0]
    root_indices = root_subgraph.nodes
    root_verts = vertices[root_indices, :]
    root_mean = np.mean(root_verts, axis=0).reshape((1, 3))
    subgraph_vectors = np.array([get_subgraph_vector(subgraph, vertices, root_mean) for subgraph in nx_subgraphs[1:]])
    return G, edges, root_indices, root_subgraph, subgraph_vectors, subgraphs, nx_subgraphs, vertices, verts_and_idx


if __name__ == "__main__":
    DIRECTORY = "brady129"
    MAX_DISTANCE = 30
    MIN_NEIGHBORS = 0
    BIDIRECTIONAL_DISTANCE = False

    link_by_distance(DIRECTORY, MAX_DISTANCE, MIN_NEIGHBORS)

    if False:
        os.unlink(pickle_name)
