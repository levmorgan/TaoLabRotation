#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
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
                  np.sqrt(vertex_properties[vertex_filter, 1]),
                  opacity=0.1)

    mlab.quiver3d(filtered_begin_verts[:, 0], filtered_begin_verts[:, 1], filtered_begin_verts[:, 2],
                  filtered_vector_components[:, 0], filtered_vector_components[:, 1],
                  filtered_vector_components[:, 2], scalars=filtered_distances, scale_mode="scalar", scale_factor=1,
                  mode="2ddash", opacity=1.)


def plot_subgraph(vertices, edges, vertex_properties, subgraph_no):
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


cycle1 = np.array(
    [[118766, 118638], [118638, 269452], [269452, 118838], [118838, 269788], [269788, 118334], [118334, 269450],
     [269450, 118682], [118682, 269451], [269451, 118713], [118713, 118844], [118844, 118840], [118840, 269837],
     [269837, 118883], [118883, 269863], [269863, 118853], [118853, 269864], [269864, 118884], [118884, 269850],
     [269850, 118855], [118855, 269760], [269760, 118850], [118850, 269849], [269849, 118836], [118836, 269851],
     [269851, 118882], [118882, 269838], [269838, 118826], [118826, 269831], [269831, 118594], [118594, 118766]])


# plot_subgraph(vertices, edges, vertex_properties, 0)

# skel1 = PlyData.read("v2highres_skel.ply")
# vertices, edges, vertex_properties = get_skel_data(skel1)
# # plot_skeleton(skel1)
# plot_edges(edges, vertices)
# plot_edges(cycle1, vertices, color=(1, 0, 0), line_width=10.0)

def get_adjacent_points(center, vertices, side_length):
    half = side_length/2.
    cx, cy, cz = center
    vx = vertices[:, 0]
    vy = vertices[:, 1]
    vz = vertices[:, 2]
    selected = vertices[np.logical_and.reduce(
        (vx >= (cx-half), vx <= (cx+half),
         vy >= (cy-half), vy <= (cy+half),
         vz >= (cz-half), vz <= (cz+half))), :]
    return selected

# SUBGRAPH_SIZE_THRESHOLD = 50


DIRECTORY = "brady129"
DISTANCE = 30
MIN_NEIGHBORS = 0

print("Loading skeletons, creating graphs")
vertices, edges, vertex_properties, G = load_skels_from_dir(DIRECTORY)

verts_and_idx = np.hstack([vertices, np.arange(len(vertices)).reshape(-1, 1)])

subgraphs = list(sorted(nx.connected_components(G), key=len, reverse=True))
sG = G.subgraph(subgraphs[0])
# root_edges = np.array(sG.edges)

# root_indices = np.unique(root_edges.copy().flatten())
root_indices = sG.nodes
root_verts = vertices[root_indices, :]
root_degrees = np.array(G.degree(root_indices))
root_leaves = root_degrees[root_degrees[:, 1] == 1, 0]
leaf_verts = vertices[root_leaves, :]
root_mean = np.mean(root_verts, axis=0)

pickle_name = "distance_matrix_{}.pickle".format(DIRECTORY)
if os.path.exists(pickle_name):
    print("Reading distance matrix from file: {}".format(pickle_name))
    with open(pickle_name, "rb") as fi:
        distance_matrix, mst = pickle.load(fi)
else:
    print("Calculating distance matrix:")
    non_root_indices = np.delete(np.arange(vertices.shape[0]), root_indices)
    nrG = G.subgraph(item for subgraph in subgraphs[1:] for item in subgraph)

    print("Finding connected component leaves")
    nr_degrees = np.array(nrG.degree())
    nr_leaf_indices = nr_degrees[nr_degrees[:, 1] == 1, 0]
    # nr_leaf_vertices = vertices[nr_leaf_indices, :]

    indices_by_subgraph = np.array([[item, idx] for idx, subgraph in enumerate(subgraphs) for item in subgraph])
    end_indices = np.concatenate([nr_leaf_indices, root_indices])
    end_indices_sub = indices_by_subgraph[np.isin(indices_by_subgraph, end_indices)[:, 0], :]

    distance_matrix = sparse.lil_matrix((vertices.shape[0], vertices.shape[0]))

    distance_matrix[edges[:, 0], edges[:, 1]] = 10 ^ -6

    ct = 0

    print("Finding neighbors")
    # TODO: Find the optimal path (according to the cost function) between all pairwise subgraphs, then optimize that
    for leaf_idx, subgraph in end_indices_sub[end_indices_sub[:, 1] != 0, :]:
        leaf = vertices[leaf_idx, :]
        _end_indices = end_indices_sub[end_indices_sub[:, 1] != subgraph, 0]
        end_verts = verts_and_idx[_end_indices, :]
        max_distance = DISTANCE
        neighbors = np.zeros((0, 0))
        neighbors = get_adjacent_points(leaf, end_verts, max_distance)
        while len(neighbors) < MIN_NEIGHBORS:
            neighbors = get_adjacent_points(leaf, end_verts, max_distance)
            #         print("Got {} neighbors, max_distance={}".format(len(neighbors), max_distance))
            max_distance = max_distance + 10
        leaf = leaf.reshape(1, -1)
        # TODO: use elliptical distance here
        distances = spatial.distance_matrix(leaf, neighbors[:, :3])
        #     print(distances)
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


root_edges = np.array(sG.edges)

plot_edges(root_edges, vertices, color=(0,0,0), line_width=3.0)


# plot_edges(mst_edges, vertices, line_width=4.0)
# plot_edges(edges, vertices)
# mlab.points3d(leaf_verts[:, 0], leaf_verts[:, 1], leaf_verts[:, 2], color=(1,0,0), scale_factor=2)

mlab.show()
