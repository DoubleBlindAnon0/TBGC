"""Generates several toy datasets for the experiments.

Authors
-------
* Mateus Riva (mateus.riva@telecom-paris.fr)"""
import numpy as np
import networkx as nx


def generate_G3(c=6, intra_cluster_prob=0.9, inter_cluster_prob=0.1):
    """Generate G3 toy graph.

    Parameters
    ----------
    c : int or list of ints
      Number of nodes in each community/cluster. If `int`, assume all nodes are equal size.

    Returns
    ----------
    G_M, A_M, L_M, G_O, A_O, L_O, vertex_labels
    """
    # Generating G3_M as a graph with 3 vertices "a", "b" and "c" and 2 edges, "ab"
    # and "bc"
    k = 3  # b = number of vertices in the model graph
    if type(c) == int:
        c = np.repeat([c], k)  # c = list of number of vertices per community
    A_M = np.array(
        [[c[0] * 2 * 0.9, c[0] * 0.1 + c[1] * 0.1, 0],
         [c[0] * 0.1 + c[1] * 0.1, c[1] * 2 * 0.9, c[1] * 0.1 + c[2] * 0.1],
         [0, c[1] * 0.1 + c[2] * 0.1, c[2] * 2 * 0.9]]
    )
    G_M = nx.Graph(A_M)
    L_M = nx.laplacian_matrix(G_M).todense()

    # Generating G_O as a graph with three communities "A", "B", and "C" with 30
    # vertices each, of which both "A" and "B", and "B" and "C" are slightly connected
    n = np.sum(c)  # n = number of vertices in the observation graph
    v_label_shape = (1, n)
    p_matrix_shape = (k, k)
    block_matrix_shape = (n, n)
    block_matrix = np.zeros(block_matrix_shape, dtype=int)
    vertex_labels = np.repeat(np.arange(k), c)  # couldn't rename communities here
    p_matrix = [
        [0.9, 0.1, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.1, 0.9]
    ]

    for row, _row in enumerate(block_matrix):
        for col, _col in enumerate(block_matrix[row]):
            community_a = vertex_labels[row]
            community_b = vertex_labels[col]

            p = np.random.random()
            val = p_matrix[community_a][community_b]

            if p <= val:
                block_matrix[row][col] = 1
    G_O = nx.from_numpy_matrix(block_matrix)
    A_O = nx.to_numpy_matrix(G_O)
    L_O = nx.laplacian_matrix(G_O).todense()

    return G_M, A_M, L_M, G_O, A_O, L_O, vertex_labels


def generate_G6(c=6, intra_cluster_prob=0.9, inter_cluster_prob=0.1):
    """Generate G6 toy graph.
  
    Parameters
    ----------
    c : int or list of ints
      Number of nodes in each community/cluster. If `int`, assume all nodes are equal size.
  
    Returns
    ----------
    G_M, A_M, L_M, G_O, A_O, L_O, vertex_labels
    """
    k = 6  # k = number of vertices in the model graph
    # Generating G_M as a graph with 5 vertices "a-e" and 6 edges, "ab", "bc", "bd",
    # "cd", "ce", "ef"
    if type(c) == int:
        c = np.repeat([c], k)
    A_M = np.array(
        [[c[0] * 2 * 0.9, c[0] * 0.1 + c[1] * 0.1, 0, 0, 0, 0],
         [c[0] * 0.1 + c[1] * 0.1, c[1] * 2 * 0.7, c[1] * 0.1 + c[2] * 0.1, c[1] * 0.1 + c[3] * 0.1, 0, 0],
         [0, c[1] * 0.1 + c[2] * 0.1, c[2] * 2 * 0.7, c[2] * 0.1 + c[3] * 0.1, c[2] * 0.1 + c[4] * 0.1, 0],
         [0, c[1] * 0.1 + c[3] * 0.1, c[2] * 0.1 + c[3] * 0.1, c[3] * 2 * 0.8, 0, 0],
         [0, 0, c[2] * 0.1 + c[4] * 0.1, 0, c[4] * 2 * 0.8, c[4] * 0.1 + c[5] * 0.1],
         [0, 0, 0, 0, c[4] * 0.1 + c[5] * 0.1, c[5] * 2 * 0.8]]
    )
    G_M = nx.Graph(A_M)
    L_M = nx.laplacian_matrix(G_M).todense()

    # Generating G_O as a graph of five communities
    n = np.sum(c)  # n = number of vertices in the observation graph
    v_label_shape = (1, n)
    p_matrix_shape = (k, k)
    block_matrix_shape = (n, n)
    block_matrix = np.zeros(block_matrix_shape, dtype=int)
    vertex_labels = np.repeat(np.arange(k), c)
    p_matrix = [
        # a---b---c---d---e---f
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],  # a
        [0.1, 0.7, 0.1, 0.1, 0.0, 0.0],  # b
        [0.0, 0.1, 0.7, 0.1, 0.1, 0.0],  # c
        [0.0, 0.1, 0.1, 0.8, 0.0, 0.0],  # d
        [0.0, 0.0, 0.1, 0.0, 0.8, 0.1],  # e
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.9]  # f
    ]

    for row, _row in enumerate(block_matrix):
        for col, _col in enumerate(block_matrix[row]):
            community_a = vertex_labels[row]
            community_b = vertex_labels[col]

            p = np.random.random()
            val = p_matrix[community_a][community_b]

            if p <= val:
                block_matrix[row][col] = 1
    G_O = nx.from_numpy_matrix(block_matrix)
    A_O = nx.to_numpy_matrix(G_O)
    L_O = nx.laplacian_matrix(G_O).todense()

    return G_M, A_M, L_M, G_O, A_O, L_O, vertex_labels


def generate_C2(c=6, intra_cluster_prob=0.48, inter_cluster_prob=0.42):
    """Generate C2 toy graph.

    Note that, optimally, `intra_cluster_prob + inter_cluster_prob = 1.0`.

    Parameters
    ----------
    c : int or list of ints
      Number of nodes in each community/cluster. If `int`, assume all nodes are equal size.
    intra_cluster_prob : float
      Probability of connecting nodes inside clusters 2 and 3.
    inter_cluster_prob : float
      Probability of connecting nodes between clusters 2 and 3.

    Returns
    ----------
    G_M, A_M, L_M, G_O, A_O, L_O, vertex_labels
    """
    # Generating G_M as a graph with 4 vertices "a-d" and 3 edges, "ab", "bc", "cd"
    k = 4  # k = number of vertices in the model graph
    if type(c) == int:
        c = np.repeat([c], k)
    A_M = np.array(
        [[c[0] * 2 * 0.9, c[1] * 0.1 + c[2] * 0.1, 0, 0],
         [c[1] * 0.1 + c[2] * 0.1, c[1] * 2 * intra_cluster_prob, c[2] * inter_cluster_prob + c[3] * inter_cluster_prob,
          0],
         [0, c[2] * inter_cluster_prob + c[3] * inter_cluster_prob, c[2] * 2 * intra_cluster_prob,
          c[2] * 0.1 + c[3] * 0.1],
         [0, 0, c[2] * 0.1 + c[3] * 0.1, c[3] * 2 * 0.9]]
    )
    G_M = nx.Graph(A_M)
    L_M = nx.laplacian_matrix(G_M).todense()

    # Generating G_O as a graph of five communities
    if type(c) == int:
        c = np.repeat([c], k)
    n = np.sum(c)  # n = number of vertices in the observation graph
    v_label_shape = (1, n)
    p_matrix_shape = (k, k)
    block_matrix_shape = (n, n)
    block_matrix = np.zeros(block_matrix_shape, dtype=int)
    vertex_labels = np.repeat(np.arange(k), c)
    p_matrix = [
        # a---b---c---d
        [0.9, 0.1, 0.0, 0.0],  # a
        [0.1, intra_cluster_prob, inter_cluster_prob, 0.0],  # b
        [0.0, inter_cluster_prob, intra_cluster_prob, 0.1],  # c
        [0.0, 0.0, 0.1, 0.9]  # d
    ]

    for row, _row in enumerate(block_matrix):
        for col, _col in enumerate(block_matrix[row]):
            community_a = vertex_labels[row]
            community_b = vertex_labels[col]

            p = np.random.random()
            val = p_matrix[community_a][community_b]

            if p <= val:
                block_matrix[row][col] = 1
    G_O = nx.from_numpy_matrix(block_matrix)
    A_O = nx.to_numpy_matrix(G_O)
    L_O = nx.laplacian_matrix(G_O).todense()

    return G_M, A_M, L_M, G_O, A_O, L_O, vertex_labels


def generate_bp(c=6, intra_cluster_prob=0.5, inter_cluster_prob=0.5):
    """Generate BP toy graph.

    Parameters
    ----------
    intra_cluster_prob : float
      Probability that any two nodes in the first community are connected
    c : int or list of ints
      Number of nodes in each community/cluster. If `int`, assume all nodes are equal size.
    inter_cluster_prob : float
      Probability that any two nodes in the distinct communities are connected

    Returns
    ----------
    G_M, A_M, L_M, G_O, A_O, L_O, vertex_labels
    """
    k = 2  # k = number of vertices in the model graph
    if type(c) == int:
        c = np.repeat([c], k)  # c = list of number of vertices per community
    A_M = np.array(
        [[c[0] * 2 * intra_cluster_prob, np.sum(c) * 2 * inter_cluster_prob],
         [np.sum(c) * 2 * inter_cluster_prob, 0]]
    )
    G_M = nx.Graph(A_M)
    L_M = nx.laplacian_matrix(G_M).todense()

    # Generating G_O as a bipartite graph
    n = np.sum(c)  # n = number of vertices in the observation graph
    block_matrix_shape = (n, n)
    A_O = np.zeros(block_matrix_shape, dtype=int)
    vertex_labels = np.repeat(np.arange(k), c)

    for row, _row in enumerate(A_O):
        for col, _col in enumerate(A_O[:row]):
            # If different communities:
            if row < c[0] <= col or row >= c[0] > col:
                if np.random.random() < inter_cluster_prob:
                    A_O[row, col] = 1
                    A_O[col, row] = 1
            # If first community:
            if row < c[0] and col < c[0]:
                if np.random.random() < intra_cluster_prob:
                    A_O[row, col] = 1
                    A_O[col, row] = 1
    G_O = nx.from_numpy_matrix(A_O)
    L_O = nx.laplacian_matrix(G_O).todense()

    return G_M, A_M, L_M, G_O, A_O, L_O, vertex_labels