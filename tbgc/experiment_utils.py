"""Utility functions for running experiments with TBGC.

Authors
-------
 * Mateus Riva (mateus.riva@telecom-paris.fr)
"""
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score

from clustering_techniques import cluster_template, cluster_spectral, cluster_modularity
from clustering_techniques import compute_matching_norm


def run_iteration(generator_function: Callable, generator_args: Sequence, cost_function: Callable = compute_matching_norm, random_state = None) -> dict:
    """Generates a graph using the specified function and arguments, realizes all clusterings and compute metrics.

    Parameters
    ----------
    generator_function : Callable
        Function for generating a graph. This function should return at least a 7-tuple with elements 1 being $A_M$, 4
        being $A_O$ and 6 being the ground truth labels of vertices.
    generator_args : Sequence
        Arguments for the generator function. For the included toy datasets, these are `c, intra_cluster_prob,
        inter_cluster_prob`.
    cost_function : Callable
        The cost function that the TBGC technique will optimize. Used for testing alternative cost functions.

    Returns
    -------
    measures : dict
        Dictionary containing each technique's measures, organized as `[technique_name][measure]`.
    """
    # Initializing random state if specified
    if random_state is not None:
        np.random.seed(random_state)

    # Generate graph
    G_M, A_M, L_M, G_O, A_O, L_O, vertex_labels = generator_function(*generator_args)
    # Computing ground-truth P from vertex labels
    k, n = A_M.shape[0], A_O.shape[0]  # Getting cluster size (amount of vertices in model) and observation size
    P_gt = np.zeros((n, k))
    P_gt[np.arange(n), vertex_labels] = 1
    P_gt = P_gt @ np.diag(1/np.sqrt(np.diag(P_gt.T@P_gt)))  # TODO: is this still correct on adjacency?
    # w/ division ^
    #template_adj 0.8404879031765299
    #template_lap 1.1156126900660381
    #spectral 1.4298900861891668
    # wo/ division
    #template_adj 18.157258517397675
    #template_lap 18.576717569039133
    #spectral 18.576468690032076

    # Run techniques
    template_adj_prediction, P_opt_template_adj, template_adj_time = \
        cluster_template(A_M, A_O, mode="adjacency", cost_function=cost_function)
    template_lap_prediction, P_opt_template_lap, template_lap_time = \
        cluster_template(A_M, A_O, mode="laplacian", cost_function=cost_function)
    spectral_prediction, P_opt_spectral, spectral_time = \
        cluster_spectral(A_M, A_O)
    modularity_prediction, _, modularity_time = \
        cluster_modularity(A_M, A_O)

    # Compute measures
    measures = {"template_adj": {}, "template_lap": {}, "spectral": {}, "modularity": {}}
    for method, prediction, features, times in zip(["template_adj", "template_lap", "spectral", "modularity"],
                                                   [template_adj_prediction, template_lap_prediction,  spectral_prediction, modularity_prediction],
                                                   [P_opt_template_adj, P_opt_template_lap, P_opt_spectral, None],
                                                   [template_adj_time, template_lap_time, spectral_time, modularity_time]):
        # Adjusted Rand Index
        measures[method]["ari"] = adjusted_rand_score(vertex_labels, prediction)
        # Distance to gt projector
        if method != "modularity":
            measures[method]["projector_distance"] = np.linalg.norm(
                np.matmul(P_gt, P_gt.T) - np.matmul(features, features.T))
        else:
            measures[method]["projector_distance"] = "N/A"
        # Computational time
        measures[method]["time"] = times

    return measures