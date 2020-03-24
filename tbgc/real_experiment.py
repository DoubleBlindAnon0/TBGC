"""Runs experiments on real datasets."""
import json
from multiprocessing import Pool

from tqdm import tqdm

from clustering_techniques import compute_matching_norm
from experiment_utils import run_iteration
from real_datasets import load_email_dataset, load_dbpl_dataset


def pooled_iteration_function(iteration):
    """Used for the multiprocessing Pool of run_iterations."""
    return run_iteration(graph_function, (10000,), random_state=iteration)

if __name__ == '__main__':
    # Experiment parameters
    graphs = list(zip(["email", "DBPL"], [load_email_dataset, load_dbpl_dataset]))
    range_of_iterations = range(20)

    # Experiment results dict
    results = {}

    for graph_name, graph_function in graphs:
        print("Experiment on {}".format(graph_name))

        experiment_measures = {"template_adj": {"ari": [], "projector_distance": [], "time": []},
                               "template_lap": {"ari": [], "projector_distance": [], "time": []},
                               "spectral": {"ari": [], "projector_distance": [], "time": []},
                               "modularity": {"ari": [], "projector_distance": [], "time": []}}

        with Pool(20) as pool:
            iteration_measures_list = list(tqdm(pool.imap(pooled_iteration_function, range_of_iterations), total=len(range_of_iterations)))
            #for iteration in tqdm(range_of_iterations):
            #    iteration_measures = run_iteration(graph_function, (c,), random_state=iteration)

            for iteration_measures in iteration_measures_list:
                for method in ["template_adj", "template_lap", "spectral", "modularity"]:
                    for measure in ["ari", "projector_distance", "time"]:
                        experiment_measures[method][measure].append(iteration_measures[method][measure])
                        print(iteration_measures)

        results[graph_name] = experiment_measures

    with open("real_results.json", "w") as fp:
        json.dump(results, fp, indent=2)