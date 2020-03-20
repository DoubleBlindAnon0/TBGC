"""Plots JSON results file from the TBGC experiments."""
import json

import matplotlib.pyplot as plt
import numpy as np


def plot_basic_results():# Plotting basic results
    graph_names = ["C2","G3","G6"]
    measures = ["ari", "projector_distance", "time"]
    techniques = ["template_adj", "template_lap", "spectral", "modularity"]
    cluster_sizes = ["5","10","20","40","80"]

    with open("basic_results.json") as fp:
        results_dict = json.load(fp)

    # BASIC PLOT SCHEMA: one plot per measure and graph type, x-axis is cluster size, lines are techniques
    for graph_name in graph_names:
        # Setting up printables name
        if graph_name == "G3": graph_print_name = "$G_3$"
        if graph_name == "G6": graph_print_name = "$G_6$"
        if graph_name == "C2": graph_print_name = "$C2$"

        for measure in measures:
            # Setting up printables name
            if measure == "ari": measure_print = "ARI"
            if measure == "projector_distance": measure_print = "Projector Distance"
            if measure == "time": measure_print = "Time"

            # Starting plot
            plt.figure()
            plt.title(graph_print_name)
            for technique in techniques:
                # Skipping modularity and PD
                if technique == "modularity" and measure == "projector_distance":
                    continue
                # Setting up printables name
                if technique == "template_adj": technique_printable = "Adjacency TB" ; color = (0.6,0,0) ; marker = "^" ; variation = -0.15
                if technique == "template_lap": technique_printable = "Laplacian TB" ; color = (1.0,0.5,0.1) ; marker = "v" ; variation = -0.05
                if technique == "spectral": technique_printable = "Spectral" ; color = "g" ; marker = "o" ; variation = 0.05
                if technique == "modularity": technique_printable = "Modularity" ; color = "b" ; marker = "D" ; variation = 0.15

                measure_list = [results_dict[graph_name][cluster_size][technique][measure] for cluster_size in cluster_sizes]

                plt.errorbar(x=np.array(range(len(cluster_sizes))) + variation,
                             y=np.mean(measure_list, axis=1),
                             yerr=np.std(measure_list, axis=1),
                             markersize=6, marker=marker, color=color, capsize=2,
                             label=technique_printable)

            plt.xticks(range(len(cluster_sizes)), labels=cluster_sizes)
            plt.ylabel(measure_print)
            plt.xlabel("Cluster size")
            plt.legend()
            plt.grid(b=True, axis="both", which="major")

            plt.tight_layout()
            plt.savefig("basic_{}_{}.eps".format(graph_name, measure))

def plot_progression_results():

    intra_cluster_probabilities = ["80.0", "75.0", "70.0", "65.0", "60.0", "55.0", "50.0", "45.0", "40.0"][::-1]
    inter_cluster_probabilities_print = ["20%", "25%", "30%", "35%", "40%", "45%", "50%", "55%", "60%"][::-1]
    measures = ["ari", "projector_distance", "time"]
    techniques = ["template_adj", "template_lap", "spectral", "modularity"]
    cluster_sizes = ["10","20","40"]

    with open("progression_results.json") as fp:
        results_dict = json.load(fp)

    # PROGRESSION PLOT SCHEMA: one plot per cluster size/measure, x-axis is intraprob, lines are techniques
    for cluster_size in cluster_sizes:
        for measure in measures:
            # Setting up printables name
            if measure == "ari": measure_print = "ARI"
            if measure == "projector_distance": measure_print = "Projector Distance"
            if measure == "time": measure_print = "Time"

            # Starting plot
            plt.figure()
            plt.title("Cluster size {}".format(cluster_size))
            for technique in techniques:
                # Skipping modularity and PD
                if technique == "modularity" and measure == "projector_distance":
                    continue
                # Setting up printables name
                if technique == "template_adj": technique_printable = "Adjacency TB"; color = (0.6, 0, 0); marker = "^"; variation = -0.15
                if technique == "template_lap": technique_printable = "Laplacian TB"; color = (1.0, 0.5, 0.1); marker = "v"; variation = -0.05
                if technique == "spectral": technique_printable = "Spectral"; color = "g"; marker = "o"; variation = 0.05
                if technique == "modularity": technique_printable = "Modularity"; color = "b"; marker = "D"; variation = 0.15

                measure_list = [results_dict[cluster_size][icp][technique][measure] for icp in
                                intra_cluster_probabilities]

                plt.errorbar(x=np.array(range(len(intra_cluster_probabilities))) + variation,
                             y=np.mean(measure_list, axis=1),
                             yerr=np.std(measure_list, axis=1),
                             markersize=6, marker=marker, color=color, capsize=2,
                             label=technique_printable)

            plt.xticks(range(len(intra_cluster_probabilities)), labels=inter_cluster_probabilities_print)
            plt.ylabel(measure_print)
            plt.xlabel("Inter-cluster Probability")
            plt.legend()
            plt.grid(b=True, axis="both", which="major")

            plt.tight_layout()
            plt.savefig("progression_{}_{}.eps".format(cluster_size, measure))


if __name__ == '__main__':
    inter_cluster_probabilities = ["80", "75", "70", "65", "60", "55", "50", "45", "40"]
    inter_cluster_probabilities_print = ["80%", "75%", "70%", "65%", "60%", "55%", "50%", "45%", "40%"]
    measures = ["ari", "projector_distance", "time"]
    techniques = ["template_adj", "template_lap", "spectral", "modularity"]
    cluster_sizes = ["10","20","40"]
    graphs = ["HUB", "BIPARTITE"]

    with open("bp_results.json") as fp:
        results_dict = json.load(fp)

    # PROGRESSION PLOT SCHEMA: one plot per graph/cluster size/measure, x-axis is interprob, lines are techniques
    for graph_name in graphs:
        # printables
        if graph_name == "HUB": graph_name_print = "Hub (50% ICP)"
        if graph_name == "BIPARTITE": graph_name_print = "Bipartite (0% ICP)"
        for cluster_size in cluster_sizes:
            for measure in measures:
                # Setting up printables name
                if measure == "ari": measure_print = "ARI"
                if measure == "projector_distance": measure_print = "Projector Distance"
                if measure == "time": measure_print = "Time"

                # Starting plot
                plt.figure()
                plt.title("{}, cluster size {}".format(graph_name_print, cluster_size))
                for technique in techniques:
                    # Skipping modularity and PD
                    if technique == "modularity" and measure == "projector_distance":
                        continue
                    # Setting up printables name
                    if technique == "template_adj": technique_printable = "Adjacency TB"; color = (0.6, 0, 0); marker = "^"; variation = -0.15
                    if technique == "template_lap": technique_printable = "Laplacian TB"; color = (1.0, 0.5, 0.1); marker = "v"; variation = -0.05
                    if technique == "spectral": technique_printable = "Spectral"; color = "g"; marker = "o"; variation = 0.05
                    if technique == "modularity": technique_printable = "Modularity"; color = "b"; marker = "D"; variation = 0.15

                    measure_list = [results_dict[graph_name][cluster_size][icp][technique][measure] for icp in
                                    inter_cluster_probabilities]

                    plt.errorbar(x=np.array(range(len(inter_cluster_probabilities))) + variation,
                                 y=np.mean(measure_list, axis=1),
                                 yerr=np.std(measure_list, axis=1),
                                 markersize=6, marker=marker, color=color, capsize=2,
                                 label=technique_printable)

                plt.xticks(range(len(inter_cluster_probabilities)), labels=inter_cluster_probabilities_print)
                plt.ylabel(measure_print)
                plt.xlabel("Inter-cluster Probability")
                plt.legend()
                plt.grid(b=True, axis="both", which="major")

                plt.tight_layout()
                plt.savefig("bipartite_{}_{}_{}.eps".format(graph_name, cluster_size, measure))
