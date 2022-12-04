import json
import math
import os

import networkx as nx
import numpy as np

from constants import EXP_INTERVAL_SCALE
from starter import read_input, read_output, score, visualize


def get_hyperparameters(name: str):
    with open("scores.json") as f:
        scores = json.load(f)
        G = read_output(read_input(f"inputs/{name}.in"), f"outputs/{name}.out")
        output_k_max = max([G.nodes[i]["team"] for i in range(len(G.nodes))])
        k_max = max(output_k_max, math.floor(2 * np.log(scores[name] / 100)))
        norm_sum_max = (np.log(scores[name]) / 70) ** 2 + 0.01
        exp_intervals = math.ceil(EXP_INTERVAL_SCALE * 70 * math.sqrt(norm_sum_max))
        return k_max, norm_sum_max, exp_intervals, scores[name]


def is_new_best(name: str, G: nx.Graph) -> bool:
    if os.path.isfile(f"outputs/{name}.out"):
        old_G = read_input(f"inputs/{name}.in")
        old_G = read_output(old_G, f"outputs/{name}.out")
        old_score = score(old_G)
        print("OLD SCORE:", old_score)
        return score(G) < old_score
    return True


def check_score(name: str) -> int:
    if os.path.isfile(f"outputs/{name}.out"):
        old_G = read_input(f"inputs/{name}.in")
        old_G = read_output(old_G, f"outputs/{name}.out")
        old_score = score(old_G)
        return old_score
    return 1000000000


def write_vars_from_graph(name: str, G: nx.Graph):
    k_max, norm_sum_max, exp_intervals, best_score = get_hyperparameters(name)
    size = len(G.nodes)

    # not an actual variable anymore
    x = [G.nodes[i]["team"] for i in range(size)]

    vars = {}

    for i in range(size):
        for j in range(1, k_max + 1):
            vars[f"x_ind_{i}_{j}"] = 1 if G.nodes[i]["team"] == j else 0

    for i in range(size):
        for j in range(i + 1, size):
            if G.has_edge(i, j):
                vars[f"b_{i}_{j}"] = 1 if x[i] == x[j] else 0

    k = 0
    for i in range(size):
        if x[i] > k:
            k = x[i]
    vars["k"] = k

    for i in range(1, k_max + 1):
        vars[f"k_ind_{i}"] = 1 if k == i else 0
    vars["k_inv"] = 1 / k
    vars["k_inv_sq"] = 1 / k**2
    vars["t"] = math.floor(100 * np.e ** (0.5 * k))
    p = [0] * k_max
    for i in range(size):
        p[x[i] - 1] += 1
    norm_sum = 0
    for i in range(1, k_max + 1):
        vars[f"p_{i}"] = p[i - 1]
        vars[f"used_ind_{i}"] = 0 if i <= k else 1
        norm_term_unsq = 1 / size * p[i - 1] - 1 / k if i <= k else 0
        vars[f"norm_term_unsq_{i}"] = norm_term_unsq
        vars[f"norm_term_{i}"] = norm_term_unsq**2
        norm_sum += vars[f"norm_term_{i}"]

    vars["norm_sum"] = norm_sum
    vars["norm_sum_sqrt"] = math.sqrt(norm_sum)
    exp_input = 70 * vars["norm_sum_sqrt"]
    vars["exp_input"] = exp_input
    vars["distribution"] = np.e**exp_input

    with open(f"solutions/{name}.mst", "w") as f:
        f.write("\n".join([f"{name} {vars[name]}" for name in vars]) + "\n")


def write_vars_from_output(name: str):
    G = read_input(f"inputs/{name}.in")
    G = read_output(G, f"outputs/{name}.out")
    write_vars_from_graph(name, G)


def write_all_vars():
    for size in ["small", "medium", "large"]:
        for i in range(1, 261):
            try:
                write_vars_from_output(f"{size}{i}")
            except:
                print(f"ERROR: {size}{i}")


def write_weights_from_input(name: str):
    G = read_input(f"inputs/{name}.in")
    with open(f"weights/{name}.txt", "w") as f:
        for i in range(len(G.nodes)):
            for j in range(len(G.nodes)):
                w = G.edges[i, j]["weight"] if G.has_edge(i, j) else 0
                f.write(f"{w}\n")


def sync_outputs():
    for name in os.listdir("outputs"):
        if name.endswith(".out"):
            print(name)
            try:
                cpp = read_input(f"inputs/{name[:-4]}.in")
                with open(f"cpp-outputs/{name}", "r") as f:
                    teams = [int(team) for team in f.read().split()]
                for i in range(len(teams)):
                    cpp.nodes[i]["team"] = teams[i]
                output = read_output(
                    read_input(f"inputs/{name[:-4]}.in"), f"outputs/{name}"
                )
                if score(output) <= score(cpp):
                    teams = [output.nodes[i]["team"] for i in range(len(output.nodes))]
                with open(f"cpp-outputs/{name}", "w") as f:
                    f.write("\n".join([str(team) for team in teams]) + "\n")
                with open(f"outputs/{name}", "w") as f:
                    f.write(json.dumps(teams))
            except Exception as e:
                print(e)


def mod():
    name = "medium107"
    G = read_output(read_input(f"inputs/{name}.in"), f"outputs/{name}.out")
    print(score(G))
    # G.nodes[0]["team"] = 8
    # 285, 290, etc.
    p = [0] * 20
    for i in range(len(G.nodes)):
        p[G.nodes[i]["team"] - 1] += 1
    print(p)
    for i, j in G.edges:
        if G.nodes[i]["team"] == G.nodes[j]["team"]:
            print(i, j, G.edges[i, j]["weight"])
    visualize(G)

    # k = max([G.nodes[i]["team"] for i in range(len(G.nodes))])
    # p = [0] * k
    # old_score = score(G)
    # for i in range(len(G.nodes)):
    #     print(i)
    #     if G.nodes[i]["team"] == 3:
    #         G.nodes[i]["team"] = 4
    #         for j in range(len(G.nodes)):
    #             if G.nodes[j]["team"] == 4:
    #                 G.nodes[j]["team"] = 1
    #                 if score(G) < old_score:
    #                     print("FOUND:", i, j)
    #                     return
    #                 G.nodes[j]["team"] = 4
    #         G.nodes[i]["team"] = 3

    # k = max([G.nodes[i]["team"] for i in range(len(G.nodes))])
    # old_score = score(G)
    # for i in range(len(G.nodes)):
    #     print(i)
    #     for j in range(i + 1, len(G.nodes)):
    #         if G.nodes[i]["team"] != G.nodes[j]["team"]:
    #             temp = G.nodes[i]["team"]
    #             G.nodes[i]["team"] = G.nodes[j]["team"]
    #             G.nodes[j]["team"] = temp
    #             if score(G) < old_score:
    #                 print(i, j)
    #                 return
    #             temp = G.nodes[i]["team"]
    #             G.nodes[i]["team"] = G.nodes[j]["team"]
    #             G.nodes[j]["team"] = temp


# mod()

# name = "small43"
# G = read_input(f"inputs/{name}.in")
# G = read_output(G, f"outputs/{name}.out")
# write_vars_from_graph(name, G)
