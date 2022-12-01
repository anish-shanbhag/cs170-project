import json
import math
import os

import networkx as nx
import numpy as np

from constants import EXP_INTERVAL_SCALE, SQ_INTERVALS, SQRT_INTERVALS
from starter import read_input, read_output, score


def get_hyperparameters(name: str):
    with open("scores.json") as f:
        scores = json.load(f)
        k_max = math.floor(2 * np.log(scores[name] / 100))
        norm_sum_max = (np.log(scores[name]) / 70) ** 2 + 0.01
        exp_intervals = math.ceil(EXP_INTERVAL_SCALE * 70 * math.sqrt(norm_sum_max))
        return k_max, norm_sum_max, exp_intervals, scores[name]


def is_new_best(name: str, G: nx.Graph) -> bool:
    if os.path.isfile(f"outputs/{name}.out"):
        old_G = read_input(f"inputs/{name}.in")
        old_G = read_output(old_G, f"outputs/{name}.out")
        old_score = score(old_G)
        print("OLD SCORE:", old_score)
        return score(G) <= old_score
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
    k_max += 1
    size = len(G.nodes)
    x = [G.nodes[i]["team"] for i in range(size)]
    vars = {f"x_{i}": x[i] for i in range(size)}

    def add_fn_approximation(ind_name, result_name, input_value, intervals, lb, rb, fn):
        input_range = rb - lb
        input_ind = 0
        for i in range(1, intervals + 2):
            val = lb + input_range * (i - 1) / intervals
            if abs(val - input_value) <= input_range / intervals:
                input_ind = i
        result = fn(lb + input_range * (input_ind - 1) / intervals)
        vars[result_name] = result
        for i in range(1, intervals + 2):
            vars[f"{ind_name}_{i}"] = 1 if input_ind == i else 0

    k = 0
    for i in range(size):
        if x[i] > k:
            k = x[i]
    vars["k"] = k

    for i in range(size):
        for j in range(1, k_max + 1):
            vars[f"x_ind_{i}_{j}"] = 1 if x[i] == j else 0
    for i in range(size):
        for j in range(i + 1, size):
            if G.has_edge(i, j):
                vars[f"w_{i}_{j}"] = G.edges[i, j]["weight"] if x[i] == x[j] else 0
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
        add_fn_approximation(
            f"norm_term_unsq_ind_{i}",
            f"norm_term_{i}",
            norm_term_unsq,
            SQ_INTERVALS,
            -0.5,
            1,
            lambda x: x**2,
        )
        norm_sum += vars[f"norm_term_{i}"]

    add_fn_approximation(
        "norm_sum_ind",
        "norm_sum_sqrt",
        norm_sum,
        SQRT_INTERVALS,
        0,
        norm_sum_max,
        lambda x: math.sqrt(abs(max(0, x))),
    )
    exp_input = 70 * vars["norm_sum_sqrt"]
    vars["exp_input"] = exp_input
    add_fn_approximation(
        "exp_input_ind",
        "distribution",
        exp_input,
        exp_intervals,
        0,
        70 * math.sqrt(norm_sum_max),
        lambda x: np.e**x,
    )

    # with open(f"solutions/{name}.mst", "r") as f:
    #     old_vars = {}
    #     for line in f:
    #         s = line.split()
    #         old_vars[s[0]] = float(s[1])
    #     for key in old_vars:
    #         if True and abs(old_vars[key] - vars[key]) > 0.00001:
    #             y = key.split("_")
    #             print(
    #                 "DIFF:",
    #                 key,
    #                 old_vars[key],
    #                 vars[key],
    #             )

    with open(f"solutions/{name}.mst", "w") as f:
        f.write("\n".join([f"{name} {vars[name]}" for name in vars]) + "\n")


def write_vars_from_output(name: str):
    G = read_input(f"inputs/{name}.in")
    G = read_output(G, f"outputs/{name}.out")
    write_vars_from_graph(name, G)


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
                f.write("\n".join(teams) + "\n")
            with open(f"outputs/{name}", "w") as f:
                f.write(json.dumps(teams) + "\n")


def write_outputs_from_cpp():
    for name in os.listdir("cpp-outputs"):
        if name.endswith(".out"):
            print(name)
            with open(f"cpp-outputs/{name}", "r") as f:
                teams = [int(team) for team in f.read().split()]
            old = read_output(read_input(f"inputs/{name[:-4]}.in"), f"outputs/{name}")
            new = read_input(f"inputs/{name[:-4]}.in")
            for i in range(len(teams)):
                new.nodes[i]["team"] = teams[i]
            if score(new) < score(old):
                with open(f"outputs/{name}", "w") as f:
                    json.dump(teams, f)


# name = "small43"
# G = read_input(f"inputs/{name}.in")
# G = read_output(G, f"outputs/{name}.out")
# write_vars_from_graph(name, G)
