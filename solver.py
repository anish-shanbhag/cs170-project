import math
import os
import json
import networkx as nx
import numpy as np
import pulp as pl
from gurobipy import GRB
from termcolor import colored

from helper import get_hyperparameters, is_new_best, write_vars_from_output
from starter import read_input, score, write_output


def log(*args):
    print(*[colored(arg, "blue") for arg in args])


def solve(name: str):
    k_max, norm_sum_max, exp_intervals, best_score = get_hyperparameters(name)
    G: nx.Graph = read_input(f"inputs/{name}.in")
    log("SOLVING:", name)
    log("K_MAX:", k_max)
    log("NORM_SUM_MAX:", norm_sum_max)
    log("EDGE COUNT:", G.number_of_edges())
    size = len(G.nodes)

    model = pl.LpProblem("CS 170 Solver", pl.LpMinimize)
    solver = pl.GUROBI(Threads=8, warmStart=True)

    last_score = -1

    def sol_callback(model, where):
        nonlocal last_score
        get_var = lambda name: model.cbGetSolution(model.getVarByName(name))
        if where == GRB.Callback.MIPSOL:
            for i in range(size):
                for k in range(1, k_max + 1):
                    if get_var(f"x_ind_{i}_{k}") == 1:
                        G.nodes[i]["team"] = k
                        break
            new_score = score(G)
            if new_score != last_score:
                last_score = new_score
                new_best = is_new_best(name, G)
                msg = "NEW BEST" if new_best else "current score"
                log(f"{name} {msg}:", score(G), "from", score(G, True))
                # log("k =", get_var("k"))

                # log("k_ind =", [get_var(f"k_ind_{i}") for i in range(1, k_max + 1)])
                # log("t =", get_var("t"))
                # log("k_inv =", get_var("k_inv"))
                # log("k_inv_sq =", get_var("k_inv_sq"))
                # for i in range(size // 30):
                #     for j in range(i + 1, size):
                #         if G.has_edge(i, j) and get_var(f"d_{i}_{j}") == 0:
                #             log(
                #                 f"d_{i}_{j} =",
                #                 get_var(f"d_{i}_{j}"),
                log("p =", [get_var(f"p_{i}") for i in range(1, k_max + 1)])
                # log("used_ind =", [get_var(f"used_ind_{i}") for i in range(1, k_max + 1)])
                # log(
                #     "norm_term_unsq =",
                #     [get_var(f"norm_term_unsq_{i}") for i in range(1, k_max + 1)],
                # )
                # log(
                #     "norm_term =",
                #     [get_var(f"norm_term_{i}") for i in range(1, k_max + 1)],
                # )
                # log("norm_sum_sqrt =", get_var("norm_sum_sqrt"))
                # log("exp_input =", get_var("exp_input"))
                # log(
                #     "exp_input_ind =",
                #     [get_var(f"exp_input_ind_{i}") for i in range(1, exp_intervals + 2)],
                # )

                log("distribution =", get_var("distribution"))
                if new_best:
                    write_output(G, f"outputs/{name}.out", True)
                    write_vars_from_output(name)

    # x_ind: penguin i is assigned to team j
    x_ind = [0] * size
    for i in range(size):
        x_ind[i] = [
            pl.LpVariable(f"x_ind_{i}_{j}", None, None, pl.LpBinary)
            for j in range(1, k_max + 1)
        ]
        model += pl.lpSum(x_ind[i]) == 1

    c = []
    for i in range(size):
        for j in range(i + 1, size):
            if G.has_edge(i, j):
                b = pl.LpVariable(f"b_{i}_{j}", None, None, pl.LpBinary)
                for k in range(k_max):
                    model += x_ind[i][k] + x_ind[j][k] - 1 <= b
                c += [G.edges[i, j]["weight"] * b]
    k = pl.LpVariable("k", 1, k_max, pl.LpInteger)
    for k_val in range(1, k_max + 1):
        for i in range(size):
            model += k >= k_val * x_ind[i][k_val - 1]
    k_ind = [
        pl.LpVariable(f"k_ind_{i}", None, None, pl.LpBinary)
        for i in range(1, k_max + 1)
    ]
    model += pl.lpSum(k_ind) == 1
    model += k == pl.lpSum(k_ind[i] * (i + 1) for i in range(k_max))
    t = pl.LpVariable("t", None, 100 * np.e ** (0.5 * (k_max + 1)), pl.LpContinuous)
    model += t == pl.lpSum(
        k_ind[i] * math.floor(100 * np.e ** (0.5 * (i + 1))) for i in range(k_max)
    )

    k_inv = pl.LpVariable("k_inv", None, None, pl.LpContinuous)
    model += k_inv == pl.lpSum(k_ind[i] * (1 / (i + 1)) for i in range(k_max))

    norm_terms = [0] * k_max
    for i in range(1, k_max + 1):
        # p: number of penguins on team i
        p = pl.LpVariable(f"p_{i}", 0, size, pl.LpInteger)
        model += p == pl.lpSum(x_ind[j][i - 1] for j in range(size))

        used_ind = pl.LpVariable(f"used_ind_{i}", None, None, pl.LpBinary)
        model += i <= k + (k_max + 1) * used_ind
        model += i >= k + 0.001 - (k_max + 1) * (1 - used_ind)

        norm_term_unsq = pl.LpVariable(f"norm_term_unsq_{i}", -0.5, 1, pl.LpContinuous)
        actual_term = 1 / size * p - k_inv
        model += norm_term_unsq >= actual_term - k_max * used_ind
        model += norm_term_unsq <= actual_term + k_max * used_ind
        model += norm_term_unsq >= -k_max * (1 - used_ind)
        model += norm_term_unsq <= actual_term + k_max * (1 + used_ind)
        norm_terms[i - 1] = pl.LpVariable(f"norm_term_{i}", -0.5, 1, pl.LpContinuous)

    norm_sum = pl.LpVariable("norm_sum", 0, None, pl.LpContinuous)
    model += norm_sum == pl.lpSum(norm_terms)
    model += norm_sum <= norm_sum_max
    norm_sum_sqrt = pl.LpVariable("norm_sum_sqrt", None, None, pl.LpContinuous)

    exp_input = pl.LpVariable(
        "exp_input", 0, 70 * math.sqrt(norm_sum_max), pl.LpContinuous
    )
    model += exp_input == 70 * norm_sum_sqrt
    distribution = pl.LpVariable("distribution", None, None, pl.LpContinuous)

    model += pl.lpSum(c) + t + distribution

    solver.buildSolverModel(model)

    for i in range(1, k_max + 1):
        unsq = model.solverModel.getVarByName(f"norm_term_unsq_{i}")
        sq = model.solverModel.getVarByName(f"norm_term_{i}")
        model.solverModel.addQConstr(sq >= unsq * unsq)

        norm_sum = model.solverModel.getVarByName("norm_sum")
        norm_sum_sqrt = model.solverModel.getVarByName("norm_sum_sqrt")
        model.solverModel.addGenConstrPow(
            norm_sum,
            norm_sum_sqrt,
            0.5,
            "sqrt",
            "FuncPieces=-1 FuncPieceError=0.001",
        )

        exp_input = model.solverModel.getVarByName("exp_input")
        distribution = model.solverModel.getVarByName("distribution")
        model.solverModel.addGenConstrExp(
            exp_input,
            distribution,
            "FuncPieces=-1 FuncPieceError=0.1",
        )

        # model.solverModel.Params.FeasibilityTol = 1e-9

    if os.path.isfile(f"solutions/{name}.mst"):
        with open(f"solutions/{name}.mst", "r") as f:
            for line in f:
                s = line.split()
                var = model.solverModel.getVarByName(s[0])
                if var:
                    var.start = float(s[1])
                else:
                    log("Variable not found:", s[0])

    model.solverModel.update()
    solver.callSolver(model, sol_callback)
    log("FINISHED SOLVING:", name)


solve("large61")
