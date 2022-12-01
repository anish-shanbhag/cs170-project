import math
import os

import networkx as nx
import numpy as np
import pulp as pl
from gurobipy import GRB
from termcolor import colored

from constants import SQ_INTERVALS, SQRT_INTERVALS
from helper import get_hyperparameters, is_new_best
from starter import read_input, score, write_output


def log(*args):
    print(*[colored(arg, "blue") for arg in args])


def add_fn_approximation(
    model, ind_name, result_name, input_value, intervals, lb, rb, fn
):
    input_ind = [
        pl.LpVariable(f"{ind_name}_{j}", None, None, pl.LpBinary)
        for j in range(1, intervals + 2)
    ]
    model += pl.lpSum(input_ind) == 1
    input_range = rb - lb
    input_ind_sum = pl.lpSum(
        input_ind[j] * (lb + input_range * j / intervals) for j in range(intervals + 1)
    )

    model += input_ind_sum - input_value <= input_range / intervals
    model += input_value - input_ind_sum <= input_range / intervals

    result = pl.LpVariable(result_name, None, None, pl.LpContinuous)
    model += result == pl.lpSum(
        input_ind[j] * fn(lb + input_range * j / intervals)
        for j in range(intervals + 1)
    )

    return result


def solve(name: str):
    k_max, norm_sum_max, exp_intervals, best_score = get_hyperparameters(name)
    G: nx.Graph = read_input(f"inputs/{name}.in")
    log("SOLVING:", name)
    log("K_MAX:", k_max)
    log("NORM_SUM_MAX:", norm_sum_max)
    log("EDGE COUNT:", G.number_of_edges())
    size = len(G.nodes)

    model = pl.LpProblem("CS 170 Solver", pl.LpMinimize)
    solver = pl.GUROBI(Threads=16, warmStart=True)

    def sol_callback(model, where):
        get_var = lambda name: model.cbGetSolution(model.getVarByName(name))
        if where == GRB.Callback.MIPSOL:
            for i in range(size):
                team = int(get_var(f"x_{i}"))
                G.nodes[i]["team"] = team
            log(f"{name} CURRENT SCORE:", score(G), "from", score(G, True))
            log("k =", get_var("k"))
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
            if is_new_best(name, G):
                write_output(G, f"outputs/{name}.out", True)
                with open(f"solutions/{name}.mst", "w") as f:
                    vars = model.getVars()
                    sol = model.cbGetSolution(vars)
                    for var, val in zip(vars, sol):
                        f.write(f"{var.VarName} {val}\n")
            else:
                log("SKIPPING OUTPUT UPDATE")

    x = [pl.LpVariable(f"x_{i}", 1, k_max, pl.LpInteger) for i in range(size)]
    c = []
    for i in range(size):
        for j in range(i + 1, size):
            if G.has_edge(i, j):
                weight_max = 1000
                b = pl.LpVariable(f"b_{i}_{j}", None, None, pl.LpBinary)
                d = pl.LpVariable(f"d_{i}_{j}", 0, k_max, pl.LpContinuous)
                d1 = x[i] - x[j]
                d2 = x[j] - x[i]
                model += d >= d1
                model += d >= d2
                model += d <= d1 + 2 * k_max * b
                model += d <= d2 + 2 * k_max * (1 - b)

                # https://or.stackexchange.com/questions/1160/how-to-linearize-min-function-as-a-constraint
                # https://math.stackexchange.com/questions/2446606/linear-programming-set-a-variable-the-max-between-two-another-variables
                weight = G.edges[i, j]["weight"]
                w = pl.LpVariable(f"w_{i}_{j}", 0, weight, pl.LpContinuous)
                model += w >= weight - weight_max * d
                c += [w]
    k = pl.LpVariable("k", 1, k_max, pl.LpInteger)
    for i in range(size):
        model += k >= x[i]
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

    # x_ind: penguin i is assigned to team j
    x_ind = [0] * size
    for i in range(size):
        x_ind[i] = [
            pl.LpVariable(f"x_ind_{i}_{j}", None, None, pl.LpBinary)
            for j in range(1, k_max + 1)
        ]
        model += pl.lpSum(x_ind[i]) == 1
        model += pl.lpSum(x_ind[i][j] * (j + 1) for j in range(k_max)) == x[i]

    k_inv = pl.LpVariable("k_inv", None, None, pl.LpContinuous)
    model += k_inv == pl.lpSum(k_ind[i] * (1 / (i + 1)) for i in range(k_max))
    k_inv_sq = pl.LpVariable("k_inv_sq", None, None, pl.LpContinuous)
    model += k_inv_sq == pl.lpSum(k_ind[i] * (1 / (i + 1) ** 2) for i in range(k_max))

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

        norm_terms[i - 1] = add_fn_approximation(
            model,
            f"norm_term_unsq_ind_{i}",
            f"norm_term_{i}",
            norm_term_unsq,
            SQ_INTERVALS,
            -0.5,
            1,
            lambda x: x**2,
        )

    norm_sum = pl.lpSum(norm_terms)
    model += norm_sum <= norm_sum_max
    norm_sum_sqrt = add_fn_approximation(
        model,
        "norm_sum_ind",
        "norm_sum_sqrt",
        norm_sum,
        SQRT_INTERVALS,
        0,
        norm_sum_max,
        lambda x: math.sqrt(x),
    )

    exp_input = pl.LpVariable("exp_input", 0, 14, pl.LpContinuous)
    model += exp_input == 70 * norm_sum_sqrt
    distribution = add_fn_approximation(
        model,
        "exp_input_ind",
        "distribution",
        exp_input,
        exp_intervals,
        0,
        70 * math.sqrt(norm_sum_max),
        lambda x: np.e**x,
    )

    objective = pl.LpVariable("objective", 0, None, pl.LpContinuous)
    model += objective == pl.lpSum(c) + t + distribution
    model += objective

    vars = {}
    for variable in model.variables():
        s = str(variable).split("_")
        b = []
        for part in s:
            if not part.isdigit():
                b.append(part)
        s = "_".join(b)
        if s not in vars:
            vars[s] = 0
        vars[s] += 1
    print([v for v in vars.items() if v[1] > 5])

    solver.buildSolverModel(model)
    if os.path.isfile(f"solutions/{name}.mst"):
        vars = {v.name: v for v in model.variables()}
        with open(f"solutions/{name}.mst", "r") as f:
            for line in f:
                s = line.split()
                vars[s[0]].setInitialValue(float(s[1]))

    solver.actualSolve(model, sol_callback)
    log("FINISHED SOLVING:", name)


solve("small236")
