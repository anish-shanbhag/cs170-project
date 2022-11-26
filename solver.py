import json
import math

import networkx as nx
import numpy as np
import pulp as pl
from gurobipy import GRB

from starter import read_input, score, write_output

SQ_INTERVALS = 500
SQRT_INTERVALS = 300
EXP_INTERVAL_SCALE = 10


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
    with open("scores.json") as f:
        scores = json.load(f)
        k_max = math.ceil(2 * np.log(scores[name] / 100))
        norm_sum_max = (np.log(scores[name]) / 70) ** 2 + 0.01
        exp_intervals = math.ceil(EXP_INTERVAL_SCALE * 70 * math.sqrt(norm_sum_max))

    G: nx.Graph = read_input(f"inputs/{name}.in")
    print("SOLVING:", name)
    print("K_MAX:", k_max)
    print("NORM_SUM_MAX:", norm_sum_max)
    print("EDGE COUNT", G.number_of_edges())
    # TODO implement this function with your solver
    # Assign a team to v with G.nodes[v]['team'] = team_id
    # Access the team of v with team_id = G.nodes[v]['team']
    size = len(G.nodes)
    model = pl.LpProblem("CS 170 Solver", pl.LpMinimize)

    def sol_callback(model, where):
        get_var = lambda name: model.cbGetSolution(model.getVarByName(name))
        if where == GRB.Callback.MIPSOL:
            for i in range(100):
                team = int(get_var(f"x_{i}"))
                G.nodes[i]["team"] = team
            print("CURRENT SCORE:", score(G), "from", score(G, True))
            print("k =", get_var("k"))
            # print("k_ind =", [get_var(f"k_ind_{i}") for i in range(1, k_max + 1)])
            # print("t =", get_var("t"))
            # document.querySelector("td:nth-child(3)")
            # print("k_inv =", get_var("k_inv"))
            # print("k_inv_sq =", get_var("k_inv_sq"))
            print("p =", [get_var(f"p_{i}") for i in range(1, k_max + 1)])
            # print("used_ind =", [get_var(f"used_ind_{i}") for i in range(1, k_max + 1)])
            print(
                "norm_term_unsq =",
                [get_var(f"norm_term_unsq_{i}") for i in range(1, k_max + 1)],
            )
            print(
                "norm_term =",
                [get_var(f"norm_term_{i}") for i in range(1, k_max + 1)],
            )
            print("norm_sum_sqrt =", get_var("norm_sum_sqrt"))
            print("exp_input =", get_var("exp_input"))
            # print(
            #     "exp_input_ind =",
            #     [get_var(f"exp_input_ind_{i}") for i in range(1, exp_intervals + 2)],
            # )
            print("distribution =", get_var("distribution"))
            write_output(G, f"outputs/{name}.out", True)

    x = [pl.LpVariable(f"x_{i}", None, None, pl.LpInteger) for i in range(size)]
    c = []
    for i in range(size):
        for j in range(i + 1, size):
            if G.has_edge(i, j):
                d = pl.LpVariable(f"d_{i}_{j}", None, None, pl.LpInteger)
                d_ind = pl.LpVariable(f"d_ind_{i}_{j}", None, None, pl.LpBinary)
                diff1 = x[i] - x[j]
                diff2 = x[j] - x[i]
                model += d >= diff1
                model += d >= diff2
                model += d <= diff1 + k_max * d_ind
                model += d <= diff2 + k_max * (1 - d_ind)
                # https://or.stackexchange.com/questions/1160/how-to-linearize-min-function-as-a-constraint
                y = pl.LpVariable(f"y_{i}_{j}", None, None, pl.LpBinary)
                model += d - 1 <= k_max * y
                model += 1 - d <= k_max * (1 - y)
                m = pl.LpVariable(f"m_{i}_{j}", None, None, pl.LpBinary)
                model += m <= d
                model += m >= 1 - k_max * (1 - y)
                model += m >= d - k_max * y
                w = pl.LpVariable(f"w_{i}_{j}", None, None, pl.LpInteger)
                weight = G.edges[i, j]["weight"]
                model += w == weight * (1 - m)
                c += [w]
    k = pl.LpVariable("k", None, None, pl.LpInteger)
    for i in range(size):
        model += k >= x[i]
    model += k <= k_max
    k_ind = [
        pl.LpVariable(f"k_ind_{i}", None, None, pl.LpBinary)
        for i in range(1, k_max + 1)
    ]
    model += pl.lpSum(k_ind) == 1
    model += k == pl.lpSum(k_ind[i] * (i + 1) for i in range(k_max))
    t = pl.LpVariable("t", None, None, pl.LpInteger)
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
        p = pl.LpVariable(f"p_{i}", None, None, pl.LpInteger)
        model += p == pl.lpSum(x_ind[j][i - 1] for j in range(size))

        used_ind = pl.LpVariable(f"used_ind_{i}", None, None, pl.LpBinary)
        model += i <= k + (k_max + 1) * used_ind
        model += i >= k + 0.001 - (k_max + 1) * (1 - used_ind)
        norm_term_unsq = pl.LpVariable(
            f"norm_term_unsq_{i}", None, None, pl.LpContinuous
        )

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

    exp_input = pl.LpVariable("exp_input", None, None, pl.LpContinuous)
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

    objective = pl.lpSum(c) + t + distribution
    model += objective

    vars = {}
    for variable in model.variables():
        s = str(variable).split("_")
        b = []
        for c in s:
            if not c.isdigit():
                b.append(c)
        s = "_".join(b)
        if s not in vars:
            vars[s] = 0
        vars[s] += 1
    print([v for v in vars.items() if v[1] > 5])

    solver = pl.GUROBI()
    solver.actualSolve(model, sol_callback)
    print([pl.value(c[i]) for i in range(len(c)) if pl.value(c[i]) > 0])
    print(pl.value(t))
    print([pl.value(x[i]) for i in range(size)])
    # print([v for v in model.solverModel.getVars() if v.varName.startswith("d_")])


solve("small123")
