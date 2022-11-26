import networkx as nx
import pulp as pl

from starter import read_input


def solve(G: nx.Graph):
    # TODO implement this function with your solver
    # Assign a team to v with G.nodes[v]['team'] = team_id
    # Access the team of v with team_id = G.nodes[v]['team']
    size = len(G.nodes)
    model = pl.LpProblem("CS 170 Solver", pl.LpMinimize)
    objective = None
    x = [pl.LpVariable(f"x_{i}", 1, size, pl.LpInteger) for i in range(1, size + 1)]
    c = []
    for i in range(len(G.nodes)):
        for j in range(i + 1, len(G.nodes)):
            if G.has_edge(i, j):
                d = pl.LpVariable(f"d_{i}_{j}", None, None, pl.LpInteger)
                model += d >= x[i] - x[j]
                model += d >= x[j] - x[i]
                m = pl.LpVariable(f"m_{i}_{j}", None, None, pl.LpBinary)
                model += m <= d
                w = pl.LpVariable(f"w_{i}_{j}", None, None, pl.LpInteger)
                weight = G.edges[i, j]["weight"]
                model += w == weight * (1 - m)
                c += [w]


G = read_input("ours/small.in")
solve(G)
