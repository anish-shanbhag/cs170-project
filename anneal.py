import json
import math
import random

import networkx as nx
import numpy as np
from simanneal import Annealer
from termcolor import colored

from helper import get_hyperparameters, is_new_best, write_vars_from_graph
from starter import read_input, score, write_output


def log(*args):
    print(*[colored(arg, "blue") for arg in args])


class PenguinProblem(Annealer):
    def __init__(self, name: str, k_max: int):
        self.k_max = k_max

        self.state: nx.Graph = read_input(f"inputs/{name}.in")
        self.size = len(self.state.nodes)
        for i in range(self.size):
            self.state.nodes[i]["team"] = (i % (self.k_max)) + 1
        log("SOLVING:", name)
        log("K_MAX:", self.k_max)
        log("EDGE COUNT:", self.state.number_of_edges())

    def move(self):
        i = random.randint(0, self.size - 1)
        # j = random.randint(0, self.size - 1)
        # self.state.nodes[i]["team"], self.state.nodes[j]["team"] = (
        #     self.state.nodes[j]["team"],
        #     self.state.nodes[i]["team"],
        # )
        self.state.nodes[i]["team"] = random.randint(1, self.k_max)

    def energy(self):
        return score(self.state)


def anneal(name: str):
    actual_k_max, *_ = get_hyperparameters(name)
    for k_max in range(max(2, actual_k_max - 4), actual_k_max + 1):
        problem = PenguinProblem(name, k_max)
        problem.Tmax = 33000
        problem.Tmin = 5.5
        problem.steps = 20000
        problem.updates = problem.steps // 300

        G = problem.anneal()[0]
        if is_new_best(name, G):
            write_output(G, f"outputs/{name}.out", True)
            write_vars_from_graph(name, G)
            log(name, "NEW BEST SCORE:", score(G))
        else:
            log(f"SKIPPING OUTPUT UPDATE for {name}:", score(G))


i = 3
for i in list(range(260))[i * 18 : (i + 1) * 18]:
    anneal("small" + str(i + 1))
    anneal("medium" + str(i + 1))
    anneal("large" + str(i + 1))
