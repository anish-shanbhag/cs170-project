import random

import networkx as nx
import numpy as np
from simanneal import Annealer
from termcolor import colored
from dataclasses import dataclass
from helper import get_hyperparameters, is_new_best, write_vars_from_graph
from starter import read_input, score, write_output


def log(*args):
    print(*[colored(arg, "blue") for arg in args])


@dataclass
class PenguinState:
    # list of team nums
    x: np.ndarray
    # list of team counts
    p: np.ndarray
    # sets of penguins on each team
    s: dict  # : dict[int, set[int]]
    # number of teams
    k: int
    # distribution score component
    d: float
    # total score
    score: float


class PenguinProblem(Annealer):
    def __init__(self, name: str, k_max: int):
        G: nx.Graph = read_input(f"inputs/{name}.in")
        self.size = len(G.nodes)
        self.k_max = k_max
        x = np.array([(i % (self.k_max)) + 1 for i in range(self.size)])
        p = np.zeros(self.k_max)
        s = {i: set() for i in range(1, self.k_max + 1)}
        for i in range(self.size):
            p[x[i] - 1] += 1
            s[x[i]].add(i)
            G.nodes[i]["team"] = int(x[i])

        self.state = PenguinState(
            x=x,
            p=p,
            s=s,
            k=self.k_max,
            d=score(G, True)[2],
            score=score(G),
            # G=G,
            # steps=0
        )
        self.G = G

        log("SOLVING:", name)
        log("K_MAX:", self.k_max)
        log("EDGE COUNT:", self.G.number_of_edges())

    def move(self):
        i = random.randint(0, self.size - 1)
        delta = 0
        # old_k = self.state.k

        b = self.state.p / self.size - 1 / self.state.k

        self.state.p[self.state.x[i] - 1] -= 1

        if self.state.p[self.state.x[i] - 1] == 0:
            while self.state.p[self.state.k - 1] == 0:
                self.state.k -= 1

        self.state.s[self.state.x[i]].remove(i)
        for j in self.state.s[self.state.x[i]]:
            if self.G.has_edge(i, j):
                delta -= self.G.edges[i, j]["weight"]

        old = self.state.x[i]
        while self.state.x[i] == old:
            self.state.x[i] = random.randint(1, self.k_max)
        new = self.state.x[i]
        self.state.p[self.state.x[i] - 1] += 1

        # if self.state.x[i] > self.state.k:
        #     self.state.k = self.state.x[i]

        self.state.s[self.state.x[i]].add(i)

        for j in self.state.s[self.state.x[i]]:
            if self.G.has_edge(i, j):
                delta += self.G.edges[i, j]["weight"]

        # if self.state.k != old_k:
        #     delta -= 100 * np.e ** (0.5 * old_k)
        #     delta += 100 * np.e ** (0.5 * self.state.k)

        delta -= self.state.d
        self.state.d = np.e ** (
            70
            * np.sqrt(
                np.inner(b, b)
                - b[old - 1] ** 2
                - b[new - 1] ** 2
                + (b[old - 1] - 1 / self.size) ** 2
                + (b[new - 1] + 1 / self.size) ** 2
            )
        )
        delta += self.state.d

        # self.G.nodes[i]["team"] = self.state.x[i]

        self.state.score += delta

        return delta

    def energy(self):
        return self.state.score


def anneal(name: str):
    actual_k_max, *_ = get_hyperparameters(name)
    for k_max in [6]:  # range(max(2, actual_k_max - 4), actual_k_max + 1):
        problem = PenguinProblem(name, k_max)
        problem.Tmax = 33000
        problem.Tmin = 5.5
        problem.steps = 50000
        problem.updates = problem.steps // 1000

        state = problem.anneal()[0]
        for i in range(len(state.x)):
            problem.G.nodes[i]["team"] = int(state.x[i])
        if is_new_best(name, problem.G):
            write_output(problem.G, f"outputs/{name}.out", True)
            write_vars_from_graph(name, problem.G)
            log(name, "NEW BEST SCORE:", score(problem.G))
        else:
            log(f"SKIPPING OUTPUT UPDATE for {name}:", score(problem.G))


i = 6
# for i in list(range(130, 260))[i * 18 : (i + 1) * 18]:
#     anneal("small" + str(i + 1))
#     anneal("medium" + str(i + 1))
#     anneal("large" + str(i + 1))

anneal("large1")
