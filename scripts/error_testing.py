import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
# from relaqs.environments.noisy_single_qubit_env import NoisySingleQubitEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import relaqs.api.gates as gates
import numpy as np
import pandas as pd
from relaqs.api.callbacks import GateSynthesisCallbacks
import os
from datetime import datetime
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from qutip.operators import *
import ast
import copy
from relaqs.api.utils import *
import re
import torch
import matplotlib.pyplot as plt
from relaqs.api.utils import *

# Function to sample numbers evenly from 0 to x


def main():
    # a = gates.Ry(lower_bound=1.20, upper_bound=1.8)
    # b = gates.Ry(lower_bound=0.20, upper_bound=0.8)
    # c = gates.Rx(lower_bound=0.2, upper_bound=0.8)
    # d = gates.Rx(lower_bound=1.2, upper_bound=1.8)
    # e = gates.Rz(lower_bound=0.2, upper_bound=0.8)
    # f = gates.Rz(lower_bound=1.2, upper_bound=1.8)
    # print(g)
    # print(gates.H().get_matrix())
    # print(gates.S().get_matrix())
    # print(gates.HS().get_matrix())
    l = gates.RandomSU2(theta_range=(0,0.5))

    gate_list = []
    for _ in range(1000):
        # gate_list.append(a.get_matrix())
        # gate_list.append(b.get_matrix())
        # gate_list.append(c.get_matrix())
        # gate_list.append(d.get_matrix())
        gate_list.append(l.get_matrix())

    visualize_gates(gate_list)
    # print(gate_list[0])
    # print(check_unitary(gates.H().get_matrix()))




if __name__ == "__main__":
    main()



