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

# Function to sample numbers evenly from 0 to x
def sample_even_distribution(n, x):
    return np.random.choice(np.arange(0, x + 1), size=n, replace=True)

def main():
    # # Set x and number of samples
    # x = 5  # Example value for x, can be changed
    # n_samples = 50000  # Number of samples to take
    #
    # # Generate samples
    # samples = sample_even_distribution(n_samples, x)
    #
    # # Count occurrences of each sampled number
    # unique_values, counts = np.unique(samples, return_counts=True)
    #
    # # Plot bar chart
    # plt.figure(figsize=(8, 5))
    # plt.bar(unique_values, counts, color='skyblue', edgecolor='black')
    #
    # # Label bars with their count
    # for i, count in zip(unique_values, counts):
    #     plt.text(i, count + 5, str(count), ha='center', fontsize=10)
    #
    # # Formatting the plot
    # plt.xlabel("Sampled Value")
    # plt.ylabel("Count")
    # plt.title(f"Count of {n_samples} Random Samples (0 to {x})")
    # plt.xticks(range(0, x + 1))
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    #
    # # Show the plot
    # plt.show()



    for _ in range(10):
        xy_gate = gates.XY_combination()
        xy_mat = xy_gate.get_matrix()
        xy_mat_conjT = xy_mat.conj().T

        unit1 = np.dot(xy_mat, xy_mat_conjT)
        unit2 = np.dot(xy_mat_conjT, xy_mat)

        print(f'\n{xy_mat}\n')



if __name__ == "__main__":
    main()



