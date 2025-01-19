import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.noisy_single_qubit_env import NoisySingleQubitEnv
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
import matplotlib.pyplot as plt



def preprocess_matrix_string(matrix_str):
    # Step 1: Remove newline characters
    matrix_str = matrix_str.replace('\n', '')
    # Step 2: Add commas where necessary after complex numbers
    matrix_str = matrix_str.replace('j ', 'j, ')
    # Step 3: Return the cleaned string
    matrix_str = matrix_str.replace('] [', '], [')
    return matrix_str

# def reprocess(matrix_str):
#     # Step 1: Remove newline characters
#     # matrix_str = matrix_str.replace('\n', '')
#     # Step 2: Add commas between rows (fixing formatting)
#     # matrix_str = matrix_str.replace('] [', '], [')
#     matrix_str = matrix_str.replace("  ", ",")
#     return matrix_str

def reprocess(matrix_str):
    matrix_str = matrix_str.strip()

    # Replace multiple spaces with a single comma
    matrix_str = re.sub(r"\s+", ",", matrix_str)

    # Ensure there are no extraneous commas (e.g., "[,0.4,...,]")
    matrix_str = matrix_str.replace("[,", "[").replace(",]", "]")
    return matrix_str


# def process_string_to_array(matrix_str):
#     try:
#         # Fix missing commas by adding a preprocessing step
#         clean_str = row.replace(" ", ",")
#         # Safely parse the string into a Python structure (list of lists)
#         parsed_data = ast.literal_eval(clean_str)
#         # Convert to a NumPy array
#         return np.array(parsed_data)
#     except Exception as e:
#         # print(f"Error processing row: {row}\n{e}")
#         return None  # Return None if there's an error



if __name__ == "__main__":

    df = pd.read_csv("/Users/vishchaudhary/rl-repo/results/2025-01-10_13-11-06/" + "env_data.csv", header=0)
    # print(df.head())
    fidelity = df.iloc[:, 0]
    print(type(fidelity[0]))
    fidelity = np.array(fidelity)
    print(fidelity.shape)
    actions = df.iloc[:, 2].apply(reprocess)

    # print(actions[0])
    # actions_list = np.array([np.array(eval(m)) for m in actions])
    # print(type(actions_list[0]))
    # print(actions_list[0])
    # print(actions_list.shape)
    # print(actions[0])
    # self_u = df.iloc[:, 4].apply(preprocess_matrix_string)
    # # print(self_u[:1])
    # self_u_list = [np.array(eval(m)) for m in self_u]
    # self_u_list = np.array(self_u_list)
    # # self_u_list = [np.array(for m in self_u)]
    # # self_u = self_u.apply(lambda x: np.array(ast.literal_eval(x)))
    # # self_u_list = [matrix for matrix in self_u]
    # # self_u_list = self_u.tolist()
    # # print(self_u_list[0])
    # print(self_u_list.shape)
    # print(type(self_u_list[0]))
    # print(self_u_list[0].shape)
    # for i in range(6):
    #     print(f'\n\n{self_u_list[i]}\n\n')



