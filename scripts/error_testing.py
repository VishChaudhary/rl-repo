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
import matplotlib.pyplot as plt

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
rho_ref = np.array([
    [1, 0],
    [0, 0]
])


def density_matrix_to_bloch(rho):
    """
    Converts a 2x2 density matrix into a Bloch vector.

    Parameters:
        rho (ndarray): 2x2 density matrix.

    Returns:
        list: Bloch vector [x, y, z].
    """
    rho_prime = rho @ rho_ref @ rho.conj().T
    # Compute the Bloch vector components
    x = np.real(np.trace(rho @ sigma_x))
    y = np.real(np.trace(rho @ sigma_y))
    z = np.real(np.trace(rho @ sigma_z))
    ans = np.array([x, y, z])
    return ans/np.linalg.norm(ans)


def plot_bloch_sphere(bloch_vectors):
    """ Helper function to plot vectors on a sphere."""
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(30, 45)

    # Draw the axes (source: https://github.com/matplotlib/matplotlib/issues/13575)
    x, y, z = np.array([[-1.5,0,0], [0,-1.5,0], [0,0,-1.5]])
    u, v, w = np.array([[3,0,0], [0,3,0], [0,0,3]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.05, color="black", linewidth=0.5)

    ax.text(0, 0, 1.7, r"|0⟩", color="black", fontsize=16)
    ax.text(0, 0, -1.9, r"|1⟩", color="black", fontsize=16)
    ax.text(1.9, 0, 0, r"|+⟩", color="black", fontsize=16)
    ax.text(-1.7, 0, 0, r"|–⟩", color="black", fontsize=16)
    ax.text(0, 1.7, 0, r"|i+⟩", color="black", fontsize=16)
    ax.text(0,-1.9, 0, r"|i–⟩", color="black", fontsize=16)

    ax.scatter(
        bloch_vectors[:,0], bloch_vectors[:,1], bloch_vectors[:, 2], c='#e29d9e', alpha=0.3
    )
    plt.show()

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
#     matrix_str = matrix_str.replace('\n', '')
#     # Step 2: Add commas between rows (fixing formatting)
#     matrix_str = matrix_str.replace('] [', '], [')
#     return matrix_str


if __name__ == "__main__":

    df = pd.read_csv("/Users/vishchaudhary/rl-repo/results/2025-01-10_13-11-06/" + "env_data.csv", header=0)
    # print(df.head())
    self_u = df.iloc[:, 4].apply(preprocess_matrix_string)
    # print(self_u[:1])
    self_u_list = [np.array(eval(m)) for m in self_u]
    self_u_list = np.array(self_u_list)
    # self_u_list = [np.array(for m in self_u)]
    # self_u = self_u.apply(lambda x: np.array(ast.literal_eval(x)))
    # self_u_list = [matrix for matrix in self_u]
    # self_u_list = self_u.tolist()
    # print(self_u_list[0])
    print(type(self_u_list))
    print(type(self_u_list[0]))
    print(self_u_list[0].shape)
    for i in range(6):
        print(f'\n\n{self_u_list[i]}\n\n')



