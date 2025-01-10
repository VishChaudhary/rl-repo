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

if __name__ == "__main__":

    # s_gate_name = gates.S()
    # s_gate = s_gate_name.get_matrix()
    # s_dagger = s_gate.conjugate().transpose()
    # unitary_s = s_dagger @ s_gate
    #
    # x_pi_4_name = gates.X_pi_4()
    # x_pi_4_gate = x_pi_4_name.get_matrix()
    # x_pi_4_dagger = x_pi_4_gate.conjugate().transpose()
    # unitary_x_pi_4 = x_pi_4_dagger @ x_pi_4_gate
    num_samples = 2000
    random_gate_name = gates.RandomSU2()
    print(random_gate_name.get_matrix())
    haar_samples = [random_gate_name.get_matrix() for _ in range(num_samples)]
    haar_bloch_vectors = np.array([density_matrix_to_bloch(s) for s in haar_samples])

    plot_bloch_sphere(haar_bloch_vectors)
    # print(f'Alleged S Gate:\n{s_gate}\n')
    # print(f'Is Alleged S Gate Unitary:\n{unitary_s}\n')
    # #
    # print(f'Alleged X_pi_4 Gate:\n{x_pi_4_gate}\n')
    # print(f'Is alleged X_pi_4 gate unitary:\n{unitary_x_pi_4}\n')
    # for _ in range(3):
    #     random_gate = random_gate_name.get_matrix()
    #     random_dagger = random_gate.conjugate().transpose()
    #     unitary_random_dagger = random_dagger @ random_gate
    #     print(f'RandomSU2 Gate:\n{random_gate}\n')
    #     print(f'Is RandomSU2 Gate Unitary:\n{unitary_random_dagger}\n')



