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
from scipy.linalg import expm
from qutip import Bloch

# Function to sample numbers evenly from 0 to x


# def main():
#     # a = gates.Ry(lower_bound=1.20, upper_bound=1.8)
#     # b = gates.Ry(lower_bound=0.20, upper_bound=0.8)
#     # c = gates.Rx(lower_bound=0.2, upper_bound=0.8)
#     # d = gates.Rx(lower_bound=1.2, upper_bound=1.8)
#     # e = gates.Rz(lower_bound=0.2, upper_bound=0.8)
#     # f = gates.Rz(lower_bound=1.2, upper_bound=1.8)
#     # print(g)
#     # print(gates.H().get_matrix())
#     # print(gates.S().get_matrix())
#     # print(gates.HS().get_matrix())
#     l = gates.RandomSU2(theta_range=(0,0.5))
#
#     gate_list = []
#     for _ in range(1000):
#         # gate_list.append(a.get_matrix())
#         # gate_list.append(b.get_matrix())
#         # gate_list.append(c.get_matrix())
#         # gate_list.append(d.get_matrix())
#         gate_list.append(l.get_matrix())
#
#     visualize_gates(gate_list)
#     # print(gate_list[0])
#     # print(check_unitary(gates.H().get_matrix()))


def main():
    # Define Pauli Matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Parameters
    steps_per_Haar = 2
    final_time = 35.5556E-9  # Evolution time

    gamma_phase_max = 1.1675 * np.pi
    gamma_magnitude_max = 1.8 * np.pi / final_time / steps_per_Haar
    alpha_max = 0.05E9

    num_samples = 3
    actions = np.linspace(-1, 1, num_samples)
    # print(actions)

    # Storage
    bloch_vectors = []
    colors = []

    # Loop over all action combinations
    for gamma_magnitude_action in actions:
        for gamma_phase_action in actions:
            for alpha_action in actions:
                gamma_magnitude = gamma_magnitude_max * gamma_magnitude_action
                gamma_phase = gamma_phase_max * gamma_phase_action
                alpha = alpha_max * alpha_action

                # Construct Hamiltonian
                H = alpha * Z + gamma_magnitude * (np.cos(gamma_phase) * X + np.sin(gamma_phase) * Y)

                # Compute Unitary Evolution
                U = expm(-1j * H * final_time)

                # Apply U to initial state |0⟩
                # psi_0 = (1/np.sqrt(2))*np.array([[1], [1]], dtype=complex)
                psi_0 = np.array([[1], [0]], dtype=complex)

                psi_final = U @ psi_0

                # Compute Density Matrix
                rho = psi_final @ psi_final.conj().T

                # Compute Bloch Vector Components
                x = np.real(np.trace(rho @ X))
                y = np.real(np.trace(rho @ Y))
                z = np.real(np.trace(rho @ Z))

                bloch_vectors.append([x, y, z])
                colors.append(alpha_action)  # Color by α

    # Convert to NumPy array
    bloch_vectors = np.array(bloch_vectors)

    # Normalize colors
    color_min, color_max = min(colors), max(colors)
    colors = [(c - color_min) / (color_max - color_min) for c in colors]

    # Use plt.get_cmap to assign colormap
    cmap = plt.get_cmap("coolwarm")  # Load the colormap correctly
    rgb_colors = [cmap(c) for c in colors]  # Apply colormap to each normalized color value

    # Plot Bloch Sphere
    b = Bloch()
    b.point_color = rgb_colors  # Assign computed RGB colors
    b.add_points(bloch_vectors.T)
    b.show()




if __name__ == "__main__":
    main()



