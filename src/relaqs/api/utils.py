import ray
import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh
from scipy.linalg import sqrtm
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ddpg import DDPGConfig
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from relaqs import RESULTS_DIR
from matplotlib import rcParams
from relaqs.quantum_noise_data.get_data import (get_month_of_all_qubit_data, get_single_qubit_detuning)
from relaqs.api.callbacks import GateSynthesisCallbacks
from relaqs import QUANTUM_NOISE_DATA_DIR
from qutip.operators import *
from datetime import datetime
import re
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import qutip
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

vec = lambda X : X.reshape(-1, 1, order="F") # vectorization operation, column-order. X is a numpy array.
vec_inverse = lambda X : X.reshape(int(np.sqrt(X.shape[0])),
                                   int(np.sqrt(X.shape[0])),
                                   order="F") # inverse vectorization operation, column-order. X is a numpy array.

def get_time():
    return datetime.now()

def preprocess_actions(matrix_str):
    matrix_str = matrix_str.strip()

    # Replace multiple spaces with a single comma
    matrix_str = re.sub(r"\s+", ",", matrix_str)

    # Ensure there are no extraneous commas (e.g., "[,0.4,...,]")
    matrix_str = matrix_str.replace("[,", "[").replace(",]", "]")
    return matrix_str

def preprocess_matrix_string(matrix_str):
    # Step 1: Remove newline characters
    matrix_str = matrix_str.replace('\n', '')
    # Step 2: Add commas where necessary after complex numbers
    matrix_str = matrix_str.replace('j ', 'j, ')
    # Step 3: Return the cleaned string
    matrix_str = matrix_str.replace('] [', '], [')
    return matrix_str


def check_unitary(matrices):
    """
    Check if a single matrix or all matrices in a list are unitary.

    Parameters:
    matrices (np.ndarray or list of np.ndarray): A single matrix or a list of matrices.

    Returns:
    bool: True if the matrix is unitary or if all matrices in the list are unitary, False otherwise.
    """

    def is_unitary(matrix):
        """Helper function to check if a single matrix is unitary."""
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        if matrix.shape[0] != matrix.shape[1]:
            return False  # Must be square

        identity_mat = np.eye(matrix.shape[0], dtype=np.complex128)
        return np.allclose(matrix.conj().T @ matrix, identity_mat) and np.allclose(matrix @ matrix.conj().T, identity_mat)

    # If input is a single matrix, return its unitary check result
    if isinstance(matrices, np.ndarray):
        return is_unitary(matrices)

    # If input is a list, return True only if all matrices are unitary
    elif isinstance(matrices, list):
        return all(is_unitary(matrix) for matrix in matrices)

    else:
        raise TypeError("Input must be a numpy array or a list of numpy arrays.")


def visualize_gates(gates, save_dir = None):
    bloch_vectors = []
    for gate in gates:
        vector = np.matmul(gate, np.array([1, 0]))
        q_obj = qutip.Qobj(vector)
        bloch_vectors.append(q_obj)

    bloch_sphere = qutip.Bloch()
    bloch_sphere.add_states(bloch_vectors)

    # Rotate view to see the back of the Bloch sphere
    bloch_sphere.view = [-60, 30]

    # Prevent cropping by increasing frame transparency and scaling
    bloch_sphere.frame_alpha = 0.3  # Make the frame slightly transparent
    bloch_sphere.font_size = 10  # Adjust label size to prevent overlap
    bloch_sphere.scale = [1.0]  # Prevent zooming in

    # # Fix the figure size and aspect ratio
    # fig = plt.figure(figsize=(6, 6))  # Adjust figure size
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Set axis limits to prevent zoom-in/cropping
    # ax.set_xlim([-1.2, 1.2])
    # ax.set_ylim([-1.2, 1.2])
    # ax.set_zlim([-1.2, 1.2])
    #
    # # Render Bloch sphere
    # bloch_sphere.fig = fig
    # bloch_sphere.axes = ax
    if save_dir:
        filename = f'Visualisation of Gates'
        bloch_sphere.save(name = os.path.join(save_dir,filename), format = "png")
    else:
        bloch_sphere.show()


def create_self_U_textfile(save_filepath, inference_gate, calculated_target, target_gate):

    super_op_target_gate = (spre(Qobj(target_gate)) * spost(Qobj(target_gate))).data.toarray()
    U_target_dagger = np.array(super_op_target_gate.conjugate().transpose())
    highest_val_self_U = np.array(calculated_target[max(calculated_target.keys())])
    fidelity = float(np.abs(np.trace(U_target_dagger @ highest_val_self_U))) / (highest_val_self_U.shape[0])
    # print(f'Super_op_target_gate:\n{np.array(super_op_target_gate)}\n\n')
    # print(f'Final Self.U:\n{np.array(highest_val_self_U)}\n\n')
    # print(f'Fidelity: {fidelity}')

    # super_op_target_gate = (spre(Qobj(target_gate)) * spost(Qobj(target_gate))).data.toarray()
    file_name = save_filepath + f"{inference_gate}_self_U.txt"
    # U_target_dagger = super_op_target_gate.conjugate().transpose()
    #
    # fidelity = float(np.abs(np.trace(U_target_dagger @ calculated_gate))) / (calculated_gate.shape[0])

    # Write to file
    with open(file_name, "w") as f:
        f.write("self.U:\n")
        f.write(f"{highest_val_self_U}\n\n")
        f.write("self.U_target:\n")
        f.write(f"{super_op_target_gate}\n\n")
        f.write(f"Fidelity:\n")
        f.write(f"{fidelity}\n")

def actions_analysis(actions_array, final_fidelity_per_episode, save_dir):
    ##Write correlation
    # action_cols =

    # 1. Convert to a DataFrame for easier analysis
    action_cols = ["gamma_magnitude", "gamma_phase", "alpha"]

    data = pd.DataFrame({
        "gamma_magnitude": actions_array[:, 0],
        "gamma_phase": actions_array[:, 1],
        "alpha": actions_array[:, 2],
        "Fidelity": final_fidelity_per_episode
    })

    # 2. Correlation Analysis
    correlations = {}
    for col in action_cols:
        corr, p_value = pearsonr(data[col], data["Fidelity"])
        correlations[col] = {"Correlation": corr, "P-Value": p_value}

    # 3. Linear Regression Analysis
    X = actions_array
    y = final_fidelity_per_episode

    # Fit the regression model
    model = LinearRegression()
    model.fit(X, y)

    # Extract coefficients
    coefficients = model.coef_

    with open(save_dir + "actions_analysis.txt", "w") as f:
        f.write("Correlation Analysis:\n")
        f.write(pd.DataFrame(correlations).T.to_string())
        f.write("\n\nLinear Regression Coefficients:\n")
        for i, action in enumerate(action_cols):
            f.write(f"{action}: {coefficients[i]:.4f}\n")
        f.write("\nInsights:\n")
        f.write("1. Pearson Correlation Coefficient:\n")
        f.write("   - Measures the strength and direction of the relationship between actions and fidelity.\n")
        f.write(
            "   - Values close to +1 indicate a positive correlation; -1 indicates a negative correlation; 0 means no correlation.\n")
        f.write("   - A high absolute value suggests a strong impact of the action on fidelity.\n")

        f.write("\n2. P-Value (Statistical Significance):\n")
        f.write("   - Determines if the correlation is statistically significant or due to chance.\n")
        f.write(
            "   - A p-value < 0.05 indicates a meaningful relationship; >= 0.05 suggests it may be due to random variations.\n")
        f.write("   - A high correlation but large p-value means uncertainty in the relationship.\n\n")

        f.write("\n3. Linear Regression Coefficients:\n")
        f.write("   - Represents the magnitude and direction of influence of actions on fidelity.\n")
        f.write("   - A larger absolute coefficient means greater impact on fidelity.\n")
        f.write(
            "   - Positive values indicate increasing the action increases fidelity, negative values indicate the opposite.\n")

        f.write("\n4. Combined Interpretation:\n")
        f.write("   - High correlation and low p-value suggest a strong and reliable relationship.\n")
        f.write("   - High regression coefficients mean actions significantly affect fidelity.\n")
        f.write("   - Consider both correlation strength and statistical significance when making decisions.\n")

def plot_overall_alg_votes(alg_votes, save_dir):
    """Efficiently plots algorithm call counts with value labels on bars."""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(alg_votes.keys(), alg_votes.values(), color="skyblue")

    # Add value labels on top of the bars
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{bar.get_height()}", ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    plt.xlabel("Algorithm ID", fontsize=12)
    plt.ylabel("Number of Calls", fontsize=12)
    plt.title("Algorithm Call Counts", fontsize=14)
    plt.xticks(list(alg_votes.keys()))

    if save_dir:
        plt.savefig(os.path.join(save_dir, "overall_alg_votes.png"))
    else:
        plt.show()

def plot_alg_gate_counts(alg_gate_counts, save_path):
    """Plots and saves a separate bar chart for each algorithm's gate counts."""
    for alg_id, gate_counts in alg_gate_counts.items():
        if not gate_counts:  # Skip empty entries
            continue

        plt.figure(figsize=(6, 4))
        gates = [str(g) for g in gate_counts.keys()]
        counts = list(gate_counts.values())

        plt.bar(gates, counts, color="skyblue")
        plt.title(f"Algorithm {alg_id} Gate Counts", fontsize=14)
        plt.xlabel("Gate Name", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # Save each figure separately
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, f"alg_{alg_id}_gate_counts.png"))
            plt.close()  # Close the figure to free memory
        else:
            plt.show()

def save_alg_names_to_file(alg_names, save_dir):
    """Saves the algorithm names from a list into a text file."""
    filepath = os.path.join(save_dir, "alg_names.txt")
    with open(filepath, "w") as file:
        for i, name in enumerate(alg_names):
            file.write(f"Alg{i}: {name}\n")


def get_poor_results(original_date, fidelity_threshold=0.9, verbose=False):
    results_path = "/Users/vishchaudhary/rl-repo/results/" + original_date + "/RandomSU2/RandomSU2_env_data.csv"
    # results_path = "/Users/vishchaudhary/rl-repo/results/" + original_date + "/env_data.csv"
    results_df = pd.read_csv(results_path, header=0)
    results_above_threshold = results_df[results_df.iloc[:, 0] >= fidelity_threshold]
    results_below_threshold = results_df[results_df.iloc[:, 0] < fidelity_threshold]

    preProcessed_u_target_below = results_below_threshold.iloc[:, 4].apply(preprocess_matrix_string)
    u_target_list_below = [np.array(eval(m)) for m in preProcessed_u_target_below]
    u_target_list_below = np.array(u_target_list_below)

    preProcessed_u_target_above = results_above_threshold.iloc[:, 4].apply(preprocess_matrix_string)
    u_target_list_above = [np.array(eval(m)) for m in preProcessed_u_target_above]
    u_target_list_above = np.array(u_target_list_above)

    if verbose:
        print(f"Number of results below threshold: {len(u_target_list_below)}\n")
        print(f"Number of results above threshold: {len(u_target_list_above)}\n")

        print(f"Type of u_target_list_below: {type(u_target_list_below)}")
        print(f"Type of u_target_list_above: {type(u_target_list_above)}")

        print(f"Type of u_target: {type(u_target_list_below[0])}\n")
        print(f"Type of u_target: {type(u_target_list_above[0])}\n")

        print(f"Result below threshold: {u_target_list_below[1]}\n")
        print(f"Result above threshold: {u_target_list_above[1]}\n")

    return u_target_list_below, u_target_list_above

def plot_multiple_visuals(df, figure_title, save_dir, plot_filename, bin_step=0.1, inf_count=None,
                          fidelity_threshold=0.8, gate=None):
    fidelities = np.array(df.iloc[:, 0])
    rewards = np.array(df.iloc[:, 1])
    episode_ids = np.array(df.iloc[:, 5])

    # preProcessed_u_target = df.iloc[:, 4].apply(preprocess_matrix_string)
    # preProcessed_actions = df.iloc[:, 2].apply(preprocess_actions)
    #
    # u_target_list = [np.array(eval(m)) for m in preProcessed_u_target]
    # u_target_list = np.array(u_target_list)
    # actions_array = [np.array(eval(m)) for m in preProcessed_actions]
    u_target_list = df.iloc[:, 4]
    actions_array = np.array(df.iloc[:, 2].tolist())

    infidelity = []

    for i in range(len(episode_ids)):
        infidelity.append(1 - fidelities[i])

    infidelity = np.array(infidelity)

    rounding_precision = 6
    final_fidelity_per_episode = np.round(fidelities, rounding_precision)
    final_infelity_per_episode = np.round(infidelity, rounding_precision)
    sum_of_rewards_per_episode = np.round(rewards, rounding_precision)

    # -------------------------------> Plotting <-------------------------------------
    rcParams['font.family'] = 'serif'
    mpl.style.use('seaborn-v0_8')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(figure_title)
    fig.set_size_inches(10, 5)

    # ----> fidelity <----
    ax1.scatter(range(len(final_fidelity_per_episode)), final_fidelity_per_episode, color="b", s=10)
    ax1.set_title("Fidelity")
    ax1.set_title("a)", loc='left', fontsize='medium')
    ax1.set_xlabel("Instance")

    # ----> infidelity <----
    ax2.scatter(range(len(final_infelity_per_episode)), final_infelity_per_episode, color="r", s=10)
    ax2.set_yscale("log")
    ax2.set_title("1 - Fidelity (log scale)")
    ax2.set_title("b)", loc='left', fontsize='medium')
    ax2.set_xlabel("Instance")

    # ----> reward <----
    ax3.scatter(range(len(sum_of_rewards_per_episode)), sum_of_rewards_per_episode, color="g", s=10)
    ax3.set_title("Sum of Rewards")
    ax3.set_title("c)", loc='left', fontsize='medium')
    ax3.set_xlabel("Instance")

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir + plot_filename)
        plt.close(fig)
    else:
        plt.show()

    ##Plot distribution

    bin_step = 0.1
    # Define bins for the fidelities with a step of 0.1
    bins = np.arange(0.0, 1.1, bin_step)  # Bins from 0.0 to 1.0 with a step of 0.1
    count_title = f'[{gate}] Number of Occurrences vs Fidelity'
    # Create a histogram
    counts, bin_edges = np.histogram(final_fidelity_per_episode, bins=bins)

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], counts, width=0.1, align='edge', edgecolor='black')

    # Add labels and title
    plt.xlabel("Fidelity")
    plt.ylabel("Number of Occurrences")
    plt.title(count_title)
    plt.xticks(bins)  # Set the x-axis ticks to match the bins
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir + "occurrences.png")
    else:
        plt.show()

    ##Write correlation
    # action_cols = ["gamma_magnitude", "gamma_phase", "alpha"]
        # 1. Convert to a DataFrame for easier analysis
    # actions_analysis(actions_array, fidelity_threshold, save_dir)

    action_cols = ["gamma_magnitude", "gamma_phase", "alpha"]

    data = pd.DataFrame({
        "gamma_magnitude": np.array(actions_array[:, 0]),
        "gamma_phase": np.array(actions_array[:, 1]),
        "alpha": np.array(actions_array[:, 2]),
        "Fidelity": final_fidelity_per_episode
    })

    # 2. Correlation Analysis
    correlations = {}
    for col in action_cols:
        corr, p_value = pearsonr(data[col], data["Fidelity"])
        correlations[col] = {"Correlation": corr, "P-Value": p_value}

    # 3. Linear Regression Analysis
    X = actions_array
    y = final_fidelity_per_episode

    # Fit the regression model
    model = LinearRegression()
    model.fit(X, y)

    # Extract coefficients
    coefficients = model.coef_

    os.makedirs(save_dir, exist_ok=True)
    analysis_file_path = os.path.join(save_dir, "inference_actions_analysis.txt")

    with open(analysis_file_path, "w") as f:
        f.write("Correlation Analysis:\n")
        f.write(pd.DataFrame(correlations).T.to_string())
        f.write("\n\nLinear Regression Coefficients:\n")
        for i, action in enumerate(action_cols):
            f.write(f"{action}: {coefficients[i]:.4f}\n")
        f.write("\nInsights:\n")
        f.write("1. Pearson Correlation Coefficient:\n")
        f.write("   - Measures the strength and direction of the relationship between actions and fidelity.\n")
        f.write(
            "   - Values close to +1 indicate a positive correlation; -1 indicates a negative correlation; 0 means no correlation.\n")
        f.write("   - A high absolute value suggests a strong impact of the action on fidelity.\n")

        f.write("\n2. P-Value (Statistical Significance):\n")
        f.write("   - Determines if the correlation is statistically significant or due to chance.\n")
        f.write(
            "   - A p-value < 0.05 indicates a meaningful relationship; >= 0.05 suggests it may be due to random variations.\n")
        f.write("   - A high correlation but large p-value means uncertainty in the relationship.\n\n")

        f.write("\n3. Linear Regression Coefficients:\n")
        f.write("   - Represents the magnitude and direction of influence of actions on fidelity.\n")
        f.write("   - A larger absolute coefficient means greater impact on fidelity.\n")
        f.write(
            "   - Positive values indicate increasing the action increases fidelity, negative values indicate the opposite.\n")

        f.write("\n4. Combined Interpretation:\n")
        f.write("   - High correlation and low p-value suggest a strong and reliable relationship.\n")
        f.write("   - High regression coefficients mean actions significantly affect fidelity.\n")
        f.write("   - Consider both correlation strength and statistical significance when making decisions.\n")

    ##Plot bloch sphere for RandomSU2

    if inf_count < 3:
        # Initialize a dictionary to store final states categorized by fidelity range
        fidelity_bins = {round(i, 1): [] for i in np.arange(0.0, 1.0, 0.1)}

        # Categorize states based on their fidelity
        for unitary, fidelity in zip(u_target_list, final_fidelity_per_episode):
            bin_range = np.floor(fidelity * 10) / 10  # Simplified binning method
            fidelity_bins[bin_range].append(np.matmul(unitary, np.array([1, 0])))

        # Generate Bloch sphere visualizations for each fidelity range
        for bin_range, states in fidelity_bins.items():
            if states:  # Only create a Bloch sphere if there are states in the bin
                q_objs = [qutip.Qobj(state) for state in states]
                bloch_sphere = qutip.Bloch()
                bloch_sphere.add_states(q_objs)

                if save_dir:
                    filename = f"Bloch_fidelity_in_bin_{str(bin_range).replace('.', '_')}"
                    bloch_sphere.save(name=os.path.join(save_dir, filename), format="png")
                else:
                    bloch_sphere.show()


def network_config_creator(alg_config):

    network_config = {
        "actor_lr":  alg_config.actor_lr,
        "actor_hidden_activation": alg_config.actor_hidden_activation,
        "critic_hidden_activation": alg_config.critic_hidden_activation,
        "critc_lr": alg_config.critic_lr,
        "actor_num_hidden_layers": len(alg_config.actor_hiddens),
        "actor_num_hidden_neurons":  alg_config.actor_hiddens[0],
        "critc_num_hidden_layers": len(alg_config.critic_hiddens),
        "critc_num_hidden_neurons":  alg_config.critic_hiddens[0],
        "num_steps_sampled_before_learning_starts": alg_config.num_steps_sampled_before_learning_starts,
        "twin_q" : alg_config.twin_q
    }

    return network_config

def sac_config_table(env_config, alg_config, filepath, continue_training=False, original_training_date = None):
    filtered_env_config = {}
    filtered_alg_config = {}

    env_config_default = {
        "num_Haar_basis": 1,
        "steps_per_Haar": 2,
        "training": True,
        "retraining": False,
    }

    for key in env_config_default.keys():
        filtered_env_config[key] = env_config[key]

    env_data = {
        "Config Name": list(filtered_env_config.keys()),
        "Current Value": list(filtered_env_config.values()),
    }

    default_lr = {
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
    }

    default_q_model_config = {
        "fcnet_hiddens": "[256, 256]",
        "fcnet_activation": "relu",
    }
    default_policy_model_config = {
        "fcnet_hiddens": "[256, 256]",
        "fcnet_activation": "relu",
    }

    other_default_config = {
        "num_steps_sampled_before_learning_starts": 1500,
        "twin_q": True
    }

    # for key in default_q_model_config.keys():
    #     filtered_alg_config[key] = alg_config["q_model_config"][key]
    #
    # for key in default_policy_model_config.keys():
    #     filtered_alg_config[key] = alg_config["policy_model_config"][key]

    for key in other_default_config.keys():
        filtered_alg_config[key] = alg_config[key]

    for key in default_lr:
        filtered_alg_config[key] = alg_config["optimization"][key]

    alg_data = {
        "Config Name": list(filtered_alg_config.keys()),
        "Current Value": list(filtered_alg_config.values()),
    }

    env_df = pd.DataFrame(env_data)
    alg_df = pd.DataFrame(alg_data)

    with open(filepath + "sac_config_table.txt", "w") as f:
        # Write the table header with a border
        f.write("+------------------------------------------------+----------------------+\n")
        f.write("|                  Config Name                   |     Current Value    |\n")
        f.write("+------------------------------------------------+----------------------+\n")

        for index, row in env_df.iterrows():
            f.write(f"| {row['Config Name']: <46} | {row['Current Value']: <21} |\n")
        f.write("+------------------------------------------------+----------------------+\n")

        for index, row in alg_df.iterrows():
            f.write(f"| {row['Config Name']: <46} | {row['Current Value']: <21} |\n")
        f.write("+------------------------------------------------+----------------------+\n")

        f.write(f"Continuation from previous training: {continue_training}\n")
        if continue_training:
            f.write(f"Training continued from results on: {original_training_date}\n")








def config_table(env_config, alg_config, filepath, continue_training=False, original_training_date = None):
    filtered_env_config = {}
    filtered_explor_config = {}
    network_config = network_config_creator(alg_config)

    env_config_default = {
        "num_Haar_basis": 1,
        "steps_per_Haar": 2,
        "training": True,
        "retraining": False,
    }


    network_config_default = {
        "actor_lr": 1e-3,
        "actor_hidden_activation": "relu",
        "critic_hidden_activation": "relu",
        "critc_lr": 1e-3,
        "actor_num_hidden_layers": "2",
        "actor_num_hidden_neurons": "[400,300]",
        "critc_num_hidden_layers": "2",
        "critc_num_hidden_neurons": "[400,300]",
        "num_steps_sampled_before_learning_starts": 1500,
        "twin_q": False
    }

    explor_config_default = {
        "random_timesteps": 1000,
        "ou_base_scale": 0.1,
        "ou_theta": 0.15,
        "ou_sigma": 0.2,
        "initial_scale": 1.0,
        "scale_timesteps": 10000
    }

    for key in env_config_default.keys():
        filtered_env_config[key] = env_config[key]

    for key in explor_config_default.keys():
        filtered_explor_config[key] = alg_config.exploration_config[key]

    env_data = {
        "Config Name": list(filtered_env_config.keys()),
        "Current Value": list(filtered_env_config.values()),
        "Default Value": list(env_config_default.values()),
    }

    network_data = {
        "Config Name": list(network_config.keys()),
        "Current Value": list(network_config.values()),
        "Default Value": list(network_config_default.values()),
    }

    explor_data = {
        "Config Name": list(filtered_explor_config.keys()),
        "Current Value": list(filtered_explor_config.values()),
        "Default Value": list(explor_config_default.values()),
    }

    env_df = pd.DataFrame(env_data)
    network_df = pd.DataFrame(network_data)
    explor_df = pd.DataFrame(explor_data)

    with open(filepath + "ddpg_config_table.txt", "w") as f:
        # Write the table header with a border
        f.write("+------------------------------------------------+----------------------+--------------------+\n")
        f.write("|                  Config Name                   |     Current Value    |    Default Value   |\n")
        f.write("+------------------------------------------------+----------------------+--------------------+\n")

        for index, row in env_df.iterrows():
            f.write(f"| {row['Config Name']: <46} | {row['Current Value']: <21} | {row['Default Value']: <18} |\n")
        f.write("+------------------------------------------------+----------------------+--------------------+\n")

        for index, row in explor_df.iterrows():
            f.write(f"| {row['Config Name']: <46} | {row['Current Value']: <21} | {row['Default Value']: <18} |\n")
        f.write("+------------------------------------------------+----------------------+--------------------+\n")

        for index, row in network_df.iterrows():
            f.write(f"| {row['Config Name']: <46} | {row['Current Value']: <21} | {row['Default Value']: <18} |\n")
        f.write("+------------------------------------------------+----------------------+--------------------+\n")
        f.write(f"Continuation from previous training: {continue_training}\n")
        if continue_training:
            f.write(f"Training continued from results on: {original_training_date}\n")


def normalize(quantity, list_of_values):
    """ normalize quantity to [0, 1] range based on list of values """
    return (quantity - min(list_of_values) + 1E-15) / (max(list_of_values) - min(list_of_values) + 1E-15)

def polar_vec_to_complex_matrix(vec, return_flat=False):
    """ 
    The intended use of this function is to convert from the representation of the unitary
    in the agent's observation back to the unitary matrxi.

    Converts a vector of polar coordinates to a unitary matrix. 
    
    The vector is of the form: [r1, phi1, r2, phi2, ...]
    
    And the matrix is then: [-1 * r1 * exp(i * phi1 * 2pi),...] """
    # Convert polar coordinates to complex numbers
    complex_data = []
    for i in range(0, len(vec), 2):
        r = vec[i]
        phi = vec[i+1]
        z = -1 * r * np.exp(1j * phi * 2*np.pi) 
        complex_data.append(z)

    # Reshape into square matrix
    if not return_flat:
        matrix_dimension = int(np.sqrt(len(vec)))
        complex_data = np.array(complex_data).reshape((matrix_dimension, matrix_dimension))

    return complex_data

def superoperator_evolution(superop, dm):
    return vec_inverse(superop @ vec(dm))

def load_pickled_env_data(data_path):
    df = pd.read_pickle(data_path)
    return df

gate_fidelity = lambda U, V: float(np.abs(np.trace(U.conjugate().transpose() @ V))) / (U.shape[0])

def dm_fidelity(rho, sigma):
    assert np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).imag < 1e-8, f"Non-negligable imaginary component {np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).imag}"
    #return np.abs(np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))))**2
    return np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).real**2

def sample_noise_parameters(t1_t2_noise_file=None, detuning_noise_file=None):
    # ---------------------> Get quantum noise data <-------------------------
    if t1_t2_noise_file is None:
        t1_list = np.random.uniform(40e-6, 200e-6, 100)
        t2_list = np.random.uniform(40e-6, 200e-6, 100)
    else:
        t1_list, t2_list = get_month_of_all_qubit_data(QUANTUM_NOISE_DATA_DIR + t1_t2_noise_file) # in seconds

    if detuning_noise_file is None:
        mean = 0
        std = 1e4
        sample_size = 100
        samples = np.random.normal(mean, std, sample_size)
        detunings = samples.tolist()
    else:
        detunings = get_single_qubit_detuning(QUANTUM_NOISE_DATA_DIR + detuning_noise_file)

    return list(t1_list), list(t2_list), detunings

# def do_inferencing(alg, n_episodes_for_inferencing, quantum_noise_file_path):
#     """
#     alg: The trained model
#     n_episodes_for_inferencing: Number of episodes to do during the training
#     """
#
#     assert n_episodes_for_inferencing > 0
#     env = return_env_from_alg(alg)
#     obs, info = env.reset()
#     t1_list, t2_list, detuning_list = sample_noise_parameters(quantum_noise_file_path)
#     env.relaxation_rates_list = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()]
#     env.detuning_list = detuning_list
#     num_episodes = 0
#     episode_reward = 0.0
#     print("Inferencing is starting ....")
#     while num_episodes < n_episodes_for_inferencing:
#         print("episode : ", num_episodes)
#         # Compute an action (`a`).
#         a = alg.compute_single_action(
#             observation=obs,
#             policy_id="default_policy",  # <- default value
#         )
#         # Send the computed action `a` to the env.
#         obs, reward, done, truncated, _ = env.step(a)
#         episode_reward += reward
#         # Is the episode `done`? -> Reset.
#         if done:
#             print(f"Episode done: Total reward = {episode_reward}")
#             obs, info = env.reset()
#             num_episodes += 1
#             episode_reward = 0.0
#     return env, alg

def load_model(path):
    "path (str): Path to the file usually beginning with the word 'checkpoint' " 
    loaded_model = Algorithm.from_checkpoint(path)
    return loaded_model

def get_best_episode_information(filename):
    df = pd.read_csv(filename, names=['Fidelity', 'Reward', 'Actions', 'Flattened U', 'Episode Id'], header=0)
    fidelity = df.iloc[:, 0]
    max_fidelity_idx = fidelity.argmax()
    fidelity = df.iloc[max_fidelity_idx, 0]
    episode = df.iloc[max_fidelity_idx, 4]
    best_episode = df[df["Episode Id"] == episode]
    return best_episode

def get_best_actions(filename):
    best_episode = get_best_episode_information(filename)
    action_str_array = best_episode['Actions'].to_numpy()

    best_actions = []
    for actions_str in action_str_array:
        # Remove the brackets and split the string by spaces
        str_values = actions_str.strip('[]').split()

        # Convert the string values to float
        float_values = [float(value) for value in str_values]

        # Convert the list to a numpy array (optional)
        best_actions.append(float_values)
    return best_actions, best_episode['Fidelity'].to_numpy() 

# def run(env_class, gate, n_training_iterations=1, noise_file=""):
#     """Args
#        gate (Gate type):
#        n_training_iterations (int)
#        noise_file (str):
#     Returns
#       alg (rllib.algorithms.algorithm)
#
#     """
#     ray.init()
#     env_config = env_class.get_default_env_config()
#     env_config["U_target"] = gate.get_matrix()
#
#     # ---------------------> Get quantum noise data <-------------------------
#     t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)
#
#     env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()] # using real T1 data
#     env_config["detuning_list"] = detuning_list
#     env_config["relaxation_ops"] = [sigmam(),sigmaz()]
#     env_config["observation_space_size"] = 2*16 + 1 + 2 + 1 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning
#     env_config["verbose"] = True
#
#     # ---------------------> Configure algorithm and Environment <-------------------------
#     alg_config = DDPGConfig()
#     alg_config.framework("torch")
#     alg_config.environment(env_class, env_config=env_config)
#     alg_config.rollouts(batch_mode="complete_episodes")
#     alg_config.callbacks(GateSynthesisCallbacks)
#     alg_config.train_batch_size = env_class.get_default_env_config()["steps_per_Haar"]
#
#     ### working 1-3 sets
#     alg_config.actor_lr = 4e-5
#     alg_config.critic_lr = 5e-4
#
#     alg_config.actor_hidden_activation = "relu"
#     alg_config.critic_hidden_activation = "relu"
#     alg_config.num_steps_sampled_before_learning_starts = 1000
#     alg_config.actor_hiddens = [30,30,30]
#     alg_config.exploration_config["scale_timesteps"] = 10000
#
#     alg = alg_config.build()
#     list_of_results = []
#     for _ in range(n_training_iterations):
#         result = alg.train()
#         list_of_results.append(result['hist_stats'])
#
#     ray.shutdown()
#
#     return alg

def return_env_from_alg(alg):
    env = alg.workers.local_worker().env
    return env

def load_and_analyze_best_unitary(data_path, U_target):
    df = pd.read_csv(data_path, names=['Fidelity', 'Reward', 'Actions', 'Flattened U', 'Episode Id'], header=0)
    
    fidelity = df["Fidelity"]
    max_fidelity_idx = fidelity.argmax()
    best_flattened_unitary = eval(df.iloc[max_fidelity_idx, 3])

    best_fidelity_unitary = np.array([complex(x) for x in best_flattened_unitary]).reshape(4, 4)
    max_fidelity = fidelity.iloc[max_fidelity_idx]

    print("Max fidelity:", max_fidelity)
    print("Max unitary:", best_fidelity_unitary)

    zero = np.array([1, 0]).reshape(-1, 1)
    zero_dm = zero @ zero.T.conjugate()
    zero_dm_flat = zero_dm.reshape(-1, 1)

    dm = best_fidelity_unitary @ zero_dm_flat
    dm = dm.reshape(2, 2)
    print("Density Matrix:\n", dm)

    # Check trace = 1
    dm_diagonal = np.diagonal(dm)
    print("diagonal:", dm_diagonal)
    trace = sum(np.diagonal(dm))
    print("trace:", trace)

    # # Check that all eigenvalues are positive
    eigenvalues = eigvalsh(dm)
    print("eigenvalues:", eigenvalues)
    #assert (0 <= eigenvalues).all()

    psi = U_target.get_matrix() @ zero
    true_dm = psi @ psi.T.conjugate()
    print("true dm\n:", true_dm)

    print("Density matrix fidelity:", dm_fidelity(true_dm, dm))
