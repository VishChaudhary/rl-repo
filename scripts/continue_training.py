import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.noisy_single_qubit_env import NoisySingleQubitEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import relaqs.api.gates as gates
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from relaqs.api.callbacks import GateSynthesisCallbacks
import os
from datetime import datetime
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from qutip.operators import *
import copy
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import qutip
from relaqs.api.utils import *


def boosted_retraining(training_alg, n_training_episodes):

    ray.init(num_cpus=14,  # change to your available number of CPUs
             num_gpus=20,
             include_dashboard=False,
             ignore_reinit_error=True,
             log_to_driver=False)

    training_start_time = get_time()

    new_alg_config = training_alg.config.copy()
    env_config = new_alg_config['env_config']

    n_training_episodes *= env_config['num_Haar_basis'] * env_config['steps_per_Haar']
    update_every_percent = 5
    results = []
    update_interval = n_training_episodes * (update_every_percent / 100)

    for i in range(n_training_episodes):
        results.append(training_alg.train())
        # Print update every x%
        if (i + 1) % int(update_interval) == 0 or (i + 1) == n_training_episodes:
            percent_complete = (i + 1) / n_training_episodes * 100
            print(f"Training Progress: {percent_complete:.0f}% complete")


    training_end_time = get_time()

    training_elapsed_time = training_end_time - training_start_time
    print(f"Training Elapsed time: {training_elapsed_time}\n")

    # return updated_model
    return training_alg, training_elapsed_time

def save_training_results(training_alg, filepath, filename, title_figure, initial_training_date):
    train_env = training_alg.workers.local_worker().env
    env_config = train_env.return_env_config()
    alg_config = training_alg.workers.local_worker().config

    # ---------------------> Save/Plot Training Results <-------------------------
    sr = SaveResults(train_env, training_alg, save_path=filepath)
    save_dir = sr.save_results()
    plot_data(save_dir, plot_filename=filename,
              figure_title=f"[NOISY] Re-Training on {title_figure}", gate_switch_array=None)
    print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    config_table(env_config=env_config, alg_config=alg_config, filepath=filepath, continue_training=True,
                 original_training_date=initial_training_date)


def inference_and_save(inference_list, save_dir, train_alg, n_episodes_for_inferencing):
    columns = ['Fidelity', 'Rewards', 'Actions', 'Self.U_Operator', 'U_target', 'Episode Id']
    inf_count = 0

    for curr_gate in inference_list:
        # train_alg = copy.deepcopy(alg)

        gate_save_dir = save_dir + f'/{curr_gate}/'
        plot_filename = f'inference_{curr_gate}.png'
        os.makedirs(gate_save_dir)

        if inf_count == 0:
            figure_title = f"[NOISY] Inferencing on Multiple Different {str(curr_gate)}."
        else:
            figure_title = f"[NOISY] Multiple Instances of Inferencing on {str(curr_gate)}"

        env_data_title = f"{curr_gate}_"
        transition_history = []

        for inference_iteration in range(n_episodes_for_inferencing):
            # -----------------------> Inferencing <---------------------------
            env = train_alg.workers.local_worker().env
            inference_env, target_gate, history = do_inferencing(env, train_alg, curr_gate)
            transition_history.append(history)

        df = pd.DataFrame(transition_history, columns=columns)
        # df.to_pickle(env_data_title + "env_data.pkl")  # easier to load than csv
        df.to_csv(gate_save_dir + env_data_title + "env_data.csv", index=False)  # backup in case pickle doesn't work
        plot_multiple_visuals(df, figure_title=figure_title, save_dir=gate_save_dir, plot_filename=plot_filename,
                              inf_count=inf_count, gate=curr_gate)

        inf_count += 1


def do_inferencing(env, train_alg, curr_gate):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """

    # Initialize a new environment for inference using this configuration
    inference_env_config = env.return_env_config()
    target_gate = curr_gate.get_matrix()  # Set new target gate for inference

    inference_env_config["U_target"] = target_gate
    inference_env_config['training'] = False
    inference_env_config['retraining'] = False
    inference_env_config['verbose'] = False
    inference_env = NoisySingleQubitEnv(inference_env_config)

    # ------------------------------------------------------------------------------------
    target_gate = np.array(target_gate)

    episode_reward = 0.0
    done = False

    ##Check this out. When we reset it computes the fidelity for the observation between the U_inital (I gate by default) and our new target for the first haar num and step per haar
    obs, info = inference_env.reset()  # Start with the inference environment
    while not done:

        # Compute an action (`a`).
        action = train_alg.compute_single_action(
            observation=obs,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `action` to the env.
        obs, reward, done, truncated, _ = inference_env.step(action)
        episode_reward += reward

        if done:
            return inference_env, target_gate, inference_env.transition_history[-1]


def main():
    original_date = "2025-02-06_19-51-54"
    model_path = "/Users/vishchaudhary/rl-repo/results/" + original_date + "/model_checkpoints"
    save_filepath = "/Users/vishchaudhary/rl-repo/results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

    # retrain_gates = [ gates.X(), gates.Y(), gates.Z()]
    # retrain_gates = [gates.X()]
    retrain_name = "RandomSU2"

    # for gate in retrain_gates:
    #     retrain_name += f"{gate}_"

    training_plot_filename = f'{retrain_name}_retraining.png'

    # Modified to be number of episodes for training (in thousands)
    n_training_iterations = 350
    n_episodes_for_inferencing = 1000

    ##RandomGate must be kept as first in the array and XY_combination MUST be kept as second in the array
    inferencing_gate = [gates.RandomSU2(), gates.XY_combination(), gates.I(), gates.X(), gates.Y(), gates.Z(), gates.H(), gates.S(),
                        gates.X_pi_4()]

    alg = load_model(model_path)

    # Retrained for all specified pauli gates
    # for gate in retrain_gates:
    #     alg = boosted_retraining(alg, gate, n_training_iterations)

    alg, training_time = boosted_retraining(alg, n_training_iterations)

    save_training_results(training_alg=alg, filepath=save_filepath,
                          filename=training_plot_filename, initial_training_date=original_date,
                          title_figure=retrain_name)

    inference_and_save(inference_list=inferencing_gate, save_dir=save_filepath, train_alg=alg,
                       n_episodes_for_inferencing = n_episodes_for_inferencing)

    print(f"Training Elapsed time: \n{training_time}\n")

# Uses the idea of retraining on just the poorly performing Pauli Gates to see if that will boost overall performance
# without hurting the inferencing of the other gates

if __name__ == "__main__":
    main()
