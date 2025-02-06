""" Learning new single qubit gates, rather than the default X gate. """

import ray
from ray.rllib.algorithms.sac import SACConfig
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
from relaqs.api.SAC_callbacks import SACGateSynthesisCallbacks
import os
from datetime import datetime
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from qutip.operators import *
import copy
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import qutip
import torch
from relaqs.api.utils import *

# path_to_relaqs_root = '/Users/vishchaudhary/rl-repo/src/relaqs'
# QUANTUM_NOISE_DATA_DIR = path_to_relaqs_root + "/quantum_noise_data/"
noise_file = "april/ibmq_belem_month_is_4.json"


def run(train_gate, inference_gate, n_training_iterations=1, n_episodes_for_inferencing = 1, save=True, plot=True, noise_file=noise_file, plot_target_change = True):

# 14 CPU Cores
# 20 GPU Cores

    ray.init(num_cpus=14,   # change to your available number of CPUs
        num_gpus=20,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False)
    # print(ray.available_resources())

    # ---------------------> Configure algorithm and Environment <-------------------------
    # Initialize default configuration
    env_config = NoisySingleQubitEnv.get_default_env_config()

    save_filepath = "/Users/vishchaudhary/rl-repo/results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    training_plot_filename = f'training_{train_gate}.png'


    # ---------------------> Get quantum noise data <-------------------------
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)

    # ---------------------> Configure Environment <-------------------------
    env_config["U_target"] = train_gate.get_matrix()
    env_config['num_Haar_basis'] = 1
    env_config['steps_per_Haar'] = 2
    env_config['training'] = True
    total_Haar_nums = env_config["steps_per_Haar"] * env_config["num_Haar_basis"]

    #************************************************************************************************************************************************#
    ###Check what the need for np.reciprocal is bc the default is not like that
    # env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(),
    #                                        np.reciprocal(t2_list).tolist()]  # using real T1 data
    env_config["relaxation_rates_list"] = [t1_list, t2_list]  # using real T1 data
    #************************************************************************************************************************************************#

    env_config["detuning_list"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]
    # env_config["fidelity_threshold"] = 0.7
    # env_config["fidelity_target_switch_case"] = 30
    # env_config["base_target_switch_case"] = 4000
    env_config["verbose"] = False


    # ---------------------> Configure algorithm<-------------------------
    alg_config = SACConfig()
    alg_config.framework("torch")
    alg_config.environment(NoisySingleQubitEnv, env_config=env_config)
    alg_config.callbacks(SACGateSynthesisCallbacks)
    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = env_config["steps_per_Haar"]

    # ---------------------> Tuned Parameters <-------------------------
    # Learning rates for actor and critic
###############################################This is not properly set#########################
    alg_config.actor_lr = 5.057359278283752e-05
    alg_config.critic_lr = 9.959658940947128e-05
    # alg_config.initial_alpha = 0.1  # Starting value for entropy coefficient

    # Hidden layer configurations for policy and Q-value models
    #Actor
    alg_config.policy_model_config = {
        "fcnet_hiddens": [256] * 10,
        "fcnet_activation": "relu",
    }
    #Critic
    alg_config.q_model_config = {
        "fcnet_hiddens": [256] * 10,
        "fcnet_activation": "relu",
    }

    # alg_config.num_steps_sampled_before_learning_starts = 10000

    # Entropy configuration (specific to SAC)
    # Automatically set to auto
    # alg_config.target_entropy = "auto"  # Automatically tune entropy


    # Twin Q-network for stability
    #Already set to true
    alg_config.twin_q = True


    alg = alg_config.build()
    # alg = alg.to(torch.device("mps"))
    # ---------------------------------------------------------------------

    training_start_time = get_time()
    # print(ray.available_resources())
    # ---------------------> Train Agent <-------------------------
    n_training_iterations *= env_config['num_Haar_basis'] * env_config['steps_per_Haar']
    results = [alg.train() for _ in range(n_training_iterations)]
    # -------------------------------------------------------------
    training_end_time = get_time()

    train_env = alg.workers.local_worker().env
    train_alg = alg

    save_dir = None

    # ---------------------> Save/Plot Training Results <-------------------------
    if save and plot is True:
        sr = SaveResults(train_env, alg, save_path=save_filepath)
        save_dir = sr.save_results()
        plot_data(save_dir, plot_filename=training_plot_filename,
                  figure_title=f"[NOISY] Training on {str(train_gate)}", gate_switch_array=None)
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    # config_table(env_config = env_config, alg_config = alg_config, filepath = save_filepath)
    columns = ['Fidelity', 'Rewards', 'Actions', 'Self.U_Operator', 'U_target', 'Episode Id']
    inf_count = 0

    for gate in inference_gate:
        # train_alg = copy.deepcopy(alg)

        gate_save_dir = save_dir+f'/{gate}/'
        plot_filename = f'inference_{gate}.png'
        os.makedirs(gate_save_dir)

        if inf_count == 0:
            figure_title = f"[NOISY] Inferencing on Multiple Different {str(gate)}. (Previously Trained on Multiple {str(train_gate)})"
        else:
            figure_title = f"[NOISY] Multiple Instances of Inferencing on {str(gate)} (Previously Trained on Multiple {str(train_gate)})"
        env_data_title = f"{gate}_"
        transition_history = []

        for inference_iteration in range(n_episodes_for_inferencing):
            # -----------------------> Inferencing <---------------------------
            env = train_alg.workers.local_worker().env
            inference_env, target_gate, history = do_inferencing(env, train_alg, gate)
            transition_history.append(history)

        df = pd.DataFrame(transition_history, columns=columns)
        # df.to_pickle(env_data_title + "env_data.pkl")  # easier to load than csv
        df.to_csv(gate_save_dir + env_data_title + "env_data.csv", index=False)  # backup in case pickle doesn't work
        plot_multiple_visuals(df, figure_title=figure_title, save_dir= gate_save_dir, plot_filename=plot_filename, inf_count = inf_count, gate=gate)

        inf_count += 1

        # ---------------------> Save/Plot Inference Results <-------------------------
        # if plot is True:
        #     plot_data(env_data_title, plot_filename=inferencing_plot_filename,
        #               figure_title=figure_title, modified_inference=True, df = df)
        # --------------------------------------------------------------



    # if plot_target_change:
    #     print(f'Original Switching Array: {original_episodes_target_switch}')
    #     print(f'Adjusted Switching Array: {episodes_target_switch}')
    # else:
    #     print(f'Not plotting when target switches. Number of times target switches: {len(original_episodes_target_switch)}')

    training_elapsed_time = training_end_time - training_start_time
    print(f"Training Elapsed time: {training_elapsed_time}\n")

    ray.shutdown()




def do_inferencing(env, alg, gate):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """

    # Initialize a new environment for inference using this configuration
    inference_env_config = env.return_env_config()
    target_gate = gate.get_matrix()  # Set new target gate for inference

    inference_env_config["U_target"] = target_gate
    inference_env_config['training'] = False
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
        action = alg.compute_single_action(
            observation=obs,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `action` to the env.
        obs, reward, done, truncated, _ = inference_env.step(action)
        episode_reward += reward

        if done:
            return inference_env, target_gate, inference_env.transition_history[-1]


def main():
    # Modified to be number of episodes for training (in thousands)
    n_training_iterations = 10
    n_episodes_for_inferencing = 1000

    save = True
    plot = True
    plot_target_change = False

    train_gate = gates.RandomSU2()

    ##RandomGate must be kept as first in the array
    inferencing_gate = [gates.RandomSU2(), gates.I(), gates.X(), gates.Y(), gates.Z(), gates.H(), gates.S(),
                        gates.X_pi_4()]

    run(train_gate, inferencing_gate, n_training_iterations, n_episodes_for_inferencing, save, plot, noise_file,
        plot_target_change)


if __name__ == "__main__":
    main()